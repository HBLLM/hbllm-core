"""
Abstract Domain Module Node.

Each domain specialization (General, Coding, Math) is a Node that wraps
the shared base LLM but dynamically activates its specific LoRA adapter
before generating text.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import torch

from hbllm.modules.lora import LoRAManager
from hbllm.network.messages import Message, MessageType, QueryPayload
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class DomainModuleNode(Node):
    """
    A network node that provides Domain-specific LLM inference.

    It listens to the bus for `domain.{name}.query` messages, activates
    its LoRA adapter on the shared base model, generates a response,
    and publishes it back.
    """

    def __init__(
        self,
        node_id: str,
        domain_name: str,
        model: torch.nn.Module | None = None,
        tokenizer: Any = None,
        lora_state_dict: dict[str, torch.Tensor] | None = None,
        capabilities: list[str] | None = None,
        llm: Any = None,
    ):
        super().__init__(
            node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=capabilities
        )
        self.domain_name = domain_name
        self.model = model
        self.tokenizer = tokenizer
        self.llm = llm

        self.has_lora = lora_state_dict is not None and self.model is not None
        if self.has_lora:
            logger.info("DomainModuleNode '%s' registering LoRA adapter...", self.domain_name)
            LoRAManager.add_adapter(self.model, self.domain_name, lora_state_dict)

        self.topic_sub = "module.evaluate"

    async def on_start(self) -> None:
        """Subscribe to domain query messages."""
        logger.info("Starting DomainModuleNode for domain '%s'", self.domain_name)
        await self.bus.subscribe(self.topic_sub, self.handle_message)

    async def on_stop(self) -> None:
        """Clean up."""
        logger.info("Stopping DomainModuleNode '%s'", self.domain_name)

    async def handle_message(self, message: Message) -> Message | None:
        """Process incoming evaluation requests from the Workspace Blackboard."""
        if message.topic != self.topic_sub:
            return None

        # Parse payload
        try:
            payload = QueryPayload(**message.payload)
        except (TypeError, ValueError, KeyError) as e:
            return message.create_error(f"Invalid QueryPayload: {e}")

        # Prefer augmented_prompt (with memory context + history) for inference,
        # fall back to raw text if not provided (e.g. for greeting fast-paths)
        payload_metadata = getattr(payload, "metadata", {}) or {}
        prompt = payload_metadata.get("augmented_prompt") or getattr(payload, "text", "")
        domain_hint = getattr(payload, "domain_hint", "general")

        # Domain eligibility check (supports hierarchical sub-domains)
        is_targeted = False
        if isinstance(domain_hint, dict):
            # Weighted MoE: check if this domain (or ancestor) appears in the blend
            for hint_domain in domain_hint:
                if (
                    self.domain_name == hint_domain
                    or hint_domain.startswith(self.domain_name + ".")
                    or self.domain_name.startswith(hint_domain + ".")
                ):
                    is_targeted = True
                    break
            # For MoE, only the highest-weighted matching domain runs inference
            if is_targeted:
                matching = {
                    d: w
                    for d, w in domain_hint.items()
                    if d == self.domain_name
                    or d.startswith(self.domain_name + ".")
                    or self.domain_name.startswith(d + ".")
                }
                if matching:
                    best = max(matching.items(), key=lambda x: x[1])[0]
                    # Only the closest matching domain should actually run
                    is_targeted = best == self.domain_name or best.startswith(
                        self.domain_name + "."
                    )
                    if is_targeted:
                        logger.info(
                            "Domain '%s' elected for MoE blend %s",
                            self.domain_name,
                            domain_hint,
                        )
        elif isinstance(domain_hint, str):
            # Hierarchical: "coding" matches "coding.python", and vice versa
            is_targeted = (
                domain_hint == self.domain_name
                or domain_hint.startswith(self.domain_name + ".")
                or self.domain_name.startswith(domain_hint + ".")
                or self.domain_name == "general"
            )

        if not is_targeted:
            return None

        if self.llm is not None:
            logger.info(
                "Domain '%s' generating response using external LLM for prompt: %s...",
                self.domain_name,
                prompt[:30],
            )
            try:
                from hbllm.brain.prompt_helper import get_dynamic_system_prompt

                tenant = message.tenant_id or "default"
                system_prompt = await get_dynamic_system_prompt(self.bus, tenant, self.node_id)

                start_time = time.monotonic()
                response_text = await self.llm.generate(
                    prompt, max_tokens=1024, system_prompt=system_prompt
                )
                latency_ms = int((time.monotonic() - start_time) * 1000)
                logger.info(
                    "Domain '%s' finished generating using external LLM in %d ms.",
                    self.domain_name,
                    latency_ms,
                )

                thought_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=message.tenant_id,
                    session_id=message.session_id,
                    topic="workspace.thought",
                    payload={
                        "type": f"intuition_{self.domain_name}",
                        "confidence": 0.8,  # Base LLM confidence
                        "content": response_text,
                        "metrics": {
                            "latency_ms": latency_ms,
                            "tokens_generated": len(response_text) // 4,
                            "domain": self.domain_name,
                        },
                    },
                    correlation_id=message.correlation_id,
                )
                await self.bus.publish("workspace.thought", thought_msg)
                return None
            except Exception as e:
                logger.error(
                    "External LLM generation failed for domain '%s': %s", self.domain_name, e
                )
                return None

        model = self.model
        tokenizer = self.tokenizer

        if model is None or tokenizer is None:
            logger.error(
                "DomainModuleNode '%s' has no model/tokenizer nor external LLM configured.",
                self.domain_name,
            )
            return None

        logger.info(
            "Domain '%s' generating response for prompt: %s...", self.domain_name, prompt[:30]
        )

        try:
            # 1. Page in LoRA (Asynchronous PCIe VRAM transfer) and set ContextVar (O(1) Lock-Free mapping)
            if self.has_lora:
                LoRAManager.page_in(model, domain_hint)
                LoRAManager.set_active_adapter(model, domain_hint)
            else:
                LoRAManager.set_active_adapter(model, None)

            # 2. Tokenize and Generate
            device = next(model.parameters()).device

            async def _generate_async() -> str:
                enc = tokenizer.encode(prompt)
                if hasattr(enc, "ids"):
                    enc = enc.ids
                input_ids = torch.tensor([enc], dtype=torch.long).to(device)

                model.eval()
                out_tokens = input_ids[0].tolist()
                past_key_values = None

                # with torch.no_grad() is thread-local. Awaiting inside it leaks the context to other coroutines!
                # Generate 30 tokens using cached autoregressive steps
                for _ in range(30):
                    # Only pass the last decoded token to the model if caching
                    model_input = input_ids[:, -1:] if past_key_values else input_ids

                    with torch.no_grad():
                        outputs = model(
                            model_input, past_key_values=past_key_values, use_cache=True
                        )

                    logits = outputs["logits"][:, -1, :]
                    past_key_values = outputs.get("past_key_values")

                    next_token = logits.argmax().item()
                    out_tokens.append(next_token)
                    input_ids = torch.cat(
                        [input_ids, torch.tensor([[next_token]], device=device)], dim=1
                    )

                    # Yield to asyncio to allow other DomainModuleNodes to compute their own tokens concurrently!
                    await asyncio.sleep(0.001)

                if hasattr(tokenizer, "decode_to_string"):
                    return str(tokenizer.decode_to_string(out_tokens))
                else:
                    return str(tokenizer.decode(out_tokens, skip_special_tokens=True))

            start_time = time.monotonic()
            response_text = await _generate_async()
            latency_ms = int((time.monotonic() - start_time) * 1000)
            logger.info("Domain '%s' finished generating in %d ms.", self.domain_name, latency_ms)

            # 3. Propose thought to Blackboard instead of creating a synchronous response
            thought_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": f"intuition_{self.domain_name}",
                    "confidence": 0.8,  # Base LLM confidence
                    "content": response_text,
                    "metrics": {
                        "latency_ms": latency_ms,
                        "tokens_generated": 30,
                        "domain": self.domain_name,
                    },
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", thought_msg)
            return None

        except (
            RuntimeError,
            torch.cuda.CudaError if hasattr(torch, "cuda") else RuntimeError,
            ValueError,
        ) as e:
            logger.error("Domain generation failed: %s", e)
            return None

        finally:
            if self.has_lora:
                LoRAManager.page_out(model, domain_hint)
