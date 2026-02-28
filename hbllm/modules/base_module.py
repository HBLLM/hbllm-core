"""
Abstract Domain Module Node.

Each domain specialization (General, Coding, Math) is a Node that wraps
the shared base LLM but dynamically activates its specific LoRA adapter 
before generating text.
"""

from __future__ import annotations

import logging
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
        model: torch.nn.Module,
        tokenizer: Any,
        lora_state_dict: dict[str, torch.Tensor] | None = None,
        capabilities: list[str] | None = None,
    ):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=capabilities)
        self.domain_name = domain_name
        self.model = model
        self.tokenizer = tokenizer
        self.lora_state_dict = lora_state_dict
        
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
        except Exception as e:
            return message.create_error(f"Invalid QueryPayload: {e}")

        prompt = payload.get("text", "")

        domain_hint = payload.get("domain_hint", "general")
        if domain_hint != self.domain_name and self.domain_name != "general":
            # To save compute, only the targeted expert or the general fallback thinks about this.
            return None

        logger.info("Domain '%s' generating response for prompt: %s...", self.domain_name, prompt[:30])

        # 1. Hot-swap the LoRA adapter
        # In a real multi-threaded environment where the base model is shared, 
        # this would need a strict lock. For prototype async queues, this is safe.
        if self.lora_state_dict is not None:
            LoRAManager.load_lora_state_dict(self.model, self.lora_state_dict)
            LoRAManager.set_active(self.model, True)
        else:
            # If no LoRA (e.g. baseline General model), deactivate adapter path
            LoRAManager.set_active(self.model, False)

        # 2. Tokenize and Generate
        # We simulate a continuous batching queue by leveraging PyTorch KV caching
        # incrementally, yielding early rather than blocking.
        try:
            import asyncio
            device = next(self.model.parameters()).device
            
            async def _generate_async() -> str:
                enc = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([enc], dtype=torch.long).to(device)
                
                self.model.eval()
                out_tokens = input_ids[0].tolist()
                past_key_values = None
                
                with torch.no_grad():
                    # Generate 30 tokens using cached autoregressive steps
                    for _ in range(30):
                        # Only pass the last decoded token to the model if caching
                        model_input = input_ids[:, -1:] if past_key_values else input_ids
                        
                        outputs = self.model(
                            model_input,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                        
                        logits = outputs["logits"][:, -1, :]
                        past_key_values = outputs.get("past_key_values")
                        
                        next_token = logits.argmax().item()
                        out_tokens.append(next_token)
                        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
                        
                        # Yield to the asyncio event loop to allow other requests to be processed
                        # in a true vLLM setup, this would schedule the next batch sequence
                        await asyncio.sleep(0.001) 
                
                return self.tokenizer.decode_to_string(out_tokens)

            response_text = await _generate_async()
            logger.info("Domain '%s' finished generating.", self.domain_name)
            
            # 3. Propose thought to Blackboard instead of creating a synchronous response
            thought_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": f"intuition_{self.domain_name}",
                    "confidence": 0.8, # Base LLM confidence
                    "content": response_text
                },
                correlation_id=message.correlation_id
            )
            await self.bus.publish("workspace.thought", thought_msg)
            return None
            
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return None
