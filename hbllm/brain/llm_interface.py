"""
Shared LLM Inference Interface.

Provides a unified, async-safe API for all cognitive nodes to call the
base language model for text generation and structured JSON extraction.
This replaces all previous hardcoded mock/stub logic throughout the system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import torch

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Wraps the shared HBLLMForCausalLM model + tokenizer for reusable inference.

    All cognitive nodes (LogicNode, FuzzyNode, CriticNode, etc.) call this
    instead of implementing their own generation loops.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        tenant_id: str | None = None,
    ) -> str:
        """
        Generate free-form text from the model given a prompt.

        Args:
            prompt: The input prompt string.
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more creative).

        Returns:
            The generated text string (excluding the prompt).
        """
        return await asyncio.to_thread(
            self._generate_sync, prompt, max_tokens, temperature, tenant_id
        )

    async def generate_json(
        self, prompt: str, max_tokens: int = 256, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """
        Generate structured JSON from the model.

        Appends a JSON instruction suffix to the prompt, generates text,
        and extracts the first valid JSON object from the output.

        Args:
            prompt: The instruction prompt (should describe the desired JSON schema).
            max_tokens: Maximum tokens for the response.

        Returns:
            Parsed JSON dict. Returns {"error": "..."} if parsing fails.
        """
        json_prompt = f"{prompt}\n\nRespond with ONLY a valid JSON object, no other text."
        raw_output = await self.generate(
            json_prompt, max_tokens=max_tokens, temperature=0.3, tenant_id=tenant_id
        )
        return self._extract_json(raw_output)

    def _generate_sync(
        self, prompt: str, max_tokens: int, temperature: float, tenant_id: str | None = None
    ) -> str:
        """Synchronous generation with KV-cache autoregressive decoding."""
        tokens = list(self._generate_stream_sync(prompt, max_tokens, temperature, tenant_id))
        return "".join(tokens)

    def _generate_stream_sync(
        self, prompt: str, max_tokens: int, temperature: float, tenant_id: str | None = None
    ) -> Iterator[str]:
        """Synchronous token-by-token generator with KV-cache decoding."""
        enc = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([enc], dtype=torch.long).to(self.device)

        # Dynamic LoRA Adapter Multiplexing
        if tenant_id and hasattr(self.model, "set_adapter"):
            try:
                self.model.set_adapter(tenant_id)
            except ValueError:
                # Fallback to base model if tenant has no specific LoRA loaded
                try:
                    self.model.set_adapter("default")
                except ValueError:
                    pass

        try:
            self.model.eval()
            past_key_values = None

            with torch.no_grad():
                for _ in range(max_tokens):
                    model_input = input_ids[:, -1:] if past_key_values else input_ids

                    outputs = self.model(
                        model_input, past_key_values=past_key_values, use_cache=True
                    )

                    logits = outputs["logits"][:, -1, :]
                    past_key_values = outputs.get("past_key_values")

                    # Temperature-scaled sampling
                    if temperature > 0:
                        scaled = logits / temperature
                        probs = torch.softmax(scaled, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token = logits.argmax().item()

                    input_ids = torch.cat(
                        [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
                    )

                    # Decode only the new token and yield it
                    token_text = self.tokenizer.decode_to_string([next_token])
                    yield token_text
        finally:
            if hasattr(self.model, "set_adapter"):
                try:
                    self.model.set_adapter("default")
                except ValueError:
                    pass

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        tenant_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields tokens one-at-a-time during decoding.

        Usage:
            async for token in llm.generate_stream("Hello"):
                print(token, end="", flush=True)
        """
        import queue
        import threading

        token_queue: queue.Queue[str | None] = queue.Queue()

        def _run() -> None:
            try:
                for token in self._generate_stream_sync(prompt, max_tokens, temperature, tenant_id):
                    token_queue.put(token)
            finally:
                token_queue.put(None)  # Sentinel

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            # Poll the queue without blocking the event loop
            try:
                token = await asyncio.to_thread(token_queue.get, timeout=30.0)
            except Exception:
                break
            if token is None:
                break
            yield token

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """
        Extract the first valid JSON object from LLM output.

        Handles common LLM quirks: markdown code fences, trailing commas, etc.
        """
        # Try to find JSON inside code fences first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return cast(dict[str, Any], json.loads(fence_match.group(1)))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return cast(dict[str, Any], json.loads(brace_match.group(0)))
            except json.JSONDecodeError:
                pass

        # Try the entire text
        try:
            return cast(dict[str, Any], json.loads(text.strip()))
        except json.JSONDecodeError:
            logger.warning("[LLMInterface] Failed to extract JSON from: %s", text[:100])
            return {"error": "Failed to parse structured output from LLM", "raw": text[:200]}
