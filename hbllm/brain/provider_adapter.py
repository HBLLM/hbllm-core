"""
Provider LLM Adapter — bridges LLMProvider with brain node interface.

Brain nodes call ``llm.generate(prompt)`` and ``llm.generate_json(prompt)``,
but LLMProvider exposes ``generate(messages)``. This adapter translates
between the two so any provider (OpenAI, Anthropic, Local) works with
the cognitive loop.

Usage::

    from hbllm.serving.provider import get_provider
    from hbllm.brain.provider_adapter import ProviderLLM

    provider = get_provider("openai/gpt-4o-mini")
    llm = ProviderLLM(provider)

    text = await llm.generate("What is 2 + 2?")
    data = await llm.generate_json("Return a JSON with key 'answer'")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from hbllm.serving.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class ProviderLLM:
    """
    Adapter that wraps an LLMProvider to expose the same interface
    as LLMInterface (generate / generate_json / generate_stream).
    
    Brain nodes accept ``llm=`` and call:
      - ``await llm.generate(prompt)`` → str
      - ``await llm.generate_json(prompt)`` → dict
      - ``async for token in llm.generate_stream(prompt)``

    This class makes any LLMProvider (OpenAI, Anthropic, Local) compatible.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._call_count = 0

    @property
    def usage(self) -> dict[str, int]:
        """Accumulated token usage stats."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "call_count": self._call_count,
        }

    def _build_messages(
        self,
        prompt: str,
        system_override: str | None = None,
    ) -> list[dict[str, str]]:
        """Convert a prompt string into chat messages."""
        system = system_override or self.system_prompt
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def _track_usage(self, response: LLMResponse) -> None:
        """Accumulate token usage from response."""
        self._total_prompt_tokens += response.usage.get("prompt_tokens", 0)
        self._total_completion_tokens += response.usage.get("completion_tokens", 0)
        self._call_count += 1

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate free-form text from the provider.

        Args:
            prompt: The input prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        messages = self._build_messages(prompt)
        response = await self.provider.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._track_usage(response)
        return response.content

    async def generate_json(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """
        Generate structured JSON from the provider.

        Appends a JSON instruction to the prompt, calls the provider,
        and extracts the first valid JSON object from the response.

        Args:
            prompt: Instruction describing the desired JSON output.
            max_tokens: Maximum tokens.

        Returns:
            Parsed JSON dict. Returns {"error": "..."} on failure.
        """
        json_prompt = f"{prompt}\n\nRespond with ONLY a valid JSON object, no other text."

        messages = self._build_messages(
            json_prompt,
            system_override="You are a precise AI that always responds with valid JSON only. No markdown, no explanation.",
        )

        response = await self.provider.generate(
            messages,
            max_tokens=max_tokens,
            temperature=0.3,  # Low temp for structured output
        )
        self._track_usage(response)
        return self._extract_json(response.content)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from the provider.

        Yields tokens one at a time. Falls back to yielding the full
        response as a single chunk if the provider doesn't support streaming.
        """
        messages = self._build_messages(prompt)
        async for token in self.provider.stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield token

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """
        Extract the first valid JSON object from LLM output.

        Handles markdown fences, trailing commas, and other LLM quirks.
        """
        # Try JSON in code fences
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try full/nested JSON (greedy — handles objects within objects)
        deep_match = re.search(r"\{.*\}", text, re.DOTALL)
        if deep_match:
            try:
                return json.loads(deep_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try shallow JSON (non-greedy — handles simple objects)
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try the entire text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("[ProviderLLM] Failed to extract JSON from: %s", text[:100])
            return {"error": "Failed to parse structured output", "raw": text[:200]}
