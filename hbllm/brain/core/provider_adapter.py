"""
Provider LLM Adapter — bridges LLMProvider with brain node interface.

Brain nodes call ``llm.generate(prompt)`` and ``llm.generate_json(prompt)``,
but LLMProvider exposes ``generate(messages)``. This adapter translates
between the two so any provider (OpenAI, Anthropic, Local) works with
the cognitive loop.

Usage::

    from hbllm.serving.provider import get_provider
    from hbllm.brain.core.provider_adapter import ProviderLLM

    provider = get_provider("openai/gpt-4o-mini")
    llm = ProviderLLM(provider)

    text = await llm.generate("What is 2 + 2?")
    data = await llm.generate_json("Return a JSON with key 'answer'")
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from hbllm.serving.provider import LLMProvider, LLMResponse

from hbllm.runtime.providers.capability import ProviderCapability
from hbllm.runtime.providers.cognition import CognitionRequest, ThoughtResult

logger = logging.getLogger(__name__)


class ProviderLLM:
    """
    Adapter that wraps an LLMProvider to expose the same interface
    as LLMInterface (generate / generate_json / generate_stream) while also
    conforming to the unified ``CognitionProvider`` protocol.

    Brain nodes accept ``llm=`` and call:
      - ``await llm.generate(prompt)`` → str
      - ``await llm.generate_json(prompt)`` → dict
      - ``async for token in llm.generate_stream(prompt)``
      - ``await llm.reason(cognition_request)`` → ThoughtResult

    This class makes any LLMProvider (OpenAI, Anthropic, Local, Ollama) compatible.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = (
            "You are Sentra, an advanced cognitive AI assistant powered by the HBLLM modular architecture. "
            "You have access to various cognitive and tool modules, including a BrowserNode (which allows "
            "you to browse the web and search for real-time information), an ExecutionNode (for running "
            "Python code in a secure sandbox), a LogicNode (powered by Z3 for symbolic reasoning), and a "
            "persistent memory node. Be helpful, precise, and accurate."
        ),
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._call_count = 0

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest conforming to Unified Cognition Provider."""
        provider_name = getattr(self.provider, "name", "llm_provider")
        return ProviderCapability(
            provider_id=f"provider_llm_{provider_name}",
            provider_type="cognition",
            capabilities=["text_reasoning", "planning", "structured_json", "streaming"],
            modalities=["text"],
            latency_profile="medium",
            quality_profile="high",
            max_input_tokens=8192,
        )

    def to_cognition_adapter(self) -> Any:
        """Convert this ProviderLLM into a unified LLMCognitionAdapter."""
        from hbllm.runtime.adapters.cognition.llm_adapter import LLMCognitionAdapter

        return LLMCognitionAdapter(
            provider_id=f"cognition_{getattr(self.provider, 'name', 'llm')}",
            underlying_provider=self.provider,
        )

    async def reason(self, request: CognitionRequest) -> ThoughtResult:
        """Execute reasoning over a structured CognitionRequest."""
        adapter = self.to_cognition_adapter()
        return await adapter.reason(request)

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
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate free-form text from the provider.

        Args:
            prompt: The input prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            system_prompt: Optional override for the system prompt.

        Returns:
            Generated text string.
        """
        messages = self._build_messages(prompt, system_override=system_prompt)
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
        max_tokens: int = 64,
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
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from the provider.

        Yields tokens one at a time. Falls back to yielding the full
        response as a single chunk if the provider doesn't support streaming.
        """
        messages = self._build_messages(prompt, system_override=system_prompt)
        async for token in self.provider.stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield token

    @staticmethod
    def _parse_json_str(raw: str) -> dict[str, Any]:
        """Parse JSON string using orjson with stdlib json fallback."""
        try:
            import orjson

            return cast(dict[str, Any], orjson.loads(raw))
        except (ImportError, Exception):
            return cast(dict[str, Any], json.loads(raw))

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
                return ProviderLLM._parse_json_str(fence_match.group(1))
            except (json.JSONDecodeError, Exception):
                pass

        # Try full/nested JSON (greedy — handles objects within objects)
        deep_match = re.search(r"\{.*\}", text, re.DOTALL)
        if deep_match:
            try:
                return ProviderLLM._parse_json_str(deep_match.group(0))
            except (json.JSONDecodeError, Exception):
                pass

        # Try shallow JSON (non-greedy — handles simple objects)
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return ProviderLLM._parse_json_str(brace_match.group(0))
            except (json.JSONDecodeError, Exception):
                pass

        # Try the entire text
        try:
            return ProviderLLM._parse_json_str(text.strip())
        except (json.JSONDecodeError, Exception):
            logger.warning("[ProviderLLM] Failed to extract JSON from: %s", text[:100])
            return {"error": "Failed to parse structured output", "raw": text[:200]}
