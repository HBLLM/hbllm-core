"""
MockProvider — A deterministic LLM provider for testing.

Returns predictable responses without network calls, enabling tests
to exercise BrainFactory.create() and SentraAgent.start() without
API keys or internet access.

Usage::

    from hbllm.testing import MockProvider
    provider = MockProvider(default_response="Hello!")
    brain = await BrainFactory.create(provider=provider)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from hbllm.serving.provider import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """
    Deterministic mock LLM provider for testing.

    All calls return the ``default_response`` text and accumulate
    call history in ``calls`` for assertion.
    """

    def __init__(
        self,
        default_response: str = "Mock response.",
        model: str = "mock/test",
    ):
        self._default_response = default_response
        self._model = model
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(
            {
                "method": "generate",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return LLMResponse(
            content=self._default_response,
            model=self._model,
            usage={
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(self._default_response.split()),
                "total_tokens": 0,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.calls.append(
            {
                "method": "stream",
                "messages": messages,
            }
        )
        for word in self._default_response.split():
            yield word + " "

    @property
    def name(self) -> str:
        return self._model
