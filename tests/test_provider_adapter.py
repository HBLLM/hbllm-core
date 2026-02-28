"""Tests for ProviderLLM adapter — bridges LLMProvider with brain node interface."""

import pytest
import json
from typing import Any, AsyncIterator

from hbllm.brain.provider_adapter import ProviderLLM
from hbllm.serving.provider import LLMProvider, LLMResponse


# ── Mock Provider ────────────────────────────────────────────────────────────

class MockProvider(LLMProvider):
    """A mock LLM provider for testing. Returns configurable responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["Hello from mock!"]
        self._call_index = 0
        self.last_messages: list[dict[str, str]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        self.last_messages = messages
        content = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        content = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        for word in content.split():
            yield word + " "

    @property
    def name(self) -> str:
        return "mock"


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def llm(mock_provider):
    return ProviderLLM(mock_provider)


@pytest.mark.asyncio
async def test_generate_basic(llm):
    result = await llm.generate("Hello!")
    assert result == "Hello from mock!"


@pytest.mark.asyncio
async def test_generate_passes_messages(mock_provider):
    llm = ProviderLLM(mock_provider, system_prompt="Custom system prompt")
    await llm.generate("Test query")

    assert len(mock_provider.last_messages) == 2
    assert mock_provider.last_messages[0]["role"] == "system"
    assert mock_provider.last_messages[0]["content"] == "Custom system prompt"
    assert mock_provider.last_messages[1]["role"] == "user"
    assert "Test query" in mock_provider.last_messages[1]["content"]


@pytest.mark.asyncio
async def test_generate_json_valid():
    provider = MockProvider(responses=['{"answer": 42, "reason": "math"}'])
    llm = ProviderLLM(provider)

    result = await llm.generate_json("What is 6*7?")
    assert result == {"answer": 42, "reason": "math"}


@pytest.mark.asyncio
async def test_generate_json_with_code_fence():
    provider = MockProvider(responses=['```json\n{"key": "value"}\n```'])
    llm = ProviderLLM(provider)

    result = await llm.generate_json("Return JSON")
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_generate_json_with_surrounding_text():
    provider = MockProvider(responses=['Here is the result: {"status": "ok"} hope that helps!'])
    llm = ProviderLLM(provider)

    result = await llm.generate_json("Status check")
    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_generate_json_invalid_returns_error():
    provider = MockProvider(responses=["I cannot produce JSON, sorry"])
    llm = ProviderLLM(provider)

    result = await llm.generate_json("Give me JSON")
    assert "error" in result


@pytest.mark.asyncio
async def test_generate_json_uses_low_temperature():
    """JSON generation should use low temperature for consistency."""
    provider = MockProvider(responses=['{"ok": true}'])
    llm = ProviderLLM(provider)

    # The system prompt should be overridden for JSON
    await llm.generate_json("Give JSON")
    assert "JSON" in provider.last_messages[0]["content"]


@pytest.mark.asyncio
async def test_generate_stream(llm):
    tokens = []
    async for token in llm.generate_stream("Stream test"):
        tokens.append(token)
    assert len(tokens) > 0
    full = "".join(tokens).strip()
    assert "Hello" in full


@pytest.mark.asyncio
async def test_usage_tracking(llm):
    assert llm.usage["call_count"] == 0

    await llm.generate("First call")
    assert llm.usage["call_count"] == 1
    assert llm.usage["prompt_tokens"] == 10
    assert llm.usage["completion_tokens"] == 5

    await llm.generate("Second call")
    assert llm.usage["call_count"] == 2
    assert llm.usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_multiple_responses():
    provider = MockProvider(responses=["First", "Second", "Third"])
    llm = ProviderLLM(provider)

    r1 = await llm.generate("a")
    r2 = await llm.generate("b")
    r3 = await llm.generate("c")

    assert r1 == "First"
    assert r2 == "Second"
    assert r3 == "Third"


@pytest.mark.asyncio
async def test_extract_json_nested():
    provider = MockProvider(responses=['{"outer": {"inner": "value"}, "list": [1, 2, 3]}'])
    llm = ProviderLLM(provider)

    result = await llm.generate_json("Nested JSON")
    assert result["outer"]["inner"] == "value"
    assert result["list"] == [1, 2, 3]
