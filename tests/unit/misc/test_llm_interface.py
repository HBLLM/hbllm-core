"""Tests for LLMInterface.

Validates the async and sync inference wrappers, JSON extraction,
and dynamic LoRA adapter loading.
"""

from __future__ import annotations

import pytest

from hbllm.brain.core.llm_interface import LLMInterface


class MockTokenizer:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=True):
        return "mock_token"


class MockModel:
    def __init__(self):
        self.device = "cpu"
        self.adapter_calls = []
        self._current_adapter = "default"

    def parameters(self):
        class MockParam:
            device = "cpu"

        yield MockParam()

    def eval(self):
        pass

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        import torch

        batch_size = input_ids.shape[0]
        # Mock logits shape: [batch, seq, vocab]
        logits = torch.randn(batch_size, 1, 100)
        return {"logits": logits, "past_key_values": "mock_cache"}

    def set_adapter(self, adapter_name):
        if adapter_name == "invalid":
            raise ValueError(f"Adapter {adapter_name} not found")
        self.adapter_calls.append(adapter_name)
        self._current_adapter = adapter_name


@pytest.fixture
def mock_llm():
    return LLMInterface(model=MockModel(), tokenizer=MockTokenizer(), device="cpu")


@pytest.mark.asyncio
class TestLLMInterfaceGeneration:
    async def test_generate_text(self, mock_llm):
        """Standard text generation should work."""
        res = await mock_llm.generate("Hello", max_tokens=3, temperature=0.0)
        assert res == "mock_tokenmock_tokenmock_token"

    async def test_generate_json(self, mock_llm):
        """JSON generation should inject instruction and extract JSON."""

        # Override generate to return a mock JSON string
        async def _mock_gen(prompt, max_tokens, temperature, tenant_id):
            return '```json\n{"status": "success", "score": 9}\n```'

        mock_llm.generate = _mock_gen
        res = await mock_llm.generate_json("Evaluate this", max_tokens=10)
        assert "status" in res
        assert res["score"] == 9

    async def test_generate_stream(self, mock_llm):
        """Streaming should yield tokens async."""
        tokens = []
        async for token in mock_llm.generate_stream("Stream this", max_tokens=2):
            tokens.append(token)
        assert len(tokens) == 2
        assert tokens == ["mock_token", "mock_token"]


class TestLLMInterfaceJSONExtraction:
    def test_extract_json_markdown_block(self):
        """Should extract JSON from markdown fences."""
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        res = LLMInterface._extract_json(text)
        assert res["key"] == "value"

    def test_extract_json_raw_braces(self):
        """Should extract JSON from raw text containing braces."""
        text = 'The answer is {"answer": 42} and nothing else.'
        res = LLMInterface._extract_json(text)
        assert res["answer"] == 42

    def test_extract_json_invalid(self):
        """Should return error dict if no valid JSON found."""
        text = "This is just plain text with no JSON."
        res = LLMInterface._extract_json(text)
        assert "error" in res
        assert "raw" in res


@pytest.mark.asyncio
class TestLLMInterfaceAdapterLoading:
    async def test_loads_tenant_adapter(self, mock_llm):
        """Should load tenant specific adapter before generation and reset after."""
        await mock_llm.generate("Hello", max_tokens=1, tenant_id="tenant_123")
        assert "tenant_123" in mock_llm.model.adapter_calls
        # Should reset to default at the end
        assert mock_llm.model.adapter_calls[-1] == "default"

    async def test_fallback_to_default_adapter(self, mock_llm):
        """Should fallback to default if tenant adapter is missing."""
        await mock_llm.generate("Hello", max_tokens=1, tenant_id="invalid")
        # Attempted invalid, caught ValueError, fallback to default, then generate, then reset to default
        assert mock_llm.model.adapter_calls == ["default", "default"]


@pytest.mark.asyncio
async def test_provider_missing_api_key_validation():
    import os

    from hbllm.serving.provider import AnthropicProvider, GroqProvider, OpenAIProvider

    # Force empty env keys
    orig_openai = os.environ.pop("OPENAI_API_KEY", None)
    orig_groq = os.environ.pop("GROQ_API_KEY", None)
    orig_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)

    try:
        # 1. Groq Provider
        groq = GroqProvider(api_key="")
        with pytest.raises(ValueError, match="API key is not set"):
            await groq.generate([{"role": "user", "content": "hi"}])
        with pytest.raises(ValueError, match="API key is not set"):
            async for _ in groq.stream([{"role": "user", "content": "hi"}]):
                pass

        # 2. OpenAI Provider with default URL
        openai = OpenAIProvider(api_key="")
        with pytest.raises(ValueError, match="API key is not set"):
            await openai.generate([{"role": "user", "content": "hi"}])
        with pytest.raises(ValueError, match="API key is not set"):
            async for _ in openai.stream([{"role": "user", "content": "hi"}]):
                pass

        # 3. Anthropic Provider
        anthropic = AnthropicProvider(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            await anthropic.generate([{"role": "user", "content": "hi"}])
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            async for _ in anthropic.stream([{"role": "user", "content": "hi"}]):
                pass

    finally:
        # Restore original env variables
        if orig_openai is not None:
            os.environ["OPENAI_API_KEY"] = orig_openai
        if orig_groq is not None:
            os.environ["GROQ_API_KEY"] = orig_groq
        if orig_anthropic is not None:
            os.environ["ANTHROPIC_API_KEY"] = orig_anthropic
