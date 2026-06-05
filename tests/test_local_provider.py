"""Tests for LocalProvider's template formatting and tokenization integration."""

import pytest
import torch

from hbllm.model.tokenizer import HBLLMTokenizer
from hbllm.serving.provider import LocalProvider


class MockModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, input_ids, **kwargs):
        # Returns the prompt tensor with one extra token appended
        return torch.cat([input_ids, torch.tensor([[999]], dtype=torch.long)], dim=1)


class MockHFTokenizer:
    def __init__(self):
        self.eos_id = 999
        self.eos_token_id = 999

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        if tokenize:
            return [101, 102, 103]
        return "mock_formatted_chat"

    def encode(self, text, **kwargs):
        if text == "mock_formatted_chat":
            return [101, 102, 103]
        raise ValueError(f"Unexpected encode input: {text}")

    def decode(self, token_ids, **kwargs):
        return "mock_decoded_response"


@pytest.mark.asyncio
async def test_local_provider_with_native_tokenizer():
    """Test LocalProvider with the native HBLLMTokenizer (returns string, no tokenize arg)."""
    model = MockModel()
    tokenizer = HBLLMTokenizer()
    provider = LocalProvider(model=model, tokenizer=tokenizer)

    messages = [{"role": "user", "content": "Hi"}]
    response = await provider.generate(messages)
    assert response.content is not None
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] == 1


@pytest.mark.asyncio
async def test_local_provider_with_hf_tokenizer():
    """Test LocalProvider with a mock HuggingFace tokenizer (accepts tokenize arg)."""
    model = MockModel()
    tokenizer = MockHFTokenizer()
    provider = LocalProvider(model=model, tokenizer=tokenizer)

    messages = [{"role": "user", "content": "Hi"}]
    response = await provider.generate(messages)
    assert response.content == "mock_decoded_response"
    assert response.usage["prompt_tokens"] == 3
    assert response.usage["completion_tokens"] == 1
