"""
Tests for hbllm.model.tokenizer — unified tokenizer with chat templates.
"""
from hbllm.model.tokenizer import (
    HBLLMTokenizer,
    BOS, EOS, PAD, SYSTEM, USER, ASSISTANT,
    SPECIAL_TOKENS,
)


class TestSpecialTokens:
    """Tests for special token definitions."""

    def test_special_tokens_exist(self):
        assert BOS
        assert EOS
        assert PAD
        assert SYSTEM
        assert USER
        assert ASSISTANT

    def test_special_tokens_are_unique(self):
        assert len(set(SPECIAL_TOKENS)) == len(SPECIAL_TOKENS)

    def test_special_tokens_list_complete(self):
        assert len(SPECIAL_TOKENS) == 6


class TestHBLLMTokenizer:
    """Tests for HBLLMTokenizer."""

    def setup_method(self):
        # Creates tokenizer in tiktoken fallback mode
        self.tok = HBLLMTokenizer()

    def test_encode_decode_roundtrip(self):
        text = "Hello, world! This is a test."
        ids = self.tok.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        decoded = self.tok.decode(ids)
        assert text in decoded or decoded.strip() == text.strip()

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.encode("Testing encoding")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_empty_string(self):
        ids = self.tok.encode("")
        assert isinstance(ids, list)

    def test_vocab_size(self):
        assert self.tok.vocab_size > 0

    def test_chat_template(self):
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = self.tok.apply_chat_template(messages)
        assert isinstance(result, str)
        assert "Hello!" in result
        assert "Hi there!" in result

    def test_chat_template_with_system(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = self.tok.apply_chat_template(messages)
        assert "helpful assistant" in result
        assert "2+2" in result

    def test_chat_template_empty(self):
        result = self.tok.apply_chat_template([])
        assert isinstance(result, str)

    def test_encode_with_special_tokens(self):
        ids = self.tok.encode("Hello", add_bos=True, add_eos=True)
        assert isinstance(ids, list)
        assert len(ids) >= 1  # at minimum the token(s) for "Hello"

    def test_from_tiktoken(self):
        tok = HBLLMTokenizer.from_tiktoken()
        assert tok.vocab_size > 0
        ids = tok.encode("test")
        assert len(ids) > 0
