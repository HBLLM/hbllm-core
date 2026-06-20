"""
Unit tests to verify mathematically-exact parity between the Rust-based BPE tokenizer
and the new pure-Python BPE fallback tokenizer.
"""

import json
import tempfile
from pathlib import Path

from hbllm.model.tokenizer import HBLLMTokenizer, PurePythonBPE

# Try to import Rust Vocab to test parity directly if compiled
try:
    from hbllm_tokenizer_rs import Vocab as RustVocab
except ImportError:
    RustVocab = None


def test_pure_python_bpe_parity():
    # 1. Create a mock vocabulary matching the custom BPE save structure
    mock_vocab = {
        "vocab_size": 1000,
        "merges": [
            # 'h' + 'e' -> 256
            {"left": 104, "right": 101, "merged": 256, "rank": 0},
            # 256 ('he') + 'l' -> 257
            {"left": 256, "right": 108, "merged": 257, "rank": 1},
            # 257 ('hel') + 'l' -> 258
            {"left": 257, "right": 108, "merged": 258, "rank": 2},
            # 258 ('hell') + 'o' -> 259
            {"left": 258, "right": 111, "merged": 259, "rank": 3},
            # 'w' + 'o' -> 260
            {"left": 119, "right": 111, "merged": 260, "rank": 4},
        ],
        "special_tokens": [
            {"token": "<|bos|>", "id": 999},
            {"token": "<|eos|>", "id": 998},
            {"token": "<|pad|>", "id": 997},
            {"token": "<|system|>", "id": 996},
            {"token": "<|user|>", "id": 995},
            {"token": "<|assistant|>", "id": 994},
        ],
    }

    # Write to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file = Path(tmpdir) / "vocab.json"
        vocab_file.write_text(json.dumps(mock_vocab), encoding="utf-8")

        # Load using PurePythonBPE
        py_bpe = PurePythonBPE(mock_vocab)

        # 2. Test encoding / decoding correctness in pure-Python
        text = "hello world"
        py_tokens = py_bpe.encode(text)

        # 'hello' should merge to token 259 ('hello'), followed by space (32), followed by 'world' ('wo' -> 260, 'r' -> 114, 'l' -> 108, 'd' -> 100)
        # So we expect: [259, 32, 260, 114, 108, 100]
        expected_tokens = [259, 32, 260, 114, 108, 100]
        assert py_tokens == expected_tokens, f"Expected {expected_tokens}, got {py_tokens}"

        # Decode round-trip
        decoded_text = py_bpe.decode(py_tokens)
        assert decoded_text == text

        # 3. Test parity with Rust BPE (if compiled)
        if RustVocab is not None:
            # Load using Rust BPE
            rust_vocab = RustVocab.load(str(vocab_file))

            # Encode using Rust
            rust_tokens = rust_vocab.encode(text)
            assert rust_tokens == py_tokens, (
                f"Token IDs mismatch! Rust: {rust_tokens}, Python: {py_tokens}"
            )

            # Decode using Rust
            rust_decoded = rust_vocab.decode(rust_tokens)
            assert rust_decoded == decoded_text, (
                f"Decoded text mismatch! Rust: {rust_decoded}, Python: {decoded_text}"
            )


def test_tokenizer_from_vocab_fallback():
    mock_vocab = {
        "vocab_size": 500,
        "merges": [],
        "special_tokens": [
            {"token": "<|bos|>", "id": 499},
            {"token": "<|eos|>", "id": 498},
            {"token": "<|pad|>", "id": 497},
            {"token": "<|system|>", "id": 496},
            {"token": "<|user|>", "id": 495},
            {"token": "<|assistant|>", "id": 494},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file = Path(tmpdir) / "vocab.json"
        vocab_file.write_text(json.dumps(mock_vocab), encoding="utf-8")

        # Load through HBLLMTokenizer
        tokenizer = HBLLMTokenizer.from_vocab(vocab_file)

        # Test Special tokens property methods
        assert tokenizer.bos_id == 256
        assert tokenizer.eos_id == 257
        assert tokenizer.pad_id == 258

        # Test encoding with special tokens added
        ids = tokenizer.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == 256
        assert ids[-1] == 257
        assert len(ids) == 4  # [BOS, 'h', 'i', EOS]


def test_zero_dependency_fallback_no_crash():
    # Force tokenizer to initialize with no vocab or tiktoken (by patching it if necessary,
    # but self._init_fallback already handles it gracefully when tiktoken is not imported/installed).
    # We can create one directly with no arguments:
    tokenizer = HBLLMTokenizer()
    assert tokenizer.vocab_size > 0

    text = "Zero-dependency test!"
    ids = tokenizer.encode(text)
    assert len(ids) > 0
    decoded = tokenizer.decode(ids)
    assert decoded == text
