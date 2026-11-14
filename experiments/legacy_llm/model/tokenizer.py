"""
Unified tokenizer for HBLLM.

Provides a high-performance Rust-based BPE Vocab engine,
with a mathematically-exact pure-Python fallback when the Rust extensions are not compiled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Try to import high-performance Rust Vocab
try:
    from hbllm_tokenizer_rs import Vocab as RustVocab  # type: ignore
except ImportError:
    RustVocab = None

logger = logging.getLogger(__name__)

# Special token strings
BOS = chr(60) + "|bos|" + chr(62)
EOS = chr(60) + "|eos|" + chr(62)
PAD = chr(60) + "|pad|" + chr(62)
SYSTEM = chr(60) + "|system|" + chr(62)
USER = chr(60) + "|user|" + chr(62)
ASSISTANT = chr(60) + "|assistant|" + chr(62)

SPECIAL_TOKENS = [BOS, EOS, PAD, SYSTEM, USER, ASSISTANT]


class PurePythonBPE:
    """
    A mathematically-exact pure-Python implementation of the custom byte-level BPE tokenizer.
    Ensures complete vocabulary and token ID parity when Rust extensions are not available.
    """

    def __init__(self, vocab_data: dict[str, Any]) -> None:
        self.vocab_size = vocab_data.get("vocab_size", 32768)

        # Base byte mappings (IDs 0-255 are raw bytes)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.bytes_to_id = {bytes([i]): i for i in range(256)}

        self.merges = vocab_data.get("merges", [])
        self.merge_ranks = {}
        self.merge_to_id = {}

        # Reconstruct merge rules in exact rank order
        for rule in self.merges:
            left = rule["left"]
            right = rule["right"]
            merged = rule["merged"]
            rank = rule["rank"]

            self.merge_ranks[(left, right)] = rank
            self.merge_to_id[(left, right)] = merged

            # Build the byte sequence representation of the merged token
            if left in self.id_to_bytes and right in self.id_to_bytes:
                merged_bytes = self.id_to_bytes[left] + self.id_to_bytes[right]
                self.id_to_bytes[merged] = merged_bytes
                self.bytes_to_id[merged_bytes] = merged

        # Reconstruct special tokens
        self.special_tokens = vocab_data.get("special_tokens", [])
        for sp in self.special_tokens:
            token_str = sp["token"]
            token_id = sp["id"]
            token_bytes = token_str.encode("utf-8")
            self.id_to_bytes[token_id] = token_bytes
            self.bytes_to_id[token_bytes] = token_id

    def __len__(self) -> int:
        return len(self.id_to_bytes)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        # Start with byte-level tokens
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            best_rank = float("inf")
            best_pair = None
            best_idx = -1

            # Find the merge pair with the lowest rank (highest priority)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i

            if best_idx == -1:
                break  # No more merge rules applicable

            assert best_pair is not None  # guaranteed by best_idx != -1
            left, right = best_pair
            merged_id = self.merge_to_id[best_pair]
            tokens[best_idx] = merged_id
            tokens.pop(best_idx + 1)

        return tokens

    def decode(self, ids: list[int]) -> str:
        res = bytearray()
        for i in ids:
            b = self.id_to_bytes.get(i)
            if b is not None:
                res.extend(b)
        return res.decode("utf-8", errors="replace")

    def decode_to_string(self, ids: list[int]) -> str:
        return self.decode(ids)


class HBLLMTokenizer:
    """
    Unified tokenizer with special token support and chat templates.

    Usage:
        tok = HBLLMTokenizer.from_vocab('path/to/vocab.json')
        ids = tok.encode('Hello world')
        text = tok.decode(ids)
        prompt = tok.apply_chat_template([{'role': 'user', 'content': 'Hi'}])
    """

    def __init__(self, vocab: Any | None = None, vocab_size: int = 32768) -> None:
        self._vocab = vocab  # Rust Vocab or PurePythonBPE or None
        self._tiktoken: Any | None = None
        self._special_ids: dict[str, int] = {}
        self.vocab_size = vocab_size

        if vocab is not None:
            self._init_from_vocab(vocab)
        else:
            self._init_fallback()

    def _init_from_vocab(self, vocab: Any) -> None:
        """Initialize from Vocab object (Rust or Python)."""
        self.vocab_size = len(vocab)
        # Map special tokens to IDs (reserve last N vocab positions)
        base = self.vocab_size - len(SPECIAL_TOKENS)
        for i, token in enumerate(SPECIAL_TOKENS):
            self._special_ids[token] = base + i
        logger.info("Tokenizer loaded from Custom Vocab (%d tokens)", self.vocab_size)

    # Class-level tiktoken cache to avoid re-initialization
    _tiktoken_cache: dict[str, Any] = {}

    def _init_fallback(self) -> None:
        """Graceful fallback: use tiktoken if installed, else a lightweight byte-level model."""
        try:
            import tiktoken

            cache_key = "cl100k_base"
            if cache_key not in HBLLMTokenizer._tiktoken_cache:
                HBLLMTokenizer._tiktoken_cache[cache_key] = tiktoken.get_encoding(cache_key)
            self._tiktoken = HBLLMTokenizer._tiktoken_cache[cache_key]
            if self._tiktoken:
                self.vocab_size = self._tiktoken.n_vocab
                logger.info("Tokenizer fallback: tiktoken cl100k_base (%d tokens)", self.vocab_size)
        except ImportError:
            # Safe zero-dependency fallback (base 256 bytes)
            self.vocab_size = 256
            logger.info("Tokenizer fallback: Zero-dependency 256-byte base vocabulary")

        # Reserve special IDs at the end of the fallback vocabulary
        base = self.vocab_size
        for i, token in enumerate(SPECIAL_TOKENS):
            self._special_ids[token] = base + i
        self.vocab_size += len(SPECIAL_TOKENS)

    @classmethod
    def from_vocab(cls, path: str | Path) -> HBLLMTokenizer:
        """Load from a BPE vocab JSON file."""
        # 1. Try high-performance Rust Vocab first
        if RustVocab is not None:
            try:
                vocab = RustVocab.load(str(path))
                return cls(vocab=vocab)
            except Exception as e:
                logger.warning("Failed to load Rust vocab: %s. Trying pure-Python loader.", e)

        # 2. Fallback to Pure-Python custom BPE loader
        try:
            with open(path, encoding="utf-8") as f:
                vocab_data = json.load(f)
            vocab = PurePythonBPE(vocab_data)
            return cls(vocab=vocab)
        except Exception as e:
            logger.warning("Failed to load custom vocab: %s. Using fallback.", e)

        return cls()

    @classmethod
    def from_tiktoken(cls) -> HBLLMTokenizer:
        """Create using tiktoken fallback."""
        return cls()

    # ─── Encode / Decode ─────────────────────────────────────────────

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs."""
        ids: list[int] = []

        if add_bos:
            ids.append(self.bos_id)

        if self._vocab is not None:
            ids.extend(self._vocab.encode(text))
        elif self._tiktoken is not None:
            ids.extend(self._tiktoken.encode(text))
        else:
            # Zero-dependency 256-byte fallback
            ids.extend(list(text.encode("utf-8")))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True, **kwargs: Any) -> str:
        """Decode token IDs to text."""
        # Filter out special tokens
        if skip_special_tokens:
            special_values = set(self._special_ids.values())
            filtered = [i for i in ids if i not in special_values]
        else:
            filtered = ids

        if self._vocab is not None:
            if hasattr(self._vocab, "decode_to_string"):
                return str(self._vocab.decode_to_string(filtered))
            return str(self._vocab.decode(filtered))
        elif self._tiktoken is not None:
            return str(self._tiktoken.decode(filtered))
        else:
            # Zero-dependency 256-byte fallback — token IDs ≥256 (e.g.
            # merged BPE tokens or special tokens) have no byte mapping here,
            # so we filter them out to avoid ValueError.
            return bytes(t for t in filtered if 0 <= t < 256).decode("utf-8", errors="replace")

    # ─── Special Token IDs ───────────────────────────────────────────

    @property
    def bos_id(self) -> int:
        return self._special_ids[BOS]

    @property
    def eos_id(self) -> int:
        return self._special_ids[EOS]

    @property
    def pad_id(self) -> int:
        return self._special_ids[PAD]

    # ─── Chat Template ───────────────────────────────────────────────

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format messages into ChatML-style prompt.

        Args:
            messages: List of {'role': 'system'|'user'|'assistant', 'content': '...'}
            add_generation_prompt: Append assistant prefix for generation

        Returns:
            Formatted prompt string
        """
        parts = []
        nl = "\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(SYSTEM + nl + content + EOS)
            elif role == "user":
                parts.append(USER + nl + content + EOS)
            elif role == "assistant":
                parts.append(ASSISTANT + nl + content + EOS)

        if add_generation_prompt:
            parts.append(ASSISTANT + nl)

        return nl.join(parts)

    def encode_chat(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Apply chat template and encode to IDs."""
        prompt = self.apply_chat_template(messages, add_generation_prompt)
        return self.encode(prompt, add_bos=True)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        backend = (
            "rust"
            if RustVocab and isinstance(self._vocab, RustVocab)
            else "python"
            if self._vocab
            else "tiktoken"
            if self._tiktoken
            else "none"
        )
        return f"HBLLMTokenizer(vocab_size={self.vocab_size}, backend={backend})"
