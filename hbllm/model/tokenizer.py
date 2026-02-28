"""
Unified tokenizer for HBLLM.

Wraps the Rust-based BPE Vocab or falls back to tiktoken.
Provides encode/decode with special tokens and chat templates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Special token strings
BOS = chr(60) + '|bos|' + chr(62)
EOS = chr(60) + '|eos|' + chr(62)
PAD = chr(60) + '|pad|' + chr(62)
SYSTEM = chr(60) + '|system|' + chr(62)
USER = chr(60) + '|user|' + chr(62)
ASSISTANT = chr(60) + '|assistant|' + chr(62)

SPECIAL_TOKENS = [BOS, EOS, PAD, SYSTEM, USER, ASSISTANT]


class HBLLMTokenizer:
    """
    Unified tokenizer with special token support and chat templates.

    Usage:
        tok = HBLLMTokenizer.from_vocab('path/to/vocab.json')
        ids = tok.encode('Hello world')
        text = tok.decode(ids)
        prompt = tok.apply_chat_template([{'role': 'user', 'content': 'Hi'}])
    """

    def __init__(self, vocab: Any = None, vocab_size: int = 32768):
        self._vocab = vocab  # Rust Vocab object or None
        self._tiktoken = None
        self._special_ids: dict[str, int] = {}
        self.vocab_size = vocab_size

        if vocab is not None:
            self._init_from_vocab(vocab)
        else:
            self._init_fallback()

    def _init_from_vocab(self, vocab: Any) -> None:
        """Initialize from Rust Vocab."""
        self.vocab_size = len(vocab)
        # Map special tokens to IDs (reserve last N vocab positions)
        base = self.vocab_size - len(SPECIAL_TOKENS)
        for i, token in enumerate(SPECIAL_TOKENS):
            self._special_ids[token] = base + i
        logger.info('Tokenizer loaded from Rust Vocab (%d tokens)', self.vocab_size)

    def _init_fallback(self) -> None:
        """Fallback: use tiktoken for encoding."""
        try:
            import tiktoken
            self._tiktoken = tiktoken.get_encoding('cl100k_base')
            self.vocab_size = self._tiktoken.n_vocab
            logger.info('Tokenizer fallback: tiktoken cl100k_base (%d tokens)', self.vocab_size)
        except ImportError:
            logger.warning('No tokenizer backend available. Encode/decode will fail.')

        # Reserve special IDs at end
        base = self.vocab_size
        for i, token in enumerate(SPECIAL_TOKENS):
            self._special_ids[token] = base + i
        self.vocab_size += len(SPECIAL_TOKENS)

    @classmethod
    def from_vocab(cls, path: str | Path) -> HBLLMTokenizer:
        """Load from a Rust vocab JSON file."""
        try:
            from hbllm_tokenizer_rs import Vocab
            vocab = Vocab.load(str(path))
            return cls(vocab=vocab)
        except ImportError:
            logger.warning('Rust tokenizer not available, using tiktoken fallback')
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
            raise RuntimeError('No tokenizer backend available')

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        # Filter out special tokens
        filtered = [i for i in ids if i not in self._special_ids.values()]

        if self._vocab is not None:
            return self._vocab.decode_to_string(filtered)
        elif self._tiktoken is not None:
            return self._tiktoken.decode(filtered)
        else:
            raise RuntimeError('No tokenizer backend available')

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
            role = msg['role']
            content = msg['content']
            if role == 'system':
                parts.append(SYSTEM + nl + content + EOS)
            elif role == 'user':
                parts.append(USER + nl + content + EOS)
            elif role == 'assistant':
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
        backend = 'rust' if self._vocab else 'tiktoken' if self._tiktoken else 'none'
        return f'HBLLMTokenizer(vocab_size={self.vocab_size}, backend={backend})'
