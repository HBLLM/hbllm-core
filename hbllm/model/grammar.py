"""
Grammar-Constrained Sampler for HBLLM Core.

Provides incremental, linear-time logit masking to enforce deterministic JSON structure.
"""

from __future__ import annotations

import torch


class GrammarState:
    """
    Incremental state machine for strict, parser-safe JSON grammar validation.
    Enforces O(1) step execution during autoregressive generation.
    """

    def __init__(self, vocab_words: dict[int, str]) -> None:
        self.vocab_words = vocab_words
        self.stack: list[str] = []
        self.inside_string = False
        self.last_char_escape = False
        # Last structural character observed: '{', '}', '[', ']', ':', ',', '"'
        self.last_structural = ""
        self.expecting_value = False
        self.is_valid = True

    def advance(self, token_id: int) -> GrammarState:
        """
        Advance the grammar state machine with the chosen token ID.
        Executes in O(1) time per step.
        """
        word = self.vocab_words.get(token_id, "")
        if not word:
            return self

        for char in word:
            if self.inside_string:
                if self.last_char_escape:
                    self.last_char_escape = False
                elif char == "\\":
                    self.last_char_escape = True
                elif char == '"':
                    self.inside_string = False
                    self.last_structural = '"'
            else:
                if char == '"':
                    self.inside_string = True
                    self.last_structural = '"'
                elif char in "{[":
                    self.stack.append("}" if char == "{" else "]")
                    self.last_structural = char
                    self.expecting_value = False
                elif char in "}]":
                    if self.stack and self.stack[-1] == char:
                        self.stack.pop()
                    else:
                        self.is_valid = False
                    self.last_structural = char
                elif char == ":":
                    self.last_structural = char
                    self.expecting_value = True
                elif char == ",":
                    self.last_structural = char
                    self.expecting_value = False

        return self

    def get_logit_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        """
        Produce a binary logit mask tensor.
        Invalid token indices are set to -inf, valid to 0.0.
        """
        mask = torch.zeros(vocab_size, device=device, dtype=torch.float32)

        # If stack is empty and we have already seen structural completion, we are done
        # Only allow whitespace or EOF/EOS
        if not self.stack and self.last_structural in "}]":
            for idx, word in self.vocab_words.items():
                if not word.strip():
                    continue
                mask[idx] = float("-inf")
            return mask

        # If inside a string, allow almost all plain text
        if self.inside_string:
            return mask

        # If outside string, restrict to valid grammar transitions
        for idx, word in self.vocab_words.items():
            stripped = word.strip()
            if not stripped:
                continue

            char = stripped[0]

            if self.last_structural in "{,":
                # Expecting a key (starts with a string quote)
                if char != '"' and char != "}":
                    mask[idx] = float("-inf")
            elif self.last_structural == '"':
                # Just closed a string (either key or value)
                if not self.expecting_value:
                    # We just closed a key string! ONLY allow colon ':'
                    if char != ":":
                        mask[idx] = float("-inf")
                else:
                    # We just closed a value string! Allow comma ',' or close brace '}'
                    if self.stack and self.stack[-1] == "}":
                        if char not in ",}":
                            mask[idx] = float("-inf")
                    else:
                        # Inside array
                        if char not in ",]":
                            mask[idx] = float("-inf")
            elif self.last_structural == ":":
                # Expecting a value: starts with string, digit, or nested structure
                if char not in '"0123456789-{[tfn':
                    mask[idx] = float("-inf")
            elif self.last_structural == "[":
                # Inside array, expecting value or close bracket
                if char not in '"0123456789-{[tfn]' and stripped != "]":
                    mask[idx] = float("-inf")

        return mask
