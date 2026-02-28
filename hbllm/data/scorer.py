"""
Data Quality Scorer.

Provides heuristics to filter out low-quality web documents (e.g., gibberish, 
excessive symbols, extremely long words) before they enter the pre-training dataset.
"""

from __future__ import annotations

import re


class QualityScorer:
    """
    Evaluates text quality based on simple statistical heuristics.
    """

    def __init__(
        self,
        min_alpha_frac: float = 0.5,
        max_symbol_frac: float = 0.2,
        min_mean_word_length: float = 3.0,
        max_mean_word_length: float = 10.0,
        max_word_length: int = 100,
    ):
        """
        Args:
            min_alpha_frac: Minimum fraction of alphabetic characters.
            max_symbol_frac: Maximum fraction of symbol characters.
            min_mean_word_length: Minimum average word length.
            max_mean_word_length: Maximum average word length.
            max_word_length: Maximum allowed length for any single word.
        """
        self.min_alpha_frac = min_alpha_frac
        self.max_symbol_frac = max_symbol_frac
        self.min_mean_word_length = min_mean_word_length
        self.max_mean_word_length = max_mean_word_length
        self.max_word_length = max_word_length

        # regex for symbols
        self.symbol_re = re.compile(r"[^\w\s]")

    def is_high_quality(self, text: str) -> bool:
        """
        Check if text passes all quality heuristics.
        Returns True if high quality, False if it should be discarded.
        """
        if not text.strip():
            return False

        # Character level checks
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        symbols = len(self.setup_finditer(text))

        if alpha_chars / total_chars < self.min_alpha_frac:
            return False

        if symbols / total_chars > self.max_symbol_frac:
            return False

        # Word level checks
        words = text.split()
        if not words:
            return False

        total_word_length = 0
        for word in words:
            length = len(word)
            if length > self.max_word_length:
                return False
            total_word_length += length

        mean_word_length = total_word_length / len(words)
        if not (self.min_mean_word_length <= mean_word_length <= self.max_mean_word_length):
            return False

        return True

    def setup_finditer(self, text: str) -> list[str]:
        return self.symbol_re.findall(text)

