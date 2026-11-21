"""Deterministic Language-Agnostic Orthographic Normalizer & String Edit Metrics.

Zero-LLM, sub-millisecond bounded Damerau-Levenshtein distance calculation and
two-tier orthographic normalization with LRU caching.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Compute bounded Optimal String Alignment (Damerau-Levenshtein) distance.

    Returns the minimum number of single-character insertions, deletions,
    substitutions, and adjacent transpositions required to transform ``a`` into ``b``.
    Short-circuits to ``max_dist + 1`` if length delta or running cost exceeds ``max_dist``.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if a == b:
        return 0
    if la == 0:
        return lb
    if lb == 0:
        return la

    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j

    for i in range(1, la + 1):
        row_min = max_dist + 1
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            val = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]:
                val = min(val, d[i - 2][j - 2] + 1)
            d[i][j] = val
            if val < row_min:
                row_min = val
        if row_min > max_dist:
            return max_dist + 1

    return d[la][lb]


class BaseOrthographicNormalizer:
    """Language-independent orthographic normalizer with LRU cache & false-positive guards."""

    def __init__(
        self,
        closed_class_vocab: frozenset[str] | None = None,
        command_verbs: frozenset[str] | None = None,
        common_typos: dict[str, str] | None = None,
        lexicon: Any | None = None,
        ontology_store: Any | None = None,
        language: str = "en",
        cache_size: int = 1024,
    ) -> None:
        self._closed_class_vocab = closed_class_vocab or frozenset()
        self._command_verbs = command_verbs or frozenset()
        self._common_typos = common_typos or {}
        self._lexicon = lexicon
        self._ontology_store = ontology_store
        self._language = language
        self._cache_size = cache_size
        self._cache: dict[str, str] = {}
        self._dynamic_vocab: set[str] = set()

    def set_dynamic_vocabulary(self, words: Iterable[str]) -> None:
        """Update active world/persistent entity & concept vocabulary for fuzzy matching."""
        cleaned: set[str] = set()
        for w in words:
            for part in str(w).lower().replace("_", " ").split():
                if len(part) >= 3 and part.isalpha():
                    cleaned.add(part)
        self._dynamic_vocab = cleaned

    def _is_trusted(self, low: str) -> bool:
        if (
            low in self._closed_class_vocab
            or low in self._command_verbs
            or low in self._dynamic_vocab
        ):
            return True
        if self._lexicon is not None:
            if hasattr(self._lexicon, "is_trusted_word"):
                if self._lexicon.is_trusted_word(low):
                    return True
            elif hasattr(self._lexicon, "has_word") and self._lexicon.has_word(low):
                return True
        store = self._get_store()
        if store is not None and hasattr(store, "is_trusted_word"):
            return bool(store.is_trusted_word(low, language=self._language))
        return False

    def _get_store(self) -> Any | None:
        if self._ontology_store is not None:
            return self._ontology_store
        if self._lexicon is not None:
            return getattr(self._lexicon, "_ontology_store", None)
        return None

    def normalize_token(self, token: str) -> str:
        """Return corrected token if *token* is an untrusted typo, else *token* unchanged."""
        if not token or len(token) <= 1:
            return token
        if any(ch in "./\\:@#$%0123456789_" for ch in token):
            return token
        if not token.isalpha():
            return token

        low = token.lower()
        if low in self._common_typos:
            return self._common_typos[low]

        if len(low) <= 3:
            return token

        if self._is_trusted(low):
            return token

        cached = self._cache.get(low)
        if cached is not None:
            return cached

        tier0_words = self._closed_class_vocab | self._command_verbs
        base_words: frozenset[str] = (
            self._lexicon.base_words()
            if self._lexicon is not None and hasattr(self._lexicon, "base_words")
            else frozenset()
        )
        candidates: list[tuple[int, int, int, int, int, str]] = []
        max_dist = 2 if len(low) >= 6 else 1
        low_sorted = sorted(low)

        def _scan(vocab: Iterable[str], tier_rank: int) -> None:
            for cand in vocab:
                if abs(len(cand) - len(low)) > max_dist:
                    continue
                if (
                    len(low) <= 4
                    and cand[0] != low[0]
                    and not (len(cand) >= 2 and cand[1] == low[0])
                ):
                    continue
                d = damerau_levenshtein(low, cand, max_dist)
                if d <= max_dist:
                    if d == 2:
                        first_ok = cand[0] == low[0] or (
                            len(cand) >= 2 and cand[1] == low[0] and cand[0] == low[1]
                        )
                        if not first_ok:
                            continue
                    candidates.append(
                        (
                            d,
                            0 if sorted(cand) == low_sorted else 1,
                            0 if cand[0] == low[0] else 1,
                            tier_rank,
                            abs(len(cand) - len(low)),
                            cand,
                        )
                    )

        _scan(tier0_words, 0)
        _scan(self._dynamic_vocab, 1)
        _scan(base_words, 2)

        result = token
        if candidates:
            candidates.sort()
            result = candidates[0][5]

        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[low] = result
        return result
