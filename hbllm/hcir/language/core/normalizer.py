"""Deterministic Language-Agnostic Orthographic Normalizer & String Edit Metrics.

Zero-LLM, sub-millisecond bounded Damerau-Levenshtein distance calculation and
O(L) SymSpell delete-1 indexed orthographic normalization with proper-noun,
frequency-threshold, and substitution false-positive guards.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


class _Delete1Index:
    """SymSpell-style O(L) delete-1 vocabulary index with tier-rank tracking."""

    __slots__ = ("exact", "del1")

    def __init__(self) -> None:
        self.exact: dict[str, int] = {}
        self.del1: dict[str, list[str]] = {}

    def clear(self) -> None:
        self.exact.clear()
        self.del1.clear()

    def add_word(self, word: str, tier_rank: int) -> None:
        if len(word) < 2 or not word.isalpha():
            return
        prev = self.exact.get(word)
        if prev is not None:
            if tier_rank < prev:
                self.exact[word] = tier_rank
            return
        self.exact[word] = tier_rank
        for i in range(len(word)):
            d = word[:i] + word[i + 1 :]
            if len(d) < 2:
                continue
            bucket = self.del1.get(d)
            if bucket is None:
                self.del1[d] = [word]
            elif len(bucket) < 32:
                bucket.append(word)


class BaseOrthographicNormalizer:
    """Language-independent orthographic normalizer with O(L) SymSpell index & safety guards."""

    def __init__(
        self,
        closed_class_vocab: frozenset[str] | None = None,
        command_verbs: frozenset[str] | None = None,
        common_typos: dict[str, str] | None = None,
        lexicon: Any | None = None,
        ontology_store: Any | None = None,
        language: str = "en",
        cache_size: int = 1024,
        frequency_threshold: int = 3,
    ) -> None:
        self._closed_class_vocab = closed_class_vocab or frozenset()
        self._command_verbs = command_verbs or frozenset()
        self._common_typos = common_typos or {}
        self._lexicon = lexicon
        self._ontology_store = ontology_store
        self._language = language
        self._cache_size = cache_size
        self._frequency_threshold = max(1, int(frequency_threshold))
        self._cache: dict[tuple[str, bool, bool], str] = {}
        self._dynamic_vocab: set[str] = set()
        self._word_frequencies: dict[str, int] = {}
        self._tier0_words: frozenset[str] = self._closed_class_vocab | self._command_verbs
        self._index = _Delete1Index()
        self._indexed_base_len: int = -1

    def _base_words(self) -> frozenset[str]:
        if self._lexicon is not None and hasattr(self._lexicon, "base_words"):
            return self._lexicon.base_words()
        return frozenset()

    def _ensure_index(self) -> None:
        base_words = self._base_words()
        if self._indexed_base_len == len(base_words):
            return
        self._index.clear()
        for w in self._tier0_words:
            self._index.add_word(w, 0)
        for w in self._dynamic_vocab:
            self._index.add_word(w, 1)
        for w in base_words:
            self._index.add_word(w, 2)
        self._indexed_base_len = len(base_words)

    def set_dynamic_vocabulary(self, words: Iterable[str]) -> None:
        """Update active world/persistent entity & concept vocabulary incrementally."""
        cleaned: set[str] = set()
        for w in words:
            for part in str(w).lower().replace("_", " ").split():
                if len(part) >= 3 and part.isalpha():
                    cleaned.add(part)
        if cleaned == self._dynamic_vocab:
            return
        added = cleaned - self._dynamic_vocab
        removed = self._dynamic_vocab - cleaned
        self._dynamic_vocab = cleaned
        self._cache.clear()
        if not removed and self._indexed_base_len >= 0:
            for w in added:
                self._index.add_word(w, 1)
        else:
            self._indexed_base_len = -1

    def set_word_frequencies(self, freqs: Mapping[str, int]) -> None:
        """Register observed word frequencies; words >= frequency_threshold are never rewritten."""
        self._word_frequencies = {str(k).lower(): int(v) for k, v in freqs.items() if k}
        self._cache.clear()

    def record_token_frequency(self, word: str, delta: int = 1) -> None:
        """Increment observed frequency for *word*."""
        low = word.strip().lower()
        if low:
            self._word_frequencies[low] = self._word_frequencies.get(low, 0) + delta

    def _is_trusted(self, low: str) -> bool:
        if (
            low in self._closed_class_vocab
            or low in self._command_verbs
            or low in self._dynamic_vocab
        ):
            return True
        if self._word_frequencies.get(low, 0) >= self._frequency_threshold:
            return True
        if self._lexicon is not None:
            if hasattr(self._lexicon, "is_trusted_word"):
                if self._lexicon.is_trusted_word(low):
                    return True
            elif hasattr(self._lexicon, "has_word") and self._lexicon.has_word(low):
                return True
        store = self._get_store()
        if store is not None:
            if hasattr(store, "is_trusted_word") and store.is_trusted_word(
                low, language=self._language
            ):
                return True
            if hasattr(store, "get_word_frequency") and (
                store.get_word_frequency(low, language=self._language) >= self._frequency_threshold
            ):
                return True
        return False

    def _get_store(self) -> Any | None:
        if self._ontology_store is not None:
            return self._ontology_store
        if self._lexicon is not None:
            return getattr(self._lexicon, "_ontology_store", None)
        return None

    def normalize_token(self, token: str, *, is_sentence_start: bool = True) -> str:
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

        is_title = token[0].isupper() and not token.isupper()
        # Never "correct" mid-sentence proper names (e.g. Mira, Darcy, Florb)
        if is_title and not is_sentence_start:
            return token

        if self._is_trusted(low):
            return token

        # Protect valid inflections of trusted base stems (e.g. "plants" -> stem "plant" is trusted)
        if low.endswith("ies") and len(low) > 4 and self._is_trusted(low[:-3] + "y"):
            return token
        if (
            low.endswith("es")
            and len(low) > 3
            and (self._is_trusted(low[:-2]) or self._is_trusted(low[:-1]))
        ):
            return token
        if (
            low.endswith("s")
            and len(low) > 3
            and not low.endswith(("ss", "us", "is"))
            and self._is_trusted(low[:-1])
        ):
            return token

        cache_key = (low, is_title, is_sentence_start)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        self._ensure_index()

        candidates: list[tuple[int, int, int, int, int, str]] = []
        seen_cands: set[str] = set()
        low_sorted = sorted(low)

        def _consider(cand: str, dist: int, tier_rank: int) -> None:
            if cand in seen_cands or cand == low:
                return
            if is_title and tier_rank != 0:
                return
            seen_cands.add(cand)
            candidates.append(
                (
                    dist,
                    0 if sorted(cand) == low_sorted else 1,
                    0 if cand[0] == low[0] else 1,
                    tier_rank,
                    abs(len(cand) - len(low)),
                    cand,
                )
            )

        exact = self._index.exact
        del1 = self._index.del1
        low_deletes = [low[:i] + low[i + 1 :] for i in range(len(low))]

        # 1a. Extra character typed (deletion in low matches exact word, e.g. lisst -> list)
        for d_str in low_deletes:
            tier_rank = exact.get(d_str)
            if tier_rank is not None and (d_str[0] == low[0] or (tier_rank == 0 and len(low) >= 5)):
                _consider(d_str, 1, tier_rank)

        # 1b. Missing character typed (low is a 1-char deletion of exact word, e.g. wher -> where)
        for cand in del1.get(low, ()):
            tier_rank = exact.get(cand, 2)
            if cand[0] == low[0] or (tier_rank == 0 and len(low) >= 5):
                _consider(cand, 1, tier_rank)

        # 1c. Adjacent character transposition (e.g. blal -> ball, focsu -> focus, explian -> explain)
        for i in range(len(low) - 1):
            if low[i] == low[i + 1]:
                continue
            trans = low[:i] + low[i + 1] + low[i] + low[i + 2 :]
            tier_rank = exact.get(trans)
            if tier_rank is not None:
                _consider(trans, 1, tier_rank)

        # 1d. Same-length single-char substitution: allowed ONLY for tier-0 closed/command words
        # (requiring matching first char when len <= 5 so invented nouns like "zint" != "lint")
        # or long (>=7 char) dynamic/base words sharing a 4-char prefix (prevents flood -> floor).
        for d_str in low_deletes:
            for cand in del1.get(d_str, ()):
                if cand == low or len(cand) != len(low):
                    continue
                tier_rank = exact.get(cand, 2)
                tier0_ok = tier_rank == 0 and (len(low) >= 6 or cand[0] == low[0])
                long_ok = len(low) >= 7 and cand[:4] == low[:4]
                if tier0_ok or long_ok:
                    if damerau_levenshtein(low, cand, 1) <= 1:
                        _consider(cand, 1, tier_rank)

        # 2. Bounded edit-distance 2 ONLY on tier-0 (~100 closed/command words), requiring
        # matching 3-char prefix and length delta <= 1 (catches explane -> explain, blocks plants -> plan).
        if not candidates and len(low) >= 6:
            for cand in self._tier0_words:
                if abs(len(cand) - len(low)) > 1:
                    continue
                if len(cand) < 4 or cand[:3] != low[:3]:
                    continue
                d = damerau_levenshtein(low, cand, 2)
                if d <= 2:
                    _consider(cand, d, 0)

        result = token
        if candidates:
            candidates.sort()
            result = candidates[0][5]

        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[cache_key] = result
        return result
