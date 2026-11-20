"""Deterministic English Orthographic Normalizer & Conversational Wrapper Stripper.

Zero-LLM, sub-millisecond normalization for typos, contractions, and noisy
conversational wrappers while protecting filenames, URLs, numbers, and valid English words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Common high-collision typos on short function words (where corpus OCR data may pollute lookups)
COMMON_TYPOS: dict[str, str] = {
    "teh": "the",
    "hte": "the",
    "taht": "that",
    "waht": "what",
    "wht": "what",
    "wher": "where",
    "hwere": "where",
    "wyh": "why",
    "hwo": "how",
    "plz": "please",
    "pls": "please",
}

CONTRACTIONS: dict[str, str] = {
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "why's": "why is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "i'm": "i am",
    "i'd": "i would",
    "wanna": "want to",
}

CLOSED_CLASS_VOCAB: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "what",
        "where",
        "why",
        "how",
        "who",
        "when",
        "which",
        "is",
        "are",
        "was",
        "were",
        "be",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "will",
        "not",
        "on",
        "in",
        "inside",
        "within",
        "under",
        "beneath",
        "above",
        "near",
        "beside",
        "from",
        "to",
        "into",
        "about",
        "for",
        "of",
        "with",
        "it",
        "them",
        "me",
        "us",
        "you",
        "they",
        "and",
        "then",
        "there",
        "please",
    }
)

COMMAND_VERB_VOCAB: frozenset[str] = frozenset(
    {
        "read",
        "open",
        "cat",
        "show",
        "display",
        "print",
        "view",
        "list",
        "verify",
        "check",
        "analyze",
        "parse",
        "inspect",
        "lint",
        "status",
        "search",
        "find",
        "locate",
        "grep",
        "run",
        "execute",
        "start",
        "launch",
        "predict",
        "evaluate",
        "simulate",
        "plan",
        "expand",
        "validate",
        "remember",
        "store",
        "memorize",
        "save",
        "recall",
        "retrieve",
        "forget",
        "ingest",
        "index",
        "explain",
        "describe",
        "tell",
        "detail",
        "summarize",
        "why",
        "how",
        "websearch",
        "google",
        "browse",
        "fetch",
        "move",
        "push",
        "put",
        "place",
        "give",
        "drop",
        "roll",
        "unlock",
        "rotate",
        "close",
    }
)

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

_LIST_MARKER_RE = re.compile(r"^\s*(?:step\s+\d+[:.)]?\s+|\d+[.)]\s+|[-*•]\s+)", flags=re.I)
_SEQ_PREFIX_RE = re.compile(
    r"^\s*(?:first|second|third|next|then|finally|after\s+that|also)\s*[,]?\s+",
    flags=re.I,
)
_POLITE_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:please|kindly)\s*[,]?\s+", flags=re.I),
    re.compile(r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?", flags=re.I),
    re.compile(r"^\s*i\s+(?:want|would\s+like|need)\s+(?:you\s+)?to\s+", flags=re.I),
    re.compile(r"^\s*help\s+me\s+(?:to\s+)?", flags=re.I),
    re.compile(r"^\s*go\s+ahead\s+and\s+", flags=re.I),
)
_INFORMAL_KNOW_RE = re.compile(
    r"^\s*(?:i|me)\s+(?:want|need)\s+(?:to\s+)?(?:know|learn|hear)\s+(?:about\s+|more\s+about\s+)?(.+?)\s*$",
    flags=re.I,
)
_TELL_WH_RE = re.compile(
    r"^\s*(?:tell\s+me|let\s+me\s+know)\s+(what|where|why|how|who|when|which)\b\s*(.+?)\s*$",
    flags=re.I,
)
_TELL_ABOUT_RE = re.compile(
    r"^\s*(?:tell\s+me|let\s+me\s+know)\s+about\s+(.+?)\s*$",
    flags=re.I,
)
_TRAILING_POLITE_RE = re.compile(
    r"(?:\s*,\s*|\s+)(?:please|thanks|thank\s+you|for\s+me)(?=\s*[.?!]*\s*$)",
    flags=re.I,
)


def damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Compute bounded Optimal String Alignment (Damerau-Levenshtein) distance."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if a == b:
        return 0
    if la == 0:
        return lb
    if lb == 0:
        return la

    # Previous two rows suffice when no adjacent transpositions, three for Damerau
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


def strip_conversational_wrappers(text: str) -> tuple[str, bool]:
    """Strip polite/conversational framing & expand contractions deterministically.

    Returns ``(cleaned_text, had_wrapper)`` while preserving trailing punctuation
    and URLs/filenames.
    """
    cleaned = text.strip()
    if not cleaned:
        return "", False

    had_wrapper = False

    # Expand safe contractions before pattern stripping
    def _expand(match: re.Match[str]) -> str:
        return CONTRACTIONS[match.group(1).lower()]

    expanded = _CONTRACTION_RE.sub(_expand, cleaned)
    if expanded != cleaned:
        cleaned = expanded
        had_wrapper = True

    # Strip leading list markers (e.g. "1. ", "- ", "Step 2: ")
    m_list = _LIST_MARKER_RE.match(cleaned)
    if m_list and len(cleaned[m_list.end() :].strip()) >= 3:
        cleaned = cleaned[m_list.end() :].strip()
        had_wrapper = True

    # Iteratively strip polite wrappers and leading sequence adverbs
    for _ in range(4):
        before = cleaned
        m_seq = _SEQ_PREFIX_RE.match(cleaned)
        if m_seq and len(cleaned[m_seq.end() :].split()) >= 2:
            cleaned = cleaned[m_seq.end() :].strip()
        for pat in _POLITE_PREFIXES:
            m = pat.match(cleaned)
            if m and len(cleaned[m.end() :].strip()) >= 2:
                cleaned = cleaned[m.end() :].strip()
        cleaned_trailing = _TRAILING_POLITE_RE.sub("", cleaned).strip()
        if cleaned_trailing != cleaned and len(cleaned_trailing) >= 2:
            cleaned = cleaned_trailing
        if cleaned == before:
            break
        had_wrapper = True

    # Strip trailing punctuation temporarily for rewrite patterns, preserving '?' if present
    end_punct = ""
    if cleaned and cleaned[-1] in ".?!":
        end_punct = cleaned[-1]
        core = cleaned[:-1].strip()
    else:
        core = cleaned

    m_inf = _INFORMAL_KNOW_RE.match(core)
    if m_inf:
        target = m_inf.group(1).strip()
        if target:
            cleaned = f"explain {target}{end_punct}"
            had_wrapper = True
            return cleaned, had_wrapper

    m_twh = _TELL_WH_RE.match(core)
    if m_twh:
        wh = m_twh.group(1).lower()
        rest = m_twh.group(2).strip()
        if rest:
            cleaned = f"{wh} {rest}{end_punct}"
            had_wrapper = True
            return cleaned, had_wrapper

    m_tabout = _TELL_ABOUT_RE.match(core)
    if m_tabout:
        target = m_tabout.group(1).strip()
        if target:
            cleaned = f"explain {target}{end_punct}"
            had_wrapper = True
            return cleaned, had_wrapper

    return cleaned, had_wrapper


class OrthographicNormalizer:
    """Two-tier zero-LLM typo normalizer with LRU cache & strict false-positive guards."""

    def __init__(
        self,
        lexicon: Any | None = None,
        ontology_store: Any | None = None,
        cache_size: int = 1024,
    ) -> None:
        self._lexicon = lexicon
        self._ontology_store = ontology_store
        self._cache_size = cache_size
        self._cache: dict[str, str] = {}
        self._dynamic_vocab: set[str] = set()

    def set_dynamic_vocabulary(self, words: Iterable[str]) -> None:
        """Update active world/persistent entity & concept vocabulary for Tier-1 fuzzy matching."""
        cleaned: set[str] = set()
        for w in words:
            for part in str(w).lower().replace("_", " ").split():
                if len(part) >= 3 and part.isalpha():
                    cleaned.add(part)
        self._dynamic_vocab = cleaned

    def _is_trusted(self, low: str) -> bool:
        if low in CLOSED_CLASS_VOCAB or low in COMMAND_VERB_VOCAB or low in self._dynamic_vocab:
            return True
        if self._lexicon is not None:
            if hasattr(self._lexicon, "is_trusted_word"):
                if self._lexicon.is_trusted_word(low):
                    return True
            elif self._lexicon.has_word(low):
                return True
        store = self._get_store()
        if store is not None and hasattr(store, "is_trusted_word"):
            return bool(store.is_trusted_word(low, language="en"))
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
        if low in COMMON_TYPOS:
            return COMMON_TYPOS[low]

        if self._is_trusted(low):
            return token

        cached = self._cache.get(low)
        if cached is not None:
            return cached

        # Tier 1: Fast in-memory closed-class + command verbs + base lexicon + dynamic world vocab
        tier0_words = CLOSED_CLASS_VOCAB | COMMAND_VERB_VOCAB
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
