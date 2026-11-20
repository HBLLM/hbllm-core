"""Deterministic English Orthographic Normalizer & Conversational Wrapper Stripper.

Zero-LLM, sub-millisecond normalization for typos, contractions, and noisy
conversational wrappers while protecting filenames, URLs, numbers, and valid English words.
"""

from __future__ import annotations

import re
from typing import Any

from hbllm.hcir.language.core.normalizer import (
    BaseOrthographicNormalizer,
    damerau_levenshtein,
)

__all__ = [
    "OrthographicNormalizer",
    "BaseOrthographicNormalizer",
    "damerau_levenshtein",
]

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


class OrthographicNormalizer(BaseOrthographicNormalizer):
    """Two-tier zero-LLM typo normalizer with LRU cache & strict false-positive guards."""

    def __init__(
        self,
        lexicon: Any | None = None,
        ontology_store: Any | None = None,
        cache_size: int = 1024,
    ) -> None:
        super().__init__(
            closed_class_vocab=CLOSED_CLASS_VOCAB,
            command_verbs=COMMAND_VERB_VOCAB,
            common_typos=COMMON_TYPOS,
            lexicon=lexicon,
            ontology_store=ontology_store,
            language="en",
            cache_size=cache_size,
        )
