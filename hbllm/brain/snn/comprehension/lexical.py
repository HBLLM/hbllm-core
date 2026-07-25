"""
Lexical analysis layers for SNN comprehension.

Layer 0 — LexicalBuffer:
    Absorbs subword tokenizer noise and emits clean word/phrase
    candidates.  Pure string manipulation, no ML.

Layer 1 — LexicalSignals:
    Computes cheap (~0.01ms) feature signals from a word for SNN
    input.  Uses frozenset lookups — no embeddings, no ML.
"""

from __future__ import annotations

import re  # noqa: F401 — used implicitly by callers for pattern work
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hbllm.modules.domain_registry import DomainRegistry

# ── Frozenset vocabularies ────────────────────────────────────────────────

_TECHNICAL_TERMS_BASE: frozenset[str] = frozenset(
    {
        "api",
        "auth",
        "database",
        "server",
        "deploy",
        "production",
        "error",
        "bug",
        "config",
        "docker",
        "kubernetes",
        "nginx",
        "firebase",
        "flutter",
        "laravel",
        "django",
        "react",
        "sql",
        "redis",
        "cache",
        "token",
        "jwt",
        "oauth",
        "ssl",
        "https",
        "cpu",
        "gpu",
        "memory",
        "thread",
        "async",
        "queue",
        "webhook",
        "endpoint",
        "middleware",
        "backend",
        "frontend",
        "runtime",
        "compiler",
        "debugger",
        "pipeline",
        "microservice",
        "container",
        "cluster",
        "schema",
        "migration",
        "graphql",
        "rest",
        "grpc",
        "websocket",
        "terraform",
        "ansible",
        "cicd",
        "prometheus",
        "grafana",
    }
)

# Mutable copy — extended at startup via populate_from_registry()
_TECHNICAL_TERMS: set[str] = set(_TECHNICAL_TERMS_BASE)

_CONSTRAINT_WORDS: frozenset[str] = frozenset(
    {
        "only",
        "except",
        "but",
        "unless",
        "however",
        "although",
        "instead",
        "rather",
        "despite",
        "without",
        "never",
        "always",
        "specifically",
        "exclusively",
    }
)

_SHIFT_WORDS: frozenset[str] = frozenset(
    {
        "and",
        "also",
        "plus",
        "additionally",
        "moreover",
        "then",
        "next",
        "after",
        "finally",
        "meanwhile",
    }
)

_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "how",
        "why",
        "where",
        "when",
        "which",
        "who",
        "can",
        "could",
        "should",
        "would",
        "does",
        "is",
    }
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "in",
        "of",
        "for",
        "on",
        "with",
        "it",
        "this",
        "that",
        "i",
        "my",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "did",
        "at",
        "by",
        "from",
        "or",
        "as",
        "if",
        "so",
        "no",
        "not",
        "its",
        "me",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "their",
    }
)


def populate_from_registry(domain_registry: DomainRegistry) -> None:
    """Extend _TECHNICAL_TERMS with words extracted from DomainRegistry centroid texts.

    This makes entity detection domain-aware without adding per-word cost.
    Called once at startup.
    """
    for _domain_name, centroid_text in domain_registry.centroid_texts().items():
        for word in centroid_text.lower().split():
            # Only add words that are meaningful (>3 chars, not stopwords)
            cleaned = word.strip(".,!?\"'()[]{}:;")
            if len(cleaned) > 3 and cleaned not in _STOPWORDS:
                _TECHNICAL_TERMS.add(cleaned)


# ── Layer 0: Subword Noise Absorption ─────────────────────────────────────


class LexicalBuffer:
    """Layer 0: Absorbs subword noise and emits clean word/phrase candidates.

    Buffers subword tokens until a word boundary is detected.
    No ML — pure string/character operations.
    """

    def __init__(self) -> None:
        self._subword_buffer: list[str] = []

    def feed(self, subword: str) -> str | None:
        """Feed a subword token.

        Returns a complete word/phrase when a boundary is detected,
        None otherwise.
        """
        # Detect subword continuation markers:
        #   - "##..." is a BERT-style continuation (e.g., "##ation" after "auth")
        #   - "▁..." is a SentencePiece word-start marker (its ABSENCE means continuation)
        # Regular standalone words (no prefix) are always new words.
        is_continuation = subword.startswith("##")

        if is_continuation and self._subword_buffer:
            # Still building the same word — merge
            clean = subword.lstrip("#")
            self._subword_buffer.append(clean)
            return None
        else:
            # New word starting — flush previous
            result = None
            if self._subword_buffer:
                result = "".join(self._subword_buffer)

            clean = subword.lstrip("#▁ ")
            self._subword_buffer = [clean] if clean else []
            return result

    def flush(self) -> str | None:
        """Flush the remaining buffer content."""
        if self._subword_buffer:
            result = "".join(self._subword_buffer)
            self._subword_buffer = []
            return result
        return None

    def reset(self) -> None:
        """Clear the buffer state."""
        self._subword_buffer = []


# ── Layer 1: Cheap Lexical Signals ────────────────────────────────────────


class LexicalSignals:
    """Computes cheap (~0.01ms) features from a word for SNN input.

    No embeddings.  No ML.  Pure lexical analysis using frozenset
    lookups and simple string operations.

    Signal channels (all in [0.0, 1.0]):
        semantic_weight — content word vs filler
        topic_shift     — new sub-topic indicator
        novelty         — word is new to current buffer
        inter_novelty   — word is new vs previous concept
        constraint      — qualifier that modifies meaning
        buffer_pressure — accumulation without spike
        punctuation     — natural pause point
    """

    @staticmethod
    def compute(
        word: str,
        buffer: list[str],
        prev_concept_words: list[str] | None,
    ) -> dict[str, float]:
        """Compute signal channels for a single word.

        Args:
            word: The current clean word.
            buffer: Words accumulated since last spike.
            prev_concept_words: Words from the previous completed concept.

        Returns:
            Dict of signal channel names to values in [0.0, 1.0].
        """
        w = word.lower().strip(".,!?\"'()[]{}:;")

        signals: dict[str, float] = {}

        # 1. Semantic weight: is this a content word or filler?
        if w in _STOPWORDS:
            signals["semantic_weight"] = 0.0
        elif w in _TECHNICAL_TERMS:
            signals["semantic_weight"] = 0.8
        elif len(w) > 4:  # Longer words tend to be more meaningful
            signals["semantic_weight"] = 0.4
        else:
            signals["semantic_weight"] = 0.2

        # 2. Topic shift: does this word signal a new sub-topic?
        if w in _SHIFT_WORDS:
            signals["topic_shift"] = 0.7
        elif w in _CONSTRAINT_WORDS:
            signals["topic_shift"] = 0.5  # Constraints modify, not shift
        else:
            signals["topic_shift"] = 0.0

        # 3. Novelty: does this word appear in the current buffer?
        buffer_words = {bw.lower() for bw in buffer}
        if w not in buffer_words and w not in _STOPWORDS:
            signals["novelty"] = 0.5
        else:
            signals["novelty"] = 0.0

        # 4. Inter-concept novelty: is this word new vs previous concept?
        if prev_concept_words:
            prev_set = {pw.lower() for pw in prev_concept_words}
            if w not in prev_set and w not in _STOPWORDS:
                signals["inter_novelty"] = 0.4
            else:
                signals["inter_novelty"] = 0.0
        else:
            signals["inter_novelty"] = 0.2  # First concept gets moderate

        # 5. Constraint signal: qualifiers that modify meaning significantly
        if w in _CONSTRAINT_WORDS:
            signals["constraint"] = 0.8
        else:
            signals["constraint"] = 0.0

        # 6. Buffer pressure: accumulation without spike
        signals["buffer_pressure"] = min(1.0, len(buffer) / 12) * 0.3

        # 7. Punctuation boundary: natural pause points
        if word.rstrip().endswith((",", ".", "!", "?", ";", ":")):
            signals["punctuation"] = 0.4
        else:
            signals["punctuation"] = 0.0

        return signals
