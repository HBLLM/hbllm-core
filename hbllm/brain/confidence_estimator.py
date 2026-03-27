"""
Confidence Estimator — scores response reliability and hallucination risk.

Multi-signal confidence scoring:
- Lexical overlap with query
- Consistency checking
- Uncertainty language detection
- Token-level entropy estimation
- Source attribution coverage
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceReport:
    """Detailed confidence analysis of a response."""
    overall: float  # 0.0 - 1.0
    relevance: float
    coherence: float
    factuality_risk: float  # 0.0 = likely factual, 1.0 = likely hallucinated
    uncertainty: float
    detail_level: float
    flags: list[str] = field(default_factory=list)


# Hedging language patterns
_HEDGE_PATTERNS = re.compile(
    r"\b(i think|maybe|perhaps|possibly|might be|not sure|could be|"
    r"i believe|it seems|approximately|roughly|around)\b",
    re.IGNORECASE,
)

# Definitive claim patterns (high confidence language)
_DEFINITIVE_PATTERNS = re.compile(
    r"\b(definitely|certainly|always|never|absolutely|exactly|"
    r"proven|guaranteed|100%)\b",
    re.IGNORECASE,
)

# Number/date patterns (claims that could be hallucinated)
_FACTUAL_CLAIM_PATTERNS = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d+\.\d+%|\$\d+|"
    r"founded in \d{4}|born in \d{4}|died in \d{4})\b",
    re.IGNORECASE,
)


class ConfidenceEstimator:
    """
    Estimates response confidence using multi-signal analysis.

    Signals:
    1. Query-response relevance (lexical overlap)
    2. Response coherence (sentence consistency)
    3. Factuality risk (hallucination indicators)
    4. Uncertainty language detection
    5. Detail level (specificity vs vagueness)
    """

    def __init__(
        self,
        hallucination_threshold: float = 0.6,
        weights: dict[str, float] | None = None,
    ):
        self.hallucination_threshold = hallucination_threshold
        self.weights = weights or {
            "relevance": 0.25,
            "coherence": 0.20,
            "factuality": 0.25,
            "uncertainty": 0.15,
            "detail": 0.15,
        }

    def estimate(self, query: str, response: str) -> ConfidenceReport:
        """
        Produce a full confidence report for a query-response pair.
        """
        relevance = self._score_relevance(query, response)
        coherence = self._score_coherence(response)
        factuality_risk = self._score_factuality_risk(response)
        uncertainty = self._score_uncertainty(response)
        detail = self._score_detail(response)

        # Combine into overall score
        overall = (
            self.weights["relevance"] * relevance
            + self.weights["coherence"] * coherence
            + self.weights["factuality"] * (1.0 - factuality_risk)
            + self.weights["uncertainty"] * (1.0 - uncertainty)
            + self.weights["detail"] * detail
        )

        flags = []
        if factuality_risk > self.hallucination_threshold:
            flags.append("high_hallucination_risk")
            overall *= 0.8
        if uncertainty > 0.6:
            flags.append("high_uncertainty")
        if relevance < 0.3:
            flags.append("low_relevance")
            overall *= 0.8
        if len(response.split()) < 5:
            flags.append("too_brief")
            overall *= 0.8
            
        overall = max(0.0, min(1.0, overall))

        return ConfidenceReport(
            overall=round(max(0.0, min(1.0, overall)), 3),
            relevance=round(relevance, 3),
            coherence=round(coherence, 3),
            factuality_risk=round(factuality_risk, 3),
            uncertainty=round(uncertainty, 3),
            detail_level=round(detail, 3),
            flags=flags,
        )

    def score(self, query: str, response: str) -> float:
        """Quick confidence score (0-1). Use estimate() for full report."""
        return self.estimate(query, response).overall

    # ─── Individual Scorers ──────────────────────────────────────────

    def _score_relevance(self, query: str, response: str) -> float:
        """Query-response relevance via lexical overlap."""
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        r_words = set(re.findall(r'\b\w+\b', response.lower()))
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                     "to", "in", "of", "for", "on", "with", "it", "this", "that"}
        q_words -= stopwords
        r_words -= stopwords

        if not q_words:
            return 0.5
        overlap = len(q_words & r_words) / len(q_words)
        return min(1.0, overlap * 1.5)

    def _score_coherence(self, response: str) -> float:
        """Score response coherence based on structure."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]

        if len(sentences) <= 1:
            return 0.6

        # Check for repetition (incoherence signal)
        unique = set(s.lower() for s in sentences)
        repetition_ratio = len(unique) / len(sentences)

        # Average sentence length consistency
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            cv = (variance ** 0.5) / max(mean_len, 1)  # coefficient of variation
            length_consistency = max(0.0, 1.0 - cv * 0.5)
        else:
            length_consistency = 0.5

        return (repetition_ratio * 0.6 + length_consistency * 0.4)

    def _score_factuality_risk(self, response: str) -> float:
        """Estimate risk of factual hallucination."""
        risk = 0.0

        # Count unverifiable factual claims
        claims = _FACTUAL_CLAIM_PATTERNS.findall(response)
        if claims:
            risk += min(0.5, len(claims) * 0.15)

        # Overly definitive language without sources
        definitive = _DEFINITIVE_PATTERNS.findall(response)
        if definitive:
            risk += min(0.3, len(definitive) * 0.1)

        # Long numeric sequences (likely made up)
        if re.search(r'\b\d{6,}\b', response):
            risk += 0.2

        return min(1.0, risk)

    def _score_uncertainty(self, response: str) -> float:
        """Detect hedging/uncertainty language."""
        hedges = _HEDGE_PATTERNS.findall(response)
        words = response.split()
        if not words:
            return 0.5
        return min(1.0, len(hedges) / max(len(words) / 20, 1))

    def _score_detail(self, response: str) -> float:
        """Score the detail/specificity level."""
        words = response.split()
        word_count = len(words)

        if word_count < 5:
            return 0.1
        if word_count < 20:
            return 0.4
        if word_count < 100:
            return 0.7
        return 0.9
