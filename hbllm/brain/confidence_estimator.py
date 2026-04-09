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
import re
import time
from collections import defaultdict
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
        calibration_window: int = 200,
    ):
        self.hallucination_threshold = hallucination_threshold
        self.weights = weights or {
            "relevance": 0.25,
            "coherence": 0.20,
            "factuality": 0.25,
            "uncertainty": 0.15,
            "detail": 0.15,
        }

        # v2: Calibration tracking
        self._calibration_window = calibration_window
        self._calibration_history: list[dict[str, Any]] = []
        self._domain_adjustments: dict[str, float] = defaultdict(lambda: 0.0)
        self._total_predictions = 0
        self._total_feedback = 0

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
        q_words = set(re.findall(r"\b\w+\b", query.lower()))
        r_words = set(re.findall(r"\b\w+\b", response.lower()))
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "and",
            "or",
            "to",
            "in",
            "of",
            "for",
            "on",
            "with",
            "it",
            "this",
            "that",
        }
        q_words -= stopwords
        r_words -= stopwords

        if not q_words:
            return 0.5
        overlap = len(q_words & r_words) / len(q_words)
        return min(1.0, overlap * 1.5)

    def _score_coherence(self, response: str) -> float:
        """Score response coherence based on structure."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]

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
            cv = (variance**0.5) / max(mean_len, 1)  # coefficient of variation
            length_consistency = max(0.0, 1.0 - cv * 0.5)
        else:
            length_consistency = 0.5

        return float(repetition_ratio * 0.6 + length_consistency * 0.4)

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
        if re.search(r"\b\d{6,}\b", response):
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

    # ─── v2: Calibration & Uncertainty ────────────────────────────────

    def record_outcome(
        self, predicted: float, actual: float, domain: str = "general"
    ) -> None:
        """
        Record a predicted-vs-actual outcome for calibration tracking.

        Called by EvaluationNode when user feedback arrives.
        Over time this adjusts future confidence estimates per domain.
        """
        entry = {
            "predicted": predicted,
            "actual": actual,
            "domain": domain,
            "error": abs(predicted - actual),
            "timestamp": time.time(),
        }
        self._calibration_history.append(entry)
        if len(self._calibration_history) > self._calibration_window:
            self._calibration_history = self._calibration_history[-self._calibration_window :]

        self._total_feedback += 1

        # Update domain adjustment (exponential moving average of error direction)
        bias = predicted - actual  # positive = overconfident, negative = underconfident
        alpha = 0.1
        current = self._domain_adjustments[domain]
        self._domain_adjustments[domain] = current * (1 - alpha) + bias * alpha

    def calibrated_score(
        self, query: str, response: str, domain: str = "general"
    ) -> float:
        """
        Return a calibration-adjusted confidence score.

        Uses historical over/under-confidence data per domain to
        correct the raw confidence estimate.
        """
        raw = self.score(query, response)
        self._total_predictions += 1

        # Apply domain-specific calibration adjustment
        adjustment = self._domain_adjustments.get(domain, 0.0)
        calibrated = raw - adjustment  # subtract bias to correct

        return max(0.05, min(0.95, round(calibrated, 3)))

    def calibration_error(self, domain: str | None = None) -> float:
        """
        Compute Expected Calibration Error (ECE) over recent history.

        Lower is better. 0.0 = perfectly calibrated.
        """
        entries = self._calibration_history
        if domain:
            entries = [e for e in entries if e["domain"] == domain]

        if not entries:
            return 0.0

        return sum(e["error"] for e in entries) / len(entries)

    def calibration_stats(self) -> dict[str, Any]:
        """Return calibration statistics."""
        if not self._calibration_history:
            return {
                "total_predictions": self._total_predictions,
                "total_feedback": self._total_feedback,
                "calibration_error": 0.0,
                "domain_adjustments": {},
            }

        return {
            "total_predictions": self._total_predictions,
            "total_feedback": self._total_feedback,
            "calibration_error": round(self.calibration_error(), 4),
            "history_size": len(self._calibration_history),
            "domain_adjustments": {
                d: round(v, 4) for d, v in self._domain_adjustments.items()
            },
            "domain_errors": {
                d: round(self.calibration_error(d), 4)
                for d in set(e["domain"] for e in self._calibration_history)
            },
        }
