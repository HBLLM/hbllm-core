"""Error Classifier — probabilistic prediction error diagnosis for A14.

Classifies prediction errors as a probability distribution over four
categories, NOT a single label.  The distribution shifts as evidence
accumulates — error classification is itself epistemic.

Categories::

    MODEL_ERROR         — internal model parameters are wrong (learning signal)
    ENVIRONMENT_CHANGE  — world genuinely changed state
    NOISE               — transient measurement error (ignore)
    NOVELTY             — genuinely new phenomenon (exploration signal)

**Critical invariant:** The classifier NEVER triggers model mutation.
It produces a classification distribution.  Only the AdaptationGate
can authorize model changes.

Architecture::

    PredictionErrorNode
            ↓
    ErrorClassifier.classify()
            ↓
    ErrorClassification(
        model_error=0.72,
        environment_change=0.18,
        noise=0.07,
        novelty=0.03,
    )
            ↓
    LearningSignalRouter / AdaptationGate
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ErrorClassification — probabilistic diagnosis
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ErrorClassification:
    """Probabilistic diagnosis of a prediction error.

    The four probabilities sum to ≈1.0 (within tolerance).
    The classifier never discards the distribution — all four
    values are valuable evidence for the later gate.
    """

    model_error: float = 0.25
    environment_change: float = 0.25
    noise: float = 0.25
    novelty: float = 0.25

    def __post_init__(self) -> None:
        total = self.model_error + self.environment_change + self.noise + self.novelty
        if abs(total - 1.0) > 0.01:
            msg = f"ErrorClassification probabilities must sum to ≈1.0, got {total:.4f}"
            raise ValueError(msg)

    @property
    def dominant_class(self) -> str:
        """Return the class with highest probability."""
        classes = {
            "model_error": self.model_error,
            "environment_change": self.environment_change,
            "noise": self.noise,
            "novelty": self.novelty,
        }
        return max(classes, key=classes.get)  # type: ignore[arg-type]

    @property
    def dominant_probability(self) -> float:
        """Return the probability of the dominant class."""
        return max(
            self.model_error,
            self.environment_change,
            self.noise,
            self.novelty,
        )

    @property
    def entropy(self) -> float:
        """Shannon entropy of the classification (lower = more certain)."""
        probs = [
            self.model_error,
            self.environment_change,
            self.noise,
            self.novelty,
        ]
        return -sum(p * math.log2(p) if p > 0 else 0.0 for p in probs)

    def as_dict(self) -> dict[str, float]:
        return {
            "model_error": self.model_error,
            "environment_change": self.environment_change,
            "noise": self.noise,
            "novelty": self.novelty,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ErrorContext — input to the classifier
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ErrorContext:
    """Context for classifying a prediction error.

    Provides the signals the classifier uses to compute
    the probabilistic diagnosis.
    """

    error_magnitude: float = 0.0  # |predicted - observed|
    prediction_confidence: float = 0.5  # How confident was the prediction?
    historical_error_rate: float = 0.0  # Error rate for this prediction type
    temporal_pattern: str = "isolated"  # "isolated", "sudden", "gradual", "recurring"
    cross_entity_correlation: float = 0.0  # Were other entities wrong too? (0-1)
    recency_weighted_frequency: float = 0.0  # How often has this error occurred recently?
    time_since_last_similar: float = float("inf")  # Seconds since last similar error
    entity_type: str = ""
    prediction_domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Error Classifier
# ═══════════════════════════════════════════════════════════════════════════


class ErrorClassifier:
    """Probabilistic prediction error classifier for A14.

    Produces a classification distribution over the four error types.
    Classification is deterministic given the same ErrorContext —
    supports replay.

    **Does NOT:**
    - Trigger model mutation
    - Own adaptation decisions
    - Interact with HCIR directly

    **Does:**
    - Compute probabilistic error diagnosis from context signals
    - Maintain classification history for trend analysis

    Usage::

        classifier = ErrorClassifier()
        classification = classifier.classify(error_context)
        # → ErrorClassification(model_error=0.72, ...)
    """

    def __init__(self) -> None:
        # Classification history for trend analysis
        self._history: list[tuple[str, ErrorClassification]] = []  # (error_id, classification)

    # ── Classification ────────────────────────────────────────────────

    def classify(
        self,
        context: ErrorContext,
        error_id: str = "",
    ) -> ErrorClassification:
        """Classify a prediction error as a probability distribution.

        Uses multiple signals to compute posterior probabilities for
        each error category.  The classification is deterministic
        given the same context.

        Args:
            context: The error context with all classification signals.
            error_id: Optional error ID for history tracking.

        Returns:
            ErrorClassification with probabilities summing to ≈1.0.
        """
        # Compute raw scores for each category
        scores = {
            "model_error": self._score_model_error(context),
            "environment_change": self._score_environment_change(context),
            "noise": self._score_noise(context),
            "novelty": self._score_novelty(context),
        }

        # Normalize to probability distribution
        total = sum(scores.values())
        if total <= 0:
            total = 1.0  # Uniform fallback

        classification = ErrorClassification(
            model_error=scores["model_error"] / total,
            environment_change=scores["environment_change"] / total,
            noise=scores["noise"] / total,
            novelty=scores["novelty"] / total,
        )

        # Record in history
        if error_id:
            self._history.append((error_id, classification))

        return classification

    # ── Scoring Functions ─────────────────────────────────────────────

    def _score_model_error(self, ctx: ErrorContext) -> float:
        """Model error signal: systematic, recurring, gradual drift."""
        score = 0.3  # Base prior

        # High historical error rate → model is systematically wrong
        score += ctx.historical_error_rate * 2.0

        # Recurring temporal pattern → model inadequacy
        if ctx.temporal_pattern == "recurring":
            score += 1.5
        elif ctx.temporal_pattern == "gradual":
            score += 1.0

        # High prediction confidence + error → model is confidently wrong
        if ctx.prediction_confidence > 0.7 and ctx.error_magnitude > 0.3:
            score += 0.8

        # Frequent recent errors of same type
        score += ctx.recency_weighted_frequency * 1.2

        # Low cross-entity correlation → entity-specific model failure
        if ctx.cross_entity_correlation < 0.2:
            score += 0.3

        return max(score, 0.01)

    def _score_environment_change(self, ctx: ErrorContext) -> float:
        """Environment change signal: sudden, correlated across entities."""
        score = 0.2  # Base prior

        # Sudden temporal pattern → world transition
        if ctx.temporal_pattern == "sudden":
            score += 1.5

        # High cross-entity correlation → global state change
        score += ctx.cross_entity_correlation * 2.0

        # Large error magnitude → something big changed
        if ctx.error_magnitude > 0.5:
            score += 0.5

        # Isolated occurrence (not recurring) → one-time world event
        if ctx.temporal_pattern == "isolated" and ctx.recency_weighted_frequency < 0.1:
            score += 0.4

        return max(score, 0.01)

    def _score_noise(self, ctx: ErrorContext) -> float:
        """Noise signal: small, isolated, uncorrelated."""
        score = 0.2  # Base prior

        # Small error magnitude → likely measurement noise
        if ctx.error_magnitude < 0.15:
            score += 1.5
        elif ctx.error_magnitude < 0.3:
            score += 0.5

        # Isolated temporal pattern → transient
        if ctx.temporal_pattern == "isolated":
            score += 0.8

        # Low cross-entity correlation → not systematic
        if ctx.cross_entity_correlation < 0.1:
            score += 0.3

        # Low recency frequency → rare
        if ctx.recency_weighted_frequency < 0.05:
            score += 0.3

        return max(score, 0.01)

    def _score_novelty(self, ctx: ErrorContext) -> float:
        """Novelty signal: no historical precedent, low prior for this domain."""
        score = 0.1  # Low base prior (most errors are not genuinely novel)

        # No similar errors in history → potentially novel
        if ctx.time_since_last_similar == float("inf"):
            score += 1.0

        # Very low historical error rate → this domain has been stable
        if ctx.historical_error_rate < 0.05:
            score += 0.5

        # Large magnitude + no precedent → genuinely new phenomenon
        if ctx.error_magnitude > 0.5 and ctx.time_since_last_similar > 300:
            score += 0.8

        return max(score, 0.01)

    # ── History / Trend Analysis ──────────────────────────────────────

    def classification_trend(
        self,
        last_n: int = 10,
    ) -> dict[str, float]:
        """Compute average classification distribution over recent history.

        Useful for detecting whether MODEL_ERROR probability is
        increasing over time (suggesting systematic model inadequacy).
        """
        if not self._history:
            return {
                "model_error": 0.25,
                "environment_change": 0.25,
                "noise": 0.25,
                "novelty": 0.25,
            }

        recent = self._history[-last_n:]
        n = len(recent)

        avg = {
            "model_error": sum(c.model_error for _, c in recent) / n,
            "environment_change": sum(c.environment_change for _, c in recent) / n,
            "noise": sum(c.noise for _, c in recent) / n,
            "novelty": sum(c.novelty for _, c in recent) / n,
        }
        return avg

    @property
    def total_classifications(self) -> int:
        return len(self._history)
