"""Prediction Evaluation Engine — meta-learning for A14.

Determines whether an adaptation actually improved prediction.
Closes the meta-learning loop:

    Adaptation → Evaluation → Was it useful? → Update adaptation policy

Without this, the system cannot distinguish beneficial adaptations
from harmful ones.

Evaluates:
- Pre-adaptation accuracy vs post-adaptation accuracy
- Cross-domain impact (adapting model A degraded domain B?)
- Overall learning trajectory

This is meta-learning of the learning process, without LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Types
# ═══════════════════════════════════════════════════════════════════════════


class AdaptationOutcome(StrEnum):
    """Outcome of evaluating an adaptation."""

    IMPROVED = "improved"
    DEGRADED = "degraded"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a single adaptation event.

    Includes cross-domain impact to detect whether adapting
    one model degraded predictions in another domain.
    """

    adaptation_id: str
    model_id: str
    outcome: AdaptationOutcome
    accuracy_before: float
    accuracy_after: float
    delta: float  # accuracy_after - accuracy_before
    sample_count: int  # Predictions evaluated
    cross_domain_impact: dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Prediction Evaluation Engine
# ═══════════════════════════════════════════════════════════════════════════


class PredictionEvaluationEngine:
    """Evaluates whether adaptations improved predictions.

    Compares pre-adaptation and post-adaptation accuracy windows
    to determine if the adaptation was beneficial, neutral, or harmful.

    Also detects cross-domain degradation (plasticity vs stability).

    Usage::

        evaluator = PredictionEvaluationEngine()

        # Before adaptation
        evaluator.record_pre_window(model_id, predictions=[...])

        # ... adaptation happens ...

        # After adaptation
        evaluator.record_post_window(model_id, predictions=[...])

        # Evaluate
        result = evaluator.evaluate(adaptation_id, model_id)
    """

    def __init__(
        self,
        improvement_threshold: float = 0.02,
        degradation_threshold: float = -0.02,
    ) -> None:
        self._improvement_threshold = improvement_threshold
        self._degradation_threshold = degradation_threshold

        # Pre/post windows: model_id → list of (predicted_correct: bool)
        self._pre_windows: dict[str, list[bool]] = {}
        self._post_windows: dict[str, list[bool]] = {}

        # Cross-domain tracking: domain → list of (predicted_correct: bool)
        self._domain_pre_accuracy: dict[str, float] = {}
        self._domain_post_accuracy: dict[str, float] = {}

        # History of evaluations
        self._history: list[EvaluationResult] = []

    # ── Window Recording ──────────────────────────────────────────────

    def record_pre_window(
        self,
        model_id: str,
        outcomes: list[bool],
    ) -> None:
        """Record prediction outcomes in the pre-adaptation window.

        Args:
            model_id: The model being evaluated.
            outcomes: List of booleans (True = correct prediction).
        """
        self._pre_windows[model_id] = list(outcomes)

    def record_post_window(
        self,
        model_id: str,
        outcomes: list[bool],
    ) -> None:
        """Record prediction outcomes in the post-adaptation window.

        Args:
            model_id: The model being evaluated.
            outcomes: List of booleans (True = correct prediction).
        """
        self._post_windows[model_id] = list(outcomes)

    def record_domain_accuracy(
        self,
        domain: str,
        pre_accuracy: float,
        post_accuracy: float,
    ) -> None:
        """Record per-domain accuracy for cross-domain impact analysis."""
        self._domain_pre_accuracy[domain] = pre_accuracy
        self._domain_post_accuracy[domain] = post_accuracy

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        adaptation_id: str,
        model_id: str,
    ) -> EvaluationResult:
        """Evaluate whether an adaptation improved predictions.

        Compares pre-adaptation and post-adaptation accuracy.
        Also checks cross-domain impact.

        Args:
            adaptation_id: The AdaptationEventNode ID.
            model_id: The adapted model ID.

        Returns:
            EvaluationResult with outcome and metrics.
        """
        pre = self._pre_windows.get(model_id, [])
        post = self._post_windows.get(model_id, [])

        accuracy_before = sum(pre) / len(pre) if pre else 0.5
        accuracy_after = sum(post) / len(post) if post else 0.5
        delta = accuracy_after - accuracy_before

        # Determine outcome
        if delta >= self._improvement_threshold:
            outcome = AdaptationOutcome.IMPROVED
        elif delta <= self._degradation_threshold:
            outcome = AdaptationOutcome.DEGRADED
        else:
            outcome = AdaptationOutcome.NEUTRAL

        # Cross-domain impact
        cross_domain: dict[str, float] = {}
        for domain in self._domain_post_accuracy:
            pre_acc = self._domain_pre_accuracy.get(domain, 0.5)
            post_acc = self._domain_post_accuracy[domain]
            cross_domain[domain] = post_acc - pre_acc

        result = EvaluationResult(
            adaptation_id=adaptation_id,
            model_id=model_id,
            outcome=outcome,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            delta=delta,
            sample_count=len(pre) + len(post),
            cross_domain_impact=cross_domain,
        )

        self._history.append(result)

        logger.debug(
            "PredictionEvaluationEngine: adaptation %s → %s (%.2f → %.2f, Δ=%.3f)",
            adaptation_id,
            outcome,
            accuracy_before,
            accuracy_after,
            delta,
        )

        return result

    # ── Evaluate from Direct Accuracy Values ──────────────────────────

    def evaluate_from_accuracy(
        self,
        adaptation_id: str,
        model_id: str,
        accuracy_before: float,
        accuracy_after: float,
        sample_count: int = 0,
        cross_domain_impact: dict[str, float] | None = None,
    ) -> EvaluationResult:
        """Evaluate using direct accuracy values (no window needed).

        Useful when accuracy is tracked externally (e.g., by
        PredictiveModelRegistry).
        """
        delta = accuracy_after - accuracy_before

        if delta >= self._improvement_threshold:
            outcome = AdaptationOutcome.IMPROVED
        elif delta <= self._degradation_threshold:
            outcome = AdaptationOutcome.DEGRADED
        else:
            outcome = AdaptationOutcome.NEUTRAL

        result = EvaluationResult(
            adaptation_id=adaptation_id,
            model_id=model_id,
            outcome=outcome,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            delta=delta,
            sample_count=sample_count,
            cross_domain_impact=cross_domain_impact or {},
        )

        self._history.append(result)
        return result

    # ── History / Analysis ────────────────────────────────────────────

    def adaptation_success_rate(self, last_n: int = 10) -> float:
        """Fraction of recent adaptations that improved predictions."""
        recent = self._history[-last_n:]
        if not recent:
            return 0.0
        improved = sum(1 for r in recent if r.outcome == AdaptationOutcome.IMPROVED)
        return improved / len(recent)

    def stability_check(self, tolerance: float = 0.02) -> bool:
        """Check if any domain degraded beyond tolerance.

        Returns True if all domains are within tolerance.
        """
        for domain, delta in self._domain_post_accuracy.items():
            pre = self._domain_pre_accuracy.get(domain, 0.5)
            if (delta - pre) < -tolerance:
                return False
        return True

    @property
    def total_evaluations(self) -> int:
        return len(self._history)

    @property
    def evaluation_history(self) -> list[EvaluationResult]:
        return list(self._history)

    def clear_windows(self, model_id: str) -> None:
        """Clear pre/post windows after evaluation."""
        self._pre_windows.pop(model_id, None)
        self._post_windows.pop(model_id, None)
        self._domain_pre_accuracy.clear()
        self._domain_post_accuracy.clear()
