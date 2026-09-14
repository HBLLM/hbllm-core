"""Metacognitive Calibration Engine (Stage D13).

Quantifies self-assessed confidence, calibration error (ECE, Brier score),
and strategic epistemic abstention when uncertainty exceeds safe action thresholds.
"""

from __future__ import annotations

import logging
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .types import (
    BeliefTransitionEvent,
    BeliefTransitionType,
    MetacognitiveReport,
    PredicateGoal,
)

logger = logging.getLogger(__name__)


class MetacognitiveEngine:
    """Evaluates self-competence, calibration, and strategic abstention."""

    def __init__(
        self,
        substrate: BlankBrainSubstrate,
        env: BabyWorldEnvironment,
        abstain_threshold: float = 0.65,
    ) -> None:
        self.substrate = substrate
        self.env = env
        self.abstain_threshold = abstain_threshold
        self.prediction_history: list[dict[str, Any]] = []

    def assess_confidence(self, goal: PredicateGoal) -> float:
        """Estimate prior probability of plan success based on acquired knowledge."""
        subj = self.env.objects.get(goal.subject_id)
        if not subj:
            return 0.1

        # Check if subject's affordances are known
        shape = subj.object_type.value
        known_affordances = self.substrate.affordances.get(shape, [])

        confidence = 0.4  # Baseline prior

        if goal.predicate == "INSIDE":
            # Requires containment schema
            if self.substrate.spatial_schemas:
                confidence += 0.3
            if known_affordances:
                confidence += 0.2
            # Check reach
            dist = self.env.agent_hand_position.distance_to(subj.position)
            if dist > self.env.REACH_DISTANCE:
                # Needs tool
                has_tool = any(obj.is_tool for obj in self.env.objects.values())
                if not has_tool:
                    confidence -= 0.3

        elif goal.predicate == "REACHABLE":
            dist = self.env.agent_hand_position.distance_to(subj.position)
            if dist <= self.env.REACH_DISTANCE:
                confidence = 0.95
            else:
                has_viable_tool = any(
                    obj.tool_length + self.env.REACH_DISTANCE >= dist
                    for obj in self.env.objects.values()
                )
                confidence = 0.85 if has_viable_tool else 0.2

        elif goal.predicate == "STATE":
            # Check if causal rule exists
            has_rule = any(
                r.get("consequence") == "OPEN" or "open" in str(r.get("consequence")).lower()
                for r in self.substrate.causal_rules
            )
            confidence = 0.9 if has_rule else 0.2

        return max(0.05, min(0.98, confidence))

    def should_abstain_or_explore(self, goal: PredicateGoal) -> bool:
        """Decide whether to execute goal or abstain due to high epistemic uncertainty."""
        conf = self.assess_confidence(goal)
        return conf < self.abstain_threshold

    def record_outcome(self, goal: PredicateGoal, confidence: float, actual_success: bool) -> None:
        """Log predicted confidence vs empirical success for calibration analysis."""
        record = {
            "goal": f"{goal.predicate}({goal.subject_id})",
            "confidence": confidence,
            "success": actual_success,
            "brier_loss": (confidence - (1.0 if actual_success else 0.0)) ** 2,
        }
        self.prediction_history.append(record)

        if hasattr(self.substrate, "profile") and hasattr(
            self.substrate.profile, "belief_transitions"
        ):
            self.substrate.profile.belief_transitions.append(
                BeliefTransitionEvent(
                    event_type=BeliefTransitionType.METACOGNITION_CALIBRATED,
                    variable="confidence_calibration",
                    condition=f"conf={confidence:.2f} -> success={actual_success}",
                    prior_confidence=confidence,
                    posterior_confidence=1.0 if actual_success else 0.0,
                    evidence=record,
                )
            )

    def compute_calibration_report(self) -> MetacognitiveReport:
        """Calculate Brier score and Expected Calibration Error (ECE)."""
        if not self.prediction_history:
            return MetacognitiveReport(
                brier_score=0.0,
                expected_calibration_error=0.0,
                abstention_accuracy=1.0,
                predictions=[],
            )

        n = len(self.prediction_history)
        brier = sum(r["brier_loss"] for r in self.prediction_history) / n

        # Binning for ECE (5 bins)
        bins: dict[int, list[dict[str, Any]]] = {i: [] for i in range(5)}
        for r in self.prediction_history:
            bin_idx = min(4, int(r["confidence"] * 5))
            bins[bin_idx].append(r)

        ece = 0.0
        for b_idx, items in bins.items():
            if not items:
                continue
            avg_conf = sum(it["confidence"] for it in items) / len(items)
            avg_acc = sum(1.0 for it in items if it["success"]) / len(items)
            ece += (len(items) / n) * abs(avg_acc - avg_conf)

        # Abstention accuracy: predictions with conf < threshold that actually failed
        low_conf = [r for r in self.prediction_history if r["confidence"] < self.abstain_threshold]
        abstain_acc = (
            sum(1.0 for r in low_conf if not r["success"]) / len(low_conf) if low_conf else 1.0
        )

        return MetacognitiveReport(
            brier_score=brier,
            expected_calibration_error=ece,
            abstention_accuracy=abstain_acc,
            predictions=self.prediction_history,
        )
