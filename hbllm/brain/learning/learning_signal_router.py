"""Learning Signal Router — routes classified errors to adaptation pathways for A14.

Implements the **hybrid fast/slow** learning distinction:

Immediate (fast) loop::

    PredictionError → classify → route → small adaptation
    Used for high-confidence model errors, safety-critical corrections

Periodic (slow) loop::

    Error history → pattern detection → cross-entity analysis
    → rule extraction → model comparison → larger adaptation
    Used for drift detection, rule discovery, recalibration

The router NEVER directly modifies models — it routes to
AdaptationGate, which is the sole authority over model mutation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from hbllm.brain.learning.adaptation_gate import (
    AdaptationGate,
    ErrorEvidenceAccumulator,
    GateDecision,
    GateVerdict,
)
from hbllm.brain.learning.error_classifier import (
    ErrorClassification,
    ErrorClassifier,
    ErrorContext,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Routing Result
# ═══════════════════════════════════════════════════════════════════════════


class RoutingAction(StrEnum):
    """The routing action determined by the router."""

    ACCUMULATE = "accumulate"  # Fast path: accumulated, gate will decide later
    ADAPT = "adapt"  # Gate authorized adaptation
    WORLD_UPDATE = "world_update"  # Route to A13 reconciler
    EXPLORE = "explore"  # Novelty — route to exploration
    REJECT = "reject"  # Noise — dropped
    DEFER = "defer"  # Insufficient evidence


@dataclass(frozen=True)
class RoutingResult:
    """Result of routing a prediction error through the A14 pipeline."""

    action: RoutingAction
    classification: ErrorClassification
    gate_verdict: GateVerdict | None = None
    model_id: str = ""
    error_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Learning Signal Router
# ═══════════════════════════════════════════════════════════════════════════


class LearningSignalRouter:
    """Routes classified prediction errors to the correct adaptation pathway.

    The router combines the fast (immediate) and slow (batch) learning
    loops.

    **Fast path** (per-error):
    1. Classify the error (ErrorClassifier)
    2. Accumulate evidence (ErrorEvidenceAccumulator)
    3. Evaluate the gate (AdaptationGate)
    4. If gate says ADAPT → return ADAPT
    5. Otherwise → return ACCUMULATE/DEFER/REJECT/WORLD_UPDATE/EXPLORE

    **Slow path** (periodic — called by CognitiveAdaptationLoop):
    1. Review all accumulated evidence
    2. Re-evaluate gates for all tracked models
    3. Return models that should be adapted

    Usage::

        router = LearningSignalRouter(classifier, accumulator, gate)

        # Fast path
        result = router.route(
            error_id="err_001",
            model_id="model_physics_01",
            context=ErrorContext(error_magnitude=0.4, ...),
        )

        # Slow path
        ready = router.batch_evaluate()
    """

    def __init__(
        self,
        classifier: ErrorClassifier,
        accumulator: ErrorEvidenceAccumulator,
        gate: AdaptationGate,
    ) -> None:
        self._classifier = classifier
        self._accumulator = accumulator
        self._gate = gate

    # ── Fast Path (per-error) ─────────────────────────────────────────

    def route(
        self,
        error_id: str,
        model_id: str,
        context: ErrorContext,
        error_magnitude: float = 0.0,
        domain: str = "",
        timestamp: float | None = None,
    ) -> RoutingResult:
        """Route a single prediction error through the A14 pipeline.

        1. Classify (probabilistic)
        2. Accumulate evidence
        3. Evaluate gate
        4. Return routing decision
        """
        # Step 1: Classify
        classification = self._classifier.classify(context, error_id=error_id)

        # Step 2: Accumulate
        self._accumulator.accumulate(
            model_id=model_id,
            error_id=error_id,
            classification=classification,
            error_magnitude=error_magnitude or context.error_magnitude,
            domain=domain or context.prediction_domain,
            timestamp=timestamp,
        )

        # Step 3: Evaluate gate
        verdict = self._gate.evaluate(model_id)

        # Step 4: Map gate decision to routing action
        action = self._map_decision(verdict.decision)

        return RoutingResult(
            action=action,
            classification=classification,
            gate_verdict=verdict,
            model_id=model_id,
            error_id=error_id,
        )

    # ── Slow Path (batch) ─────────────────────────────────────────────

    def batch_evaluate(self) -> list[RoutingResult]:
        """Re-evaluate all tracked models for adaptation readiness.

        Called periodically by CognitiveAdaptationLoop.

        Returns:
            List of models that should be adapted.
        """
        results: list[RoutingResult] = []

        for model_id in self._accumulator.tracked_models:
            verdict = self._gate.evaluate(model_id)
            action = self._map_decision(verdict.decision)

            if action in (RoutingAction.ADAPT, RoutingAction.EXPLORE, RoutingAction.WORLD_UPDATE):
                evidence = self._accumulator.get_evidence(model_id)
                # Use the last classification from evidence
                last_classification = (
                    evidence.classifications[-1]
                    if evidence and evidence.classifications
                    else ErrorClassification()
                )
                results.append(RoutingResult(
                    action=action,
                    classification=last_classification,
                    gate_verdict=verdict,
                    model_id=model_id,
                ))

        return results

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _map_decision(decision: GateDecision) -> RoutingAction:
        """Map a gate decision to a routing action."""
        mapping = {
            GateDecision.ADAPT: RoutingAction.ADAPT,
            GateDecision.REJECT: RoutingAction.REJECT,
            GateDecision.DEFER: RoutingAction.DEFER,
            GateDecision.EXPLORE: RoutingAction.EXPLORE,
            GateDecision.WORLD_UPDATE: RoutingAction.WORLD_UPDATE,
        }
        return mapping.get(decision, RoutingAction.DEFER)
