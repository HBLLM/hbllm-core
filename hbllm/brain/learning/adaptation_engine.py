"""Adaptation Engine — model mutation executor for A14.

Executes adaptations ONLY when authorized by AdaptationGate.
Three adaptation pathways:

1. Parameter update — adjust transition probabilities, decay rates
2. Rule extraction — when error patterns reveal an if→then regularity
3. Model selection — mark a model as degraded (future: switch models)

Every adaptation produces an AdaptationEventNode in HCIR,
enabling deterministic replay and provenance queries.

**Adaptive learning rate:**

    effective_lr = base_lr × error_surprise × model_uncertainty
                   × novelty_factor × evidence_quality

    min_lr ≤ effective_lr ≤ max_lr

**Stability factor:** Many recent adaptations → reduce magnitude.
Persistent systematic error → increase magnitude.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.learning.adaptation_gate import ErrorEvidence
from hbllm.hcir.graph import (
    AdaptationEventNode,
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    LearnedRuleNode,
    PredictiveModelNode,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Adaptation Record — before/after snapshot
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AdaptationRecord:
    """Record of a single adaptation with before/after state."""

    adaptation_event_id: str
    model_id: str
    adaptation_type: str  # "parameter_update", "rule_extraction", "model_selection"
    parameters_before: dict[str, Any]
    parameters_after: dict[str, Any]
    learning_rate_used: float
    evidence_count: int
    trigger_error_ids: list[str] = field(default_factory=list)
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Adaptation Engine
# ═══════════════════════════════════════════════════════════════════════════


class AdaptationEngine:
    """Executes model adaptations, authorized by AdaptationGate.

    **Does NOT:**
    - Decide when to adapt (that's AdaptationGate)
    - Classify errors (that's ErrorClassifier)
    - Route signals (that's LearningSignalRouter)

    **Does:**
    - Update model parameters (Markov transitions, decay rates, confidence)
    - Extract LearnedRuleNodes from error patterns
    - Record AdaptationEventNode in HCIR with full provenance

    Usage::

        engine = AdaptationEngine(graph)

        # Only called after gate authorizes adaptation
        record = engine.adapt_parameters(
            model_node=model,
            evidence=evidence,
            learning_rate=0.05,
        )

        # Rule extraction
        rule = engine.extract_rule(
            condition="support_absent",
            prediction="object_falls",
            evidence=evidence,
        )
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        base_lr: float = 0.05,
        min_lr: float = 0.001,
        max_lr: float = 0.3,
    ) -> None:
        self._graph = graph
        self._base_lr = base_lr
        self._min_lr = min_lr
        self._max_lr = max_lr

    # ── Parameter Adaptation ──────────────────────────────────────────

    def adapt_parameters(
        self,
        model_node: PredictiveModelNode,
        evidence: ErrorEvidence,
        timestamp: float | None = None,
    ) -> AdaptationRecord:
        """Update a model's parameters based on accumulated error evidence.

        Computes an adaptive learning rate, applies parameter deltas,
        and records the adaptation as an AdaptationEventNode in HCIR.

        Args:
            model_node: The model to adapt.
            evidence: Accumulated error evidence.
            timestamp: Optional override for event timestamp.

        Returns:
            AdaptationRecord with before/after state.
        """
        now = timestamp if timestamp is not None else time.time()

        # Compute adaptive learning rate
        lr = self._compute_learning_rate(model_node, evidence)

        # Snapshot before state
        params_before = dict(model_node.parameters)

        # Compute parameter deltas from evidence
        params_after = self._compute_parameter_update(
            model_node,
            evidence,
            lr,
        )

        # Apply update to model node
        model_node.parameters = params_after
        model_node.adaptation_count += 1
        model_node.last_adapted_at = now
        model_node.learning_rate = lr

        # Update accuracy based on evidence
        if evidence.occurrences > 0:
            # Blend old accuracy with error signal
            error_signal = evidence.total_magnitude / evidence.occurrences
            model_node.error_rate = error_signal
            # Accuracy decreases proportional to error rate
            model_node.accuracy = max(0.0, model_node.accuracy * (1 - lr * error_signal))

        self._graph.upsert_node(model_node)

        # Record AdaptationEventNode in HCIR
        event_node = AdaptationEventNode(
            model_id=model_node.id,
            adaptation_type="parameter_update",
            parameters_before=params_before,
            parameters_after=params_after,
            learning_rate_used=lr,
            evidence_count=evidence.occurrences,
            trigger_error_ids=list(evidence.error_ids),
            tags=["a14_adaptation", "parameter_update", model_node.domain],
        )
        self._graph.add_node(event_node)

        # ADAPTS edge: AdaptationEvent → PredictiveModel
        self._graph.add_edge(
            HCIREdge(
                edge_type=HCIREdgeType.ADAPTS,
                sources=[event_node.id],
                targets=[model_node.id],
            )
        )

        # ADAPTED_BY edge: PredictiveModel → AdaptationEvent
        self._graph.add_edge(
            HCIREdge(
                edge_type=HCIREdgeType.ADAPTED_BY,
                sources=[model_node.id],
                targets=[event_node.id],
            )
        )

        # LEARNED_FROM edges: AdaptationEvent → PredictionErrors
        for error_id in evidence.error_ids:
            if self._graph.get_node(error_id) is not None:
                self._graph.add_edge(
                    HCIREdge(
                        edge_type=HCIREdgeType.LEARNED_FROM,
                        sources=[event_node.id],
                        targets=[error_id],
                    )
                )

        logger.debug(
            "AdaptationEngine: adapted model %s (lr=%.4f, evidence=%d)",
            model_node.id,
            lr,
            evidence.occurrences,
        )

        return AdaptationRecord(
            adaptation_event_id=event_node.id,
            model_id=model_node.id,
            adaptation_type="parameter_update",
            parameters_before=params_before,
            parameters_after=params_after,
            learning_rate_used=lr,
            evidence_count=evidence.occurrences,
            trigger_error_ids=list(evidence.error_ids),
            timestamp=now,
        )

    # ── Rule Extraction ───────────────────────────────────────────────

    def extract_rule(
        self,
        condition: str,
        prediction: str,
        evidence: ErrorEvidence,
        domain: str = "",
        applies_to_types: list[str] | None = None,
        timestamp: float | None = None,
    ) -> LearnedRuleNode:
        """Extract a LearnedRuleNode from error pattern evidence.

        Creates explicit if→then knowledge in HCIR — substantially
        more powerful than mere parameter updates.

        Args:
            condition: The if-clause (e.g., "support_surface = absent").
            prediction: The then-clause (e.g., "object_motion = downward").
            evidence: The error evidence that revealed this pattern.
            domain: Domain scope for the rule.
            applies_to_types: Entity types this rule applies to.
            timestamp: Optional override.

        Returns:
            The created LearnedRuleNode.
        """

        # Compute rule confidence from evidence
        confidence = min(0.95, 0.5 + (evidence.occurrences * 0.05))

        rule = LearnedRuleNode(
            condition=condition,
            prediction=prediction,
            confidence=confidence,
            source_error_count=evidence.occurrences,
            domain=domain or evidence.domain,
            applies_to_types=applies_to_types or [],
            tags=["a14_learned_rule", domain or evidence.domain],
        )
        self._graph.add_node(rule)

        # LEARNED_FROM edges: LearnedRule → PredictionErrors
        for error_id in evidence.error_ids:
            if self._graph.get_node(error_id) is not None:
                self._graph.add_edge(
                    HCIREdge(
                        edge_type=HCIREdgeType.LEARNED_FROM,
                        sources=[rule.id],
                        targets=[error_id],
                    )
                )
                # Bidirectional: CONTRIBUTED_TO
                self._graph.add_edge(
                    HCIREdge(
                        edge_type=HCIREdgeType.CONTRIBUTED_TO,
                        sources=[error_id],
                        targets=[rule.id],
                    )
                )

        # Record extraction as an AdaptationEventNode
        event_node = AdaptationEventNode(
            model_id="",
            adaptation_type="rule_extraction",
            parameters_after={"condition": condition, "prediction": prediction},
            evidence_count=evidence.occurrences,
            trigger_error_ids=list(evidence.error_ids),
            tags=["a14_adaptation", "rule_extraction"],
        )
        self._graph.add_node(event_node)

        logger.debug(
            "AdaptationEngine: extracted rule '%s → %s' (confidence=%.2f, errors=%d)",
            condition,
            prediction,
            confidence,
            evidence.occurrences,
        )

        return rule

    # ── Learning Rate Computation ─────────────────────────────────────

    def _compute_learning_rate(
        self,
        model: PredictiveModelNode,
        evidence: ErrorEvidence,
    ) -> float:
        """Compute adaptive learning rate.

        effective_lr = base_lr × error_surprise × model_uncertainty
                       × evidence_quality × stability_factor

        Returns:
            Clamped learning rate in [min_lr, max_lr].
        """
        # Error surprise: how surprising is this error given model accuracy?
        error_surprise = 1.0 + (1.0 - model.accuracy) if model.accuracy < 0.9 else 1.0

        # Model uncertainty: uncertain models should learn faster
        model_uncertainty = 1.0 + (1.0 - model.calibration)

        # Evidence quality: more evidence → more confident adaptation
        evidence_quality = min(2.0, math.log2(max(1, evidence.occurrences)))

        # Stability factor: recent adaptations → reduce magnitude
        stability = 1.0 / (1.0 + 0.5 * evidence.recent_adaptation_count)

        effective_lr = (
            self._base_lr * error_surprise * model_uncertainty * evidence_quality * stability
        )

        return max(self._min_lr, min(self._max_lr, effective_lr))

    # ── Parameter Update Logic ────────────────────────────────────────

    def _compute_parameter_update(
        self,
        model: PredictiveModelNode,
        evidence: ErrorEvidence,
        lr: float,
    ) -> dict[str, Any]:
        """Compute updated parameters from evidence.

        Performs Hebbian-style delta updates — incremental,
        not full retraining.
        """
        params = dict(model.parameters)

        # For Markov models: adjust transition probabilities
        if model.model_type == "markov":
            transitions = params.get("transitions", {})
            # Increase probability of observed transitions,
            # decrease probability of incorrectly predicted transitions
            for classification in evidence.classifications:
                if classification.model_error > 0.5:
                    # This error indicates the model's transitions need adjustment
                    # Apply a general accuracy penalty
                    for state, probs in transitions.items():
                        if isinstance(probs, dict):
                            # Slightly flatten confident predictions
                            max_prob = max(probs.values()) if probs else 0.5
                            if max_prob > 0.7:
                                for target in probs:
                                    probs[target] = (
                                        probs[target] * (1 - lr) + (1.0 / max(1, len(probs))) * lr
                                    )
            params["transitions"] = transitions

        # For permanence models: adjust decay rates
        elif model.model_type == "permanence":
            decay_rates = params.get("decay_rates", {})
            avg_magnitude = (
                evidence.total_magnitude / evidence.occurrences if evidence.occurrences > 0 else 0.0
            )
            # If errors are large, decay is too slow (increase rate)
            # If errors are small, decay may be too fast (decrease rate)
            for dim, rate in decay_rates.items():
                if avg_magnitude > 0.3:
                    decay_rates[dim] = rate * (1 + lr * avg_magnitude)
                else:
                    decay_rates[dim] = rate * (1 - lr * 0.1)
            params["decay_rates"] = decay_rates

        # For spatial models: adjust confidence weights
        elif model.model_type == "spatial":
            weights = params.get("relation_weights", {})
            for rel, weight in weights.items():
                # Reduce confidence in relation types that produce errors
                weights[rel] = weight * (1 - lr * 0.1)
            params["relation_weights"] = weights

        # Generic: adjust base confidence
        base_conf = params.get("base_confidence", 0.5)
        avg_error = (
            evidence.total_magnitude / evidence.occurrences if evidence.occurrences > 0 else 0.0
        )
        params["base_confidence"] = base_conf * (1 - lr * avg_error)

        return params
