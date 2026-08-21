"""Discovery Belief Manager — evolves existing BeliefStore into discovery cognition.

This module wraps the existing ``BeliefStore`` to add discovery-aware
belief lifecycle management.  It does NOT replace or duplicate the
BeliefStore — it extends it.

Architecture::

    DiscoveryBeliefManager
            │
            ├── wraps BeliefStore (storage, retrieval, indexing)
            ├── adds belief lifecycle tracking
            ├── adds Bayesian confidence updates
            ├── adds falsification candidate identification
            └── adds prediction-based revision

The belief lifecycle in discovery mode::

    Belief Created
        │
    Evidence Added
        │
    Prediction Generated
        │
    Prediction Tested
        │
    Confidence Updated
        │
    Belief Revised / Falsified / Strengthened

Design principle: "Evolve existing cognition, don't build a second brain."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.epistemics.interfaces import BeliefRevision, PredictionOutcome
from hbllm.hcir.graph import (
    BeliefNode,
    BeliefTransitionNode,
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.hcir.types import (
    BeliefTransition,
    EvidenceAssessment,
    FalsificationStatus,
    PropositionLikelihood,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian Update Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BayesianConfig:
    """Configuration for Bayesian confidence updates.

    Controls how aggressively beliefs are updated in response to
    new evidence or prediction outcomes.
    """

    # How much a single supporting evidence item can raise confidence
    max_support_delta: float = 0.15

    # How much a single contradicting evidence item can lower confidence
    max_contradict_delta: float = 0.20

    # How much a correct prediction raises hypothesis confidence
    prediction_success_delta: float = 0.10

    # How much an incorrect prediction lowers hypothesis confidence
    prediction_failure_delta: float = 0.15

    # Minimum confidence below which a belief is considered falsified
    falsification_threshold: float = 0.1

    # Minimum confidence above which a belief is considered corroborated
    corroboration_threshold: float = 0.7

    # Evidence strength weights (maps EvidenceStrength to multiplier)
    strength_weights: dict[str, float] = field(
        default_factory=lambda: {
            "anecdotal": 0.3,
            "observational": 0.5,
            "correlational": 0.7,
            "experimental": 0.9,
            "meta_analytic": 0.95,
            "replicated": 1.0,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# Discovery Belief Manager
# ═══════════════════════════════════════════════════════════════════════════


class DiscoveryBeliefManager:
    """Manages beliefs with discovery-aware lifecycle.

    Wraps the shared CognitiveGraph to provide:
    - Odds-space Bayesian belief updates with Likelihood Ratios
    - Evidence-based confidence updates (Bayesian)
    - Prediction-based belief revision
    - Falsification candidate identification
    - Event-sourced BeliefTransitionNode tracking

    This is NOT a separate belief store — it operates on BeliefNodes
    in the shared HCIR graph.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        config: BayesianConfig | None = None,
    ) -> None:
        self._graph = graph
        self._config = config or BayesianConfig()

    # ── Odds-Space Bayesian Revision ──────────────────────────────────

    async def revise(
        self,
        belief_id: str,
        proposition_likelihood: PropositionLikelihood,
        evidence_assessment: EvidenceAssessment | None = None,
        rationale: str = "",
    ) -> BeliefTransition:
        """Revise belief confidence using Bayesian odds-space updating.

        O(H|E) = O(H) * LR
        P(H|E) = O(H|E) / (1 + O(H|E))

        Emits and commits an event-sourced BeliefTransitionNode into HCIR.
        """
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            raise ValueError(f"Node {belief_id} is not a BeliefNode")

        prior_confidence = float(node.uncertainty.confidence)
        prior_revision = node.current_revision

        # Check if update should be skipped
        if proposition_likelihood.status in ("insufficient", "redundant"):
            logger.debug(
                "Skipping belief update for %s: status=%s",
                belief_id,
                proposition_likelihood.status,
            )
            return BeliefTransition(
                transition_id="",
                belief_id=belief_id,
                prior_confidence=prior_confidence,
                posterior_confidence=prior_confidence,
                delta=0.0,
                prior_revision=prior_revision,
                posterior_revision=prior_revision,
                likelihood_ratio=proposition_likelihood.likelihood_ratio,
                source_evidence_id=proposition_likelihood.evidence_id,
                rationale=f"No update: status={proposition_likelihood.status}",
            )

        # 1. Compute prior odds (clamped to prevent div by zero)
        p_prior = max(0.001, min(0.999, prior_confidence))
        prior_odds = p_prior / (1.0 - p_prior)

        # 2. Update posterior odds via Likelihood Ratio (LR)
        lr = proposition_likelihood.likelihood_ratio
        posterior_odds = prior_odds * lr

        # 3. Convert back to posterior probability
        posterior_confidence = posterior_odds / (1.0 + posterior_odds)
        posterior_confidence = float(max(0.01, min(0.99, posterior_confidence)))

        delta = posterior_confidence - prior_confidence
        posterior_revision = prior_revision + 1

        # 4. Update BeliefNode state
        node.uncertainty.confidence = posterior_confidence
        node.current_revision = posterior_revision

        if delta > 0:
            if proposition_likelihood.evidence_id not in node.evidence_sources:
                node.evidence_sources.append(proposition_likelihood.evidence_id)
        elif delta < 0:
            if proposition_likelihood.evidence_id not in node.counter_evidence:
                node.counter_evidence.append(proposition_likelihood.evidence_id)

        # Update falsification status
        if posterior_confidence < self._config.falsification_threshold:
            node.falsification_status = FalsificationStatus.FALSIFIED
        elif posterior_confidence >= self._config.corroboration_threshold:
            node.falsification_status = FalsificationStatus.CORROBORATED
        elif delta < 0:
            node.falsification_status = FalsificationStatus.WEAKENED

        # 5. Create immutable BeliefTransitionNode in HCIR
        transition_id = f"trans_{int(time.time() * 1000)}_{belief_id}"
        node.latest_transition_id = transition_id

        transition_record = BeliefTransition(
            transition_id=transition_id,
            belief_id=belief_id,
            prior_confidence=prior_confidence,
            posterior_confidence=posterior_confidence,
            delta=delta,
            prior_revision=prior_revision,
            posterior_revision=posterior_revision,
            likelihood_ratio=lr,
            source_evidence_id=proposition_likelihood.evidence_id,
            timestamp=time.time(),
            rationale=rationale or f"LR={lr:.2f} ({proposition_likelihood.status})",
        )

        transition_node = BeliefTransitionNode(
            id=transition_id,
            belief_id=belief_id,
            prior_confidence=prior_confidence,
            posterior_confidence=posterior_confidence,
            delta=delta,
            prior_revision=prior_revision,
            posterior_revision=posterior_revision,
            likelihood_ratio=lr,
            source_evidence_id=proposition_likelihood.evidence_id,
            rationale=transition_record.rationale,
        )
        self._graph.upsert_node(transition_node)

        # Append lightweight history entry to node
        node.revision_history.append({
            "timestamp": transition_record.timestamp,
            "transition_id": transition_id,
            "prior": prior_confidence,
            "posterior": posterior_confidence,
            "delta": delta,
            "evidence_id": proposition_likelihood.evidence_id,
        })
        self._graph.upsert_node(node)

        # 6. Commit epistemic edge
        edge_type = HCIREdgeType.STRENGTHENS if delta >= 0 else HCIREdgeType.WEAKENS
        edge = HCIREdge(
            edge_type=edge_type,
            sources=[proposition_likelihood.evidence_id],
            targets=[belief_id],
        )
        try:
            self._graph.add_edge(edge)
        except ValueError:
            pass

        return transition_record

    # ── Evidence-Based Revision ───────────────────────────────────────

    async def revise_belief(
        self,
        belief_id: str,
        evidence_id: str,
        direction: str,  # "supporting" | "contradicting"
        evidence_strength: str = "observational",
    ) -> BeliefRevision:
        """Update a belief's confidence based on new evidence.

        Uses a Bayesian-inspired update weighted by evidence strength.
        Records the revision in the belief's history.
        """
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            return BeliefRevision(
                belief_id=belief_id,
                reason=f"Node {belief_id} is not a BeliefNode",
            )

        old_confidence = node.uncertainty.confidence
        strength_weight = self._config.strength_weights.get(evidence_strength, 0.5)

        if direction == "supporting":
            delta = self._config.max_support_delta * strength_weight
            # Diminishing returns as confidence approaches 1.0
            effective_delta = delta * (1.0 - old_confidence)
            new_confidence = min(1.0, old_confidence + effective_delta)
            node.evidence_sources.append(evidence_id)
        elif direction == "contradicting":
            delta = self._config.max_contradict_delta * strength_weight
            # Stronger effect as evidence strength grows
            effective_delta = delta * old_confidence
            new_confidence = max(0.0, old_confidence - effective_delta)
            node.counter_evidence.append(evidence_id)
        else:
            return BeliefRevision(
                belief_id=belief_id,
                reason=f"Unknown direction: {direction}",
            )

        # Update the node
        node.uncertainty.confidence = new_confidence
        node.current_revision += 1

        # Update falsification status
        if new_confidence < self._config.falsification_threshold:
            node.falsification_status = FalsificationStatus.FALSIFIED
        elif new_confidence >= self._config.corroboration_threshold:
            node.falsification_status = FalsificationStatus.CORROBORATED
        elif old_confidence > new_confidence:
            node.falsification_status = FalsificationStatus.WEAKENED

        # Record revision history
        transition_id = f"trans_{int(time.time() * 1000)}_{belief_id}"
        node.latest_transition_id = transition_id

        revision_entry = {
            "timestamp": time.time(),
            "transition_id": transition_id,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "reason": f"{direction} evidence ({evidence_strength})",
            "evidence_id": evidence_id,
        }
        node.revision_history.append(revision_entry)

        transition_node = BeliefTransitionNode(
            id=transition_id,
            belief_id=belief_id,
            prior_confidence=old_confidence,
            posterior_confidence=new_confidence,
            delta=new_confidence - old_confidence,
            prior_revision=node.current_revision - 1,
            posterior_revision=node.current_revision,
            source_evidence_id=evidence_id,
            rationale=f"{direction} evidence ({evidence_strength})",
        )
        self._graph.upsert_node(transition_node)
        self._graph.upsert_node(node)

        # Create epistemic edge
        edge_type = HCIREdgeType.STRENGTHENS if direction == "supporting" else HCIREdgeType.WEAKENS
        edge = HCIREdge(
            edge_type=edge_type,
            sources=[evidence_id],
            targets=[belief_id],
        )
        try:
            self._graph.add_edge(edge)
        except ValueError:
            pass  # Evidence node may not be in graph yet

        revision = BeliefRevision(
            belief_id=belief_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            reason=f"{direction} evidence ({evidence_strength}), "
            f"δ={new_confidence - old_confidence:+.4f}",
            evidence_id=evidence_id,
        )

        logger.debug(
            "Belief %s revised: %.3f → %.3f (%s)",
            belief_id,
            old_confidence,
            new_confidence,
            direction,
        )
        return revision

    # ── Prediction-Based Revision ─────────────────────────────────────

    async def revise_from_prediction(
        self,
        belief_id: str,
        prediction_outcome: PredictionOutcome,
    ) -> BeliefRevision:
        """Update a belief based on a prediction outcome.

        If a prediction derived from this belief was correct, confidence
        rises.  If incorrect, confidence falls.  This implements the
        Popperian falsification cycle at the belief level.
        """
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            return BeliefRevision(
                belief_id=belief_id,
                reason=f"Node {belief_id} is not a BeliefNode",
            )

        old_confidence = node.uncertainty.confidence

        if prediction_outcome.correct is True:
            delta = self._config.prediction_success_delta * (1.0 - old_confidence)
            new_confidence = min(1.0, old_confidence + delta)
            node.prediction_score += 1.0
            reason = "Prediction confirmed"
        elif prediction_outcome.correct is False:
            delta = self._config.prediction_failure_delta * old_confidence
            new_confidence = max(0.0, old_confidence - delta)
            node.prediction_score -= 1.0
            reason = "Prediction failed"
        else:
            return BeliefRevision(
                belief_id=belief_id,
                reason="Prediction outcome is inconclusive",
            )

        node.uncertainty.confidence = new_confidence

        # Update falsification status
        if new_confidence < self._config.falsification_threshold:
            node.falsification_status = FalsificationStatus.FALSIFIED
        elif new_confidence >= self._config.corroboration_threshold:
            node.falsification_status = FalsificationStatus.CORROBORATED

        # Record revision
        node.revision_history.append(
            {
                "timestamp": time.time(),
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "reason": reason,
                "evidence_id": prediction_outcome.prediction_id,
            }
        )

        self._graph.upsert_node(node)

        return BeliefRevision(
            belief_id=belief_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            reason=f"{reason}, δ={new_confidence - old_confidence:+.4f}",
            evidence_id=prediction_outcome.prediction_id,
        )

    # ── Falsification Candidates ──────────────────────────────────────

    async def get_falsification_candidates(
        self,
        domain: str = "",
        min_confidence: float = 0.3,
        max_confidence: float = 0.9,
    ) -> list[str]:
        """Return belief IDs that are good candidates for falsification testing.

        Good candidates are beliefs that:
        1. Have moderate confidence (not too certain, not too weak)
        2. Have NOT been falsified
        3. Have few or no linked predictions (untested)
        4. Optionally filtered by domain
        """
        candidates: list[tuple[str, float]] = []

        for node in self._graph.nodes_by_type(HCIRNodeType.BELIEF):
            if not isinstance(node, BeliefNode):
                continue

            conf = node.uncertainty.confidence
            if not (min_confidence <= conf <= max_confidence):
                continue

            if node.falsification_status == FalsificationStatus.FALSIFIED:
                continue

            if domain and node.belief_type != domain:
                continue

            # Score: prefer beliefs with fewer tests and moderate confidence
            # Moderate confidence = closer to 0.5 is better for information gain
            info_gain = 1.0 - abs(conf - 0.5) * 2  # Max at 0.5, min at 0.0/1.0
            test_penalty = min(1.0, len(node.revision_history) * 0.1)
            score = info_gain * (1.0 - test_penalty)

            candidates.append((node.id, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    # ── Belief Query Utilities ────────────────────────────────────────

    async def get_beliefs_by_status(
        self,
        status: FalsificationStatus,
    ) -> list[BeliefNode]:
        """Return all beliefs with a given falsification status."""
        results: list[BeliefNode] = []
        for node in self._graph.nodes_by_type(HCIRNodeType.BELIEF):
            if isinstance(node, BeliefNode) and node.falsification_status == status:
                results.append(node)
        return results

    async def get_contested_beliefs(
        self,
        min_counter_evidence: int = 1,
    ) -> list[BeliefNode]:
        """Return beliefs that have counter-evidence."""
        results: list[BeliefNode] = []
        for node in self._graph.nodes_by_type(HCIRNodeType.BELIEF):
            if isinstance(node, BeliefNode) and len(node.counter_evidence) >= min_counter_evidence:
                results.append(node)
        return results

    async def get_belief_summary(self, belief_id: str) -> dict[str, Any]:
        """Get a comprehensive summary of a belief's epistemic state."""
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            return {}

        return {
            "belief_id": belief_id,
            "claim": node.claim,
            "confidence": node.uncertainty.confidence,
            "belief_type": node.belief_type,
            "falsification_status": node.falsification_status.value,
            "prediction_score": node.prediction_score,
            "evidence_count": len(node.evidence_sources),
            "counter_evidence_count": len(node.counter_evidence),
            "revision_count": len(node.revision_history),
            "last_revision": node.revision_history[-1] if node.revision_history else None,
        }
