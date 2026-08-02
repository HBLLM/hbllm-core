"""Prediction Tracker — tracks competing predictions against observations.

Supports rival hypotheses with conflicting predictions::

    Hypothesis A → "X will increase"
    Hypothesis B → "X will decrease"
    → Conflict detected → discriminative experiment opportunity

Prediction outcomes update ``BeliefConfidence.prediction_accuracy``
for linked beliefs.

Architecture::

    PredictionTracker
        ├── register_prediction         → PredictionNode in graph
        ├── register_competing          → detect rival predictions
        ├── check_prediction            → compare against observation
        ├── find_competing_predictions  → pairs for experiments
        ├── check_expired_predictions   → time-horizon-based eval
        └── update linked BeliefConfidence.prediction_accuracy

Usage::

    tracker = PredictionTracker(graph=graph, belief_manager=manager)
    pred_id = await tracker.register_prediction(
        hypothesis_id, "Blood pressure will decrease", "decrease",
    )
    outcome = await tracker.check_prediction(pred_id, "decreased by 12%")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.brain.epistemics.interfaces import PredictionOutcome
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    HypothesisNode,
    PredictionNode,
)

logger = logging.getLogger(__name__)


class PredictionTracker:
    """Tracks predictions against observations (Popperian science).

    Implements the ``IPredictionTracker`` protocol.

    Every hypothesis must produce testable predictions.  This tracker
    monitors prediction outcomes and triggers belief/hypothesis updates.

    The tracker is domain-neutral — it tracks claim strings and outcomes,
    never interpreting domain-specific content.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        belief_manager: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        """Initialize the prediction tracker.

        Args:
            graph: The shared HCIR cognitive graph.
            belief_manager: Optional DiscoveryBeliefManager for cascade
                updates to belief confidence.
            llm: Optional LLM for semantic outcome comparison.
        """
        self._graph = graph
        self._belief_manager = belief_manager
        self._llm = llm

    async def register_prediction(
        self,
        hypothesis_id: str,
        prediction_claim: str,
        predicted_outcome: str,
        time_horizon_ms: int = 0,
    ) -> str:
        """Register a new prediction derived from a hypothesis.

        Args:
            hypothesis_id: The hypothesis that generates this prediction.
            prediction_claim: Human-readable prediction claim.
            predicted_outcome: Expected outcome value/state.
            time_horizon_ms: Time window for evaluation (0 = no deadline).

        Returns:
            The PredictionNode ID.
        """
        node = PredictionNode(
            claim=prediction_claim,
            predicted_outcome=predicted_outcome,
            hypothesis_id=hypothesis_id,
            time_horizon_ms=time_horizon_ms,
        )

        self._graph.upsert_node(node)

        # Link hypothesis → prediction
        self._graph.add_edge(HCIREdge(
            sources=[hypothesis_id],
            targets=[node.id],
            edge_type=HCIREdgeType.PREDICTS,
        ))

        # Also link to hypothesis node's linked_predictions
        hyp_node = self._graph.get_node(hypothesis_id)
        if isinstance(hyp_node, HypothesisNode):
            hyp_node.linked_predictions.append(node.id)
            self._graph.upsert_node(hyp_node)

        logger.info(
            "Registered prediction for hypothesis %s: %s",
            hypothesis_id, prediction_claim[:60],
        )
        return node.id

    async def register_competing_predictions(
        self,
        predictions: list[tuple[str, str, str]],
    ) -> list[str]:
        """Register multiple predictions from rival hypotheses.

        Args:
            predictions: List of (hypothesis_id, claim, predicted_outcome).

        Returns:
            List of prediction node IDs.
        """
        pred_ids: list[str] = []
        for hyp_id, claim, outcome in predictions:
            pred_id = await self.register_prediction(hyp_id, claim, outcome)
            pred_ids.append(pred_id)

        if len(predictions) > 1:
            logger.info(
                "Registered %d competing predictions across %d hypotheses",
                len(predictions),
                len(set(p[0] for p in predictions)),
            )

        return pred_ids

    async def check_prediction(
        self,
        prediction_id: str,
        observed_outcome: str,
    ) -> PredictionOutcome:
        """Check a prediction against an observed outcome.

        Compares the predicted outcome with the observed outcome,
        and updates related hypothesis/belief confidence.

        Args:
            prediction_id: The PredictionNode ID.
            observed_outcome: The actual observed outcome.

        Returns:
            A PredictionOutcome with correctness and confidence delta.
        """
        node = self._graph.get_node(prediction_id)
        if not isinstance(node, PredictionNode):
            logger.warning("Node %s is not a PredictionNode", prediction_id)
            return PredictionOutcome(prediction_id=prediction_id)

        predicted = node.predicted_outcome
        hypothesis_id = node.hypothesis_id

        # Determine correctness
        correct = await self._evaluate_correctness(predicted, observed_outcome)

        # Calculate confidence delta
        confidence_delta = self._compute_confidence_delta(correct)

        # Update prediction node
        node.observed_outcome = observed_outcome
        node.prediction_correct = correct
        node.verified = True
        node.verification_timestamp = time.time()
        self._graph.upsert_node(node)

        # Update hypothesis confidence
        if hypothesis_id:
            await self._update_hypothesis_confidence(
                hypothesis_id, correct, confidence_delta,
            )

        outcome = PredictionOutcome(
            prediction_id=prediction_id,
            hypothesis_id=hypothesis_id,
            predicted=predicted,
            observed=observed_outcome,
            correct=correct,
            confidence_delta=confidence_delta,
        )

        logger.info(
            "Prediction %s: correct=%s, delta=%.3f",
            prediction_id, correct, confidence_delta,
        )
        return outcome

    async def find_competing_predictions(self) -> list[tuple[str, str]]:
        """Find pairs of predictions from rival hypotheses that conflict.

        Returns:
            List of (prediction_id_a, prediction_id_b) tuples that
            are from different hypotheses and predict conflicting outcomes.
        """
        # Collect all active predictions grouped by hypothesis
        hyp_predictions: dict[str, list[tuple[str, str]]] = {}

        for node in self._graph.all_nodes():
            if not isinstance(node, PredictionNode):
                continue

            if node.verified:
                continue  # Already verified

            hyp_id = node.hypothesis_id
            outcome = node.predicted_outcome
            if hyp_id and outcome:
                hyp_predictions.setdefault(hyp_id, []).append((node.id, outcome))

        # Find conflicts across hypotheses
        conflicts: list[tuple[str, str]] = []
        hyp_ids = list(hyp_predictions.keys())

        for i in range(len(hyp_ids)):
            for j in range(i + 1, len(hyp_ids)):
                preds_a = hyp_predictions[hyp_ids[i]]
                preds_b = hyp_predictions[hyp_ids[j]]

                for pred_id_a, outcome_a in preds_a:
                    for pred_id_b, outcome_b in preds_b:
                        if self._outcomes_conflict(outcome_a, outcome_b):
                            conflicts.append((pred_id_a, pred_id_b))

        return conflicts

    async def get_pending_predictions(
        self,
        hypothesis_id: str = "",
    ) -> list[str]:
        """Return IDs of predictions that haven't been verified yet.

        Args:
            hypothesis_id: Optional filter to a specific hypothesis.
        """
        pending: list[str] = []

        for node in self._graph.all_nodes():
            if not isinstance(node, PredictionNode):
                continue

            if node.verified:
                continue  # Already verified

            if hypothesis_id and node.hypothesis_id != hypothesis_id:
                continue

            pending.append(node.id)

        return pending

    async def check_expired_predictions(self) -> list[PredictionOutcome]:
        """Find and evaluate predictions past their time horizon.

        Expired predictions without observed outcomes are marked as
        untestable.
        """
        now = time.time()
        outcomes: list[PredictionOutcome] = []

        for node in self._graph.all_nodes():
            if not isinstance(node, PredictionNode):
                continue

            if node.verified:
                continue

            if node.time_horizon_ms <= 0:
                continue

            # Check if expired
            created_at = node.created_at if hasattr(node, "created_at") else 0.0
            expires_at = created_at + node.time_horizon_ms / 1000.0
            if expires_at > now:
                continue

            # Expired prediction — mark as untestable
            outcome = PredictionOutcome(
                prediction_id=node.id,
                hypothesis_id=node.hypothesis_id,
                predicted=node.predicted_outcome,
                observed="(expired — no observation recorded)",
                correct=None,
                confidence_delta=0.0,
            )

            node.verified = True
            node.verification_timestamp = now
            self._graph.upsert_node(node)

            outcomes.append(outcome)

        if outcomes:
            logger.info("Found %d expired predictions", len(outcomes))

        return outcomes

    # ── Internal Methods ───────────────────────────────────────────────

    async def _evaluate_correctness(
        self, predicted: str, observed: str,
    ) -> bool | None:
        """Compare predicted vs observed outcomes."""
        if not predicted or not observed:
            return None

        # Simple string comparison
        pred_lower = predicted.strip().lower()
        obs_lower = observed.strip().lower()

        if pred_lower == obs_lower:
            return True

        # Check for obvious contradictions
        opposites = {
            ("increase", "decrease"), ("decrease", "increase"),
            ("yes", "no"), ("no", "yes"),
            ("true", "false"), ("false", "true"),
            ("positive", "negative"), ("negative", "positive"),
        }
        for a, b in opposites:
            if a in pred_lower and b in obs_lower:
                return False

        # Partial match
        if pred_lower in obs_lower or obs_lower in pred_lower:
            return True

        # LLM comparison if available
        if self._llm is not None:
            return await self._llm_compare(predicted, observed)

        return None  # Can't determine

    async def _llm_compare(
        self, predicted: str, observed: str,
    ) -> bool | None:
        """Use LLM to compare predicted vs observed outcomes."""
        prompt = (
            f"Did this prediction come true?\n"
            f"Predicted: {predicted}\n"
            f"Observed: {observed}\n\n"
            f"Answer YES, NO, or UNCLEAR."
        )
        try:
            response = await self._llm.generate(prompt)
            text = response if isinstance(response, str) else str(response)
            text_lower = text.strip().lower()
            if "yes" in text_lower:
                return True
            if "no" in text_lower:
                return False
            return None
        except Exception:
            return None

    def _compute_confidence_delta(self, correct: bool | None) -> float:
        """Calculate the confidence change based on prediction correctness."""
        if correct is True:
            return 0.10  # Confirmed prediction boosts confidence
        elif correct is False:
            return -0.15  # Failed prediction reduces confidence (asymmetric)
        return 0.0  # Inconclusive → no change

    async def _update_hypothesis_confidence(
        self,
        hypothesis_id: str,
        correct: bool | None,
        confidence_delta: float,
    ) -> None:
        """Update hypothesis confidence based on prediction outcome."""
        node = self._graph.get_node(hypothesis_id)
        if not isinstance(node, HypothesisNode):
            return

        # Update the uncertainty confidence
        old_conf = node.uncertainty.confidence
        new_conf = min(1.0, max(0.0, old_conf + confidence_delta))
        node.uncertainty.confidence = new_conf
        self._graph.upsert_node(node)

        logger.debug(
            "Hypothesis %s confidence: %.3f → %.3f (prediction %s)",
            hypothesis_id, old_conf, new_conf,
            "confirmed" if correct else "falsified" if correct is False else "inconclusive",
        )

    def _outcomes_conflict(self, outcome_a: str, outcome_b: str) -> bool:
        """Check if two predicted outcomes are conflicting."""
        a = outcome_a.strip().lower()
        b = outcome_b.strip().lower()

        if a == b:
            return False

        # Check for known opposites
        opposites = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("yes", "no"),
            ("true", "false"),
            ("higher", "lower"),
            ("more", "less"),
        ]
        for x, y in opposites:
            if (x in a and y in b) or (y in a and x in b):
                return True

        return False
