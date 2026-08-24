"""
Prediction Operator — wraps MarkovPredictor as a ReasoningOperator.

Delegates to the existing ``CognitivePredictors`` / ``MarkovPredictor``
for order-N Markov next-state prediction, but wraps the result as a
proper ``CognitiveResult`` with provenance and proposed HCIR transitions.

The wrapper reads observations and beliefs from the frozen HCIR view,
feeds them to a local MarkovPredictor, and returns predictions as
proposed PredictionNodes.

Independence Level: L1 (no LLM)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    ProblemType,
    ProvenanceChain,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.brain.reasoning.prediction import MarkovPredictor
from hbllm.hcir.graph import (
    EventNode,
    HCIRNodeType,
    ObservationNode,
    PredictionNode,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)


class PredictionOperator:
    """Wraps MarkovPredictor as a reasoning operator.

    Reads event sequences from the frozen HCIR view, trains a
    transient MarkovPredictor on them, and returns predictions
    as proposed PredictionNodes.
    """

    def __init__(self, order: int = 3) -> None:
        self._order = order

    @property
    def operator_id(self) -> str:
        return "prediction"

    @property
    def operator_name(self) -> str:
        return "Markov Prediction Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.PREDICTION: 0.9,
            ProblemType.TEMPORAL: 0.5,
            ProblemType.PLANNING: 0.4,
            ProblemType.EXPLANATION: 0.2,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        n_events = len(view.nodes_by_type(HCIRNodeType.EVENT))
        n_obs = len(view.nodes_by_type(HCIRNodeType.OBSERVATION))
        total = n_events + n_obs

        if total < 2:
            return 0.0

        return min(1.0, base + min(0.3, total * 0.02))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = context.graph_view.node_count
        return ResourceCost(
            wall_clock_ms=max(1.0, n * 0.1),
            nodes_read=n,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Train on HCIR event sequence and produce predictions."""
        start = time.time()
        view = context.graph_view

        # ── Extract event sequence (sorted by timestamp) ─────────────
        events = self._extract_sequence(view)

        if len(events) < 2:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "Insufficient event sequence"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Train transient predictor ────────────────────────────────
        predictor = MarkovPredictor(order=self._order)
        for state_label in events:
            predictor.train(state_label)

        # ── Predict ──────────────────────────────────────────────────
        predictions = predictor.predict_top_k(k=5)
        entropy = predictor.entropy()

        if not predictions:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "Predictor returned empty distribution"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Build result ─────────────────────────────────────────────
        proposed_ops: list[TransactionOperation] = []
        provenance_chains: list[ProvenanceChain] = []

        for predicted_state, probability in predictions:
            pred_node = PredictionNode(
                claim=f"Next state predicted: {predicted_state}",
                predicted_outcome=predicted_state,
                provenance=Provenance(
                    created_by=self.operator_id,
                    source_type="inferred",
                    reason=(
                        f"Markov order-{self._order} prediction: "
                        f"P({predicted_state})={probability:.3f}"
                    ),
                ),
            )
            pred_node.uncertainty.confidence = probability

            proposed_ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=pred_node.model_dump(),
                )
            )

            provenance_chains.append(
                ProvenanceChain(
                    conclusion=f"Predicted: {predicted_state} (p={probability:.3f})",
                    operator_id=self.operator_id,
                    reasoning_steps=[
                        f"Trained on {len(events)} events",
                        f"Distribution entropy: {entropy:.3f} bits",
                    ],
                    confidence=probability,
                )
            )

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "predictions": [{"state": s, "probability": round(p, 4)} for s, p in predictions],
                "entropy": round(entropy, 3),
                "sequence_length": len(events),
            },
            confidence=predictions[0][1] if predictions else 0.0,
            assumptions=["Markov assumption: future depends only on recent context"],
            proposed_transitions=proposed_ops,
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
            ),
        )

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_sequence(view: Any) -> list[str]:
        """Extract a temporal event sequence from the frozen view."""
        timed_events: list[tuple[float, str]] = []

        for node in view.nodes_by_type(HCIRNodeType.EVENT):
            if isinstance(node, EventNode):
                label = node.event_kind or node.id
                timed_events.append((node.event_timestamp, label))

        for node in view.nodes_by_type(HCIRNodeType.OBSERVATION):
            if isinstance(node, ObservationNode):
                ts = node.provenance.timestamp if node.provenance.timestamp else 0.0
                label = node.payload.get("type", node.id) if node.payload else node.id
                timed_events.append((ts, str(label)))

        timed_events.sort(key=lambda x: x[0])
        return [label for _, label in timed_events]
