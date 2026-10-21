"""World Prediction Receipt — Structured Replay & Trace Receipt for Predictions.

Deterministic replay and execution trace receipts lifted to formal CausalPredicates
and atomic TransactionOperations for state-space auditability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.prediction_types import PredictionProvenance


@dataclass
class WorldPredictionReceipt:
    """Deterministic trace receipt of a prediction execution."""

    receipt_id: str
    prediction_id: str
    model_version: str
    provenance: PredictionProvenance
    input_state_hash: str
    predicted_outcome: str
    actual_outcome: str | None = None
    calibrated_confidence: float = 0.90
    surprise_score: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export trace receipt as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"receipt:{self.receipt_id}",
            operator="==",
            value=self.predicted_outcome,
            relation=self.actual_outcome or "pending",
            target_entity=self.prediction_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording replay receipt."""
        return [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=f"receipt_{self.receipt_id}",
                node_data={
                    "receipt_id": self.receipt_id,
                    "prediction_id": self.prediction_id,
                    "model_version": self.model_version,
                    "input_state_hash": self.input_state_hash,
                    "predicted_outcome": self.predicted_outcome,
                    "actual_outcome": self.actual_outcome,
                    "calibrated_confidence": self.calibrated_confidence,
                    "surprise_score": self.surprise_score,
                    "created_at": self.created_at,
                },
            )
        ]
