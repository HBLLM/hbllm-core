"""Prediction Error Node & Typology — Distinguishes Model Error from Environment Change.

Captures expectation versus reality variance, error root causes, and transitions
to declarative CausalPredicates and atomic TransactionOperations for state correction.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


class PredictionErrorTypology(StrEnum):
    """Typology classifying prediction error root cause."""

    MODEL_ERROR = "model_error"
    ENVIRONMENT_CHANGE = "environment_change"
    LATENT_CONFOUNDER = "latent_confounder"


@dataclass
class WorldPredictionError:
    """Error record capturing expectation vs reality variance and error typology."""

    error_id: str = field(default_factory=lambda: f"perr_{uuid.uuid4().hex[:8]}")
    prediction_id: str = ""
    prediction_source: str = "physics"
    typology: PredictionErrorTypology = PredictionErrorTypology.MODEL_ERROR
    expected_state: dict[str, Any] = field(default_factory=dict)
    actual_state: dict[str, Any] = field(default_factory=dict)
    error_variance: float = 0.0
    cause_hypothesis: str = "unknown"
    confidence_before: float = 0.90
    confidence_after: float = 0.70
    weight_delta: float = -0.10
    created_at: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export prediction error as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"prediction_error:{self.error_id}",
            operator="!=",
            value=self.error_variance,
            relation=str(self.typology),
            target_entity=self.prediction_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording prediction error."""
        return [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=f"error_{self.error_id}",
                node_data={
                    "error_id": self.error_id,
                    "prediction_id": self.prediction_id,
                    "prediction_source": self.prediction_source,
                    "typology": str(self.typology),
                    "error_variance": self.error_variance,
                    "cause_hypothesis": self.cause_hypothesis,
                    "confidence_before": self.confidence_before,
                    "confidence_after": self.confidence_after,
                    "weight_delta": self.weight_delta,
                    "created_at": self.created_at,
                },
            )
        ]
