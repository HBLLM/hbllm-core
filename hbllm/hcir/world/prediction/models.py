"""Prediction Types, Provenance, Receipts & Error Models."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.causal import ActionAffordance


@dataclass(frozen=True)
class PredictionProvenance:
    """Provenance chain tracking requestor, tenant, world, and predictor models used."""

    requestor: str = "executive_controller"
    tenant_id: str = "default_tenant"
    world_id: str = "default_world"
    entity_ids: Sequence[str] = field(default_factory=tuple)
    predictors_used: Sequence[str] = field(default_factory=tuple)
    model_version: str = "world-model-v1.0"
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export provenance tracking as a declarative Pearl causal predicate."""
        return CausalPredicate(
            variable="prediction_provenance",
            operator="==",
            value=self.model_version,
            relation=self.requestor,
            target_entity=self.world_id,
        )


@dataclass
class EnsemblePrediction:
    """Aggregated prediction result from ensemble reality predictors."""

    prediction_id: str
    action_intent: str
    predicted_state: dict[str, Any]
    calibrated_confidence: float
    component_predictions: dict[str, Any] = field(default_factory=dict)
    provenance: PredictionProvenance = field(default_factory=PredictionProvenance)

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Lift predicted state variables into declarative causal predicates."""
        preds: list[CausalPredicate] = []
        for k, v in self.predicted_state.items():
            if isinstance(v, (int, float, str, bool)):
                preds.append(
                    CausalPredicate(
                        variable=k,
                        operator="==",
                        value=v,
                        relation=self.action_intent,
                        target_entity=self.provenance.world_id,
                    )
                )
            elif (
                isinstance(v, (tuple, list))
                and len(v) == 2
                and all(isinstance(x, (int, float)) for x in v)
            ):
                preds.append(
                    CausalPredicate(
                        variable=k,
                        operator="==",
                        value=(int(v[0]), int(v[1])),
                        relation=self.action_intent,
                        target_entity=self.provenance.world_id,
                    )
                )
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction committing prediction to working memory."""
        serializable_state = {
            k: v for k, v in self.predicted_state.items() if not isinstance(v, np.ndarray)
        }
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"pred_{self.prediction_id}",
                node_data={
                    "prediction_id": self.prediction_id,
                    "action_intent": self.action_intent,
                    "predicted_state": serializable_state,
                    "calibrated_confidence": self.calibrated_confidence,
                    "world_id": self.provenance.world_id,
                    "model_version": self.provenance.model_version,
                },
            )
        ]

    def to_affordance(self) -> ActionAffordance:
        """Synthesize action affordance corresponding to predicted action intent."""
        from hbllm.hcir.world.causal import ActionAffordance

        return ActionAffordance(
            action_id=self.action_intent,
            name=f"predicted_{self.action_intent}",
            confidence=self.calibrated_confidence,
            metadata={
                "prediction_id": self.prediction_id,
                "world_id": self.provenance.world_id,
            },
        )


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
                    "id": f"error_{self.error_id}",
                    "node_type": "prediction_error",
                    "category": "reasoning",
                    "error_id": self.error_id,
                    "prediction_id": self.prediction_id,
                    "prediction_source": self.prediction_source,
                    "typology": str(self.typology),
                    "error_variance": self.error_variance,
                    "delta": self.error_variance,
                    "error_magnitude": self.error_variance,
                    "cause_hypothesis": self.cause_hypothesis,
                    "suspected_cause": self.cause_hypothesis,
                    "confidence_before": self.confidence_before,
                    "confidence_after": self.confidence_after,
                    "weight_delta": self.weight_delta,
                    "created_at": self.created_at,
                },
            )
        ]


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
                    "id": f"receipt_{self.receipt_id}",
                    "node_type": "prediction_receipt",
                    "category": "meta",
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


__all__ = [
    "EnsemblePrediction",
    "PredictionErrorTypology",
    "PredictionProvenance",
    "WorldPredictionError",
    "WorldPredictionReceipt",
]
