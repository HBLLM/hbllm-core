"""Prediction Types & Provenance — Structured Containers for Ensemble Predictions & Provenance.

Provides multi-horizon ensemble prediction structures, provenance tracking, and conversions
to declarative CausalPredicates, ActionAffordances, and atomic TransactionOperations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.cortex_causal import ActionAffordance


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
            k: v for k, v in self.predicted_state.items() if not hasattr(v, "shape")
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
        from hbllm.hcir.world.cortex_causal import ActionAffordance

        return ActionAffordance(
            action_id=self.action_intent,
            name=f"predicted_{self.action_intent}",
            confidence=self.calibrated_confidence,
            metadata={
                "prediction_id": self.prediction_id,
                "world_id": self.provenance.world_id,
            },
        )
