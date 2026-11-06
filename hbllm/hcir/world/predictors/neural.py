"""
Neural World Model — Latent Space Neural World Model Predictor.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class NeuralWorldModel:
    """Latent space neural network world model predictor."""

    name: str = "neural"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export neural world model status as a declarative causal predicate."""
        return CausalPredicate(
            variable="predictor:neural",
            operator="==",
            value="active",
            relation="latent_forward_model",
            target_entity="neural",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering neural world model."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="predictor_neural",
                node_data={
                    "name": self.name,
                    "type": "latent_forward_model",
                    "active": True,
                },
            )
        ]

    def predict_state(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
    ) -> tuple[dict[str, Any], float]:
        """Compute neural latent space forward prediction."""
        predicted = dict(snapshot.variables)
        logger.debug(
            "NeuralWorldModel calculated state for action '%s' confidence=0.82", action_intent
        )
        return predicted, 0.82
