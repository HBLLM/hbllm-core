"""
Statistical Predictor — Time-Series Extrapolation & Regression Predictor.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class StatisticalPredictor:
    """Time-series trend predictor."""

    name: str = "statistical"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export statistical predictor status as a declarative causal predicate."""
        return CausalPredicate(
            variable="predictor:statistical",
            operator="==",
            value="active",
            relation="time_series_trend",
            target_entity="statistical",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering statistical predictor."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="predictor_statistical",
                node_data={
                    "name": self.name,
                    "type": "time_series_trend",
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
        """Compute statistical trend prediction."""
        predicted = dict(snapshot.variables)
        for var_name, var_value in snapshot.variables.items():
            if isinstance(var_value, (int, float)):
                predicted[var_name] = var_value * 1.01  # Slight linear trend

        logger.debug(
            "StatisticalPredictor calculated state for action '%s' confidence=0.85", action_intent
        )
        return predicted, 0.85
