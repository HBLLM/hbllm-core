"""Predictor Weight Manager — Domain-Keyed Predictor Weight Manager with Exploration Scoring.

Manages domain-scoped predictor weights with exploration scoring, lifted to
declarative CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.world_context import WorldModelScope

logger = logging.getLogger(__name__)


@dataclass
class PredictorWeightState:
    """Weight and accuracy metrics for a specific predictor within a domain scope."""

    predictor_name: str
    accuracy_weight: float = 0.50
    exploration_bonus: float = 0.20
    uncertainty: float = 0.10
    total_evaluations: int = 0

    def compute_score(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1) -> float:
        """Compute exploration-aware selection score: (alpha*acc + beta*expl) / (gamma*unc + 1e-5)."""
        num = (alpha * self.accuracy_weight) + (beta * self.exploration_bonus)
        den = max(0.01, gamma * (1.0 + self.uncertainty))
        return num / den

    def to_causal_predicate(self) -> CausalPredicate:
        """Export predictor weight state as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"weight:{self.predictor_name}",
            operator="==",
            value=self.accuracy_weight,
            relation="accuracy_weight",
            target_entity=self.predictor_name,
        )

    def to_transaction_operation(self, scope_key: str = "global") -> TransactionOperation:
        """Propose atomic HCIR transaction updating predictor weight in cognitive kernel."""
        return TransactionOperation(
            op=TransactionOp.UPSERT_NODE,
            node_id=f"weight_{scope_key}_{self.predictor_name}",
            node_data={
                "predictor_name": self.predictor_name,
                "scope_key": scope_key,
                "accuracy_weight": self.accuracy_weight,
                "exploration_bonus": self.exploration_bonus,
                "uncertainty": self.uncertainty,
                "total_evaluations": self.total_evaluations,
            },
        )


class PredictorWeightManager:
    """Manages domain-keyed predictor weights with exploration scoring."""

    def __init__(self) -> None:
        self._weights: dict[str, dict[str, PredictorWeightState]] = {}

    def get_predictor_weight(
        self, scope: WorldModelScope, predictor_name: str
    ) -> PredictorWeightState:
        """Retrieve or initialize predictor weight state for given domain scope."""
        key = scope.to_key()
        if key not in self._weights:
            self._weights[key] = {}

        if predictor_name not in self._weights[key]:
            self._weights[key][predictor_name] = PredictorWeightState(predictor_name=predictor_name)

        return self._weights[key][predictor_name]

    def update_weight(
        self,
        scope: WorldModelScope,
        predictor_name: str,
        accuracy_delta: float,
        lr: float = 0.05,
    ) -> PredictorWeightState:
        """Update predictor weight using accuracy delta rule."""
        state = self.get_predictor_weight(scope, predictor_name)
        new_acc = max(0.01, min(1.0, state.accuracy_weight + (lr * accuracy_delta)))
        state.accuracy_weight = new_acc
        state.total_evaluations += 1
        state.exploration_bonus = max(
            0.01, state.exploration_bonus * 0.95
        )  # Decay exploration bonus over time
        logger.info(
            "PredictorWeightManager updated '%s' for scope '%s': new_acc=%.4f",
            predictor_name,
            scope.to_key(),
            new_acc,
        )
        return state

    def export_causal_predicates(self, scope: WorldModelScope) -> list[CausalPredicate]:
        """Export all predictor weights for a scope as causal predicates."""
        key = scope.to_key()
        states = self._weights.get(key, {})
        return [st.to_causal_predicate() for st in states.values()]

    def to_transaction_operations(self, scope: WorldModelScope) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting weights for a domain scope."""
        key = scope.to_key()
        states = self._weights.get(key, {})
        return [st.to_transaction_operation(key) for st in states.values()]
