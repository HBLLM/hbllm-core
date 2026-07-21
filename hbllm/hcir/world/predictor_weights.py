"""
Predictor Weight Manager — Domain-Keyed Predictor Weight Manager with Exploration Scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
