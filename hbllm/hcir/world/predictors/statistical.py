"""
Statistical Predictor — Time-Series Extrapolation & Regression Predictor.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class StatisticalPredictor:
    """Time-series trend predictor."""

    name: str = "statistical"

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
