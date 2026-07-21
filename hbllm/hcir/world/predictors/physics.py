"""
Physics Predictor — Deterministic Physics Differential & State Transition Calculator.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class PhysicsPredictor:
    """Physics-based forward state transition predictor."""

    name: str = "physics"

    def predict_state(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
    ) -> tuple[dict[str, Any], float]:
        """Compute physics forward state transition and return (predicted_variables, confidence)."""
        predicted = dict(snapshot.variables)
        time_sec = horizon_ms / 1000.0

        for var_name, var_value in snapshot.variables.items():
            if isinstance(var_value, (int, float)):
                if "temp" in var_name.lower():
                    if "cool" in action_intent.lower() or "reduce" in action_intent.lower():
                        predicted[var_name] = max(20.0, var_value - 0.1 * time_sec)
                    else:
                        predicted[var_name] = var_value + 0.05 * time_sec
                elif "pressure" in var_name.lower():
                    if "vent" in action_intent.lower():
                        predicted[var_name] = max(1.0, var_value - 0.02 * time_sec)

        logger.debug(
            "PhysicsPredictor calculated state for action '%s' confidence=0.92", action_intent
        )
        return predicted, 0.92
