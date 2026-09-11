"""Developmental Action Adapter for BabyWorld.

Pure decoupled device driver bridging HCIR ActionNodes to low-level
BabyWorldEnvironment motor commands without domain-specific cheats.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import ActionNode

from .environment import BabyWorldEnvironment
from .types import BabyActionType, SensoryObservation

logger = logging.getLogger(__name__)


class DevelopmentalActionAdapter:
    """Pure device driver for BabyWorld motor execution."""

    def __init__(self, environment: BabyWorldEnvironment) -> None:
        self.env = environment

    def execute_action_node(
        self,
        action_node: ActionNode,
        neutral_to_raw_id_map: dict[str, str] | None = None,
    ) -> tuple[SensoryObservation, float, bool, dict[str, Any]]:
        """Dispatch an HCIR ActionNode to the physical environment."""
        action_name = action_node.properties.get("action_type", "")
        target_neutral_id = action_node.properties.get("target_id", "")
        params = action_node.properties.get("parameters", {})

        # Map neutral ID back to environment object ID if mapping provided
        raw_target_id = target_neutral_id
        if neutral_to_raw_id_map and target_neutral_id in neutral_to_raw_id_map:
            raw_target_id = neutral_to_raw_id_map[target_neutral_id]

        try:
            baby_action = BabyActionType(action_name.upper())
        except ValueError:
            logger.warning("Unrecognized action type '%s', defaulting to LOOK", action_name)
            baby_action = BabyActionType.LOOK

        return self.env.step(action=baby_action, target_id=raw_target_id, parameters=params)

    def dispatch_primitive(
        self,
        action_type: BabyActionType,
        target_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[SensoryObservation, float, bool, dict[str, Any]]:
        """Direct primitive dispatch helper for exploratory motor babbling."""
        return self.env.step(action=action_type, target_id=target_id, parameters=parameters)
