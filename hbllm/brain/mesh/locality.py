"""Cognitive Locality Engine.

Determines what minimum context is required for a delegated task,
preventing over-sharing of cognition and maintaining local sovereignty.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.mesh.capsule import CognitiveOwnership, TaskCapsule
from hbllm.brain.mesh.registry import TaskPriorityClass

logger = logging.getLogger(__name__)


class CognitiveLocalityEngine:
    """Calculates the minimal required subgraph for delegation."""

    def __init__(self, local_node_id: str) -> None:
        self.local_node_id = local_node_id

    def create_task_capsule(
        self,
        goal_id: str,
        target_node_id: str,
        authority_node: str,
        priority: TaskPriorityClass,
        world_state_subgraph: dict[str, Any],
        causal_edges: list[dict[str, Any]],
        utility_constraints: dict[str, float],
        permissions_scope: list[str],
    ) -> TaskCapsule:
        """Packages only the necessary context for the target node."""

        # Determine strict context boundary
        # For privacy, if the target is a CLOUD_SERVER, we might anonymize or drop PII
        filtered_state = self._apply_locality_filters(world_state_subgraph, target_node_id)

        capsule = TaskCapsule(
            goal_id=goal_id,
            ownership=CognitiveOwnership(
                origin_node=self.local_node_id,
                authority_node=authority_node,
                execution_node=target_node_id,
                verification_node=self.local_node_id,
            ),
            priority=priority,
            required_entities=filtered_state,
            causal_dependencies=causal_edges,
            utility_constraints=utility_constraints,
            permissions_scope=permissions_scope,
        )

        return capsule

    def _apply_locality_filters(self, state: dict[str, Any], target_node: str) -> dict[str, Any]:
        """Strip highly sensitive local data before sending it out of the personal cluster."""
        # Simple placeholder for data minimization rules
        filtered = {}
        for key, value in state.items():
            if key.startswith("biometric") and "cloud" in target_node.lower():
                continue  # Do not send biometrics to cloud
            filtered[key] = value
        return filtered
