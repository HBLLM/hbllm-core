"""
World Kernel — Predictive Cognitive Subsystem & Forward World Model.

Enables predictive cognition:

    Observation
        │
    World State (WorldVariable, PhysicalEntity, EnvironmentState)
        │
    Prediction Model (forward state transition: state + action → next_state)
        │
    Simulation (FORK branch)
        │
    Counterfactual Evaluation
        │
    Decision

Functions:
    - ``get_current_world_state(workspace)``
    - ``predict(workspace, action_node, time_horizon_ms)`` -> PredictionNode
    - ``compare_outcomes(candidate_outcomes)`` -> Ranks predictions by utility
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import (
    ActionNode,
    EnvironmentStateNode,
    HCIRNodeType,
    NodeLifecycle,
    PhysicalEntityNode,
    PredictionNode,
    WorldVariableNode,
)
from hbllm.hcir.types import Provenance, Scope, UncertaintyVector
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class WorldStateSummary:
    """Aggregated summary of environmental state, variables, and physical entities."""

    environment_name: str = "default_env"
    overall_status: str = "nominal"
    variables: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, str] = field(default_factory=dict)  # entity_name → status
    timestamp: float = field(default_factory=time.time)


class WorldKernel:
    """Predictive cognitive kernel for world model state estimation and forward prediction.

    Usage::

        world_kernel = WorldKernel(workspace)
        state = world_kernel.get_current_world_state()
        prediction = world_kernel.predict(
            action=ActionNode(intent="increase_irrigation"),
            time_horizon_ms=3600000,
        )
    """

    def __init__(self, workspace: HCIRWorkspaceState) -> None:
        self._workspace = workspace

    def get_current_world_state(self) -> WorldStateSummary:
        """Aggregate current world variables and physical entities from the workspace graph."""
        summary = WorldStateSummary()

        # Environmental state nodes
        env_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.ENVIRONMENT_STATE)
        if env_nodes and isinstance(env_nodes[0], EnvironmentStateNode):
            summary.environment_name = env_nodes[0].environment_name
            summary.overall_status = env_nodes[0].overall_status

        # World variables
        var_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.WORLD_VARIABLE)
        for node in var_nodes:
            if isinstance(node, WorldVariableNode):
                summary.variables[node.variable_name] = node.value

        # Physical entities
        entity_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)
        for node in entity_nodes:
            if isinstance(node, PhysicalEntityNode):
                summary.entities[node.entity_name] = node.status

        return summary

    def predict(
        self,
        action: ActionNode,
        time_horizon_ms: int = 3600000,
        confidence: float = 0.82,
        author: str = "world_kernel",
        tenant_id: str = "default",
    ) -> PredictionNode:
        """Forward state transition: state + action -> predicted_outcome.

        Generates a PredictionNode stored in the workspace graph.
        """
        current_state = self.get_current_world_state()

        # Heuristic / capability prediction simulation
        predicted_outcome = (
            f"Action '{action.intent}' on environment '{current_state.environment_name}' "
            f"with active variables {list(current_state.variables.keys())} "
            f"is predicted to complete nominally within {time_horizon_ms // 1000}s horizon."
        )

        node_id = f"pred_{uuid.uuid4().hex[:8]}"
        pred_node = PredictionNode(
            id=node_id,
            claim=f"Outcome of {action.intent}",
            predicted_outcome=predicted_outcome,
            time_horizon_ms=time_horizon_ms,
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=confidence),
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["prediction", "world_kernel", action.intent],
        )

        self._workspace.upsert_node(pred_node, author=author)
        logger.info("Generated PredictionNode '%s' for action '%s'", node_id, action.intent)
        return pred_node

    @staticmethod
    def compare_outcomes(candidates: list[tuple[ActionNode, PredictionNode]]) -> tuple[ActionNode, PredictionNode]:
        """Rank candidate (Action, Prediction) pairs and return the optimal candidate."""
        if not candidates:
            raise ValueError("No candidates provided for comparison")

        # Utility score = prediction confidence * (1.0 - cost_factor)
        best_candidate = max(
            candidates,
            key=lambda item: item[1].uncertainty.confidence * (1.0 / (1.0 + item[0].estimated_cost * 0.01)),
        )
        return best_candidate
