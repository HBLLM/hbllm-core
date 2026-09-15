"""
World Kernel — Predictive Cognitive OS Substrate & Forward Reality Model.
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
from hbllm.hcir.world.active_inference import ActiveInferenceEngine
from hbllm.hcir.world.counterfactual_graph import CounterfactualGraph
from hbllm.hcir.world.digital_twin import DigitalTwinRegistry
from hbllm.hcir.world.prediction_promotion import PredictionPromotion
from hbllm.hcir.world.predictive_reality import PredictiveRealityModel
from hbllm.hcir.world.surprise_engine import SurpriseEngine
from hbllm.hcir.world.verification_gate import VerificationGate
from hbllm.hcir.world.world_belief import WorldBeliefGraph
from hbllm.hcir.world.world_causal import WorldCausalGraph
from hbllm.hcir.world.world_state_interpreter import WorldStateInterpreter
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class WorldStateSummary:
    """Aggregated summary of environmental state, variables, and physical entities."""

    environment_name: str = "default_env"
    overall_status: str = "nominal"
    variables: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class WorldKernel:
    """Predictive cognitive kernel for world model state estimation and forward prediction."""

    def __init__(self, workspace: HCIRWorkspaceState) -> None:
        self._workspace = workspace
        branch = getattr(workspace, "branch_name", "main")
        self.digital_twin = DigitalTwinRegistry(world_id=branch)
        self.interpreter = WorldStateInterpreter()
        self.belief_graph = WorldBeliefGraph(world_id=branch)
        self.causal_graph = WorldCausalGraph(world_id=branch)
        self.counterfactual_graph = CounterfactualGraph(world_id=branch)
        self.predictive_reality = PredictiveRealityModel()
        self.surprise_engine = SurpriseEngine()
        self.active_inference = ActiveInferenceEngine()
        self.verification_gate = VerificationGate()
        self.promotion = PredictionPromotion(self.verification_gate)

    def get_current_world_state(self) -> WorldStateSummary:
        """Aggregate current world variables and physical entities from workspace graph."""
        summary = WorldStateSummary()

        env_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.ENVIRONMENT_STATE)
        if env_nodes and isinstance(env_nodes[0], EnvironmentStateNode):
            summary.environment_name = env_nodes[0].environment_name
            summary.overall_status = env_nodes[0].overall_status

        var_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.WORLD_VARIABLE)
        for node in var_nodes:
            if isinstance(node, WorldVariableNode):
                summary.variables[node.variable_name] = node.value

        entity_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)
        for node in entity_nodes:
            if isinstance(node, PhysicalEntityNode):
                summary.entities[node.entity_name] = node.status

        return summary

    def sync_from_workspace(self) -> None:
        """Sync variables and physical entities from workspace graph into digital twin."""
        for node in self._workspace.graph.nodes_by_type(HCIRNodeType.WORLD_VARIABLE):
            if isinstance(node, WorldVariableNode):
                self.digital_twin.sync_sensor_telemetry(node.variable_name, node.value)
        for node in self._workspace.graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if isinstance(node, PhysicalEntityNode):
                entity = self.digital_twin.register_entity(node.id, node.entity_name, node.status)
                entity.telemetry = dict(node.properties)

    def predict(
        self,
        action: ActionNode,
        time_horizon_ms: int = 3600000,
        confidence: float = 0.82,
        author: str = "world_kernel",
        tenant_id: str = "default",
    ) -> PredictionNode:
        """Forward state transition: state + action -> predicted_outcome."""
        self.sync_from_workspace()
        snapshot = self.digital_twin.create_snapshot()
        if getattr(action, "properties", None):
            updated_vars = dict(snapshot.variables)
            updated_vars.update(action.properties)
            snapshot = WorldStateSnapshot(
                world_id=snapshot.world_id,
                timestamp=snapshot.timestamp,
                variables=updated_vars,
                entity_states=snapshot.entity_states,
            )

        ensemble_pred = self.predictive_reality.predict(snapshot, action.intent, time_horizon_ms)

        node_id = f"pred_{uuid.uuid4().hex[:8]}"
        pred_node = PredictionNode(
            id=node_id,
            claim=f"Outcome of {action.intent}",
            predicted_outcome=str(ensemble_pred.predicted_state),
            time_horizon_ms=time_horizon_ms,
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=ensemble_pred.calibrated_confidence),
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["prediction", "world_kernel", action.intent],
            properties={"predicted_state": ensemble_pred.predicted_state},
        )

        self._workspace.upsert_node(pred_node, author=author)
        logger.info(
            "WorldKernel generated PredictionNode '%s' for action '%s'", node_id, action.intent
        )
        return pred_node

    @staticmethod
    def compare_outcomes(
        candidates: list[tuple[ActionNode, PredictionNode]],
    ) -> tuple[ActionNode, PredictionNode]:
        """Rank candidate (Action, Prediction) pairs and return optimal candidate."""
        if not candidates:
            raise ValueError("No candidates provided for comparison")

        best_candidate = max(
            candidates,
            key=lambda item: (
                item[1].uncertainty.confidence
                * (1.0 / (1.0 + getattr(item[0], "estimated_cost", 10.0) * 0.01))
            ),
        )
        return best_candidate
