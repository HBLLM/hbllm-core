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
    BeliefNode,
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
from hbllm.hcir.world.surprise_engine import SurpriseEngine, SurpriseEvaluation
from hbllm.hcir.world.verification_gate import VerificationGate
from hbllm.hcir.world.world_belief import WorldBeliefGraph, WorldBeliefNode
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
    latent_beliefs: dict[str, Any] = field(default_factory=dict)
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

        for b in self.belief_graph.get_latent_beliefs():
            summary.latent_beliefs[b.subject] = {
                "value": b.value,
                "confidence": b.confidence,
                "distribution": dict(b.distribution),
            }

        return summary

    def sync_from_workspace(self) -> None:
        """Sync variables, physical entities, and beliefs from workspace graph into digital twin and belief graph."""
        for node in self._workspace.graph.nodes_by_type(HCIRNodeType.WORLD_VARIABLE):
            if isinstance(node, WorldVariableNode):
                self.digital_twin.sync_sensor_telemetry(node.variable_name, node.value)
        for node in self._workspace.graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            if isinstance(node, PhysicalEntityNode):
                entity = self.digital_twin.register_entity(node.id, node.entity_name, node.status)
                entity.telemetry = dict(node.properties)
        for node in self._workspace.graph.nodes_by_type(HCIRNodeType.BELIEF):
            if isinstance(node, BeliefNode):
                is_latent = bool(node.properties.get("is_latent", False))
                dist = dict(node.properties.get("distribution", {}))
                subj = str(node.properties.get("subject", node.id))
                val = node.properties.get("value", node.claim)
                conf = node.uncertainty.confidence if node.uncertainty else 0.9
                if node.id not in self.belief_graph._beliefs:
                    self.belief_graph.add_belief(
                        WorldBeliefNode(
                            belief_id=node.id,
                            subject=subj,
                            predicate=str(node.properties.get("predicate", "has_state")),
                            value=val,
                            confidence=conf,
                            evidence_sources=list(node.evidence_sources),
                            is_latent=is_latent,
                            distribution=dist,
                        )
                    )

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
        updated_vars = dict(snapshot.variables)
        if getattr(action, "properties", None):
            updated_vars.update(action.properties)

        latent_beliefs = self.belief_graph.get_latent_beliefs()
        if latent_beliefs:
            updated_vars["latent_beliefs"] = {
                b.subject: {
                    "value": b.value,
                    "confidence": b.confidence,
                    "distribution": dict(b.distribution),
                }
                for b in latent_beliefs
            }

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

    def observe_and_update(
        self,
        action: ActionNode,
        actual_state: dict[str, Any],
        prediction: PredictionNode | None = None,
        prediction_source: str = "physics",
        author: str = "world_kernel",
    ) -> tuple[SurpriseEvaluation, WorldBeliefNode | None]:
        """Evaluate prediction error / surprise upon observing actual_state after an action.

        If persistent surprise is detected (PredictionErrorTypology.LATENT_CONFOUNDER),
        induces a latent variable belief to model the unobserved confounder.
        """
        expected_state: dict[str, Any] = {}
        prediction_id = f"pred_eval_{uuid.uuid4().hex[:6]}"
        confidence = 0.85

        if prediction is not None:
            prediction_id = prediction.id
            confidence = prediction.uncertainty.confidence if prediction.uncertainty else 0.85
            if "predicted_state" in prediction.properties and isinstance(
                prediction.properties["predicted_state"], dict
            ):
                expected_state = prediction.properties["predicted_state"]
            else:
                expected_state = {
                    k: v for k, v in prediction.properties.items() if k != "predicted_state"
                }
        elif getattr(action, "properties", None) and "predicted_state" in action.properties:
            if isinstance(action.properties["predicted_state"], dict):
                expected_state = action.properties["predicted_state"]

        context_sig = f"{action.intent}:{prediction_source}"
        surprise_eval = self.surprise_engine.evaluate_surprise(
            prediction_id=prediction_id,
            expected_state=expected_state,
            actual_state=actual_state,
            confidence=confidence,
            prediction_source=prediction_source,
            context_signature=context_sig,
        )

        latent_belief: WorldBeliefNode | None = None
        if surprise_eval.is_persistent:
            latent_belief = self.induce_latent_variable(
                action_intent=action.intent,
                expected_state=expected_state,
                actual_state=actual_state,
                author=author,
            )

        return surprise_eval, latent_belief

    def induce_latent_variable(
        self,
        action_intent: str,
        expected_state: dict[str, Any],
        actual_state: dict[str, Any],
        author: str = "world_kernel",
    ) -> WorldBeliefNode:
        """Discover divergent state variables and induce a latent confounder belief node."""
        divergent_keys = [
            k
            for k in sorted(set(expected_state.keys()) | set(actual_state.keys()))
            if expected_state.get(k) != actual_state.get(k)
        ]
        key_desc = "_".join(divergent_keys) if divergent_keys else "state_variance"
        latent_name = f"latent_{action_intent}_{key_desc}"
        belief_id = f"belief_latent_{uuid.uuid4().hex[:8]}"

        actual_val = str(actual_state.get(divergent_keys[0])) if divergent_keys else "unexpected"
        expected_val = str(expected_state.get(divergent_keys[0])) if divergent_keys else "nominal"
        initial_dist = {
            expected_val: 0.5,
            actual_val: 0.5,
        }
        if len(initial_dist) < 2:
            initial_dist = {"nominal": 0.5, "anomaly": 0.5}

        world_belief = WorldBeliefNode(
            belief_id=belief_id,
            subject=latent_name,
            predicate="confounds",
            value=actual_val if actual_val in initial_dist else list(initial_dist.keys())[0],
            confidence=0.5,
            evidence_sources=[f"surprise_engine:{action_intent}"],
            is_latent=True,
            distribution=initial_dist,
        )
        self.belief_graph.add_belief(world_belief)

        graph_belief = BeliefNode(
            id=belief_id,
            claim=f"Latent confounder for {action_intent} affecting {key_desc}",
            belief_type="causal",
            evidence_sources=[f"surprise_engine:{action_intent}"],
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=0.5),
            provenance=Provenance(created_by=author),
            tags=["latent", "belief", "pomdp", action_intent],
            properties={
                "subject": latent_name,
                "value": world_belief.value,
                "is_latent": True,
                "distribution": initial_dist,
                "divergent_keys": divergent_keys,
            },
        )
        self._workspace.upsert_node(graph_belief, author=author)
        self.digital_twin.sync_sensor_telemetry(latent_name, world_belief.value)

        logger.info(
            "WorldKernel induced latent variable '%s' (id=%s) with distribution=%s",
            latent_name,
            belief_id,
            initial_dist,
        )
        return world_belief

    def update_latent_belief(
        self,
        latent_name_or_id: str,
        likelihoods: dict[str, float],
        author: str = "world_kernel",
    ) -> WorldBeliefNode:
        """Update latent belief distribution via Bayesian posterior inference."""
        belief = self.belief_graph.get_belief(latent_name_or_id)
        if belief is None:
            matching = [
                b for b in self.belief_graph.get_latent_beliefs() if b.subject == latent_name_or_id
            ]
            if matching:
                belief = matching[0]

        if belief is None:
            raise KeyError(f"No latent belief found with subject or id '{latent_name_or_id}'")

        belief.update_distribution(likelihoods)
        self.digital_twin.sync_sensor_telemetry(belief.subject, belief.value)

        workspace_node = self._workspace.graph.get_node(belief.belief_id)
        if workspace_node and isinstance(workspace_node, BeliefNode):
            workspace_node.uncertainty.confidence = belief.confidence
            workspace_node.properties["value"] = belief.value
            workspace_node.properties["distribution"] = dict(belief.distribution)
            self._workspace.upsert_node(workspace_node, author=author)

        logger.info(
            "WorldKernel updated latent belief '%s': MAP=%s (confidence=%.3f, dist=%s)",
            belief.subject,
            belief.value,
            belief.confidence,
            belief.distribution,
        )
        return belief

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
