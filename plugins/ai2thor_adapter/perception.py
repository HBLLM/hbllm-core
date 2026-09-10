"""
AI2-THOR Perception Adapter.

Translates 3D spatial metadata, poses, and receptacle containment trees into
hierarchical HCIR CognitiveGraphs and 3D scene-graph topological representations.
"""

from __future__ import annotations

import logging

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)

from .types import (
    AI2ThorGoal,
    AI2ThorObjectMetadata,
    AI2ThorObservation,
)

logger = logging.getLogger(__name__)


class AI2ThorPerceptionAdapter:
    """
    Ingests AI2-THOR 3D object metadata and maintains an authoritative CognitiveGraph
    with 3D coordinates, visibility, and hierarchical receptacle containment edges.
    """

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()

    def ingest_observation(
        self, obs: AI2ThorObservation, goal: AI2ThorGoal | None = None
    ) -> CognitiveGraph:
        """Update CognitiveGraph with 3D poses, scene graph containment, and goal."""
        ap = obs.agent_pose.position

        # 1. Update Agent Node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="ai2thor_agent",
            entity_type="agent",
            properties={
                "x": ap.x,
                "y": ap.y,
                "z": ap.z,
                "rotation": obs.agent_pose.rotation,
                "horizon": obs.agent_pose.horizon,
                "held_object_id": obs.held_object_id,
                "reach_distance": 1.6,
                "step_count": obs.step_count,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("agent"):
            ex = self.graph.get_node("agent")
            if isinstance(ex, PhysicalEntityNode):
                ex.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # 2. Ingest Scene Objects
        for obj in obs.objects:
            node = PhysicalEntityNode(
                id=obj.objectId,
                entity_name=obj.objectType.lower(),
                entity_type="receptacle" if obj.isReceptacle else "object",
                properties={
                    "object_type": obj.objectType,
                    "x": obj.position.x,
                    "y": obj.position.y,
                    "z": obj.position.z,
                    "distance": obj.distance,
                    "is_pickupable": obj.isPickupable,
                    "is_receptacle": obj.isReceptacle,
                    "is_openable": obj.isOpenable,
                    "is_opened": obj.isOpened,
                    "is_toggleable": obj.isToggleable,
                    "is_toggled": obj.isToggled,
                    "parent_receptacles": list(obj.parentReceptacles),
                    "contained_objects": list(obj.receptacleObjectIds),
                },
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            if self.graph.has_node(obj.objectId):
                ex = self.graph.get_node(obj.objectId)
                if isinstance(ex, PhysicalEntityNode):
                    ex.properties.update(node.properties)
            else:
                self.graph.add_node(node)

            # Assert containment edges
            for parent_id in obj.parentReceptacles:
                edge_id = f"edge_inside_{obj.objectId}_{parent_id}"
                if not self.graph.has_edge(edge_id) and self.graph.has_node(parent_id):
                    self.graph.add_edge(
                        HCIREdge(
                            id=edge_id,
                            edge_type=HCIREdgeType.PART_OF,
                            sources=[obj.objectId],
                            targets=[parent_id],
                            properties={"relation": "inside"},
                        )
                    )

        # 3. Held Object Edge
        if obs.held_object_id and self.graph.has_node(obs.held_object_id):
            edge_id = f"edge_holds_agent_{obs.held_object_id}"
            if not self.graph.has_edge(edge_id):
                self.graph.add_edge(
                    HCIREdge(
                        id=edge_id,
                        edge_type=HCIREdgeType.DEPENDS_ON,
                        sources=["agent"],
                        targets=[obs.held_object_id],
                    )
                )

        # 4. Ingest Goal if provided
        if goal is not None:
            self.ingest_goal(goal)

        return self.graph

    def ingest_goal(self, goal: AI2ThorGoal) -> GoalNode:
        """Translate AI2ThorGoal specification into an active HCIR GoalNode."""
        target_conditions: list[str] = []

        if goal.target_receptacle_id and not goal.target_object_id:
            # Receptacle open or toggle goal
            rec_node = self.graph.get_node(goal.target_receptacle_id)
            rec_props = rec_node.properties if rec_node and hasattr(rec_node, "properties") else {}
            if rec_props.get("is_toggleable"):
                target_conditions.append(f"is_toggled({goal.target_receptacle_id})")
            else:
                target_conditions.append(f"is_opened({goal.target_receptacle_id})")
        elif goal.target_object_id and not goal.target_receptacle_id:
            # Pickup goal
            target_conditions.append(f"holds({goal.target_object_id})")
        elif goal.target_object_id and goal.target_receptacle_id:
            # Relocation goal
            target_conditions.append(
                f"inside({goal.target_object_id}, {goal.target_receptacle_id})"
            )

        goal_node = GoalNode(
            id="goal_active",
            description=f"AI2-THOR Goal: {target_conditions}",
            properties={
                "target_conditions": target_conditions,
                "target_object_id": goal.target_object_id,
                "target_receptacle_id": goal.target_receptacle_id,
            },
        )
        if self.graph.has_node("goal_active"):
            existing = self.graph.get_node("goal_active")
            if isinstance(existing, GoalNode):
                existing.properties.update(goal_node.properties)
                existing.description = goal_node.description
        else:
            self.graph.add_node(goal_node)

        return goal_node

    def find_object(self, obs: AI2ThorObservation, object_id: str) -> AI2ThorObjectMetadata | None:
        for obj in obs.objects:
            if obj.objectId == object_id:
                return obj
        return None
