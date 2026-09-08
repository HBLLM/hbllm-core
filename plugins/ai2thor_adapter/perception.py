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
    PhysicalEntityNode,
)

from .types import (
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

    def ingest_observation(self, obs: AI2ThorObservation) -> CognitiveGraph:
        """Update CognitiveGraph with 3D poses and scene graph containment."""
        ap = obs.agent_pose.position

        # Update Agent Node
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

        # Ingest Scene Objects
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

        return self.graph

    def find_object(self, obs: AI2ThorObservation, object_id: str) -> AI2ThorObjectMetadata | None:
        for obj in obs.objects:
            if obj.objectId == object_id:
                return obj
        return None
