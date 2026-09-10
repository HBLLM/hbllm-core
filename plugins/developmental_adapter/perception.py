"""Developmental Perception Adapter for BabyWorld.

Translates raw multi-modal sensory observations into typed HCIR CognitiveGraphs
while enforcing strict scientific invariants against semantic leakage.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)

from .types import SensoryObservation

logger = logging.getLogger(__name__)


# Forbidden semantic keys that must NEVER be injected by perception
FORBIDDEN_SEMANTIC_KEYS = frozenset(
    {
        "concept",
        "category",
        "affordance",
        "affordances",
        "pushable",
        "rollable",
        "graspable",
        "openable",
        "is_container",
        "is_tool",
        "is_obstacle",
        "functional_role",
    }
)


class SemanticLeakageViolationError(AssertionError):
    """Raised when perception produces developer-supplied semantic labels or affordances."""


class DevelopmentalPerceptionAdapter:
    """Bridges BabyWorld SensoryObservations to HCIR CognitiveGraphs."""

    def __init__(self, neutral_id_prefix: str = "entity_") -> None:
        self.neutral_id_prefix = neutral_id_prefix
        self._entity_id_map: dict[str, str] = {}  # Maps raw percept_id to neutral ID
        self._unobserved_beliefs: dict[str, dict[str, Any]] = {}  # Object permanence memory

    def observe(self, obs: SensoryObservation) -> CognitiveGraph:
        """Convert multi-modal sensory observation into a canonical HCIR CognitiveGraph.

        Enforces:
        1. Neutral entity IDs (no semantic hints).
        2. Strictly physical properties (shape, extent, position, mass sensation).
        3. Object permanence tracking for occluded objects.
        4. Zero pre-baked affordances or semantic labels.
        """
        graph = CognitiveGraph()

        # 1. Agent Effector Node
        agent_node = PhysicalEntityNode(
            id="agent_effector",
            properties={
                "position": obs.proprioception.get("effector_position", (0.0, 0.0)),
                "is_agent": True,
                "touch_contact": obs.touch,
                "effort_expended": obs.proprioception.get("effort_expended", 0.0),
            },
        )
        graph.add_node(agent_node)

        currently_visible_ids: set[str] = set()

        # 2. Ingest Visible Entities
        for idx, percept in enumerate(obs.vision):
            raw_id = percept["percept_id"]
            if raw_id not in self._entity_id_map:
                neutral_id = f"{self.neutral_id_prefix}{len(self._entity_id_map) + 1:03d}"
                self._entity_id_map[raw_id] = neutral_id
            neutral_id = self._entity_id_map[raw_id]
            currently_visible_ids.add(raw_id)

            # Store in permanence tracking
            self._unobserved_beliefs[raw_id] = {
                "neutral_id": neutral_id,
                "last_position": percept["spatial_coordinates"],
                "shape": percept["shape"],
                "color": percept["color"],
                "mass_sensation": percept["mass_sensation"],
                "surface_friction": percept.get("surface_friction", 1.0),
                "last_seen_step": obs.step_index,
            }

            entity_node = PhysicalEntityNode(
                id=neutral_id,
                properties={
                    "shape": percept["shape"],
                    "color": percept["color"],
                    "size_extent": percept["size_extent"],
                    "spatial_coordinates": percept["spatial_coordinates"],
                    "velocity": percept["velocity"],
                    "mass_sensation": percept["mass_sensation"],
                    "surface_friction": percept.get("surface_friction", 1.0),
                    "is_held": percept["is_held"],
                    "is_observed": True,
                    "depth_distance": obs.depth.get(raw_id, 0.0),
                },
            )
            graph.add_node(entity_node)

            # Proximity relation to agent
            depth = obs.depth.get(raw_id, 99.0)
            if depth <= 0.6:
                graph.add_edge(
                    HCIREdge(
                        edge_type=HCIREdgeType.NEAR,
                        sources=["agent_effector"],
                        targets=[neutral_id],
                        weight=1.0,
                        properties={"depth": depth},
                    )
                )

        # 3. Object Permanence: Keep tracked beliefs for entities not currently in line-of-sight
        for raw_id in obs.occluded_entity_ids:
            if raw_id in self._unobserved_beliefs:
                belief = self._unobserved_beliefs[raw_id]
                neutral_id = belief["neutral_id"]
                occluded_node = PhysicalEntityNode(
                    id=neutral_id,
                    properties={
                        "shape": belief["shape"],
                        "color": belief["color"],
                        "last_known_position": belief["last_position"],
                        "mass_sensation": belief["mass_sensation"],
                        "surface_friction": belief.get("surface_friction", 1.0),
                        "is_observed": False,
                        "occluded": True,
                        "steps_since_seen": obs.step_index - belief["last_seen_step"],
                    },
                )
                graph.add_node(occluded_node)

        # 4. Enforce strict scientific invariant
        self.validate_perception_invariance(graph)

        return graph

    @staticmethod
    def validate_perception_invariance(graph: CognitiveGraph) -> None:
        """Validate that no semantic labels or affordances exist in perception nodes.

        Raises SemanticLeakageViolationError if any forbidden keys or values appear.
        """
        for node in graph.all_nodes():
            if node.id == "agent_effector":
                continue
            props = node.properties

            # Check property keys
            for key in props:
                lower_key = key.lower()
                for forbidden in FORBIDDEN_SEMANTIC_KEYS:
                    if forbidden in lower_key:
                        raise SemanticLeakageViolationError(
                            f"Semantic leakage detected: key '{key}' found in node '{node.id}'. "
                            f"Perception must not supply semantic labels or affordances!"
                        )

            # Check string property values for pre-baked labels
            for val in props.values():
                if isinstance(val, str):
                    lower_val = val.lower()
                    if lower_val in ("rollable", "pushable", "openable", "container", "tool"):
                        raise SemanticLeakageViolationError(
                            f"Semantic leakage detected: value '{val}' found in node '{node.id}'. "
                            f"Perception must not supply semantic concept names!"
                        )
