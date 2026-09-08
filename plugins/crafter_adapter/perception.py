"""
Crafter Perception Adapter.

Projects Crafter 2D semantic grid, vitals, and inventory into
HBLLM CognitiveGraph and EpistemicSpatialGrid.
"""

from __future__ import annotations

import logging

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    PhysicalEntityNode,
)
from hbllm.perception import EpistemicSpatialGrid

from .types import (
    CrafterObject,
    CrafterObservation,
)

logger = logging.getLogger(__name__)


class CrafterPerceptionAdapter:
    """
    Translates raw Crafter observations into typed HCIR representations:
    - EpistemicSpatialGrid for 2D spatial pathfinding and obstacle detection.
    - CognitiveGraph for vital tracking, inventory state, and resource entities.
    """

    def __init__(
        self,
        graph: CognitiveGraph | None = None,
        width: int = 64,
        height: int = 64,
    ) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        self.grid = EpistemicSpatialGrid(self.graph)
        self.width = width
        self.height = height

    def ingest_observation(self, obs: CrafterObservation) -> CognitiveGraph:
        """Update spatial grid and cognitive graph from observation."""
        px, py = obs.player_pos

        # Update Agent Node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="player",
            entity_type="agent",
            properties={
                "x": px,
                "y": py,
                "facing": obs.player_facing,
                "health": obs.vitals.health,
                "food": obs.vitals.food,
                "drink": obs.vitals.drink,
                "energy": obs.vitals.energy,
                "inventory": obs.inventory.to_dict(),
                "achievements": [a.value for a in obs.achievements],
                "step_count": obs.step_count,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("agent"):
            existing = self.graph.get_node("agent")
            if isinstance(existing, PhysicalEntityNode):
                existing.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # Ingest nearby objects into CognitiveGraph
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])
        search_radius = 16

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    obj_id = obs.semantic_grid[y][x]
                    if obj_id in (
                        CrafterObject.TREE,
                        CrafterObject.WATER,
                        CrafterObject.STONE,
                        CrafterObject.COAL,
                        CrafterObject.IRON,
                        CrafterObject.DIAMOND,
                        CrafterObject.CRAFTING_TABLE,
                        CrafterObject.FURNACE,
                        CrafterObject.COW,
                        CrafterObject.ZOMBIE,
                    ):
                        node_id = f"ent_{CrafterObject(obj_id).name.lower()}_{x}_{y}"
                        dist = abs(px - x) + abs(py - y)
                        node = PhysicalEntityNode(
                            id=node_id,
                            entity_name=CrafterObject(obj_id).name.lower(),
                            entity_type="resource" if obj_id < 14 else "entity",
                            properties={
                                "obj_type": obj_id,
                                "x": x,
                                "y": y,
                                "distance": dist,
                                "coords": (x, y),
                                "passable": False,
                            },
                            entity_lifecycle=EntityLifecycle.TRACKED,
                        )
                        if self.graph.has_node(node_id):
                            ex = self.graph.get_node(node_id)
                            if isinstance(ex, PhysicalEntityNode):
                                ex.properties.update(node.properties)
                        else:
                            self.graph.add_node(node)

        return self.graph

    def find_nearest_object(
        self, obs: CrafterObservation, target_type: CrafterObject
    ) -> tuple[int, int] | None:
        """Find the coordinates of the nearest instance of target_type."""
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])
        best_pos = None
        best_dist = float("inf")

        radius = 24
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    if obs.semantic_grid[y][x] == target_type:
                        dist = abs(px - x) + abs(py - y)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = (x, y)

        return best_pos
