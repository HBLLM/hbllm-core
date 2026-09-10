"""
NetHack Perception Adapter.

Projects NetHack glyph grid, ASCII characters, and bottom-line stats into
HBLLM CognitiveGraph and EpistemicSpatialGrid with fog-of-war tracking.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    PhysicalEntityNode,
)
from hbllm.perception import EpistemicSpatialGrid

from .types import (
    NetHackGlyph,
    NetHackObservation,
)

logger = logging.getLogger(__name__)


class NetHackPerceptionAdapter:
    """
    Translates NetHack partial dungeon observations into typed HCIR representations:
    - EpistemicSpatialGrid for fog of war, corridor pathfinding, and frontier exploration.
    - CognitiveGraph for player vitals, keys, doors, monsters, and staircases.
    """

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        self.grid = EpistemicSpatialGrid(self.graph)

    def ingest_observation(self, obs: NetHackObservation) -> CognitiveGraph:
        """Update CognitiveGraph and spatial memory from NetHack observation."""
        px, py = obs.player_pos

        # Update Agent Node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="rogue_player",
            entity_type="agent",
            properties={
                "x": px,
                "y": py,
                "hp": obs.stats.hp,
                "max_hp": obs.stats.max_hp,
                "dungeon_level": obs.stats.dungeon_level,
                "inventory": list(obs.inventory),
                "gold": obs.stats.gold,
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

        # Scan visible glyphs
        height = len(obs.glyphs)
        width = len(obs.glyphs[0])

        for y in range(height):
            for x in range(width):
                glyph = obs.glyphs[y][x]

                if glyph == NetHackGlyph.STAIRS_DOWN:
                    node = PhysicalEntityNode(
                        id=f"stairs_down_{x}_{y}",
                        entity_name="stairs_down",
                        entity_type="stairs",
                        properties={"x": x, "y": y, "coords": (x, y)},
                        entity_lifecycle=EntityLifecycle.TRACKED,
                    )
                    if not self.graph.has_node(node.id):
                        self.graph.add_node(node)

                elif glyph == NetHackGlyph.DOOR_CLOSED:
                    node = PhysicalEntityNode(
                        id=f"door_closed_{x}_{y}",
                        entity_name="door_closed",
                        entity_type="door",
                        properties={"x": x, "y": y, "coords": (x, y), "state": "closed"},
                        entity_lifecycle=EntityLifecycle.TRACKED,
                    )
                    if not self.graph.has_node(node.id):
                        self.graph.add_node(node)

                elif glyph == NetHackGlyph.MONSTER:
                    node = PhysicalEntityNode(
                        id=f"monster_{x}_{y}",
                        entity_name="monster",
                        entity_type="monster",
                        properties={"x": x, "y": y, "coords": (x, y)},
                        entity_lifecycle=EntityLifecycle.TRACKED,
                    )
                    if not self.graph.has_node(node.id):
                        self.graph.add_node(node)

                elif glyph == NetHackGlyph.KEY:
                    node = PhysicalEntityNode(
                        id=f"key_{x}_{y}",
                        entity_name="key",
                        entity_type="item",
                        properties={"x": x, "y": y, "coords": (x, y)},
                        entity_lifecycle=EntityLifecycle.TRACKED,
                    )
                    if not self.graph.has_node(node.id):
                        self.graph.add_node(node)

        return self.graph

    def ingest_goal(self, goal: Any | None, obs: NetHackObservation) -> GoalNode:
        """Create active GoalNode for dungeon navigation."""
        target_conds = ["descended"]
        if goal and getattr(goal, "target_action", "") == "pickup_key":
            target_conds = ["has(key)"]

        goal_node = GoalNode(
            id="goal_active",
            properties={"target_conditions": target_conds},
        )
        if self.graph.has_node("goal_active"):
            ex = self.graph.get_node("goal_active")
            if isinstance(ex, GoalNode):
                ex.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)
        return goal_node
