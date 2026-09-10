"""
Sokoban Perception Adapter.

Projects raw Sokoban grid observations into structured representations, identifying
agent position, active box states, target assignments, and static deadlock taboo cells.
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

from .types import SokobanObservation, SokobanTile

logger = logging.getLogger(__name__)


class SokobanPerceptionAdapter:
    """Extracts topological and semantic knowledge from Sokoban observations into CognitiveGraph."""

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        self.static_deadlock_cells: set[tuple[int, int]] = set()
        self._analyzed_layout = False

    def reset(self) -> None:
        """Reset internal perceptual state."""
        self.graph = CognitiveGraph()
        self.static_deadlock_cells.clear()
        self._analyzed_layout = False

    def process_observation(self, obs: SokobanObservation) -> dict[str, Any]:
        """Convert raw observation into perceptual feature map and taboo dead-end set."""
        if not self._analyzed_layout:
            self._analyze_static_deadlocks(obs)
            self._analyzed_layout = True

        boxes = set(obs.boxes)
        targets = set(obs.targets)

        solved_boxes = boxes.intersection(targets)
        unsolved_boxes = boxes.difference(targets)

        return {
            "player_pos": obs.player_pos,
            "boxes": boxes,
            "targets": targets,
            "solved_boxes": solved_boxes,
            "unsolved_boxes": unsolved_boxes,
            "deadlock_taboo_cells": set(self.static_deadlock_cells),
            "step_count": obs.step_count,
            "grid": obs.grid,
        }

    def ingest_observation(self, obs: SokobanObservation) -> CognitiveGraph:
        """Update CognitiveGraph with agent, boxes, targets, and topological state."""
        pr, pc = obs.player_pos
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="player",
            entity_type="agent",
            properties={"coords": (pr, pc), "x": pc, "y": pr, "step_count": obs.step_count},
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("agent"):
            ex = self.graph.get_node("agent")
            if isinstance(ex, PhysicalEntityNode):
                ex.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # Clear existing box and target nodes to reflect current state
        stale = [
            n.id
            for n in self.graph.all_nodes()
            if n.id.startswith("box_") or n.id.startswith("target_")
        ]
        for sid in stale:
            self.graph.remove_node(sid)

        for br, bc in obs.boxes:
            box_node = PhysicalEntityNode(
                id=f"box_{br}_{bc}",
                entity_name="box",
                entity_type="box",
                properties={"coords": (br, bc), "x": bc, "y": br},
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            self.graph.add_node(box_node)

        for tr, tc in obs.targets:
            tgt_node = PhysicalEntityNode(
                id=f"target_{tr}_{tc}",
                entity_name="target",
                entity_type="target",
                properties={"coords": (tr, tc), "x": tc, "y": tr},
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            self.graph.add_node(tgt_node)

        return self.graph

    def ingest_goal(self, obs: SokobanObservation | None = None) -> GoalNode:
        """Create active GoalNode for Sokoban requiring boxes on targets."""
        goal_node = GoalNode(
            id="goal_active",
            properties={"target_conditions": ["box_on_target"]},
        )
        if self.graph.has_node("goal_active"):
            ex = self.graph.get_node("goal_active")
            if isinstance(ex, GoalNode):
                ex.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)
        return goal_node

    def _analyze_static_deadlocks(self, obs: SokobanObservation) -> None:
        """
        Pre-compute static non-goal dead-end cells.
        Any empty cell that forms a corner with two adjacent walls and is NOT a target
        is a permanent static taboo cell for boxes.
        """
        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0
        targets = set(obs.targets)

        walls = set()
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(SokobanTile.WALL):
                    walls.add((r, c))

        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if (r, c) in walls or (r, c) in targets:
                    continue

                up_wall = (r - 1, c) in walls
                down_wall = (r + 1, c) in walls
                left_wall = (r, c - 1) in walls
                right_wall = (r, c + 1) in walls

                # Corner condition
                if (up_wall or down_wall) and (left_wall or right_wall):
                    self.static_deadlock_cells.add((r, c))
