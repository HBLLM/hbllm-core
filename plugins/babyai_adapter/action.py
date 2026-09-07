"""
BabyAI Action Adapter — translates high-level HCIR goals and graph state into MiniGrid actions.

Performs topological/grid path planning (BFS/A*) over the CognitiveGraph's known spatial layout,
generating exact sequences of discrete MiniGrid actions (LEFT, RIGHT, FORWARD, PICKUP, DONE).
"""

from __future__ import annotations

import collections
import logging
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode

from .types import (
    DIR_TO_VEC,
    BabyAIGoal,
    MiniGridAction,
    MiniGridDirection,
)

logger = logging.getLogger(__name__)


class BabyAIActionAdapter:
    """Computes navigation paths and converts them into discrete MiniGrid actions."""

    def __init__(self, room_size: int = 8) -> None:
        self.room_size = room_size

    def find_target_entity(
        self, graph: CognitiveGraph, goal: BabyAIGoal
    ) -> PhysicalEntityNode | None:
        """Find the matching entity in the CognitiveGraph satisfying the goal criteria."""
        candidates: list[PhysicalEntityNode] = []
        for node in graph.all_nodes():
            if not isinstance(node, PhysicalEntityNode) or node.entity_type == "agent":
                continue

            entity_type = node.entity_type
            color = node.properties.get("color")

            if goal.matches_attributes(entity_type, color):
                candidates.append(node)

        if not candidates:
            return None

        # If multiple candidates, pick the first (or closest to agent)
        return candidates[0]

    def get_agent_state(
        self, graph: CognitiveGraph
    ) -> tuple[tuple[int, int], int, dict[str, Any] | None]:
        """Extract current agent position, direction, and carrying state from graph."""
        agent_node = None
        for node in graph.all_nodes():
            if isinstance(node, PhysicalEntityNode) and node.entity_type == "agent":
                agent_node = node
                break

        if agent_node is None:
            return ((1, 1), 0, None)

        coords = agent_node.properties.get("coords", (1, 1))
        direction = agent_node.properties.get("direction", 0)
        carrying = agent_node.properties.get("carrying")
        return (coords, direction, carrying)

    def plan_trajectory(
        self,
        graph: CognitiveGraph,
        goal: BabyAIGoal,
    ) -> list[MiniGridAction]:
        """Plan a complete discrete action sequence to fulfill the goal."""
        target_entity = self.find_target_entity(graph, goal)
        if target_entity is None:
            # Target not yet spotted in graph: rotate to explore
            return [MiniGridAction.LEFT, MiniGridAction.LEFT]

        agent_pos, agent_dir, carrying = self.get_agent_state(graph)
        target_pos = target_entity.properties.get("coords")
        if not target_pos:
            return [MiniGridAction.DONE]

        # Build obstacle set from known impassable entities in graph
        obstacles: set[tuple[int, int]] = set()
        # Outer walls
        for x in range(self.room_size):
            for y in range(self.room_size):
                if x == 0 or x == self.room_size - 1 or y == 0 or y == self.room_size - 1:
                    obstacles.add((x, y))

        # Graph entities that are not passable and not the target
        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_type != "agent"
                and node.id != target_entity.id
            ):
                if not node.properties.get("passable", False):
                    c = node.properties.get("coords")
                    if c:
                        obstacles.add(c)

        # BFS state space: (x, y, dir)
        start_state = (agent_pos[0], agent_pos[1], agent_dir)
        queue: collections.deque[tuple[tuple[int, int, int], list[MiniGridAction]]] = (
            collections.deque([(start_state, [])])
        )
        visited: set[tuple[int, int, int]] = {start_state}

        def is_goal_satisfied(x: int, y: int, d: int) -> bool:
            fwd = DIR_TO_VEC[MiniGridDirection(d)]
            front = (x + fwd[0], y + fwd[1])
            # For pickup or go_to, standing adjacent and facing the target is valid
            return front == target_pos

        final_actions: list[MiniGridAction] | None = None

        while queue:
            (curr_x, curr_y, curr_d), actions = queue.popleft()

            if is_goal_satisfied(curr_x, curr_y, curr_d):
                final_actions = list(actions)
                break

            # Limit search depth to prevent infinite loops
            if len(actions) > 50:
                continue

            # Successor 1: TURN LEFT
            left_d = (curr_d - 1) % 4
            left_state = (curr_x, curr_y, left_d)
            if left_state not in visited:
                visited.add(left_state)
                queue.append((left_state, actions + [MiniGridAction.LEFT]))

            # Successor 2: TURN RIGHT
            right_d = (curr_d + 1) % 4
            right_state = (curr_x, curr_y, right_d)
            if right_state not in visited:
                visited.add(right_state)
                queue.append((right_state, actions + [MiniGridAction.RIGHT]))

            # Successor 3: MOVE FORWARD
            fwd = DIR_TO_VEC[MiniGridDirection(curr_d)]
            next_x, next_y = curr_x + fwd[0], curr_y + fwd[1]
            if (next_x, next_y) not in obstacles and (next_x, next_y) != target_pos:
                fwd_state = (next_x, next_y, curr_d)
                if fwd_state not in visited:
                    visited.add(fwd_state)
                    queue.append((fwd_state, actions + [MiniGridAction.FORWARD]))

        if final_actions is None:
            # Fallback if no direct path found: try simple forward or done
            return [MiniGridAction.DONE]

        # Append terminal interaction action
        if goal.action == "pickup":
            final_actions.append(MiniGridAction.PICKUP)
        final_actions.append(MiniGridAction.DONE)

        return final_actions

    def plan_next_action(
        self,
        graph: CognitiveGraph,
        goal: BabyAIGoal,
    ) -> MiniGridAction:
        """Get the single immediate next action."""
        trajectory = self.plan_trajectory(graph, goal)
        return trajectory[0] if trajectory else MiniGridAction.DONE
