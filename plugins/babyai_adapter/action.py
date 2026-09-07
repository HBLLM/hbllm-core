"""
BabyAI Action Adapter — translates high-level HCIR goals and graph state into MiniGrid actions.

Performs topological/grid path planning (BFS/A*) over the CognitiveGraph's known spatial layout,
supporting:
- Tier 1: Single-room navigation, object pickup, obstacle avoidance
- Tier 2: Physical door state changes (TOGGLE), multi-room navigation through doors, and exploration
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

    def __init__(self, room_size: int | tuple[int, int] = 16) -> None:
        if isinstance(room_size, tuple):
            self.width, self.height = room_size
        else:
            self.width = self.height = room_size

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

    def find_closed_door(self, graph: CognitiveGraph) -> PhysicalEntityNode | None:
        """Find any known closed door to explore adjacent rooms."""
        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.properties.get("is_door")
                and node.properties.get("state") != "open"
            ):
                return node
        return None

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
            # If target is not in current room, check for a closed door to explore
            closed_door = self.find_closed_door(graph)
            if closed_door:
                explore_goal = BabyAIGoal(
                    action="open",
                    target_type="door",
                    target_color=closed_door.properties.get("color"),
                )
                door_traj = self.plan_trajectory(graph, explore_goal)
                if door_traj and door_traj != [MiniGridAction.DONE]:
                    # Open the door and take one step through to explore
                    return door_traj[:-1] + [MiniGridAction.FORWARD]

            # Otherwise, rotate to explore current room
            return [MiniGridAction.LEFT]

        agent_pos, agent_dir, carrying = self.get_agent_state(graph)
        target_pos = target_entity.properties.get("coords")
        if not target_pos:
            return [MiniGridAction.DONE]

        # Calculate bounding dimensions
        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode)
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        # Separate walls (permanent obstacles) and closed doors (traversable via TOGGLE)
        walls: set[tuple[int, int]] = set()
        closed_doors: set[tuple[int, int]] = set()

        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_type != "agent"
                and node.id != target_entity.id
            ):
                c = node.properties.get("coords")
                if not c:
                    continue
                if node.properties.get("is_door"):
                    if node.properties.get("state") != "open":
                        closed_doors.add(c)
                elif not node.properties.get("passable", False):
                    walls.add(c)

        # BFS state space: (curr_x, curr_y, curr_d, opened_doors: frozenset[tuple[int, int]])
        start_state = (agent_pos[0], agent_pos[1], agent_dir, frozenset())
        queue: collections.deque[
            tuple[tuple[int, int, int, frozenset[tuple[int, int]]], list[MiniGridAction]]
        ] = collections.deque([(start_state, [])])
        visited: set[tuple[int, int, int, frozenset[tuple[int, int]]]] = {start_state}

        def is_goal_satisfied(x: int, y: int, d: int) -> bool:
            fwd = DIR_TO_VEC[MiniGridDirection(d)]
            front = (x + fwd[0], y + fwd[1])
            return front == target_pos

        final_actions: list[MiniGridAction] | None = None

        while queue:
            (curr_x, curr_y, curr_d, opened_doors), actions = queue.popleft()

            if is_goal_satisfied(curr_x, curr_y, curr_d):
                final_actions = list(actions)
                break

            # Limit search depth
            if len(actions) > 80:
                continue

            # Successor 1: TURN LEFT
            left_d = (curr_d - 1) % 4
            left_state = (curr_x, curr_y, left_d, opened_doors)
            if left_state not in visited:
                visited.add(left_state)
                queue.append((left_state, actions + [MiniGridAction.LEFT]))

            # Successor 2: TURN RIGHT
            right_d = (curr_d + 1) % 4
            right_state = (curr_x, curr_y, right_d, opened_doors)
            if right_state not in visited:
                visited.add(right_state)
                queue.append((right_state, actions + [MiniGridAction.RIGHT]))

            fwd = DIR_TO_VEC[MiniGridDirection(curr_d)]
            front_pos = (curr_x + fwd[0], curr_y + fwd[1])

            # Successor 3: TOGGLE (if facing a closed door that hasn't been opened yet)
            if front_pos in closed_doors and front_pos not in opened_doors:
                new_opened = opened_doors | {front_pos}
                toggle_state = (curr_x, curr_y, curr_d, new_opened)
                if toggle_state not in visited:
                    visited.add(toggle_state)
                    queue.append((toggle_state, actions + [MiniGridAction.TOGGLE]))

            # Successor 4: MOVE FORWARD
            if 0 <= front_pos[0] < max_x and 0 <= front_pos[1] < max_y:
                is_closed_door = front_pos in closed_doors and front_pos not in opened_doors
                is_wall = front_pos in walls
                is_solid_target = front_pos == target_pos and target_entity.entity_type != "door"

                if not is_wall and not is_closed_door and not is_solid_target:
                    fwd_state = (front_pos[0], front_pos[1], curr_d, opened_doors)
                    if fwd_state not in visited:
                        visited.add(fwd_state)
                        queue.append((fwd_state, actions + [MiniGridAction.FORWARD]))

        if final_actions is None:
            return [MiniGridAction.DONE]

        # Append terminal interaction action
        if goal.action == "open":
            final_actions.append(MiniGridAction.TOGGLE)
        elif goal.action == "pickup":
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
