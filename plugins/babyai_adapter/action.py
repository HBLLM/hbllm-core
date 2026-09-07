"""
BabyAI Action Adapter — translates high-level HCIR goals and graph state into MiniGrid actions.

Performs topological/grid path planning (BFS/A*) over the CognitiveGraph's known spatial layout,
supporting:
- Tier 1: Single-room navigation, object pickup, obstacle avoidance
- Tier 2: Physical door state changes (TOGGLE), multi-room navigation through doors, and exploration
- Tier 3: Key-door unlocking, prerequisite causal subgoaling, and distractor discrimination
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

    def _simulate_pose(
        self,
        start_pos: tuple[int, int],
        start_dir: int,
        actions: list[MiniGridAction],
    ) -> tuple[tuple[int, int], int]:
        """Compute the resulting (position, direction) after applying actions."""
        curr_x, curr_y = start_pos
        curr_d = start_dir
        for act in actions:
            if act == MiniGridAction.LEFT:
                curr_d = (curr_d - 1) % 4
            elif act == MiniGridAction.RIGHT:
                curr_d = (curr_d + 1) % 4
            elif act == MiniGridAction.FORWARD:
                fwd = DIR_TO_VEC[MiniGridDirection(curr_d)]
                curr_x += fwd[0]
                curr_y += fwd[1]
        return ((curr_x, curr_y), curr_d)

    def _plan_direct_path(
        self,
        graph: CognitiveGraph,
        start_pos: tuple[int, int],
        start_dir: int,
        target_pos: tuple[int, int],
        target_entity_type: str,
        target_entity_id: str,
        terminal_action: MiniGridAction | None = None,
        carrying_key: str | None = None,
        ignored_entity_ids: set[str] | None = None,
    ) -> list[MiniGridAction] | None:
        """BFS path planner from (start_pos, start_dir) to facing target_pos."""
        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode)
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        walls: set[tuple[int, int]] = set()
        closed_doors: set[tuple[int, int]] = set()

        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_type != "agent"
                and node.id != target_entity_id
                and (ignored_entity_ids is None or node.id not in ignored_entity_ids)
            ):
                c = node.properties.get("coords")
                if not c:
                    continue
                if node.properties.get("is_door"):
                    door_state = node.properties.get("state")
                    door_col = node.properties.get("color")
                    if door_state == "open":
                        pass  # open door is passable
                    elif door_state == "locked":
                        # Locked door can only be toggled if carrying matching key
                        if carrying_key is not None and (
                            carrying_key == door_col or door_col is None
                        ):
                            closed_doors.add(c)
                        else:
                            walls.add(c)
                    else:  # closed
                        closed_doors.add(c)
                elif not node.properties.get("passable", False):
                    walls.add(c)

        # BFS state: (curr_x, curr_y, curr_d, opened_doors: frozenset[tuple[int, int]])
        start_state = (start_pos[0], start_pos[1], start_dir, frozenset())
        queue: collections.deque[
            tuple[tuple[int, int, int, frozenset[tuple[int, int]]], list[MiniGridAction]]
        ] = collections.deque([(start_state, [])])
        visited: set[tuple[int, int, int, frozenset[tuple[int, int]]]] = {start_state}

        def is_facing_target(x: int, y: int, d: int) -> bool:
            fwd = DIR_TO_VEC[MiniGridDirection(d)]
            front = (x + fwd[0], y + fwd[1])
            return front == target_pos

        max_depth = max(100, (max_x + max_y) * 4)
        final_actions: list[MiniGridAction] | None = None

        while queue:
            (curr_x, curr_y, curr_d, opened_doors), actions = queue.popleft()

            if is_facing_target(curr_x, curr_y, curr_d):
                final_actions = list(actions)
                break

            if len(actions) > max_depth:
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

            # Successor 3: TOGGLE
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
                is_solid_target = front_pos == target_pos and target_entity_type != "door"

                if not is_wall and not is_closed_door and not is_solid_target:
                    fwd_state = (front_pos[0], front_pos[1], curr_d, opened_doors)
                    if fwd_state not in visited:
                        visited.add(fwd_state)
                        queue.append((fwd_state, actions + [MiniGridAction.FORWARD]))

        if final_actions is None:
            return None

        if terminal_action is not None:
            final_actions.append(terminal_action)
        final_actions.append(MiniGridAction.DONE)

        return final_actions

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

        # Check for Locked Door Prerequisite (Tier 3 Causal Subgoaling)
        if goal.action == "open" and target_entity.entity_type == "door":
            door_state = target_entity.properties.get("state")
            door_col = target_entity.properties.get("color")
            if door_state == "open":
                return [MiniGridAction.DONE]
            if door_state == "locked":
                # Check carrying
                has_matching_key = (
                    carrying is not None
                    and carrying.get("type") == "key"
                    and (carrying.get("color") == door_col or door_col is None)
                )
                if not has_matching_key:
                    # Formulate prerequisite subgoal: fetch key of matching color
                    key_goal = BabyAIGoal(
                        action="pickup",
                        target_type="key",
                        target_color=door_col,
                    )
                    key_node = self.find_target_entity(graph, key_goal)
                    if key_node is not None:
                        key_pos = key_node.properties.get("coords")
                        if key_pos:
                            key_traj = self._plan_direct_path(
                                graph=graph,
                                start_pos=agent_pos,
                                start_dir=agent_dir,
                                target_pos=key_pos,
                                target_entity_type="key",
                                target_entity_id=key_node.id,
                                terminal_action=MiniGridAction.PICKUP,
                                carrying_key=None,
                            )
                            if key_traj is not None:
                                key_actions = [a for a in key_traj if a != MiniGridAction.DONE]
                                end_pos, end_dir = self._simulate_pose(
                                    agent_pos, agent_dir, key_actions
                                )
                                # Plan trajectory from key pose to locked door
                                door_traj = self._plan_direct_path(
                                    graph=graph,
                                    start_pos=end_pos,
                                    start_dir=end_dir,
                                    target_pos=target_pos,
                                    target_entity_type="door",
                                    target_entity_id=target_entity.id,
                                    terminal_action=MiniGridAction.TOGGLE,
                                    carrying_key=door_col,
                                    ignored_entity_ids={key_node.id},
                                )
                                if door_traj is not None:
                                    return key_actions + door_traj
                                return key_traj
                    # If key not yet observed in graph, rotate to locate it
                    return [MiniGridAction.LEFT]

        # Standard direct path execution
        terminal_action = None
        if goal.action == "open":
            terminal_action = MiniGridAction.TOGGLE
        elif goal.action == "pickup":
            terminal_action = MiniGridAction.PICKUP

        car_key_col = carrying.get("color") if carrying and carrying.get("type") == "key" else None
        traj = self._plan_direct_path(
            graph=graph,
            start_pos=agent_pos,
            start_dir=agent_dir,
            target_pos=target_pos,
            target_entity_type=target_entity.entity_type,
            target_entity_id=target_entity.id,
            terminal_action=terminal_action,
            carrying_key=car_key_col,
        )
        return traj if traj is not None else [MiniGridAction.DONE]

    def plan_next_action(
        self,
        graph: CognitiveGraph,
        goal: BabyAIGoal,
    ) -> MiniGridAction:
        """Get the single immediate next action."""
        trajectory = self.plan_trajectory(graph, goal)
        return trajectory[0] if trajectory else MiniGridAction.DONE
