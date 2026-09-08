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

from hbllm.hcir.graph import CognitiveGraph, EntityLifecycle, PhysicalEntityNode

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
        self.visited_positions: set[tuple[int, int]] = set()
        self.consecutive_rotations: int = 0
        self.active_frontier: tuple[int, int] | None = None

    def reset(self) -> None:
        """Reset internal exploration history, rotation counters, and frontier commitments."""
        self.visited_positions.clear()
        self.consecutive_rotations = 0
        self.active_frontier = None

    def find_target_entity(
        self, graph: CognitiveGraph, goal: BabyAIGoal
    ) -> PhysicalEntityNode | None:
        """Find the matching entity in the CognitiveGraph satisfying the goal criteria."""
        candidates: list[PhysicalEntityNode] = []
        for node in graph.all_nodes():
            if (
                not isinstance(node, PhysicalEntityNode)
                or node.entity_type == "agent"
                or node.entity_lifecycle == EntityLifecycle.FORGOTTEN
            ):
                continue

            if goal.target_id and node.id != goal.target_id:
                continue

            # If goal is to open a door, ignore doors that are already open
            if (
                goal.action == "open"
                and node.properties.get("is_door")
                and node.properties.get("state") == "open"
            ):
                continue

            entity_type = node.entity_type
            color = node.properties.get("color")

            if goal.matches_attributes(entity_type, color, entity_id=node.id):
                candidates.append(node)

        if not candidates:
            return None

        # Sort by distance to agent to prioritize the closest matching target
        agent_pos, _, _ = self.get_agent_state(graph)
        candidates.sort(
            key=lambda n: (
                abs(n.properties.get("coords", (0, 0))[0] - agent_pos[0])
                + abs(n.properties.get("coords", (0, 0))[1] - agent_pos[1])
            )
        )
        return candidates[0]

    def find_closed_door(
        self, graph: CognitiveGraph, only_accessible: bool = True
    ) -> PhysicalEntityNode | None:
        """Find any known closed door to explore adjacent rooms, prioritizing accessible doors."""
        unlocked: list[PhysicalEntityNode] = []
        locked_with_key: list[PhysicalEntityNode] = []
        locked_other: list[PhysicalEntityNode] = []

        agent_pos, agent_dir, carrying = self.get_agent_state(graph)
        car_key = carrying.get("color") if carrying and carrying.get("type") == "key" else None

        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                and node.properties.get("is_door")
                and node.properties.get("state") != "open"
            ):
                door_state = node.properties.get("state")
                door_col = node.properties.get("color")
                if door_state == "closed":
                    unlocked.append(node)
                elif door_state == "locked":
                    has_key = car_key == door_col or door_col is None
                    key_visible = (
                        self.find_target_entity(
                            graph,
                            BabyAIGoal(
                                action="pickup",
                                target_type="key",
                                target_color=door_col,
                            ),
                        )
                        is not None
                    )
                    if has_key or key_visible:
                        locked_with_key.append(node)
                    else:
                        locked_other.append(node)

        # Sort by distance to agent
        def dist_to_agent(n: PhysicalEntityNode) -> int:
            c = n.properties.get("coords", (0, 0))
            return abs(c[0] - agent_pos[0]) + abs(c[1] - agent_pos[1])

        unlocked.sort(key=dist_to_agent)
        locked_with_key.sort(key=dist_to_agent)
        locked_other.sort(key=dist_to_agent)

        # Priority 1: Unlocked doors (can explore immediately)
        if unlocked:
            return unlocked[0]

        # Priority 2: Locked doors where we have or see the matching key
        if locked_with_key:
            return locked_with_key[0]

        # Priority 3: Other locked doors (only if non-accessible doors allowed)
        if not only_accessible and locked_other:
            return locked_other[0]

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
        allow_walk_onto_target: bool = False,
    ) -> list[MiniGridAction] | None:
        """BFS path planner from (start_pos, start_dir) to facing or stepping onto target_pos."""
        is_waypoint = allow_walk_onto_target or (
            terminal_action is None and target_entity_type in ("floor", "waypoint", "door")
        )

        def is_at_target(x: int, y: int, d: int) -> bool:
            if is_waypoint:
                return (x, y) == target_pos
            fwd = DIR_TO_VEC[MiniGridDirection(d)]
            front = (x + fwd[0], y + fwd[1])
            return front == target_pos

        if is_at_target(start_pos[0], start_pos[1], start_dir):
            final_actions: list[MiniGridAction] = []
            if terminal_action is not None:
                final_actions.append(terminal_action)
            final_actions.append(MiniGridAction.DONE)
            return final_actions

        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode) and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        walls: set[tuple[int, int]] = set()
        closed_doors: set[tuple[int, int]] = set()

        seen_cells: set[tuple[int, int]] = set()
        agent_node = None
        for node in graph.all_nodes():
            if isinstance(node, PhysicalEntityNode) and node.entity_type == "agent":
                agent_node = node
                break
        if agent_node:
            seen_cells = set(agent_node.properties.get("seen_cells", []))
            seen_cells.update(agent_node.properties.get("visited_cells", []))

        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
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

        max_depth = max(100, (max_x + max_y) * 4)
        final_actions = None

        while queue:
            (curr_x, curr_y, curr_d, opened_doors), actions = queue.popleft()

            if is_at_target(curr_x, curr_y, curr_d):
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
                if terminal_action != MiniGridAction.TOGGLE or front_pos == target_pos:
                    new_opened = opened_doors | {front_pos}
                    toggle_state = (curr_x, curr_y, curr_d, new_opened)
                    if toggle_state not in visited:
                        visited.add(toggle_state)
                        queue.append((toggle_state, actions + [MiniGridAction.TOGGLE]))

            # Successor 4: MOVE FORWARD
            if 0 <= front_pos[0] < max_x and 0 <= front_pos[1] < max_y:
                is_closed_door = front_pos in closed_doors and front_pos not in opened_doors
                is_wall = front_pos in walls
                is_solid_target = (
                    front_pos == target_pos and target_entity_type != "door" and not is_waypoint
                )

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

    def find_fixed_entity(
        self, graph: CognitiveGraph, goal: BabyAIGoal
    ) -> PhysicalEntityNode | None:
        """Find the matching fixed landmark entity in the CognitiveGraph for PutNext."""
        for node in graph.all_nodes():
            if (
                not isinstance(node, PhysicalEntityNode)
                or node.entity_type == "agent"
                or node.entity_lifecycle == EntityLifecycle.FORGOTTEN
            ):
                continue
            entity_type = node.entity_type
            color = node.properties.get("color")
            if goal.matches_fixed_attributes(entity_type, color):
                return node
        return None

    def is_subgoal_satisfied(self, graph: CognitiveGraph, subgoal: BabyAIGoal) -> bool:
        """Check if a subgoal's target condition is currently satisfied in the graph."""
        agent_pos, agent_dir, carrying = self.get_agent_state(graph)
        target = self.find_target_entity(graph, subgoal)

        if subgoal.action == "pickup":
            if carrying and subgoal.matches_attributes(
                carrying.get("type", ""), carrying.get("color")
            ):
                return True
            return False

        if subgoal.action == "open":
            for node in graph.all_nodes():
                if (
                    isinstance(node, PhysicalEntityNode)
                    and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                    and node.properties.get("is_door")
                ):
                    if subgoal.target_id and node.id != subgoal.target_id:
                        continue
                    if subgoal.matches_attributes(
                        node.entity_type, node.properties.get("color"), entity_id=node.id
                    ):
                        if node.properties.get("state") == "open":
                            return True
            return False

        if subgoal.action == "go_to":
            if target:
                t_pos = target.properties.get("coords")
                fwd = DIR_TO_VEC[MiniGridDirection(agent_dir)]
                front = (agent_pos[0] + fwd[0], agent_pos[1] + fwd[1])
                return front == t_pos
            return False

        if subgoal.action == "put_next":
            fixed = self.find_fixed_entity(graph, subgoal)
            if target and fixed:
                tp = target.properties.get("coords")
                fp = fixed.properties.get("coords")
                if tp and fp:
                    return abs(tp[0] - fp[0]) + abs(tp[1] - fp[1]) == 1
            return False

        return False

    def find_unexplored_frontier(
        self, graph: CognitiveGraph
    ) -> tuple[tuple[int, int], PhysicalEntityNode | None] | None:
        """Find an untraversed open door or nearest passable cell bordering unobserved space."""
        agent_pos, agent_dir, _ = self.get_agent_state(graph)
        agent_node = None
        for node in graph.all_nodes():
            if isinstance(node, PhysicalEntityNode) and node.entity_type == "agent":
                agent_node = node
                break

        visited = set(self.visited_positions)
        visited.add(agent_pos)
        seen = set()
        if agent_node:
            visited.update(agent_node.properties.get("visited_cells", []))
            seen.update(agent_node.properties.get("seen_cells", []))

        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode) and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        walls: set[tuple[int, int]] = set()
        closed_doors: set[tuple[int, int]] = set()
        open_doors: dict[tuple[int, int], PhysicalEntityNode] = {}

        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                and node.entity_type != "agent"
            ):
                c = node.properties.get("coords")
                if not c:
                    continue
                if node.properties.get("is_door"):
                    if node.properties.get("state") == "open":
                        open_doors[c] = node
                    else:
                        closed_doors.add(c)
                elif not node.properties.get("passable", False):
                    walls.add(c)

        # Sort open doors by proximity to agent
        sorted_doors = sorted(
            open_doors.items(),
            key=lambda item: abs(item[0][0] - agent_pos[0]) + abs(item[0][1] - agent_pos[1]),
        )

        # Priority 1: Open doors leading into unvisited cells (closest first)
        for dc, door_node in sorted_doors:
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nc = (dc[0] + ddx, dc[1] + ddy)
                if (
                    0 < nc[0] < max_x - 1
                    and 0 < nc[1] < max_y - 1
                    and nc not in visited
                    and nc not in walls
                    and nc not in closed_doors
                ):
                    return (nc, door_node)

        # Priority 2: Open door cells themselves that haven't been stepped on
        for dc, door_node in sorted_doors:
            if dc not in visited:
                return (dc, door_node)

        # Priority 3: BFS for nearest passable cell bordering unseen space
        queue = collections.deque([agent_pos])
        bfs_visited = {agent_pos}
        while queue:
            cx, cy = queue.popleft()
            borders_unseen = False
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    0 < nx < max_x - 1
                    and 0 < ny < max_y - 1
                    and (nx, ny) not in walls
                    and (nx, ny) not in seen
                ):
                    borders_unseen = True
                    break

            if borders_unseen and (cx, cy) != agent_pos:
                return ((cx, cy), None)

            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    0 < nx < max_x - 1
                    and 0 < ny < max_y - 1
                    and (nx, ny) not in bfs_visited
                    and (nx, ny) not in walls
                    and (nx, ny) not in closed_doors
                ):
                    bfs_visited.add((nx, ny))
                    queue.append((nx, ny))

        return None

    def _plan_explore_frontier(self, graph: CognitiveGraph) -> list[MiniGridAction] | None:
        """Plan a path to the nearest open portal or frontier cell with target commitment."""
        agent_pos, agent_dir, _ = self.get_agent_state(graph)

        # If we have an active frontier target, keep pursuing it until reached or visited
        if self.active_frontier is not None:
            visited_set = set(self.visited_positions)
            agent_node = None
            for node in graph.all_nodes():
                if isinstance(node, PhysicalEntityNode) and node.entity_type == "agent":
                    agent_node = node
                    break
            if agent_node:
                visited_set.update(agent_node.properties.get("visited_cells", []))

            if agent_pos == self.active_frontier or self.active_frontier in visited_set:
                self.active_frontier = None
            else:
                traj = self._plan_direct_path(
                    graph=graph,
                    start_pos=agent_pos,
                    start_dir=agent_dir,
                    target_pos=self.active_frontier,
                    target_entity_type="waypoint",
                    target_entity_id="",
                    terminal_action=None,
                    allow_walk_onto_target=True,
                )
                if traj and traj != [MiniGridAction.DONE]:
                    return traj
                self.active_frontier = None

        frontier = self.find_unexplored_frontier(graph)
        if frontier is None:
            return None
        target_pos, target_node = frontier
        target_type = target_node.entity_type if target_node else "floor"
        target_id = target_node.id if target_node else ""
        traj = self._plan_direct_path(
            graph=graph,
            start_pos=agent_pos,
            start_dir=agent_dir,
            target_pos=target_pos,
            target_entity_type=target_type,
            target_entity_id=target_id,
            terminal_action=None,
            allow_walk_onto_target=True,
        )
        if traj and traj != [MiniGridAction.DONE]:
            self.active_frontier = target_pos
            return traj
        return None

    def _find_blocking_obstacle(
        self,
        graph: CognitiveGraph,
        target_pos: tuple[int, int],
        carrying_key: str | None = None,
    ) -> PhysicalEntityNode | None:
        """Find any pickupable obstacle directly adjacent to target_pos or along the path to it."""
        # 1. First check immediate 1-step adjacency to target_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            check_pos = (target_pos[0] + dx, target_pos[1] + dy)
            for node in graph.all_nodes():
                if (
                    isinstance(node, PhysicalEntityNode)
                    and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                    and node.entity_type in ("ball", "box", "key")
                    and node.properties.get("coords") == check_pos
                    and node.properties.get("pickupable", False)
                ):
                    return node

        # 2. Path-level relaxed BFS
        agent_pos, agent_dir, _ = self.get_agent_state(graph)
        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode) and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        walls: set[tuple[int, int]] = set()
        closed_doors: set[tuple[int, int]] = set()
        movable_obstacles: dict[tuple[int, int], PhysicalEntityNode] = {}

        seen_cells: set[tuple[int, int]] = set()
        agent_node = None
        for node in graph.all_nodes():
            if isinstance(node, PhysicalEntityNode) and node.entity_type == "agent":
                agent_node = node
                break
        if agent_node:
            seen_cells = set(agent_node.properties.get("seen_cells", []))
            seen_cells.update(agent_node.properties.get("visited_cells", []))

        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                and node.entity_type != "agent"
            ):
                c = node.properties.get("coords")
                if not c:
                    continue
                if node.properties.get("is_door"):
                    if node.properties.get("state") == "locked":
                        if carrying_key is not None and (
                            carrying_key == node.properties.get("color")
                            or node.properties.get("color") is None
                        ):
                            closed_doors.add(c)
                        else:
                            walls.add(c)
                    elif node.properties.get("state") == "closed":
                        closed_doors.add(c)
                elif node.properties.get("pickupable", False) and node.entity_type in (
                    "ball",
                    "box",
                    "key",
                ):
                    movable_obstacles[c] = node
                elif not node.properties.get("passable", False):
                    walls.add(c)

        start_state = (agent_pos[0], agent_pos[1], agent_dir, frozenset())
        queue = collections.deque([(start_state, [])])
        visited = {start_state}

        def is_facing_or_at(x: int, y: int, d: int) -> bool:
            if (x, y) == target_pos:
                return True
            fwd = DIR_TO_VEC[MiniGridDirection(d)]
            return (x + fwd[0], y + fwd[1]) == target_pos

        max_depth = max(100, (max_x + max_y) * 4)
        while queue:
            (cx, cy, cd, op_doors), path = queue.popleft()
            if is_facing_or_at(cx, cy, cd):
                for step_pos in path:
                    if step_pos in movable_obstacles:
                        return movable_obstacles[step_pos]
                break

            if len(path) > max_depth:
                continue

            for nd in [(cd - 1) % 4, (cd + 1) % 4]:
                st = (cx, cy, nd, op_doors)
                if st not in visited:
                    visited.add(st)
                    queue.append((st, path))

            fwd = DIR_TO_VEC[MiniGridDirection(cd)]
            front_pos = (cx + fwd[0], cy + fwd[1])
            if 0 <= front_pos[0] < max_x and 0 <= front_pos[1] < max_y:
                is_cd = front_pos in closed_doors and front_pos not in op_doors
                if is_cd:
                    st = (cx, cy, cd, op_doors | {front_pos})
                    if st not in visited:
                        visited.add(st)
                        queue.append((st, path))
                elif front_pos not in walls:
                    st = (front_pos[0], front_pos[1], cd, op_doors)
                    if st not in visited:
                        visited.add(st)
                        queue.append((st, path + [front_pos]))

        return None

    def _plan_unblock_obstacle(
        self,
        graph: CognitiveGraph,
        blocker: PhysicalEntityNode,
        critical_pos: tuple[int, int],
    ) -> list[MiniGridAction] | None:
        """Plan a sequence of actions to pick up a blocking obstacle and relocate it to a free cell."""
        agent_pos, agent_dir, carrying = self.get_agent_state(graph)
        blocker_pos = blocker.properties.get("coords")
        if not blocker_pos:
            return None

        prefix_actions: list[MiniGridAction] = []
        curr_pos = agent_pos
        curr_dir = agent_dir

        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode) and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        occupied: set[tuple[int, int]] = set()
        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                and node.entity_type != "agent"
            ):
                c = node.properties.get("coords")
                if c:
                    occupied.add(c)

        # Phase 1: If already carrying something, drop it on an adjacent empty tile first
        if carrying is not None:
            drop_candidates = [
                (curr_pos[0] + dx, curr_pos[1] + dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 < curr_pos[0] + dx < max_x - 1
                and 0 < curr_pos[1] + dy < max_y - 1
                and (curr_pos[0] + dx, curr_pos[1] + dy) not in occupied
                and (curr_pos[0] + dx, curr_pos[1] + dy) != critical_pos
                and (curr_pos[0] + dx, curr_pos[1] + dy) != blocker_pos
            ]
            if not drop_candidates:
                return None
            drop_cell = drop_candidates[0]
            drop_traj = self._plan_direct_path(
                graph=graph,
                start_pos=curr_pos,
                start_dir=curr_dir,
                target_pos=drop_cell,
                target_entity_type="floor",
                target_entity_id="",
                terminal_action=MiniGridAction.DROP,
            )
            if drop_traj is None:
                return None
            prefix_actions = [a for a in drop_traj if a != MiniGridAction.DONE]
            curr_pos, curr_dir = self._simulate_pose(curr_pos, curr_dir, prefix_actions)

        # Phase 2: Pick up the blocker
        unblock_pickup = self._plan_direct_path(
            graph=graph,
            start_pos=curr_pos,
            start_dir=curr_dir,
            target_pos=blocker_pos,
            target_entity_type=blocker.entity_type,
            target_entity_id=blocker.id,
            terminal_action=MiniGridAction.PICKUP,
        )
        if unblock_pickup is None:
            return None

        u_acts = [a for a in unblock_pickup if a != MiniGridAction.DONE]
        p_end, d_end = self._simulate_pose(curr_pos, curr_dir, u_acts)

        # Phase 3: Relocate the blocker away from critical_pos and blocker_pos
        occupied.discard(blocker_pos)
        forbidden_cells: set[tuple[int, int]] = {critical_pos, blocker_pos}
        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
            ):
                if node.properties.get("is_door") and node.properties.get("state") != "open":
                    dc = node.properties.get("coords")
                    if dc:
                        forbidden_cells.add(dc)
                        for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            forbidden_cells.add((dc[0] + ddx, dc[1] + ddy))
                if node.entity_type == "key":
                    kc = node.properties.get("coords")
                    if kc:
                        forbidden_cells.add(kc)

        relocate_candidates = [
            (p_end[0] + dx, p_end[1] + dy)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if 0 < p_end[0] + dx < max_x - 1
            and 0 < p_end[1] + dy < max_y - 1
            and (p_end[0] + dx, p_end[1] + dy) not in occupied
            and (p_end[0] + dx, p_end[1] + dy) not in forbidden_cells
        ]

        for rc in relocate_candidates:
            drop_traj = self._plan_direct_path(
                graph=graph,
                start_pos=p_end,
                start_dir=d_end,
                target_pos=rc,
                target_entity_type="floor",
                target_entity_id="",
                terminal_action=MiniGridAction.DROP,
                ignored_entity_ids={blocker.id},
            )
            if drop_traj is not None:
                return prefix_actions + u_acts + drop_traj

        return None

    def _plan_put_next(
        self, graph: CognitiveGraph, goal: BabyAIGoal
    ) -> list[MiniGridAction] | None:
        """Plan trajectory to pick up target object and drop it adjacent to fixed landmark."""
        agent_pos, agent_dir, carrying = self.get_agent_state(graph)
        target_node = self.find_target_entity(graph, goal)
        fixed_node = self.find_fixed_entity(graph, goal)

        if fixed_node is None:
            return None
        fixed_pos = fixed_node.properties.get("coords")
        if not fixed_pos:
            return None

        # Phase 1: Ensure carrying target object
        is_carrying_target = carrying is not None and goal.matches_attributes(
            carrying.get("type", ""), carrying.get("color")
        )

        curr_pos = agent_pos
        curr_dir = agent_dir
        prefix_actions: list[MiniGridAction] = []

        if not is_carrying_target:
            if target_node is None:
                return [MiniGridAction.LEFT]
            target_pos = target_node.properties.get("coords")
            if not target_pos:
                return [MiniGridAction.LEFT]

            pickup_traj = self._plan_direct_path(
                graph=graph,
                start_pos=agent_pos,
                start_dir=agent_dir,
                target_pos=target_pos,
                target_entity_type=target_node.entity_type,
                target_entity_id=target_node.id,
                terminal_action=MiniGridAction.PICKUP,
                carrying_key=None,
            )
            if pickup_traj is None:
                return [MiniGridAction.LEFT]
            prefix_actions = [a for a in pickup_traj if a != MiniGridAction.DONE]
            curr_pos, curr_dir = self._simulate_pose(agent_pos, agent_dir, prefix_actions)

        # Phase 2: Identify candidate empty drop cells adjacent to fixed landmark
        fx, fy = fixed_pos
        candidate_neighbors = [
            (fx + 1, fy),
            (fx - 1, fy),
            (fx, fy + 1),
            (fx, fy - 1),
        ]

        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode) and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        occupied_coords: set[tuple[int, int]] = set()
        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                and node.entity_type != "agent"
            ):
                if target_node and node.id == target_node.id and prefix_actions:
                    continue
                c = node.properties.get("coords")
                if c:
                    occupied_coords.add(c)

        valid_drop_cells = [
            (nx, ny)
            for (nx, ny) in candidate_neighbors
            if 0 < nx < max_x - 1 and 0 < ny < max_y - 1 and (nx, ny) not in occupied_coords
        ]

        if not valid_drop_cells:
            return prefix_actions if prefix_actions else [MiniGridAction.LEFT]

        best_drop_traj: list[MiniGridAction] | None = None
        ignored_ids = {target_node.id} if target_node else set()

        for drop_cell in valid_drop_cells:
            traj = self._plan_direct_path(
                graph=graph,
                start_pos=curr_pos,
                start_dir=curr_dir,
                target_pos=drop_cell,
                target_entity_type="floor",
                target_entity_id="",
                terminal_action=MiniGridAction.DROP,
                ignored_entity_ids=ignored_ids,
            )
            if traj is not None:
                if best_drop_traj is None or len(traj) < len(best_drop_traj):
                    best_drop_traj = traj

        if best_drop_traj is not None:
            return prefix_actions + best_drop_traj

        return prefix_actions if prefix_actions else [MiniGridAction.LEFT]

    def _plan_drop_carried_obstacle(
        self, graph: CognitiveGraph, curr_pos: tuple[int, int], curr_dir: int
    ) -> list[MiniGridAction] | None:
        """Drop an unwanted carried obstacle onto an empty adjacent cell."""
        node_coords = [
            n.properties.get("coords", (0, 0))
            for n in graph.all_nodes()
            if isinstance(n, PhysicalEntityNode) and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
        ]
        max_x = max([self.width] + [c[0] + 2 for c in node_coords])
        max_y = max([self.height] + [c[1] + 2 for c in node_coords])

        occupied: set[tuple[int, int]] = set()
        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
                and node.entity_type != "agent"
            ):
                c = node.properties.get("coords")
                if c:
                    occupied.add(c)

        forbidden_cells: set[tuple[int, int]] = set()
        for node in graph.all_nodes():
            if (
                isinstance(node, PhysicalEntityNode)
                and node.entity_lifecycle != EntityLifecycle.FORGOTTEN
            ):
                if node.properties.get("is_door"):
                    dc = node.properties.get("coords")
                    if dc:
                        forbidden_cells.add(dc)
                        if node.properties.get("state") != "open":
                            for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                forbidden_cells.add((dc[0] + ddx, dc[1] + ddy))
                if node.entity_type == "key":
                    kc = node.properties.get("coords")
                    if kc:
                        forbidden_cells.add(kc)

        drop_candidates = [
            (curr_pos[0] + dx, curr_pos[1] + dy)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if 0 < curr_pos[0] + dx < max_x - 1
            and 0 < curr_pos[1] + dy < max_y - 1
            and (curr_pos[0] + dx, curr_pos[1] + dy) not in occupied
            and (curr_pos[0] + dx, curr_pos[1] + dy) not in forbidden_cells
        ]
        if not drop_candidates:
            # Fall back to any non-occupied cell if forbidden filter is too strict
            drop_candidates = [
                (curr_pos[0] + dx, curr_pos[1] + dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 < curr_pos[0] + dx < max_x - 1
                and 0 < curr_pos[1] + dy < max_y - 1
                and (curr_pos[0] + dx, curr_pos[1] + dy) not in occupied
            ]
        if not drop_candidates:
            return None

        # Prefer front cell if available so agent can drop without rotating
        fwd = DIR_TO_VEC[MiniGridDirection(curr_dir)]
        front_pos = (curr_pos[0] + fwd[0], curr_pos[1] + fwd[1])
        if front_pos in drop_candidates:
            drop_candidates.remove(front_pos)
            drop_candidates.insert(0, front_pos)

        for drop_cell in drop_candidates:
            traj = self._plan_direct_path(
                graph=graph,
                start_pos=curr_pos,
                start_dir=curr_dir,
                target_pos=drop_cell,
                target_entity_type="floor",
                target_entity_id="",
                terminal_action=MiniGridAction.DROP,
            )
            if traj is not None:
                return traj
        return None

    def plan_trajectory(
        self,
        graph: CognitiveGraph,
        goal: BabyAIGoal,
    ) -> list[MiniGridAction]:
        """Plan a complete discrete action sequence to fulfill the goal."""
        # Check Compound / Sequential Instructions (Tier 6)
        if goal.is_compound():
            while not goal.is_all_completed() and self.is_subgoal_satisfied(
                graph, goal.get_active_subgoal()
            ):
                goal.advance_subgoal()

            if goal.is_all_completed():
                return [MiniGridAction.DONE]

            active = goal.get_active_subgoal()
            return self.plan_trajectory(graph, active)

        # Check PutNext (Tier 4)
        if goal.action == "put_next":
            put_traj = self._plan_put_next(graph, goal)
            if put_traj:
                return put_traj

        agent_pos, agent_dir, carrying = self.get_agent_state(graph)
        self.visited_positions.add(agent_pos)

        # Hands-full safety check: If agent is carrying an unwanted obstacle,
        # drop it immediately on an adjacent free cell before continuing!
        if carrying is not None:
            is_goal_target = goal.matches_attributes(
                carrying.get("type", ""), carrying.get("color")
            )
            has_door = self.find_closed_door(graph) is not None
            is_needed_key = carrying.get("type") == "key" and has_door
            if not is_goal_target and not is_needed_key:
                drop_traj = self._plan_drop_carried_obstacle(graph, agent_pos, agent_dir)
                if drop_traj:
                    return drop_traj

        target_entity = self.find_target_entity(graph, goal)
        if target_entity is None:
            # 1. Check for an accessible closed door to explore
            closed_door = self.find_closed_door(graph, only_accessible=True)
            if closed_door:
                self.active_frontier = None
                explore_goal = BabyAIGoal(
                    action="open",
                    target_type="door",
                    target_color=closed_door.properties.get("color"),
                    target_id=closed_door.id,
                )
                door_traj = self.plan_trajectory(graph, explore_goal)
                if door_traj and door_traj != [MiniGridAction.DONE]:
                    # Only append FORWARD if door_traj actually toggled the door open
                    if len(door_traj) >= 2 and door_traj[-2] == MiniGridAction.TOGGLE:
                        return door_traj[:-1] + [MiniGridAction.FORWARD]
                    return door_traj

            # 2. Check for unexplored frontiers or open portals
            frontier_traj = self._plan_explore_frontier(graph)
            if frontier_traj:
                self.consecutive_rotations = 0
                return frontier_traj

            # 3. Fallback rotation sweep with stuck-prevention
            self.consecutive_rotations += 1
            if self.consecutive_rotations > 4:
                self.consecutive_rotations = 0
                fwd = DIR_TO_VEC[MiniGridDirection(agent_dir)]
                front = (agent_pos[0] + fwd[0], agent_pos[1] + fwd[1])
                occupied_or_walls = {
                    n.properties.get("coords")
                    for n in graph.all_nodes()
                    if isinstance(n, PhysicalEntityNode)
                    and n.entity_lifecycle != EntityLifecycle.FORGOTTEN
                    and (not n.properties.get("passable", False) or n.entity_type == "wall")
                }
                if front not in occupied_or_walls:
                    return [MiniGridAction.FORWARD]
            return [MiniGridAction.LEFT]

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
                                # If door path blocked from key pose, check unblocking
                                blocker = self._find_blocking_obstacle(
                                    graph, target_pos, carrying_key=door_col
                                )
                                if blocker is not None:
                                    unblock_traj = self._plan_unblock_obstacle(
                                        graph, blocker, critical_pos=target_pos
                                    )
                                    if unblock_traj:
                                        return unblock_traj
                                return key_traj
                            else:
                                # Key might be blocked
                                key_blocker = self._find_blocking_obstacle(graph, key_pos)
                                if key_blocker is not None:
                                    unblock_traj = self._plan_unblock_obstacle(
                                        graph, key_blocker, critical_pos=key_pos
                                    )
                                    if unblock_traj:
                                        return unblock_traj
                    # If key not yet observed in graph, explore frontiers to discover it!
                    frontier_traj = self._plan_explore_frontier(graph)
                    if frontier_traj:
                        return frontier_traj
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

        # If direct path failed or requires an extreme detour, check for Causal Obstacle Unblocking (Tier 5)
        m_dist = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
        if traj is None or len(traj) > max(10, m_dist * 3):
            blocker = self._find_blocking_obstacle(graph, target_pos, carrying_key=car_key_col)
            if blocker is not None:
                unblock_traj = self._plan_unblock_obstacle(graph, blocker, critical_pos=target_pos)
                if unblock_traj:
                    return unblock_traj

            # If path failed and no blocker, explore frontiers
            frontier_traj = self._plan_explore_frontier(graph)
            if frontier_traj:
                return frontier_traj

        return traj if traj is not None else [MiniGridAction.DONE]

    def plan_next_action(
        self,
        graph: CognitiveGraph,
        goal: BabyAIGoal,
    ) -> MiniGridAction:
        """Get the single immediate next action."""
        trajectory = self.plan_trajectory(graph, goal)
        return trajectory[0] if trajectory else MiniGridAction.DONE
