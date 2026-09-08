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

from hbllm.actions import CausalObstacleResolver, TopologicalFrontierNavigator
from hbllm.hcir.graph import CognitiveGraph, EntityLifecycle, PhysicalEntityNode
from hbllm.perception import EpistemicSpatialGrid

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
        self.frontier_navigator = TopologicalFrontierNavigator()
        self.obstacle_resolver = CausalObstacleResolver(
            grid=EpistemicSpatialGrid(default_bounds=(self.width, self.height))
        )

    @property
    def active_frontier(self) -> tuple[int, int] | None:
        """Active exploration frontier target coordinate, managed with hysteresis."""
        return self.frontier_navigator.active_frontier

    @active_frontier.setter
    def active_frontier(self, val: tuple[int, int] | None) -> None:
        self.frontier_navigator.active_frontier = val

    def reset(self) -> None:
        """Reset internal exploration history, rotation counters, and frontier commitments."""
        self.visited_positions.clear()
        self.consecutive_rotations = 0
        self.frontier_navigator.reset()

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

        combined_ignored = set(ignored_entity_ids or ())
        if target_entity_id:
            combined_ignored.add(target_entity_id)

        grid = EpistemicSpatialGrid(graph, default_bounds=(self.width, self.height))
        partition = grid.partition_cells(
            ignored_entity_ids=combined_ignored,
            carrying_key=carrying_key,
        )
        walls = partition.blocking_cells
        closed_doors = partition.closed_doors
        max_x, max_y = partition.max_x, partition.max_y

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
        agent_pos, _, _ = self.get_agent_state(graph)
        grid = EpistemicSpatialGrid(graph, default_bounds=(self.width, self.height))
        return grid.find_frontiers(agent_pos, visited_cells=self.visited_positions)

    def _plan_explore_frontier(self, graph: CognitiveGraph) -> list[MiniGridAction] | None:
        """Plan a path to the nearest open portal or frontier cell with target commitment."""
        agent_pos, agent_dir, _ = self.get_agent_state(graph)

        visited_set = set(self.visited_positions)
        agent_node = None
        for node in graph.all_nodes():
            if isinstance(node, PhysicalEntityNode) and node.entity_type == "agent":
                agent_node = node
                break
        if agent_node:
            visited_set.update(agent_node.properties.get("visited_cells", []))

        # If we have an active frontier target, keep pursuing it until reached or visited
        if self.frontier_navigator.is_valid(agent_pos, visited_set):
            target_pos = self.frontier_navigator.active_frontier
            assert target_pos is not None
            traj = self._plan_direct_path(
                graph=graph,
                start_pos=agent_pos,
                start_dir=agent_dir,
                target_pos=target_pos,
                target_entity_type="waypoint",
                target_entity_id="",
                terminal_action=None,
                allow_walk_onto_target=True,
            )
            if traj and traj != [MiniGridAction.DONE]:
                return traj
            self.frontier_navigator.invalidate()

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
            self.frontier_navigator.commit(target_pos)
            return traj
        return None

    def _find_blocking_obstacle(
        self,
        graph: CognitiveGraph,
        target_pos: tuple[int, int],
        carrying_key: str | None = None,
    ) -> PhysicalEntityNode | None:
        """Find any pickupable obstacle directly adjacent to target_pos or along the path to it."""
        agent_pos, agent_dir, _ = self.get_agent_state(graph)
        return self.obstacle_resolver.find_blocking_obstacle(
            graph=graph,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            target_pos=target_pos,
            carrying_key=carrying_key,
            dir_to_vec={d.value: v for d, v in DIR_TO_VEC.items()},
        )

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

        # Phase 1: If already carrying something, drop it on an adjacent empty tile first
        if carrying is not None:
            drop_candidates = self.obstacle_resolver.find_safe_drop_candidates(
                graph=graph,
                curr_pos=curr_pos,
                critical_positions={critical_pos, blocker_pos},
            )
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
        relocate_candidates = self.obstacle_resolver.find_safe_drop_candidates(
            graph=graph,
            curr_pos=p_end,
            critical_positions={critical_pos, blocker_pos},
        )

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

        grid = EpistemicSpatialGrid(graph, default_bounds=(self.width, self.height))
        max_x, max_y = grid.get_bounds()
        partition = grid.partition_cells(
            ignored_entity_ids={target_node.id} if (target_node and prefix_actions) else None
        )

        valid_drop_cells = [
            (nx, ny)
            for (nx, ny) in candidate_neighbors
            if 0 < nx < max_x - 1
            and 0 < ny < max_y - 1
            and (nx, ny) not in partition.occupied_coords
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
        drop_candidates = self.obstacle_resolver.find_safe_drop_candidates(
            graph=graph,
            curr_pos=curr_pos,
            critical_positions=set(),
            curr_dir=curr_dir,
            dir_to_vec={d.value: v for d, v in DIR_TO_VEC.items()},
        )
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
        if traj is None or self.obstacle_resolver.is_extreme_detour(len(traj), m_dist):
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
