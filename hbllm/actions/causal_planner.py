"""
Causal Planner — Domain-agnostic obstacle unblocking, prerequisite chaining, and frontier commitment.

Provides core primitives for embodied agents:
1. Obstacle Detection: Relaxed BFS finding movable entities obstructing access to a goal or portal.
2. Safe Relocation Planning: Identifying non-critical free cells (outside doorways and critical paths) for object offloading.
3. Frontier Commitment & Hysteresis: Preventing junction oscillation in closed-loop navigation.
"""

from __future__ import annotations

import collections
import logging

from hbllm.hcir.graph import CognitiveGraph, EntityLifecycle, PhysicalEntityNode
from hbllm.perception.spatial_grid import EpistemicSpatialGrid

logger = logging.getLogger(__name__)


class CausalObstacleResolver:
    """Detects physical obstructions along spatial trajectories and computes non-interfering relocation cells."""

    def __init__(self, grid: EpistemicSpatialGrid | None = None) -> None:
        self.grid = grid or EpistemicSpatialGrid()

    @staticmethod
    def is_extreme_detour(
        direct_path_len: int,
        manhattan_dist: int,
        multiplier: int = 3,
        min_threshold: int = 10,
    ) -> bool:
        """Evaluate if an open path is an extreme detour around an obstacle blocking direct access."""
        return direct_path_len > max(min_threshold, manhattan_dist * multiplier)

    def find_blocking_obstacle(
        self,
        graph: CognitiveGraph,
        agent_pos: tuple[int, int],
        agent_dir: int,
        target_pos: tuple[int, int],
        carrying_key: str | None = None,
        dir_to_vec: dict[int, tuple[int, int]] | None = None,
    ) -> PhysicalEntityNode | None:
        """Find any pickupable obstacle directly adjacent to target_pos or along the path to it."""
        vecs = dir_to_vec or {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        grid = EpistemicSpatialGrid(graph, default_bounds=(self.grid.width, self.grid.height))
        partition = grid.partition_cells(carrying_key=carrying_key)
        max_x, max_y = partition.max_x, partition.max_y
        max_depth = max(100, (max_x + max_y) * 4)

        start_state = (agent_pos[0], agent_pos[1], agent_dir, frozenset())

        def is_facing_or_at(x: int, y: int, d: int) -> bool:
            if (x, y) == target_pos:
                return True
            fwd = vecs[d % 4]
            return (x + fwd[0], y + fwd[1]) == target_pos

        # 1. Path-level obstacle detection using relaxed BFS (finds obstacle closest to agent)
        queue = collections.deque([(start_state, [])])
        visited = {start_state}
        while queue:
            (cx, cy, cd, op_doors), path = queue.popleft()
            if is_facing_or_at(cx, cy, cd):
                for step_pos in path:
                    if step_pos in partition.movable_obstacles:
                        return partition.movable_obstacles[step_pos]
                break

            if len(path) > max_depth:
                continue

            for nd in [(cd - 1) % 4, (cd + 1) % 4]:
                st = (cx, cy, nd, op_doors)
                if st not in visited:
                    visited.add(st)
                    queue.append((st, path))

            fwd = vecs[cd % 4]
            front_pos = (cx + fwd[0], cy + fwd[1])
            if 0 <= front_pos[0] < max_x and 0 <= front_pos[1] < max_y:
                is_cd = front_pos in partition.closed_doors and front_pos not in op_doors
                if is_cd:
                    st = (cx, cy, cd, op_doors | {front_pos})
                    if st not in visited:
                        visited.add(st)
                        queue.append((st, path))
                elif front_pos not in partition.walls:
                    st = (front_pos[0], front_pos[1], cd, op_doors)
                    if st not in visited:
                        visited.add(st)
                        queue.append((st, path + [front_pos]))

        # 2. Fallback: check immediate 1-step adjacency to target_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            check_pos = (target_pos[0] + dx, target_pos[1] + dy)
            if check_pos in partition.movable_obstacles:
                return partition.movable_obstacles[check_pos]

        return None

    def find_candidate_blocking_obstacles(
        self,
        graph: CognitiveGraph,
        agent_pos: tuple[int, int],
        agent_dir: int,
        target_pos: tuple[int, int],
        carrying_key: str | None = None,
        dir_to_vec: dict[int, tuple[int, int]] | None = None,
    ) -> list[PhysicalEntityNode]:
        """Find candidate blocking obstacles along paths to target, prioritized by directness and proximity."""
        vecs = dir_to_vec or {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        grid = EpistemicSpatialGrid(graph, default_bounds=(self.grid.width, self.grid.height))
        partition = grid.partition_cells(carrying_key=carrying_key)
        max_x, max_y = partition.max_x, partition.max_y
        max_depth = max(100, (max_x + max_y) * 4)

        start_state = (agent_pos[0], agent_pos[1], agent_dir, frozenset())

        def is_facing_or_at(x: int, y: int, d: int) -> bool:
            if (x, y) == target_pos:
                return True
            fwd = vecs[d % 4]
            return (x + fwd[0], y + fwd[1]) == target_pos

        candidate_obstacles: list[PhysicalEntityNode] = []
        seen_ids: set[str] = set()

        queue = collections.deque([(start_state, [])])
        visited = {start_state}
        best_path_len: int | None = None

        while queue:
            (cx, cy, cd, op_doors), path = queue.popleft()
            if is_facing_or_at(cx, cy, cd):
                obs_on_path = [
                    partition.movable_obstacles[step_pos]
                    for step_pos in path
                    if step_pos in partition.movable_obstacles
                ]
                if not obs_on_path:
                    # An open, obstacle-free path exists to the target! Target is not blocked!
                    return []
                if best_path_len is None:
                    best_path_len = len(path)
                first_obs = obs_on_path[0]
                if first_obs.id not in seen_ids:
                    candidate_obstacles.append(first_obs)
                    seen_ids.add(first_obs.id)
                if len(candidate_obstacles) >= 3 or (
                    best_path_len is not None and len(path) > best_path_len + 2
                ):
                    continue

            if best_path_len is not None and len(path) > best_path_len + 2:
                continue

            if len(path) > max_depth:
                continue

            for nd in [(cd - 1) % 4, (cd + 1) % 4]:
                st = (cx, cy, nd, op_doors)
                if st not in visited:
                    visited.add(st)
                    queue.append((st, path))

            fwd = vecs[cd % 4]
            front_pos = (cx + fwd[0], cy + fwd[1])
            if 0 <= front_pos[0] < max_x and 0 <= front_pos[1] < max_y:
                is_cd = front_pos in partition.closed_doors and front_pos not in op_doors
                if is_cd:
                    st = (cx, cy, cd, op_doors | {front_pos})
                    if st not in visited:
                        visited.add(st)
                        queue.append((st, path))
                elif front_pos not in partition.walls:
                    st = (front_pos[0], front_pos[1], cd, op_doors)
                    if st not in visited:
                        visited.add(st)
                        queue.append((st, path + [front_pos]))

        # Fallback: check immediate 1-step adjacency to target_pos
        if not candidate_obstacles:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (target_pos[0] + dx, target_pos[1] + dy)
                if check_pos in partition.movable_obstacles:
                    obs = partition.movable_obstacles[check_pos]
                    if obs.id not in seen_ids:
                        candidate_obstacles.append(obs)
                        seen_ids.add(obs.id)

        return candidate_obstacles

    def find_safe_drop_candidates(
        self,
        graph: CognitiveGraph,
        curr_pos: tuple[int, int],
        critical_positions: set[tuple[int, int]],
        curr_dir: int | None = None,
        dir_to_vec: dict[int, tuple[int, int]] | None = None,
        traversable_positions: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Identify candidate adjacent or reachable empty tiles for object relocation, strictly forbidding doorways."""
        grid = EpistemicSpatialGrid(graph, default_bounds=(self.grid.width, self.grid.height))
        max_x, max_y = grid.get_bounds()

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

        forbidden_cells: set[tuple[int, int]] = set(critical_positions)
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

        # 1. First check immediate 1-step adjacent cells
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
                and (curr_pos[0] + dx, curr_pos[1] + dy) not in critical_positions
            ]

        # 2. If tight pocket or corner (no adjacent candidates), expand via BFS to reachable empty tiles
        if not drop_candidates:
            trav = set(traversable_positions or ())
            queue: collections.deque[tuple[tuple[int, int], int]] = collections.deque(
                [(curr_pos, 0)]
            )
            visited: set[tuple[int, int]] = {curr_pos}
            bfs_candidates: list[tuple[int, int]] = []
            max_depth = max(8, max_x + max_y)

            while queue:
                pos, d = queue.popleft()
                if d > 0 and pos not in occupied and pos not in forbidden_cells:
                    bfs_candidates.append(pos)
                    if len(bfs_candidates) >= 5:
                        break

                if d >= max_depth:
                    continue

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    np = (pos[0] + dx, pos[1] + dy)
                    if 0 < np[0] < max_x - 1 and 0 < np[1] < max_y - 1 and np not in visited:
                        visited.add(np)
                        if np not in occupied or np in trav:
                            queue.append((np, d + 1))

            drop_candidates = bfs_candidates

        if curr_dir is not None and dir_to_vec is not None and drop_candidates:
            fwd = dir_to_vec[curr_dir % 4]
            front_pos = (curr_pos[0] + fwd[0], curr_pos[1] + fwd[1])
            if front_pos in drop_candidates:
                drop_candidates.remove(front_pos)
                drop_candidates.insert(0, front_pos)

        return drop_candidates


class TopologicalFrontierNavigator:
    """Manages active exploration commitments to prevent junction oscillations."""

    def __init__(self) -> None:
        self.active_frontier: tuple[int, int] | None = None

    def reset(self) -> None:
        """Clear active exploration commitments."""
        self.active_frontier = None

    def commit(self, frontier_pos: tuple[int, int]) -> None:
        """Commit to a target exploration waypoint."""
        self.active_frontier = frontier_pos

    def invalidate(self) -> None:
        """Explicitly invalidate current exploration target (e.g. upon opening a new door)."""
        self.active_frontier = None

    def is_valid(
        self,
        agent_pos: tuple[int, int],
        visited_cells: set[tuple[int, int]],
    ) -> bool:
        """Check if active frontier commitment remains valid."""
        if self.active_frontier is None:
            return False
        if agent_pos == self.active_frontier or self.active_frontier in visited_cells:
            self.active_frontier = None
            return False
        return True
