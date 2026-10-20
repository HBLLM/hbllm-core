"""Topological Graph & Geodesic Maze Induction Engine (Milestone 3 Phase 2).

Inductively discovers non-local topological geodesic mechanisms:
1. GridTopologyGraph: Graph abstraction over 2D grids computing shortest geodesic paths.
2. GeodesicPathOperator: Parameterized operator connecting landmark markers avoiding obstacles.
3. InductiveTopologicalGraphSynthesizer: Synthesizes graph operators from demonstration residual events.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, label

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.grid_operator import GridOperator, OperatorBinding

logger = logging.getLogger(__name__)


class GridTopologyGraph:
    """Computes shortest geodesic paths through 2D grid manifolds."""

    @staticmethod
    def find_all_shortest_paths(
        grid: np.ndarray,
        start_coord: tuple[int, int],
        goal_coord: tuple[int, int],
        traversable_colors: set[int],
        connectivity: str = "4_connected",
    ) -> list[list[tuple[int, int]]]:
        """Find all minimum-hop paths from start to goal through traversable cells."""
        H, W = grid.shape
        sr, sc = start_coord
        gr, gc = goal_coord

        if (sr, sc) == (gr, gc):
            return [[(sr, sc)]]

        offsets = (
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if connectivity == "4_connected"
            else [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        dist: dict[tuple[int, int], int] = {(sr, sc): 0}
        parents: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
        queue = collections.deque([(sr, sc)])

        found_goal = False
        goal_dist = float("inf")

        while queue:
            curr_r, curr_c = queue.popleft()
            d = dist[(curr_r, curr_c)]
            if d >= goal_dist:
                continue

            for dr, dc in offsets:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell_val = int(grid[nr, nc])
                    if (nr, nc) == (gr, gc) or cell_val in traversable_colors:
                        new_d = d + 1
                        if (nr, nc) not in dist:
                            dist[(nr, nc)] = new_d
                            parents[(nr, nc)].append((curr_r, curr_c))
                            queue.append((nr, nc))
                            if (nr, nc) == (gr, gc):
                                found_goal = True
                                goal_dist = new_d
                        elif dist[(nr, nc)] == new_d:
                            parents[(nr, nc)].append((curr_r, curr_c))

        if not found_goal:
            return []

        # Backtrack all shortest paths from goal to start
        paths: list[list[tuple[int, int]]] = []

        def backtrack(curr: tuple[int, int], current_path: list[tuple[int, int]]) -> None:
            if curr == (sr, sc):
                paths.append(list(reversed(current_path)))
                return
            for p in parents[curr]:
                backtrack(p, current_path + [p])

        backtrack((gr, gc), [(gr, gc)])
        return paths

    @staticmethod
    def shortest_path(
        grid: np.ndarray,
        start_coord: tuple[int, int],
        goal_coord: tuple[int, int],
        traversable_colors: set[int],
        connectivity: str = "4_connected",
    ) -> list[tuple[int, int]] | None:
        """Find a canonical shortest path from start to goal."""
        all_paths = GridTopologyGraph.find_all_shortest_paths(
            grid, start_coord, goal_coord, traversable_colors, connectivity
        )
        return all_paths[0] if all_paths else None

    @staticmethod
    def shortest_path_on_mask(
        passable_mask: np.ndarray,
        start_coord: tuple[int, int],
        goal_coord: tuple[int, int],
        connectivity: str = "4_connected",
    ) -> list[tuple[int, int]] | None:
        """Find minimum-hop shortest path from start to goal through a boolean traversable mask."""
        H, W = passable_mask.shape
        sr, sc = start_coord
        gr, gc = goal_coord
        if (sr, sc) == (gr, gc):
            return [(sr, sc)]
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return None

        offsets = (
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if connectivity == "4_connected"
            else [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        queue: collections.deque[tuple[int, int]] = collections.deque([(sr, sc)])
        visited: set[tuple[int, int]] = {(sr, sc)}
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        found = False
        while queue:
            curr_r, curr_c = queue.popleft()
            if (curr_r, curr_c) == (gr, gc):
                found = True
                break
            for dr, dc in offsets:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    if (nr, nc) == (gr, gc) or bool(passable_mask[nr, nc]):
                        visited.add((nr, nc))
                        parent[(nr, nc)] = (curr_r, curr_c)
                        queue.append((nr, nc))

        if not found:
            return None

        curr = (gr, gc)
        path = [curr]
        while curr != (sr, sc):
            curr = parent[curr]
            path.append(curr)
        path.reverse()
        return path

    @staticmethod
    def shortest_path_with_actions(
        passable_mask: np.ndarray,
        start_coord: tuple[int, int],
        goal_coord: tuple[int, int],
        actions_with_deltas: list[tuple[Any, int, int]],
        goal_radius: int = 0,
        body_offsets: Sequence[tuple[int, int]] | None = None,
        failed_transitions: set[tuple[tuple[int, int], Any]] | None = None,
    ) -> list[tuple[Any, tuple[int, int]]] | None:
        """Find minimum-hop shortest action sequence directly in action transition space.

        Returns list of (action, next_coord) pairs.
        """
        H, W = passable_mask.shape
        sr, sc = start_coord
        gr, gc = goal_coord

        def _at_goal(r: int, c: int) -> bool:
            return abs(r - gr) <= goal_radius and abs(c - gc) <= goal_radius

        def _cell_walkable(r: int, c: int) -> bool:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if body_offsets:
                for b_dr, b_dc in body_offsets:
                    br, bc = r + b_dr, c + b_dc
                    if not (0 <= br < H and 0 <= bc < W):
                        return False
                    if not passable_mask[br, bc] and not _at_goal(br, bc):
                        return False
                return True
            return bool(passable_mask[r, c] or _at_goal(r, c))

        if _at_goal(sr, sc):
            return []
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return None

        valid_actions = [(a, dr, dc) for a, dr, dc in actions_with_deltas if dr != 0 or dc != 0]
        if not valid_actions:
            return None

        queue: collections.deque[tuple[int, int]] = collections.deque([(sr, sc)])
        visited: set[tuple[int, int]] = {(sr, sc)}
        parent: dict[tuple[int, int], tuple[tuple[int, int], Any]] = {}

        reached_goal: tuple[int, int] | None = None
        while queue:
            curr_r, curr_c = queue.popleft()
            if _at_goal(curr_r, curr_c):
                reached_goal = (curr_r, curr_c)
                break

            for act, dr, dc in valid_actions:
                if failed_transitions and ((curr_r, curr_c), act) in failed_transitions:
                    continue
                nr, nc = curr_r + dr, curr_c + dc
                if (nr, nc) not in visited:
                    step_mag = max(abs(dr), abs(dc))
                    blocked = False
                    if step_mag > 1:
                        for s_i in range(1, step_mag + 1):
                            inter_r = curr_r + int(round(dr * s_i / step_mag))
                            inter_c = curr_c + int(round(dc * s_i / step_mag))
                            if not _cell_walkable(inter_r, inter_c):
                                blocked = True
                                break
                    elif not _cell_walkable(nr, nc):
                        blocked = True

                    if not blocked:
                        visited.add((nr, nc))
                        parent[(nr, nc)] = ((curr_r, curr_c), act)
                        queue.append((nr, nc))
                        if _at_goal(nr, nc):
                            reached_goal = (nr, nc)
                            break
            if reached_goal is not None:
                break

        if reached_goal is None:
            return None

        plan: list[tuple[Any, tuple[int, int]]] = []
        curr = reached_goal
        while curr != (sr, sc):
            prev_pos, act = parent[curr]
            plan.append((act, curr))
            curr = prev_pos
        plan.reverse()
        return plan


class GeodesicManifoldRouter:
    """Substrate routing engine computing deterministic action-space geodesics over traversable 2D manifolds."""

    @staticmethod
    def route_geodesic(
        curr_grid: np.ndarray,
        start_pos: tuple[int, int],
        goals: Sequence[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        grid_walls: set[tuple[int, int]],
        unknown_entity_cells: set[tuple[int, int]],
        oriented_threats: Sequence[Any],
        movable_actions: Sequence[tuple[Any, int, int]],
        bg: int,
        barrier_features: set[int],
        avatar_features: set[int],
        max_candidate_goals: int = 5,
        failed_transitions: set[tuple[tuple[int, int], Any]] | None = None,
    ) -> list[tuple[Any, tuple[int, int]]] | None:
        """Find deterministic action-space shortest path across the traversable manifold.

        1. Constructs boolean passable mask from static barriers, walls, and unknown obstacles.
        2. Projects dynamic threat trajectories into evasion hazard cones.
        3. Harmonizes multi-cell action strides.
        4. Dispatches action-transition BFS search to nearest reachable goals.
        """
        H, W = curr_grid.shape
        passable_mask = np.ones((H, W), dtype=bool)
        for br, bc in static_barriers:
            if 0 <= br < H and 0 <= bc < W:
                passable_mask[br, bc] = False
        for wr, wc in grid_walls:
            if 0 <= wr < H and 0 <= wc < W:
                passable_mask[wr, wc] = False
        for ur, uc in unknown_entity_cells:
            if 0 <= ur < H and 0 <= uc < W and (ur, uc) not in goals:
                passable_mask[ur, uc] = False

        hazard_cells: set[tuple[int, int]] = set()
        effective_traj_barriers = set(barrier_features)
        if not effective_traj_barriers:
            non_passable_feats = set(int(v) for v in np.unique(curr_grid[~passable_mask]))
            effective_traj_barriers.update(non_passable_feats - avatar_features - {bg})

        for t in oriented_threats:
            hazard_cells.add(t.pos)
            tr, tc = t.pos
            for dr_t in (-1, 0, 1):
                for dc_t in (-1, 0, 1):
                    rr_t, cc_t = tr + dr_t, tc + dc_t
                    if 0 <= rr_t < H and 0 <= cc_t < W:
                        if int(curr_grid[rr_t, cc_t]) == t.feature_id:
                            hazard_cells.add((rr_t, cc_t))

            if t.facing != (0, 0):
                gr, gc = getattr(t, "gaze_pos", (t.pos[0] + t.facing[0], t.pos[1] + t.facing[1]))
                d_r, d_c = t.facing
                curr_r, curr_c = gr, gc
                while 0 <= curr_r < H and 0 <= curr_c < W:
                    if (curr_r, curr_c) in static_barriers or (curr_r, curr_c) in grid_walls:
                        break
                    if int(curr_grid[curr_r, curr_c]) in effective_traj_barriers:
                        break
                    hazard_cells.add((curr_r, curr_c))
                    curr_r += d_r
                    curr_c += d_c

        for hz_r, hz_c in hazard_cells:
            if (
                0 <= hz_r < H
                and 0 <= hz_c < W
                and (hz_r, hz_c) != start_pos
                and (hz_r, hz_c) not in goals
            ):
                passable_mask[hz_r, hz_c] = False

        passable_mask[start_pos[0], start_pos[1]] = True

        sorted_goals = sorted(
            goals,
            key=lambda g: abs(g[0] - start_pos[0]) + abs(g[1] - start_pos[1]),
        )

        effective_actions = list(movable_actions)
        max_stride = max((max(abs(dr), abs(dc)) for _, dr, dc in effective_actions), default=1)
        if max_stride >= 2:
            harmonized_actions: list[tuple[Any, int, int]] = []
            for act, dr, dc in effective_actions:
                clean_dr, clean_dc = dr, dc
                if abs(clean_dc) >= 2 * abs(clean_dr):
                    clean_dr = 0
                elif abs(clean_dr) >= 2 * abs(clean_dc):
                    clean_dc = 0
                mag = max(abs(clean_dr), abs(clean_dc))
                if 0 < mag < max_stride:
                    scale = max_stride // mag if max_stride % mag == 0 else max_stride / mag
                    new_dr = int(round(clean_dr * scale))
                    new_dc = int(round(clean_dc * scale))
                    harmonized_actions.append((act, new_dr, new_dc))
                elif mag <= 1 and max_stride >= 2:
                    harmonized_actions.append((act, clean_dr * max_stride, clean_dc * max_stride))
                else:
                    harmonized_actions.append((act, clean_dr, clean_dc))
        effective_goal_radius = 0 if max_stride <= 1 else max(1, max_stride - 1)

        body_offsets: list[tuple[int, int]] | None = None
        if avatar_features:
            av_mask = np.isin(curr_grid, list(avatar_features))
            if np.any(av_mask):
                dilated = binary_dilation(av_mask, structure=np.ones((3, 3)))
                lbl, num_comp = label(dilated, structure=np.ones((3, 3)))
                best_comp: np.ndarray | None = None
                min_d = float("inf")
                for c_i in range(1, num_comp + 1):
                    c_pts = np.argwhere(av_mask & (lbl == c_i))
                    if len(c_pts) == 0:
                        continue
                    cent = np.mean(c_pts, axis=0)
                    d = (cent[0] - start_pos[0]) ** 2 + (cent[1] - start_pos[1]) ** 2
                    if d < min_d:
                        min_d = d
                        best_comp = c_pts
                if best_comp is not None and 1 < len(best_comp) < 300:
                    body_offsets = [
                        (int(r - start_pos[0]), int(c - start_pos[1])) for r, c in best_comp
                    ]
                    for r, c in best_comp:
                        passable_mask[r, c] = True

        for cand_goal in sorted_goals[:max_candidate_goals]:
            passable_mask[cand_goal[0], cand_goal[1]] = True
            action_plan = GridTopologyGraph.shortest_path_with_actions(
                passable_mask=passable_mask,
                start_coord=start_pos,
                goal_coord=cand_goal,
                actions_with_deltas=effective_actions,
                goal_radius=effective_goal_radius,
                body_offsets=body_offsets,
                failed_transitions=failed_transitions,
            )
            if action_plan:
                return action_plan

        # Fallback: if rigid body envelope proved overconstrained on traversable manifold,
        # route center-mass trajectory.
        if body_offsets is not None:
            for cand_goal in sorted_goals[:max_candidate_goals]:
                action_plan = GridTopologyGraph.shortest_path_with_actions(
                    passable_mask=passable_mask,
                    start_coord=start_pos,
                    goal_coord=cand_goal,
                    actions_with_deltas=effective_actions,
                    goal_radius=effective_goal_radius,
                    body_offsets=None,
                    failed_transitions=failed_transitions,
                )
                if action_plan:
                    return action_plan

        return None

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export router capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="geodesic_manifold_router",
            operator="==",
            value="active",
            relation="action_transition_routing",
            target_entity="manifold_router",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering geodesic manifold router."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="geodesic_manifold_router_state",
                node_data={"status": "active"},
            )
        ]


class GeodesicPathOperator(GridOperator):
    """Operator that connects landmark markers with a geodesic shortest path avoiding obstacle colors."""

    def __init__(
        self,
        name: str,
        start_color: int,
        goal_color: int,
        path_color: int,
        traversable_colors: tuple[int, ...] = (0,),
        obstacle_colors: tuple[int, ...] = (),
        include_endpoints: bool = False,
        connectivity: str = "4_connected",
        complexity: float = 2.4,
    ) -> None:
        self.name = name
        self.start_color = start_color
        self.goal_color = goal_color
        self.path_color = path_color
        self.traversable_colors = traversable_colors
        self.obstacle_colors = obstacle_colors
        self.include_endpoints = include_endpoints
        self.connectivity = connectivity
        self.complexity = complexity

    def to_causal_predicate(self) -> CausalPredicate:
        """Export geodesic path operator as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"geodesic_path:{self.start_color}->{self.goal_color}",
            operator="==",
            value=self.path_color,
            relation=f"conn:{self.connectivity}",
            target_entity=self.name,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording geodesic path operator."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"op_{self.name}",
                node_data={
                    "name": self.name,
                    "start_color": self.start_color,
                    "goal_color": self.goal_color,
                    "path_color": self.path_color,
                    "traversable_colors": list(self.traversable_colors),
                    "obstacle_colors": list(self.obstacle_colors),
                    "connectivity": self.connectivity,
                    "complexity": self.complexity,
                },
            )
        ]

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:

        return [
            OperatorBinding(
                operator_name=self.name,
                params={
                    "start_color": self.start_color,
                    "goal_color": self.goal_color,
                    "path_color": self.path_color,
                    "traversable_colors": list(self.traversable_colors),
                    "obstacle_colors": list(self.obstacle_colors),
                    "include_endpoints": self.include_endpoints,
                    "connectivity": self.connectivity,
                },
                description=f"{self.name}(start={self.start_color}, goal={self.goal_color}, path={self.path_color})",
                complexity=self.complexity,
            )
        ]

    def _find_endpoints(
        self, grid: np.ndarray
    ) -> tuple[int | None, int | None, int | None, int | None]:
        """Locate start and goal coordinates on the grid."""
        if self.start_color == self.goal_color:
            coords = np.argwhere(grid == self.start_color)
            if len(coords) < 2:
                return None, None, None, None
            return int(coords[0][0]), int(coords[0][1]), int(coords[1][0]), int(coords[1][1])
        start_coords = np.argwhere(grid == self.start_color)
        goal_coords = np.argwhere(grid == self.goal_color)
        if len(start_coords) == 0 or len(goal_coords) == 0:
            return None, None, None, None
        return (
            int(start_coords[0][0]),
            int(start_coords[0][1]),
            int(goal_coords[0][0]),
            int(goal_coords[0][1]),
        )

    def has_multiple_shortest_paths(self, grid: np.ndarray) -> bool:
        """Check if grid contains ambiguous multiple shortest paths between endpoints."""
        sr, sc, gr, gc = self._find_endpoints(grid)
        if sr is None or sc is None or gr is None or gc is None:
            return False
        paths = GridTopologyGraph.find_all_shortest_paths(
            grid=grid,
            start_coord=(sr, sc),
            goal_coord=(gr, gc),
            traversable_colors=set(self.traversable_colors),
            connectivity=self.connectivity,
        )
        return len(paths) > 1

    def apply(self, grid: np.ndarray, binding: OperatorBinding | dict[str, Any]) -> np.ndarray:
        out = np.copy(grid)
        sr, sc, gr, gc = self._find_endpoints(grid)
        if sr is None or sc is None or gr is None or gc is None:
            return out

        path = GridTopologyGraph.shortest_path(
            grid=grid,
            start_coord=(sr, sc),
            goal_coord=(gr, gc),
            traversable_colors=set(self.traversable_colors),
            connectivity=self.connectivity,
        )

        if path:
            for pr, pc in path:
                if self.include_endpoints or ((pr, pc) != (sr, sc) and (pr, pc) != (gr, gc)):
                    out[pr, pc] = self.path_color

        return out


class InductiveTopologicalGraphSynthesizer:
    """Induces topological geodesic path operators directly from demonstration residuals."""

    @classmethod
    def synthesize_from_residuals(
        cls, train_pairs: list[tuple[np.ndarray, np.ndarray]]
    ) -> GeodesicPathOperator | None:
        """Deduce landmark endpoints, obstacle sets, and shortest path parameters."""
        if not train_pairs:
            return None

        x0, y0 = train_pairs[0]
        diff_mask = x0 != y0
        residual_coords = [tuple(int(c) for c in pt) for pt in np.argwhere(diff_mask)]

        if len(residual_coords) == 0:
            return None

        # Residual path must all be colored with a single path color in y0
        path_colors = np.unique(y0[diff_mask])
        if len(path_colors) != 1:
            return None
        path_color = int(path_colors[0])

        # Traversable medium in x0
        traversed_colors = tuple(sorted(set(int(c) for c in np.unique(x0[diff_mask]))))

        # Identify terminal endpoints of the residual path (degree <= 1 in residual graph)
        residual_set = set(residual_coords)
        endpoints = [
            (r, c)
            for (r, c) in residual_coords
            if sum(
                1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if (r + dr, c + dc) in residual_set
            )
            <= 1
        ]

        H, W = x0.shape
        candidate_landmarks: set[int] = set()

        for er, ec in endpoints:
            # Check if endpoint itself was originally a landmark in x0
            orig_col = int(x0[er, ec])
            if orig_col not in traversed_colors and orig_col != 0:
                candidate_landmarks.add(orig_col)

            # Check adjacent cells outside residual path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = er + dr, ec + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in residual_set:
                    col = int(x0[nr, nc])
                    if col not in traversed_colors and col != 0:
                        candidate_landmarks.add(col)

        # Also fallback: any non-zero, non-traversed color with count <= 4 in x0
        for col in np.unique(x0):
            c_int = int(col)
            if c_int != 0 and c_int not in traversed_colors:
                if np.sum(x0 == c_int) <= 4:
                    candidate_landmarks.add(c_int)

        if not candidate_landmarks:
            return None

        landmarks = sorted(list(candidate_landmarks))
        candidate_pairs: list[tuple[int, int]] = []

        # Distinct landmark pairs
        for i in range(len(landmarks)):
            for j in range(len(landmarks)):
                if i != j:
                    candidate_pairs.append((landmarks[i], landmarks[j]))
                else:
                    # Same color pair (e.g. connect two dots of same color)
                    if np.sum(x0 == landmarks[i]) >= 2:
                        candidate_pairs.append((landmarks[i], landmarks[i]))

        all_colors = set(int(c) for c in np.unique(x0)) - {0}

        # Test candidate operators
        for start_c, goal_c in candidate_pairs:
            obstacle_candidates = tuple(
                sorted(all_colors - set(traversed_colors) - {start_c, goal_c})
            )

            for include_endpoints in [False, True]:
                for conn in ["4_connected", "8_connected"]:
                    candidate_op = GeodesicPathOperator(
                        name="synthesized_geodesic_shortest_path",
                        start_color=start_c,
                        goal_color=goal_c,
                        path_color=path_color,
                        traversable_colors=traversed_colors,
                        obstacle_colors=obstacle_candidates,
                        include_endpoints=include_endpoints,
                        connectivity=conn,
                    )

                    # Verify on Demo 0
                    b0 = candidate_op.propose({}, {"train_pairs": train_pairs})[0]
                    pred0 = candidate_op.apply(x0, b0)
                    if not np.array_equal(pred0, y0):
                        continue

                    # Out-of-construction verification on Demo 1..k
                    verified = True
                    for xi, yi in train_pairs[1:]:
                        pred_i = candidate_op.apply(xi, b0)
                        if not np.array_equal(pred_i, yi):
                            verified = False
                            break

                    if verified:
                        logger.info(
                            "Successfully induced topological geodesic operator: start=%d, goal=%d, path=%d, traversable=%s, obstacles=%s",
                            start_c,
                            goal_c,
                            path_color,
                            traversed_colors,
                            obstacle_candidates,
                        )
                        return candidate_op

        return None
