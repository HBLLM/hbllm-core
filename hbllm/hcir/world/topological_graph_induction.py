"""Topological Graph & Geodesic Maze Induction Engine (Milestone 3 Phase 2).

Inductively discovers non-local topological geodesic mechanisms:
- GridTopologyGraph & GeodesicManifoldRouter: Graph abstraction & Betti-guided manifold routing over 2D grids.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

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
    ) -> list[tuple[Any, tuple[int, int]]] | None:
        """Find minimum-hop shortest action sequence directly in action transition space.

        Returns list of (action, next_coord) pairs.
        """
        H, W = passable_mask.shape
        sr, sc = start_coord
        gr, gc = goal_coord

        def _at_goal(r: int, c: int) -> bool:
            return abs(r - gr) <= goal_radius and abs(c - gc) <= goal_radius

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
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    step_mag = max(abs(dr), abs(dc))
                    blocked = False
                    if step_mag > 1:
                        for s_i in range(1, step_mag + 1):
                            inter_r = curr_r + int(round(dr * s_i / step_mag))
                            inter_c = curr_c + int(round(dc * s_i / step_mag))
                            if not (0 <= inter_r < H and 0 <= inter_c < W) or (
                                not passable_mask[inter_r, inter_c]
                                and not _at_goal(inter_r, inter_c)
                            ):
                                blocked = True
                                break
                    elif not passable_mask[nr, nc] and not _at_goal(nr, nc):
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
        prune_cavities: bool = False,
        preserve_goal_order: bool = False,
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
            if 0 <= ur < H and 0 <= uc < W:
                passable_mask[ur, uc] = False

        hazard_cells: set[tuple[int, int]] = set()
        effective_traj_barriers = set(barrier_features)
        if not effective_traj_barriers:
            effective_traj_barriers.add(bg)
            non_passable_feats = set(int(v) for v in np.unique(curr_grid[~passable_mask]))
            effective_traj_barriers.update(non_passable_feats - avatar_features)

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

        # Homological cul-de-sac cavity pruning via Betti invariants (Area 7 / VIP)
        if prune_cavities:
            traversable_cells = {(r, c) for r in range(H) for c in range(W) if passable_mask[r, c]}
            if len(traversable_cells) > 4:
                from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer

                betti = TopologicalCutSetAnalyzer.compute_betti_invariants(
                    (H, W), traversable_cells
                )
                goal_set = set(goals)
                for cavity in betti.cavities:
                    if start_pos not in cavity and not (cavity & goal_set):
                        for cr, cc in cavity:
                            passable_mask[cr, cc] = False

        candidate_goals = (
            list(goals)
            if preserve_goal_order
            else sorted(
                goals,
                key=lambda g: abs(g[0] - start_pos[0]) + abs(g[1] - start_pos[1]),
            )
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
            effective_actions = harmonized_actions

        effective_goal_radius = 0 if max_stride <= 1 else max(1, max_stride // 2)

        for cand_goal in candidate_goals[:max_candidate_goals]:
            cand_mask = passable_mask.copy()
            cand_mask[cand_goal[0], cand_goal[1]] = True
            action_plan = GridTopologyGraph.shortest_path_with_actions(
                passable_mask=cand_mask,
                start_coord=start_pos,
                goal_coord=cand_goal,
                actions_with_deltas=effective_actions,
                goal_radius=effective_goal_radius,
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

    @staticmethod
    def compute_manifold_homology(
        grid_shape: tuple[int, int],
        passable_mask: np.ndarray,
    ) -> Any:
        """Compute persistent Betti invariants over a traversable manifold mask."""
        from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer

        H, W = grid_shape
        traversable_cells = {(r, c) for r in range(H) for c in range(W) if passable_mask[r, c]}
        return TopologicalCutSetAnalyzer.compute_betti_invariants((H, W), traversable_cells)

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
