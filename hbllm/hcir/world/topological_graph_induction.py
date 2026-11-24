"""Topological Graph & Geodesic Maze Induction Engine (Milestone 3 Phase 2).

Inductively discovers non-local topological geodesic mechanisms:
- GridTopologyGraph & GeodesicManifoldRouter: Graph abstraction & Betti-guided manifold routing over 2D grids.
"""

from __future__ import annotations

import collections
import heapq
import logging
from collections.abc import Callable, Mapping, Sequence
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
    def shortest_path_from_kb(
        grid: np.ndarray,
        start_coord: tuple[int, int],
        goal_coords: set[tuple[int, int]],
        action_deltas: Mapping[Any, tuple[int, int]],
        evaluate_edge: Callable[[tuple[int, int], int], Any],
        *,
        max_expansions: int = 2000,
        failed_transitions: set[tuple[tuple[int, int], Any]] | None = None,
        inhibited_actions: set[Any] | None = None,
        blocked_destinations: set[tuple[int, int]] | None = None,
        allow_goal_entry: bool = True,
        inter_step_cost: Callable[[tuple[int, int], int], float] | None = None,
        budget: Any | None = None,
    ) -> list[tuple[Any, tuple[int, int]]] | None:
        """Find minimum-expected-cost path querying only empirical KB transition evaluations."""
        if not goal_coords or grid.ndim < 2:
            return None
        H, W = int(grid.shape[0]), int(grid.shape[1])
        sr, sc = int(start_coord[0]), int(start_coord[1])
        if (sr, sc) in goal_coords:
            return []
        if not (0 <= sr < H and 0 <= sc < W):
            return None

        valid_actions = [
            (act, int(d[0]), int(d[1]))
            for act, d in action_deltas.items()
            if len(d) >= 2
            and (int(d[0]) != 0 or int(d[1]) != 0)
            and (inhibited_actions is None or act not in inhibited_actions)
        ]
        if not valid_actions:
            return None

        max_stride = max(abs(dr) + abs(dc) for _, dr, dc in valid_actions)
        inv_stride = 1.0 / max(1, max_stride)

        def _heuristic(r: int, c: int) -> float:
            return min(abs(r - gr) + abs(c - gc) for gr, gc in goal_coords) * inv_stride

        grid_rows = grid.tolist()
        open_heap: list[tuple[float, float, int, int, int]] = [(_heuristic(sr, sc), 0.0, 0, sr, sc)]
        best_cost: dict[tuple[int, int], float] = {(sr, sc): 0.0}
        came_from: dict[tuple[int, int], tuple[tuple[int, int], Any]] = {}
        counter = 0
        expansions = 0

        while open_heap and (max_expansions <= 0 or expansions < max_expansions):
            if budget is not None:
                if not budget.allows("cheap"):
                    return None
                budget.charge(1)
            _, g_cost, _, cr, cc = heapq.heappop(open_heap)
            if g_cost > best_cost.get((cr, cc), float("inf")) + 1e-9:
                continue
            if (cr, cc) in goal_coords:
                plan: list[tuple[Any, tuple[int, int]]] = []
                cur = (cr, cc)
                while cur != (sr, sc):
                    prev_coord, via_act = came_from[cur]
                    plan.append((via_act, cur))
                    cur = prev_coord
                plan.reverse()
                return plan
            expansions += 1

            for act, dr, dc in valid_actions:
                if failed_transitions is not None and ((cr, cc), act) in failed_transitions:
                    continue
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue

                next_coord = (nr, nc)
                if blocked_destinations is not None and next_coord in blocked_destinations:
                    continue

                steps = max(abs(dr), abs(dc))
                ray_clear = True
                extra_inter_cost = 0.0
                if steps > 1 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                    step_r = (dr > 0) - (dr < 0)
                    step_c = (dc > 0) - (dc < 0)
                    for k in range(1, steps):
                        ir, ic = cr + step_r * k, cc + step_c * k
                        if not (0 <= ir < H and 0 <= ic < W):
                            ray_clear = False
                            break
                        mid_feat = int(grid_rows[ir][ic])
                        mid_ev = evaluate_edge((ir, ic), mid_feat)
                        if (mid_ev.risk_cost >= 40.0 and mid_ev.goal_value <= 0.0) or (
                            not mid_ev.traversable
                            and (not allow_goal_entry or (ir, ic) not in goal_coords)
                        ):
                            ray_clear = False
                            break
                        if inter_step_cost is not None:
                            extra_inter_cost += float(inter_step_cost((ir, ic), mid_feat))
                        else:
                            extra_inter_cost += float(mid_ev.risk_cost)
                if not ray_clear:
                    continue

                edge_ev = evaluate_edge(next_coord, int(grid_rows[nr][nc]))
                is_goal = next_coord in goal_coords
                if (edge_ev.risk_cost >= 40.0 and edge_ev.goal_value <= 0.0) or (
                    not edge_ev.traversable
                    and not (is_goal and (edge_ev.goal_value > 0.0 or allow_goal_entry))
                ):
                    continue

                move_cost = (
                    (float(edge_ev.step_cost) if edge_ev.traversable else 1.0)
                    + float(edge_ev.risk_cost)
                    + extra_inter_cost
                )
                new_g = g_cost + move_cost
                if new_g < best_cost.get(next_coord, float("inf")) - 1e-9:
                    best_cost[next_coord] = new_g
                    came_from[next_coord] = ((cr, cc), act)
                    counter += 1
                    heapq.heappush(open_heap, (new_g + _heuristic(nr, nc), new_g, counter, nr, nc))

        return None

    @staticmethod
    def shortest_path_with_causal_ops(
        grid: np.ndarray,
        start_coord: tuple[int, int],
        goal_coords: set[tuple[int, int]],
        action_deltas: Mapping[Any, tuple[int, int]],
        transition_kb: Any,
        *,
        initial_state_tokens: frozenset[str] = frozenset(),
        failed_transitions: set[tuple[tuple[int, int], Any]] | None = None,
        inhibited_actions: set[Any] | None = None,
        unknown_traversable: bool = True,
        max_expansions: int = 2000,
    ) -> list[tuple[Any, tuple[int, int]]] | None:
        """Find minimum-expected-cost plan over (coord, active_state_tokens) via KB causal rules."""
        if not goal_coords or grid.ndim < 2 or transition_kb is None:
            return None
        H, W = int(grid.shape[0]), int(grid.shape[1])
        sr, sc = int(start_coord[0]), int(start_coord[1])
        if (sr, sc) in goal_coords:
            return []
        if not (0 <= sr < H and 0 <= sc < W):
            return None

        valid_actions = [
            (act, int(d[0]), int(d[1]))
            for act, d in action_deltas.items()
            if len(d) >= 2
            and (int(d[0]) != 0 or int(d[1]) != 0)
            and (inhibited_actions is None or act not in inhibited_actions)
        ]
        if not valid_actions:
            return None

        locus_rules: dict[tuple[int, int], list[Any]] = {}
        feature_rules: dict[int, list[Any]] = {}
        for rule in transition_kb.query_causal_rules():
            trig = rule.trigger_key
            if isinstance(trig, tuple) and len(trig) == 2:
                locus_rules.setdefault((int(trig[0]), int(trig[1])), []).append(rule)
            elif isinstance(trig, int) and not isinstance(trig, bool):
                feature_rules.setdefault(int(trig), []).append(rule)

        max_stride = max(abs(dr) + abs(dc) for _, dr, dc in valid_actions)
        inv_stride = 1.0 / max(1, max_stride)

        def _heuristic(r: int, c: int) -> float:
            return min(abs(r - gr) + abs(c - gc) for gr, gc in goal_coords) * inv_stride

        def _is_unlocked_by_locus_token(
            ev: Any, pos: tuple[int, int], tokens: frozenset[str]
        ) -> bool:
            return bool(
                not ev.traversable and float(ev.risk_cost) < 40.0 and f"unlocks:{pos}" in tokens
            )

        start_state: tuple[tuple[int, int], frozenset[str]] = ((sr, sc), initial_state_tokens)
        open_heap: list[tuple[float, float, int, tuple[int, int], frozenset[str]]] = [
            (_heuristic(sr, sc), 0.0, 0, (sr, sc), initial_state_tokens)
        ]
        best_cost: dict[tuple[tuple[int, int], frozenset[str]], float] = {start_state: 0.0}
        came_from: dict[
            tuple[tuple[int, int], frozenset[str]],
            tuple[tuple[tuple[int, int], frozenset[str]], Any],
        ] = {}
        counter = 0
        expansions = 0

        while open_heap and (max_expansions <= 0 or expansions < max_expansions):
            _, g_cost, _, cur_coord, cur_tokens = heapq.heappop(open_heap)
            cur_state = (cur_coord, cur_tokens)
            if g_cost > best_cost.get(cur_state, float("inf")) + 1e-9:
                continue
            if cur_coord in goal_coords:
                plan: list[tuple[Any, tuple[int, int]]] = []
                trace_state = cur_state
                while trace_state != start_state:
                    prev_state, via_act = came_from[trace_state]
                    plan.append((via_act, trace_state[0]))
                    trace_state = prev_state
                plan.reverse()
                return plan

            expansions += 1
            cr, cc = cur_coord

            for act, dr, dc in valid_actions:
                if failed_transitions is not None and (cur_coord, act) in failed_transitions:
                    continue
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue

                steps = max(abs(dr), abs(dc))
                ray_clear = True
                extra_inter_cost = 0.0
                if steps > 1 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                    step_r = (dr > 0) - (dr < 0)
                    step_c = (dc > 0) - (dc < 0)
                    for k in range(1, steps):
                        ir, ic = cr + step_r * k, cc + step_c * k
                        if not (0 <= ir < H and 0 <= ic < W):
                            ray_clear = False
                            break
                        mid_pos = (ir, ic)
                        mid_ev = transition_kb.evaluate_transition(
                            int(grid[ir, ic]),
                            locus=mid_pos,
                            active_state_tokens=cur_tokens,
                            unknown_traversable=unknown_traversable,
                        )
                        if not mid_ev.traversable and not _is_unlocked_by_locus_token(
                            mid_ev, mid_pos, cur_tokens
                        ):
                            ray_clear = False
                            break
                        extra_inter_cost += float(mid_ev.risk_cost)
                if not ray_clear:
                    continue

                next_coord = (nr, nc)
                dest_feat = int(grid[nr, nc])
                edge_ev = transition_kb.evaluate_transition(
                    dest_feat,
                    locus=next_coord,
                    active_state_tokens=cur_tokens,
                    unknown_traversable=unknown_traversable,
                )
                unlocked_here = _is_unlocked_by_locus_token(edge_ev, next_coord, cur_tokens)
                if not edge_ev.traversable and not unlocked_here:
                    continue

                next_token_set = set(cur_tokens)
                for rule in (*locus_rules.get(next_coord, ()), *feature_rules.get(dest_feat, ())):
                    eff = str(rule.effect_signature)
                    if not eff or eff in next_token_set:
                        continue
                    if not rule.precondition_tokens or all(
                        tok in cur_tokens for tok in rule.precondition_tokens
                    ):
                        next_token_set.add(eff)
                next_tokens = (
                    frozenset(next_token_set)
                    if len(next_token_set) != len(cur_tokens)
                    else cur_tokens
                )

                step_cost = 1.0 if unlocked_here else float(edge_ev.step_cost)
                move_cost = step_cost + float(edge_ev.risk_cost) + extra_inter_cost
                new_g = g_cost + move_cost
                next_state = (next_coord, next_tokens)
                if new_g < best_cost.get(next_state, float("inf")) - 1e-9:
                    best_cost[next_state] = new_g
                    came_from[next_state] = (cur_state, act)
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (new_g + _heuristic(nr, nc), new_g, counter, next_coord, next_tokens),
                    )

        return None

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
    def compute_soft_avoidance_and_shroud(
        engine: Any,
        curr_grid: np.ndarray,
        barrier_feats: set[int],
        bg: int,
        unknown_entity_cells: set[tuple[int, int]],
        grid_walls: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        """Compute dynamic mobile-agent and stochastic-slip soft avoidance zones + shroud walls."""
        cargo_feats = set(engine.symbolic_theory.cargo_features) | set(
            engine.learned_cargo_features
        )
        soft_avoid = set(
            engine.hazard_tracker.get_mobile_avoidance_cells(
                curr_grid, barrier_feats, bg, cargo_features=cargo_feats
            )
        )
        hz_seeds = set(engine.hazard_tracker.static_lethal_positions)
        if any(
            float(getattr(d, "slip_rate", 0.0) or 0.0) > 0.05
            for d in engine.action_dynamics.values()
        ):
            hz_seeds |= set(unknown_entity_cells)
        hz_feats = set(engine.hazard_tracker.known_lethal_features)
        rp_map = engine.working_memory.concept_role_priors
        if rp_map:
            for rc_k, rp_v in rp_map.items():
                if rp_v.get("hazard", 0.0) >= 0.65 and rp_v.get("goal", 0.0) < 0.50:
                    hz_feats.add(int(rc_k))
        for l_f in hz_feats:
            for lr_i, lc_i in np.argwhere(curr_grid == l_f):
                pt_h = (int(lr_i), int(lc_i))
                hz_seeds.add(pt_h)
                soft_avoid.add(pt_h)
        for hr_s, hc_s in hz_seeds:
            for dr_s, dc_s in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                soft_avoid.add((hr_s + dr_s, hc_s + dc_s))
        shroud_f = engine.working_memory.shroud_feature
        if shroud_f is not None:
            for sr_i, sc_i in np.argwhere(curr_grid == shroud_f):
                grid_walls.add((int(sr_i), int(sc_i)))
        return soft_avoid

    @staticmethod
    def compute_reachable_frontiers(
        engine: Any,
        curr_grid: np.ndarray,
        bg: int,
        prior_hazard_feats: set[int],
        interactive_cluster_loci: set[tuple[int, int]] | tuple[tuple[int, int], ...],
        scipy_label_fn: Any,
    ) -> list[tuple[int, int]]:
        """Compute reachable unvisited frontier cells prioritizing shroud-revealing boundaries."""
        H, W = curr_grid.shape
        avatar_pos = engine.avatar_pos
        if avatar_pos is None or not (0 <= avatar_pos[0] < H and 0 <= avatar_pos[1] < W):
            return []
        wm = engine.working_memory
        visited: set[tuple[int, int]] = set(wm.visited_cells)
        for c_pos, cnt in engine.position_visit_counts.items():
            if cnt > 0:
                visited.add(c_pos)
        visited.add(avatar_pos)

        theory = engine.symbolic_theory
        lethal = engine.hazard_tracker.known_lethal_features
        shroud_f = wm.shroud_feature
        allowed_feats = (set(theory.walkable_features) | {int(bg)}) - (
            set(theory.barrier_features) | set(lethal) | prior_hazard_feats
        )
        if shroud_f is not None:
            allowed_feats.discard(int(shroud_f))
        passable_mask = np.isin(curr_grid, list(allowed_feats))
        passable_mask[avatar_pos[0], avatar_pos[1]] = True
        labeled_reach, _ = scipy_label_fn(passable_mask)
        av_lbl = int(labeled_reach[avatar_pos[0], avatar_pos[1]])
        reach_mask = labeled_reach == av_lbl

        visited_mask = np.zeros((H, W), dtype=bool)
        for vr, vc in visited:
            if 0 <= vr < H and 0 <= vc < W:
                visited_mask[vr, vc] = True
        unvisited_mask = reach_mask & (~visited_mask)
        for er, ec in interactive_cluster_loci:
            if 0 <= er < H and 0 <= ec < W:
                unvisited_mask[er, ec] = False

        if shroud_f is not None:
            shroud_mask = curr_grid == int(shroud_f)
            adj_shroud = np.zeros((H, W), dtype=bool)
            adj_shroud[1:, :] |= shroud_mask[:-1, :]
            adj_shroud[:-1, :] |= shroud_mask[1:, :]
            adj_shroud[:, 1:] |= shroud_mask[:, :-1]
            adj_shroud[:, :-1] |= shroud_mask[:, 1:]
            shroud_front = reach_mask & adj_shroud & (~visited_mask)
            if np.any(shroud_front):
                sr_r, sr_c = np.nonzero(shroud_front)
                cands = list(zip(sr_r.tolist(), sr_c.tolist()))
                cands.sort(key=lambda p: abs(p[0] - avatar_pos[0]) + abs(p[1] - avatar_pos[1]))
                return cands

        adj_visited = np.zeros((H, W), dtype=bool)
        adj_visited[1:, :] |= visited_mask[:-1, :]
        adj_visited[:-1, :] |= visited_mask[1:, :]
        adj_visited[:, 1:] |= visited_mask[:, :-1]
        adj_visited[:, :-1] |= visited_mask[:, 1:]
        frontier_mask = unvisited_mask & adj_visited
        if not np.any(frontier_mask):
            frontier_mask = unvisited_mask
        fr_r, fr_c = np.nonzero(frontier_mask)
        frontier_candidates = list(zip(fr_r.tolist(), fr_c.tolist()))

        opt_eng = engine.option_discovery
        if (
            opt_eng is not None
            and bool(opt_eng.live_pipeline_enabled)
            and opt_eng.discovered_options
        ):
            bottleneck_frontiers = [
                opt.bottleneck_state
                for opt in opt_eng.discovered_options.values()
                if opt.bottleneck_state is not None
                and 0 <= opt.bottleneck_state[0] < H
                and 0 <= opt.bottleneck_state[1] < W
                and bool(unvisited_mask[opt.bottleneck_state[0], opt.bottleneck_state[1]])
            ]
            if bottleneck_frontiers:
                frontier_candidates = bottleneck_frontiers

        frontier_candidates.sort(
            key=lambda p: abs(p[0] - avatar_pos[0]) + abs(p[1] - avatar_pos[1])
        )
        return frontier_candidates

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
        soft_avoid_cells: set[tuple[int, int]] | None = None,
        transition_kb: Any | None = None,
    ) -> list[tuple[Any, tuple[int, int]]] | None:
        """Find deterministic action-space shortest path across the traversable manifold.

        1. Constructs boolean passable mask from static barriers, walls, and unknown obstacles.
        2. Applies locus-scoped transition KB beliefs when supplied.
        3. Projects dynamic threat trajectories into evasion hazard cones.
        4. Harmonizes multi-cell action strides and delegates shortest-path search.
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
        if transition_kb is not None:
            for locus_pos in transition_kb._locus_beliefs:
                lr, lc = int(locus_pos[0]), int(locus_pos[1])
                if not (0 <= lr < H and 0 <= lc < W):
                    continue
                l_ev = transition_kb.evaluate_transition(int(curr_grid[lr, lc]), locus=(lr, lc))
                if not l_ev.traversable or l_ev.risk_cost >= 40.0:
                    passable_mask[lr, lc] = False
                elif (
                    l_ev.traversable
                    and l_ev.confidence > 0.0
                    and (lr, lc) not in unknown_entity_cells
                ):
                    passable_mask[lr, lc] = True

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

        def _search_mask_for_goal(
            mask: np.ndarray, cand_goal: tuple[int, int]
        ) -> list[tuple[Any, tuple[int, int]]] | None:
            if transition_kb is not None and effective_goal_radius == 0:
                from hbllm.hcir.world.causal.transition_kb import TransitionEdgeEvaluation

                action_map = {act: (int(dr), int(dc)) for act, dr, dc in effective_actions}
                mask_rows = mask.tolist()

                def _eval_mask_kb(pos: tuple[int, int], feat: int) -> TransitionEdgeEvaluation:
                    if pos == cand_goal:
                        ev = transition_kb.evaluate_transition(feat, locus=pos)
                        if ev.risk_cost >= 40.0:
                            return ev
                        return TransitionEdgeEvaluation(
                            traversable=True,
                            step_cost=1.0,
                            risk_cost=0.0,
                            goal_value=1.0,
                            info_gain=0.0,
                            mutates_state=False,
                            confidence=1.0,
                        )
                    if not bool(mask_rows[pos[0]][pos[1]]):
                        return TransitionEdgeEvaluation(
                            traversable=False,
                            step_cost=float("inf"),
                            risk_cost=0.0,
                            goal_value=0.0,
                            info_gain=0.0,
                            mutates_state=False,
                            confidence=1.0,
                        )
                    return transition_kb.evaluate_transition(feat, locus=pos)

                return GridTopologyGraph.shortest_path_from_kb(
                    curr_grid,
                    start_pos,
                    {cand_goal},
                    action_map,
                    _eval_mask_kb,
                    max_expansions=0,
                    allow_goal_entry=True,
                )

            return GridTopologyGraph.shortest_path_with_actions(
                passable_mask=mask,
                start_coord=start_pos,
                goal_coord=cand_goal,
                actions_with_deltas=effective_actions,
                goal_radius=effective_goal_radius,
            )

        if soft_avoid_cells:
            for cand_goal in candidate_goals[:max_candidate_goals]:
                if not (0 <= cand_goal[0] < H and 0 <= cand_goal[1] < W):
                    continue
                buf_mask = passable_mask.copy()
                for sr, sc in soft_avoid_cells:
                    if 0 <= sr < H and 0 <= sc < W and (sr, sc) != start_pos:
                        buf_mask[sr, sc] = False
                buf_mask[cand_goal[0], cand_goal[1]] = True
                buf_plan = _search_mask_for_goal(buf_mask, cand_goal)
                if buf_plan:
                    return buf_plan

        for cand_goal in candidate_goals[:max_candidate_goals]:
            if not (0 <= cand_goal[0] < H and 0 <= cand_goal[1] < W):
                continue
            cand_mask = passable_mask.copy()
            cand_mask[cand_goal[0], cand_goal[1]] = True
            action_plan = _search_mask_for_goal(cand_mask, cand_goal)
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

    @staticmethod
    def route_continuous_kinematic(
        engine: Any,
        available_actions: Sequence[Any],
        curr_grid: np.ndarray | None = None,
        goal_radius: float = 0.55,
        max_expansions: int = 512,
    ) -> list[tuple[Any, tuple[int, int]]] | None:
        """Plan a multi-step kinematic trajectory in continuous metric space using grounded velocity vectors."""
        import heapq
        import math

        wm = engine.working_memory
        if wm is None:
            return None
        c_pos = wm.continuous_agent_pos
        c_goal = wm.continuous_goal_pos
        rf = wm.reference_frame
        if c_pos is None or len(c_pos) < 2:
            return None

        vel_map: dict[Any, tuple[float, float]] = {}
        for act in available_actions:
            dyn = engine.action_dynamics.get(act)
            if dyn is not None:
                vx, vy = dyn.get_continuous_velocity()
                if (vx * vx + vy * vy) > 1e-6:
                    vel_map[act] = (float(vx), float(vy))

        if not vel_map or c_goal is None or len(c_goal) < 2:
            return None

        bounds = rf.bounds if rf is not None and rf.bounds else ((0.0, 10.0), (0.0, 10.0))
        x_min, x_max = (
            (float(bounds[0][0]), float(bounds[0][1])) if len(bounds) >= 1 else (0.0, 10.0)
        )
        y_min, y_max = (
            (float(bounds[1][0]), float(bounds[1][1])) if len(bounds) >= 2 else (0.0, 10.0)
        )
        origin_desc = (
            rf.origin_description
            if rf is not None and rf.origin_description
            else "bottom_left_metric"
        )

        gx, gy = float(c_goal[0]), float(c_goal[1])
        sx, sy = float(c_pos[0]), float(c_pos[1])

        def _to_lattice(x: float, y: float) -> tuple[int, int]:
            if curr_grid is None or curr_grid.ndim < 2:
                return (0, 0)
            H, W = curr_grid.shape
            c_idx = int(np.clip(round((x - x_min) / max(1e-6, x_max - x_min) * W - 0.5), 0, W - 1))
            if "bottom_left" in origin_desc:
                r_idx = int(
                    np.clip(round((y_max - y) / max(1e-6, y_max - y_min) * H - 0.5), 0, H - 1)
                )
            else:
                r_idx = int(
                    np.clip(round((y - y_min) / max(1e-6, y_max - y_min) * H - 0.5), 0, H - 1)
                )
            return (r_idx, c_idx)

        border_f: int | None = None
        if curr_grid is not None and curr_grid.ndim >= 2 and curr_grid.shape[0] >= 3:
            b0 = int(curr_grid[0, 0])
            if b0 == int(curr_grid[-1, -1]) and b0 == int(curr_grid[0, -1]):
                border_f = b0

        def _is_blocked(x: float, y: float, margin: float = 0.0) -> bool:
            if not (x_min + 0.4 <= x <= x_max - 0.4 and y_min + 0.4 <= y <= y_max - 0.4):
                return True
            if curr_grid is not None and curr_grid.ndim >= 2:
                pts = (
                    [(x, y), (x - margin, y), (x + margin, y), (x, y - margin), (x, y + margin)]
                    if margin > 0.0
                    else [(x, y)]
                )
                for px, py in pts:
                    lr, lc = _to_lattice(px, py)
                    feat = int(curr_grid[lr, lc])
                    if (
                        feat == border_f
                        or engine.symbolic_theory.is_barrier(feat)
                        or (lr, lc) in engine.learned_barriers
                    ):
                        return True
            return False

        for pad in (0.28, 0.0):
            start_key = (round(sx * 4), round(sy * 4))
            pq: list[tuple[float, int, float, float, list[tuple[Any, tuple[int, int]]]]] = [
                (math.hypot(gx - sx, gy - sy), 0, sx, sy, [])
            ]
            visited: dict[tuple[int, int], int] = {start_key: 0}
            expansions = 0
            while pq and expansions < max_expansions:
                _, cost, cx, cy, path = heapq.heappop(pq)
                expansions += 1
                if math.hypot(gx - cx, gy - cy) <= goal_radius and path:
                    return path
                for act, (vx, vy) in vel_map.items():
                    nx, ny = cx + vx, cy + vy
                    if _is_blocked(nx, ny, pad):
                        continue
                    nkey = (round(nx * 4), round(ny * 4))
                    ncost = cost + 1
                    if nkey in visited and visited[nkey] <= ncost:
                        continue
                    visited[nkey] = ncost
                    lat_p = _to_lattice(nx, ny)
                    act_out = int(act) if isinstance(act, (int, np.integer)) else act
                    h = math.hypot(gx - nx, gy - ny) / max(0.25, math.hypot(vx, vy))
                    heapq.heappush(pq, (ncost + h, ncost, nx, ny, [*path, (act_out, lat_p)]))
        return None
