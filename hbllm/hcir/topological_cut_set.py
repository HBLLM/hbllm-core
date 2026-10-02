"""Topological Cut-Set & Gate Analyzer — Spatial Partitioning and Obstruction Bottleneck Discovery.

Identifies:
1. Disconnected spatial partitions (C_start vs C_goal) separated by impassable barriers.
2. The critical barrier cut-set (the separating barrier boundary).
3. The optimal gate/bottleneck cell to unlock or penetrate.
4. Accessible approach coordinates from the avatar's reachable component.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CutSetResult:
    """Result of topological graph cut-set analysis."""

    is_partitioned: bool
    start_component: set[tuple[int, int]]
    goal_component: set[tuple[int, int]]
    barrier_cut_set: set[tuple[int, int]]
    best_gate_cell: tuple[int, int] | None
    approach_cell: (
        tuple[int, int] | None
    )  # Accessible cell in start_component adjacent to best_gate_cell
    all_gates: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)


class TopologicalCutSetAnalyzer:
    """Performs graph cut-set analysis on spatial grids to find partition bottlenecks."""

    @staticmethod
    def get_reachable_component(
        start: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
    ) -> set[tuple[int, int]]:
        """Compute the flood-fill reachable connected component from start."""
        H, W = grid_shape
        sr, sc = int(round(start[0])), int(round(start[1]))
        if (sr, sc) in barrier_cells or not (0 <= sr < H and 0 <= sc < W):
            return set()

        visited = {(sr, sc)}
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        step = max(1, int(step_size))
        deltas = [(-step, 0), (step, 0), (0, -step), (0, step)]

        while queue:
            r, c = queue.popleft()
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in visited and (nr, nc) not in barrier_cells:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return visited

    @staticmethod
    def analyze_cut_set(
        start: tuple[int, int],
        goal: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
        movable_barriers: set[tuple[int, int]] | None = None,
        occupied_cells: set[tuple[int, int]] | None = None,
    ) -> CutSetResult:
        """Analyze whether start and goal are in disjoint components and locate the barrier cut-set."""
        H, W = grid_shape
        c_start = TopologicalCutSetAnalyzer.get_reachable_component(
            start, barrier_cells, grid_shape, step_size
        )
        barriers_for_goal = set(barrier_cells)
        if goal in barriers_for_goal:
            barriers_for_goal.remove(goal)
        c_goal = TopologicalCutSetAnalyzer.get_reachable_component(
            goal, barriers_for_goal, grid_shape, step_size
        )

        # If goal is in start component or within step tolerance, they are mutually reachable
        goal_reached = (goal in c_start) or any(
            math.hypot(r - goal[0], c - goal[1]) < max(1.5, step_size * 0.9) for r, c in c_start
        )
        if goal_reached:
            return CutSetResult(
                is_partitioned=False,
                start_component=c_start,
                goal_component=c_goal,
                barrier_cut_set=set(),
                best_gate_cell=None,
                approach_cell=None,
            )

        # Find barrier cells adjacent to c_start
        deltas = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]
        cut_set: set[tuple[int, int]] = set()
        barrier_to_approach: dict[tuple[int, int], tuple[int, int]] = {}

        for r, c in c_start:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in barrier_cells:
                    adj_to_goal = any(
                        (nr + gdr, nc + gdc) in c_goal
                        for gdr, gdc in [
                            (-step_size, 0),
                            (step_size, 0),
                            (0, -step_size),
                            (0, step_size),
                            (-1, 0),
                            (1, 0),
                            (0, -1),
                            (0, 1),
                        ]
                    )
                    if adj_to_goal:
                        cut_set.add((nr, nc))
                        if (nr, nc) not in barrier_to_approach:
                            barrier_to_approach[(nr, nc)] = (r, c)

        if not cut_set:
            return CutSetResult(
                is_partitioned=True,
                start_component=c_start,
                goal_component=c_goal,
                barrier_cut_set=set(),
                best_gate_cell=None,
                approach_cell=None,
            )

        # Rank cut_set cells: prioritize unoccupied gates, movable barriers, adjacency to c_goal, grid alignment, proximity
        def gate_score(b_cell: tuple[int, int]) -> tuple[int, int, int, int, float]:
            br, bc = b_cell
            is_occ = (
                1
                if (
                    occupied_cells
                    and any(
                        math.hypot(br - oc[0], bc - oc[1]) < max(2.5, step_size * 0.8)
                        for oc in occupied_cells
                    )
                )
                else 0
            )
            is_movable = 0 if (movable_barriers and b_cell in movable_barriers) else 1
            # Check if b_cell is adjacent to c_goal (direct separating barrier)
            adj_to_goal_comp = any(
                (br + dr, bc + dc) in c_goal
                for dr, dc in [
                    (-step_size, 0),
                    (step_size, 0),
                    (0, -step_size),
                    (0, step_size),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ]
            )
            is_aligned = 0 if (br % step_size == 0 and bc % step_size == 0) else 1
            dist_start = math.hypot(br - start[0], bc - start[1])
            dist_goal = math.hypot(br - goal[0], bc - goal[1])
            total_dist = dist_start + dist_goal
            return (0 if adj_to_goal_comp else 1, is_movable, is_occ, is_aligned, total_dist)

        ranked_gates = []
        for b_cell in cut_set:
            if b_cell in barrier_to_approach:
                appr = barrier_to_approach[b_cell]
                score = gate_score(b_cell)
                ranked_gates.append((score, b_cell, appr))
        ranked_gates.sort(key=lambda x: x[0])

        best_gate = ranked_gates[0][1] if ranked_gates else None
        approach = ranked_gates[0][2] if ranked_gates else None
        all_gates = [(gate, appr) for _, gate, appr in ranked_gates]

        return CutSetResult(
            is_partitioned=True,
            start_component=c_start,
            goal_component=c_goal,
            barrier_cut_set=cut_set,
            best_gate_cell=best_gate,
            approach_cell=approach,
            all_gates=all_gates,
        )
