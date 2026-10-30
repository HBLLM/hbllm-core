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
from collections.abc import Iterator
from dataclasses import dataclass, field

try:
    import hbllm_hcir_graph as _rust_graph  # type: ignore[import-not-found]
except ImportError:
    _rust_graph = None

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


@dataclass
class BettiInvariants:
    """Persistent homology invariants for 2D spatial manifolds."""

    betti_0: int  # Number of connected components
    betti_1: int  # 1D cycle rank (number of independent loops/tunnels)
    euler_characteristic: int  # V - E
    cavities: list[set[tuple[int, int]]] = field(
        default_factory=list
    )  # Dead-end cavities/cul-de-sacs
    bottlenecks: set[tuple[int, int]] = field(
        default_factory=set
    )  # Articulation points / narrow corridors


@dataclass
class MinCutMaxFlowResult:
    """Result of max-flow min-cut network analysis on spatial partition."""

    max_flow: float
    min_cut_capacity: float
    cut_edges: list[tuple[tuple[int, int], tuple[int, int]]]
    source_partition: set[tuple[int, int]]
    sink_partition: set[tuple[int, int]]
    critical_cut_cells: set[tuple[int, int]]


@dataclass
class ManifoldTopologyResult:
    """Integrated spatial topology and homology analysis."""

    cut_set_result: CutSetResult
    betti_invariants: BettiInvariants
    min_cut_result: MinCutMaxFlowResult | None


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

        if step_size == 1 and _rust_graph is not None:
            try:
                rust_comps = _rust_graph.topological_connected_components(
                    H, W, [(int(r), int(c)) for r, c in barrier_cells if 0 <= r < H and 0 <= c < W]
                )
                for comp in rust_comps:
                    comp_set = {(int(r), int(c)) for r, c in comp}
                    if (sr, sc) in comp_set:
                        return comp_set
                return set()
            except Exception:
                pass

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

    @staticmethod
    def compute_betti_invariants(
        grid_shape: tuple[int, int],
        traversable_cells: set[tuple[int, int]],
    ) -> BettiInvariants:
        """Compute 0-th and 1-st Betti numbers and detect cavities/bottlenecks in traversable space."""
        V = len(traversable_cells)
        if V == 0:
            return BettiInvariants(
                betti_0=0,
                betti_1=0,
                euler_characteristic=0,
                cavities=[],
                bottlenecks=set(),
            )

        # Build adjacency graph
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in traversable_cells}
        edge_count = 0
        deltas = [(0, 1), (1, 0)]  # Count each undirected edge once
        for r, c in traversable_cells:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in traversable_cells:
                    adj[(r, c)].append((nr, nc))
                    adj[(nr, nc)].append((r, c))
                    edge_count += 1

        # 1. Connected components (b_0)
        visited: set[tuple[int, int]] = set()
        components: list[set[tuple[int, int]]] = []
        for cell in traversable_cells:
            if cell not in visited:
                comp: set[tuple[int, int]] = set()
                queue: deque[tuple[int, int]] = deque([cell])
                visited.add(cell)
                while queue:
                    curr = queue.popleft()
                    comp.add(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(comp)

        b_0 = len(components)
        chi = V - edge_count
        b_1 = max(0, edge_count - V + b_0)

        # 2. Articulation points / Bottlenecks (cells whose removal increases b_0)
        # Computed iteratively to avoid recursion depth limits on large grid manifolds
        bottlenecks: set[tuple[int, int]] = set()
        tin: dict[tuple[int, int], int] = {}
        low: dict[tuple[int, int], int] = {}
        timer = 0

        for comp in components:
            if not comp:
                continue
            root = next(iter(comp))
            timer += 1
            tin[root] = low[root] = timer
            # Stack stores: [node, parent, neighbor_iterator, children_count]
            stack: list[
                tuple[tuple[int, int], tuple[int, int] | None, Iterator[tuple[int, int]], int]
            ] = [(root, None, iter(adj[root]), 0)]

            while stack:
                u, p, it, children = stack[-1]
                try:
                    v = next(it)
                    if v == p:
                        continue
                    if v in tin:
                        low[u] = min(low[u], tin[v])
                    else:
                        timer += 1
                        tin[v] = low[v] = timer
                        stack.append((v, u, iter(adj[v]), 0))
                except StopIteration:
                    stack.pop()
                    if p is None:
                        if children > 1:
                            bottlenecks.add(u)
                    else:
                        pu, pp, pit, pchildren = stack[-1]
                        low[pu] = min(low[pu], low[u])
                        if low[u] >= tin[pu] and pp is not None:
                            bottlenecks.add(pu)
                        stack[-1] = (pu, pp, pit, pchildren + 1)

        # 3. Cavities (dead-end pockets / cul-de-sacs with degree <= 1)
        cavities: list[set[tuple[int, int]]] = []
        dead_ends = [cell for cell, nbrs in adj.items() if len(nbrs) <= 1]
        for de in dead_ends:
            cavity: set[tuple[int, int]] = set()
            curr = de
            prev: tuple[int, int] | None = None
            while curr not in bottlenecks and len(adj[curr]) <= 2:
                if curr in cavity:
                    break
                cavity.add(curr)
                next_cells = [n for n in adj[curr] if n != prev]
                if not next_cells:
                    break
                prev = curr
                curr = next_cells[0]
            if cavity and cavity not in cavities:
                cavities.append(cavity)

        return BettiInvariants(
            betti_0=b_0,
            betti_1=b_1,
            euler_characteristic=chi,
            cavities=cavities,
            bottlenecks=bottlenecks,
        )

    @staticmethod
    def compute_min_cut_max_flow(
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_shape: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        traversable_cells: set[tuple[int, int]] | None = None,
    ) -> MinCutMaxFlowResult:
        """Compute the s-t min-cut and bottleneck barrier partition via Edmonds-Karp."""
        H, W = grid_shape
        trav = (
            traversable_cells
            if traversable_cells is not None
            else {(r, c) for r in range(H) for c in range(W) if (r, c) not in barrier_cells}
        )

        all_nodes: set[tuple[int, int]] = set(trav) | set(barrier_cells)
        if start not in all_nodes or goal not in all_nodes:
            return MinCutMaxFlowResult(
                max_flow=0.0,
                min_cut_capacity=0.0,
                cut_edges=[],
                source_partition={start},
                sink_partition=all_nodes - {start},
                critical_cut_cells=set(),
            )

        # Build capacity graph
        INF_CAP = 1000.0
        capacities: dict[tuple[int, int], dict[tuple[int, int], float]] = {u: {} for u in all_nodes}
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r, c in all_nodes:
            is_barrier_u = (r, c) in barrier_cells
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in all_nodes:
                    is_barrier_v = (nr, nc) in barrier_cells
                    cap = 1.0 if (is_barrier_u or is_barrier_v) else INF_CAP
                    capacities[(r, c)][(nr, nc)] = cap

        # Residual network copy
        residual: dict[tuple[int, int], dict[tuple[int, int], float]] = {
            u: dict(nbrs) for u, nbrs in capacities.items()
        }
        for u in all_nodes:
            for v in capacities[u]:
                if u not in residual[v]:
                    residual[v][u] = 0.0

        # Edmonds-Karp BFS augmenting path
        max_flow = 0.0
        while True:
            parent: dict[tuple[int, int], tuple[int, int]] = {}
            queue: deque[tuple[int, int]] = deque([start])
            while queue and goal not in parent:
                curr = queue.popleft()
                for nxt, res_cap in residual[curr].items():
                    if res_cap > 1e-6 and nxt not in parent and nxt != start:
                        parent[nxt] = curr
                        queue.append(nxt)

            if goal not in parent:
                break

            # Find bottleneck along path
            path_flow = float("inf")
            curr = goal
            while curr != start:
                p = parent[curr]
                path_flow = min(path_flow, residual[p][curr])
                curr = p

            # Augment flow
            curr = goal
            while curr != start:
                p = parent[curr]
                residual[p][curr] -= path_flow
                residual[curr][p] += path_flow
                curr = p

            max_flow += path_flow
            if max_flow > 10000.0:  # Direct path without barriers
                break

        # Compute S (reachable from start in residual) and T (sink partition)
        source_partition: set[tuple[int, int]] = set()
        queue_reach = deque([start])
        source_partition.add(start)
        while queue_reach:
            curr = queue_reach.popleft()
            for nxt, res_cap in residual[curr].items():
                if res_cap > 1e-6 and nxt not in source_partition:
                    source_partition.add(nxt)
                    queue_reach.append(nxt)

        sink_partition = all_nodes - source_partition
        cut_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        min_cut_capacity = 0.0
        critical_cut_cells: set[tuple[int, int]] = set()

        for u in source_partition:
            for v, orig_cap in capacities[u].items():
                if v in sink_partition and orig_cap > 0:
                    cut_edges.append((u, v))
                    min_cut_capacity += orig_cap
                    if u in barrier_cells:
                        critical_cut_cells.add(u)
                    if v in barrier_cells:
                        critical_cut_cells.add(v)

        return MinCutMaxFlowResult(
            max_flow=max_flow,
            min_cut_capacity=min_cut_capacity,
            cut_edges=cut_edges,
            source_partition=source_partition,
            sink_partition=sink_partition,
            critical_cut_cells=critical_cut_cells,
        )

    @staticmethod
    def analyze_manifold_topology(
        start: tuple[int, int],
        goal: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
        movable_barriers: set[tuple[int, int]] | None = None,
        occupied_cells: set[tuple[int, int]] | None = None,
    ) -> ManifoldTopologyResult:
        """Perform comprehensive persistent homology and min-cut manifold analysis."""
        cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
            start=start,
            goal=goal,
            barrier_cells=barrier_cells,
            grid_shape=grid_shape,
            step_size=step_size,
            movable_barriers=movable_barriers,
            occupied_cells=occupied_cells,
        )
        H, W = grid_shape
        trav = {(r, c) for r in range(H) for c in range(W) if (r, c) not in barrier_cells}
        betti = TopologicalCutSetAnalyzer.compute_betti_invariants(grid_shape, trav)
        min_cut = None
        if cut_res.is_partitioned:
            min_cut = TopologicalCutSetAnalyzer.compute_min_cut_max_flow(
                start=start,
                goal=goal,
                grid_shape=grid_shape,
                barrier_cells=barrier_cells,
                traversable_cells=trav,
            )
        return ManifoldTopologyResult(
            cut_set_result=cut_res,
            betti_invariants=betti,
            min_cut_result=min_cut,
        )
