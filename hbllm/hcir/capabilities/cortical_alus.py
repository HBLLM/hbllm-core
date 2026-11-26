"""Stateless Layer-0 Cortical Math & Physics ALUs (Phase M5).

Pure numerical and spatial kernels for:
1. Conjugate Dirichlet/Beta transition posterior updates (`math.bayesian_posterior_update`)
2. Multi-source Dijkstra geodesic wavefront fields (`math.geodesic_field`)
3. Risk-sensitive finite-horizon Bellman value backups (`math.bellman_risk_backup`)
4. Forward spatiotemporal patrol & collision-cone prediction (`math.collision_cone`)

100% deterministic, stateless, domain-blind, and LLM-free.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CARDINAL_DELTAS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
)
_DIRICHLET_TOKEN_INDEX: dict[str, int] = {
    "pass": 0,
    "traversed": 0,
    "block": 1,
    "blocked": 1,
    "lethal": 2,
    "terminal_fail": 2,
    "fail": 2,
    "goal": 3,
    "terminal_win": 3,
    "win": 3,
}


def compute_bayesian_posterior_update(
    alpha_pass: float = 0.5,
    beta_block: float = 0.5,
    gamma_lethal: float = 0.05,
    delta_goal: float = 0.05,
    *,
    observations: Sequence[str] = (),
    risk_weight: float = 50.0,
) -> dict[str, Any]:
    """Update conjugate Dirichlet outcome parameters `(alpha_pass, beta_block, gamma_lethal, delta_goal)`."""
    counts = [
        max(0.0, float(alpha_pass)),
        max(0.0, float(beta_block)),
        max(0.0, float(gamma_lethal)),
        max(0.0, float(delta_goal)),
    ]
    for raw_obs in observations:
        idx = _DIRICHLET_TOKEN_INDEX.get(str(raw_obs).strip().lower())
        if idx is not None:
            counts[idx] += 1.0
    a_pass, b_block, g_lethal, d_goal = counts
    total = max(1e-9, a_pass + b_block + g_lethal + d_goal)
    probs = (a_pass / total, b_block / total, g_lethal / total, d_goal / total)
    entropy_bits = -sum(p * math.log2(p) for p in probs if p > 0.0)
    uncertainty = float(1.0 / (1.0 + max(0.0, total - 1.1)))
    risk_cost = float(risk_weight) * probs[2]
    efe = probs[3] + 0.5 * uncertainty - risk_cost
    return {
        "alpha_pass": float(a_pass),
        "beta_block": float(b_block),
        "gamma_lethal": float(g_lethal),
        "delta_goal": float(d_goal),
        "p_pass": float(probs[0]),
        "p_block": float(probs[1]),
        "p_lethal": float(probs[2]),
        "p_goal": float(probs[3]),
        "epistemic_uncertainty": uncertainty,
        "entropy_bits": float(entropy_bits),
        "risk_cost": float(risk_cost),
        "expected_free_energy": float(efe),
    }


def _to_shape(grid: Any, grid_shape: Sequence[int] | None = None) -> tuple[int, int]:
    if grid_shape is not None and len(grid_shape) >= 2:
        return max(1, int(grid_shape[0])), max(1, int(grid_shape[1]))
    arr = np.asarray(grid)
    if arr.ndim == 3:
        arr = arr[-1]
    return int(arr.shape[0]), int(arr.shape[1])


def _cell_set(cells: Sequence[Sequence[int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for pt in cells:
        if len(pt) >= 2:
            r, c = int(pt[0]), int(pt[1])
            if 0 <= r < h and 0 <= c < w:
                out.add((r, c))
    return out


def _coerce_cost_map(
    step_costs: Mapping[Any, float] | None,
) -> dict[tuple[int, int], float]:
    if not step_costs:
        return {}
    out: dict[tuple[int, int], float] = {}
    for raw_k, raw_v in step_costs.items():
        if isinstance(raw_k, (tuple, list)) and len(raw_k) >= 2:
            out[(int(raw_k[0]), int(raw_k[1]))] = max(1.0, float(raw_v))
    return out


def compute_geodesic_field(
    grid: Any,
    goal_cells: Sequence[Sequence[int]],
    *,
    impassable_cells: Sequence[Sequence[int]] = (),
    step_costs: Mapping[Any, float] | None = None,
    start_cell: Sequence[int] | None = None,
    step_size: int = 1,
) -> dict[str, Any]:
    """Compute multi-source Dijkstra distance field from `goal_cells` across a 2D lattice."""
    h, w = _to_shape(grid)
    stride = max(1, int(step_size))
    goals = _cell_set(goal_cells, h, w)
    blocked = _cell_set(impassable_cells, h, w) - goals
    costs = _coerce_cost_map(step_costs)

    dist = np.full((h, w), math.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for gr, gc in sorted(goals):
        dist[gr, gc] = 0.0
        heapq.heappush(heap, (0.0, gr, gc))

    deltas = tuple((dr * stride, dc * stride) for _, (dr, dc) in CARDINAL_DELTAS)
    while heap:
        cur_d, r, c = heapq.heappop(heap)
        if cur_d > dist[r, c]:
            continue
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in blocked:
                continue
            edge_cost = costs.get((r, c), 1.0)
            nd = cur_d + edge_cost
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))

    finite_mask = np.isfinite(dist)
    start_dist = math.inf
    best_step: list[int] | None = None
    nearest_goal: list[int] | None = None

    if start_cell is not None and len(start_cell) >= 2:
        sr, sc = int(start_cell[0]), int(start_cell[1])
        if 0 <= sr < h and 0 <= sc < w:
            start_dist = float(dist[sr, sc])
            best_neighbor_d = start_dist
            for dr, dc in deltas:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blocked:
                    cand_d = float(dist[nr, nc])
                    if cand_d < best_neighbor_d:
                        best_neighbor_d = cand_d
                        best_step = [dr, dc]
            if goals:
                g_near = min(goals, key=lambda g: abs(g[0] - sr) + abs(g[1] - sc))
                nearest_goal = [g_near[0], g_near[1]]
    elif goals:
        g0 = min(goals)
        nearest_goal = [g0[0], g0[1]]

    return {
        "grid_shape": [h, w],
        "reachable_count": int(np.count_nonzero(finite_mask)),
        "start_distance": float(start_dist),
        "nearest_goal": nearest_goal,
        "best_step": best_step,
        "distances": dist.tolist(),
    }


def compute_bellman_risk_backup(
    grid_shape: Sequence[int],
    goal_cells: Sequence[Sequence[int]],
    *,
    lethal_cells: Sequence[Sequence[int]] = (),
    blocked_cells: Sequence[Sequence[int]] = (),
    transition_noise: float = 0.0,
    gamma: float = 0.95,
    goal_reward: float = 10.0,
    step_penalty: float = 0.1,
    risk_penalty: float = 50.0,
    horizon: int = 24,
    start_cell: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run finite-horizon risk-sensitive Bellman value iteration with stochastic slip modeling."""
    h, w = max(1, int(grid_shape[0])), max(1, int(grid_shape[1]))
    goals = _cell_set(goal_cells, h, w)
    lethals = _cell_set(lethal_cells, h, w)
    blocked = _cell_set(blocked_cells, h, w) - goals

    slip_total = max(0.0, min(0.75, float(transition_noise)))
    p_intended = 1.0 - slip_total
    p_other = slip_total / 3.0 if slip_total > 0.0 else 0.0
    disc = max(0.0, min(0.999, float(gamma)))

    values = np.zeros((h, w), dtype=np.float64)
    terminal_mask = np.zeros((h, w), dtype=bool)
    for gr, gc in goals:
        values[gr, gc] = float(goal_reward)
        terminal_mask[gr, gc] = True
    for lr, lc in lethals:
        values[lr, lc] = -float(risk_penalty)
        terminal_mask[lr, lc] = True

    def _q_cell(val_grid: np.ndarray, r: int, c: int, action_idx: int) -> float:
        q = 0.0
        for cand_idx, (_, (dr, dc)) in enumerate(CARDINAL_DELTAS):
            prob = p_intended if cand_idx == action_idx else p_other
            if prob <= 0.0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in blocked:
                nr, nc = r, c
            q += prob * (-float(step_penalty) + disc * float(val_grid[nr, nc]))
        return q

    for _ in range(max(1, int(horizon))):
        nxt = values.copy()
        for r in range(h):
            for c in range(w):
                if terminal_mask[r, c] or (r, c) in blocked:
                    continue
                nxt[r, c] = max(_q_cell(values, r, c, idx) for idx in range(len(CARDINAL_DELTAS)))
        values = nxt

    start_val = 0.0
    action_qs: dict[str, float] = {}
    best_delta: list[int] | None = None
    safe_actions: list[list[int]] = []

    if start_cell is not None and len(start_cell) >= 2:
        sr, sc = int(start_cell[0]), int(start_cell[1])
        if 0 <= sr < h and 0 <= sc < w:
            start_val = float(values[sr, sc])
            best_q = -math.inf
            for idx, (aname, (dr, dc)) in enumerate(CARDINAL_DELTAS):
                q_val = _q_cell(values, sr, sc, idx)
                action_qs[aname] = float(q_val)
                nr, nc = sr + dr, sc + dc
                in_bounds = 0 <= nr < h and 0 <= nc < w
                is_open = in_bounds and (nr, nc) not in blocked and (nr, nc) not in lethals
                if is_open and q_val > -float(risk_penalty) * 0.5:
                    safe_actions.append([dr, dc])
                if is_open and q_val > best_q:
                    best_q = q_val
                    best_delta = [dr, dc]

    return {
        "values": values.tolist(),
        "start_value": start_val,
        "action_q_values": action_qs,
        "best_delta": best_delta,
        "safe_actions": safe_actions,
    }


def compute_collision_cone(
    grid_shape: Sequence[int],
    avatar_trajectory: Sequence[Sequence[int]],
    threat_states: Sequence[Mapping[str, Any]],
    *,
    blocked_cells: Sequence[Sequence[int]] = (),
    step_size: int = 1,
) -> dict[str, Any]:
    """Simulate patrol/threat motion forward along an avatar trajectory with edge-swap detection."""
    h, w = max(1, int(grid_shape[0])), max(1, int(grid_shape[1]))
    stride = max(1, int(step_size))
    half_step = max(1, stride // 2)
    walls = _cell_set(blocked_cells, h, w)

    def _step_blocked(pos: tuple[int, int], facing: tuple[int, int]) -> bool:
        probe = (pos[0] + facing[0] * half_step, pos[1] + facing[1] * half_step)
        dest = (pos[0] + facing[0] * stride, pos[1] + facing[1] * stride)
        return any(not (0 <= q[0] < h and 0 <= q[1] < w) or q in walls for q in (probe, dest))

    sim_threats: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for t in threat_states:
        pos = t.get("pos") or t.get("position")
        facing = t.get("facing") or (0, 0)
        if (
            isinstance(pos, Sequence)
            and len(pos) >= 2
            and isinstance(facing, Sequence)
            and len(facing) >= 2
        ):
            sim_threats.add(((int(pos[0]), int(pos[1])), (int(facing[0]), int(facing[1]))))

    if not sim_threats or not avatar_trajectory:
        return {
            "is_safe": True,
            "first_collision_step": None,
            "collision_coord": None,
            "collision_cells": [],
            "recommended_wait_steps": 0,
        }

    traj = [(int(pt[0]), int(pt[1])) for pt in avatar_trajectory if len(pt) >= 2]
    start_av = traj[0]
    future_av = traj[1:] if len(traj) > 1 else traj

    first_collision_step: int | None = None
    collision_coord: list[int] | None = None
    collision_cells: list[list[int]] = []
    cur_av: tuple[int, int] = start_av

    for step_idx, avatar_dest in enumerate(future_av):
        next_threats: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        hit_at: tuple[int, int] | None = None
        for pat_pos, pat_facing in sim_threats:
            if pat_facing == (0, 0):
                next_threats.add((pat_pos, pat_facing))
                if pat_pos == avatar_dest:
                    hit_at = avatar_dest
                continue
            dest = (pat_pos[0] + pat_facing[0] * stride, pat_pos[1] + pat_facing[1] * stride)
            if not (0 <= dest[0] < h and 0 <= dest[1] < w) or dest in walls:
                rev_facing = (-pat_facing[0], -pat_facing[1])
                next_threats.add((pat_pos, rev_facing))
                if pat_pos == avatar_dest:
                    hit_at = avatar_dest
                continue
            if dest == avatar_dest or (pat_pos == avatar_dest and dest == cur_av):
                hit_at = avatar_dest
            next_f = (
                (-pat_facing[0], -pat_facing[1]) if _step_blocked(dest, pat_facing) else pat_facing
            )
            next_threats.add((dest, next_f))
            collision_cells.append([dest[0], dest[1]])

        if hit_at is not None and first_collision_step is None:
            first_collision_step = step_idx
            collision_coord = [hit_at[0], hit_at[1]]
            break
        sim_threats = next_threats
        cur_av = avatar_dest

    is_safe = first_collision_step is None
    return {
        "is_safe": is_safe,
        "first_collision_step": first_collision_step,
        "collision_coord": collision_coord,
        "collision_cells": collision_cells,
        "recommended_wait_steps": 1 if not is_safe else 0,
    }


__all__ = [
    "CARDINAL_DELTAS",
    "compute_bayesian_posterior_update",
    "compute_bellman_risk_backup",
    "compute_collision_cone",
    "compute_geodesic_field",
]
