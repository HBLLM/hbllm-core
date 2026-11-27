"""Entity Risk Estimation & Stance Priority Scoring."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hbllm.hcir.world.objects.ledger import ObjectStateLedger

logger = logging.getLogger(__name__)


def _extract_structural_signature(
    cells: set[tuple[int, int]],
    feature_id: int,
    curr_grid: np.ndarray | None,
    bg: int,
) -> dict[str, float]:
    """Compute color-invariant topological/geometric signature for analogical risk matching."""
    area = float(max(1, len(cells)))
    if not cells:
        return {
            "aspect_ratio": 1.0,
            "norm_size": math.log1p(1.0),
            "open_deg4": 1.0,
            "corner_pin_risk": 0.0,
        }
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    h_span = float(max(rs) - min(rs) + 1)
    w_span = float(max(cs) - min(cs) + 1)
    aspect_ratio = w_span / max(1.0, h_span)
    open_deg4 = 1.0
    corner_pin_risk = 0.0
    high_freq_adj4 = 0.0
    if curr_grid is not None and curr_grid.ndim == 2:
        H, W = curr_grid.shape
        r0 = int(round(sum(rs) / area))
        c0 = int(round(sum(cs) / area))
        if 0 <= r0 < H and 0 <= c0 < W:
            vals, counts = np.unique(curr_grid, return_counts=True)
            counts_map = {
                int(v): int(cnt) for v, cnt in zip(vals.tolist(), counts.tolist(), strict=False)
            }
            high_thresh = max(H, W)
            n_open = 0
            n_hf = 0
            barrier_adj = [False, False, False, False]
            for idx, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < H and 0 <= nc < W:
                    nv = int(curr_grid[nr, nc])
                    if nv == bg:
                        n_open += 1
                    elif nv != feature_id and counts_map.get(nv, 1) >= high_thresh:
                        n_hf += 1
                        barrier_adj[idx] = True
                else:
                    n_hf += 1
                    barrier_adj[idx] = True
            open_deg4 = n_open / 4.0
            high_freq_adj4 = n_hf / 4.0
            if (barrier_adj[0] or barrier_adj[1]) and (barrier_adj[2] or barrier_adj[3]):
                corner_pin_risk = 1.0
    return {
        "aspect_ratio": aspect_ratio,
        "norm_size": math.log1p(area),
        "open_deg4": open_deg4,
        "high_freq_adj4": high_freq_adj4,
        "corner_pin_risk": corner_pin_risk,
    }


def _analogous_concept_hazard_risk(
    engine: Any,
    feature_id: int,
    cells: set[tuple[int, int]],
    curr_grid: np.ndarray | None,
    bg: int,
) -> float:
    """Estimate hazard risk via structural analogy to role-annotated GroundedConcepts."""
    store = engine.working_concept_store
    concepts = store.get_all_concepts()
    if not concepts:
        return 0.0
    role_concepts = [
        c
        for c in concepts
        if any(aff in ("role:hazard", "role:goal", "role:actuator") for aff in c.affordances)
        and c.feature_distributions
    ]
    if not role_concepts:
        return 0.0

    sig = _extract_structural_signature(cells, feature_id, curr_grid, bg)
    # Match structural indices corresponding to geometric/topological dimensions (dim_0..dim_9)
    sig_dims = {
        "dim_0": sig["aspect_ratio"],
        "dim_3": sig["norm_size"],
        "dim_4": sig["open_deg4"],
        "dim_7": sig["corner_pin_risk"],
        "dim_8": sig["high_freq_adj4"],
    }
    best_haz_sim = 0.0
    best_safe_sim = 0.0
    for concept in role_concepts:
        sq_dist = 0.0
        n_dims = 0
        for dim_k, val in sig_dims.items():
            stats = concept.feature_distributions.get(dim_k)
            if stats and "mean" in stats:
                mean = float(stats["mean"])
                var = max(float(stats.get("var", 0.25)), 0.05)
                sq_dist += ((val - mean) ** 2) / var
                n_dims += 1
        if n_dims == 0:
            continue
        sim = math.exp(-sq_dist / max(1.0, float(n_dims)))
        if "role:hazard" in concept.affordances:
            best_haz_sim = max(best_haz_sim, sim)
        if "role:goal" in concept.affordances or "role:actuator" in concept.affordances:
            best_safe_sim = max(best_safe_sim, sim)

    if best_haz_sim > best_safe_sim and best_haz_sim >= 0.55:
        return round(min(0.9, best_haz_sim * 0.85), 4)
    return 0.0


def estimate_entity_risk(
    engine: Any,
    feature_id: int,
    cells: set[tuple[int, int]],
    curr_grid: np.ndarray | None,
    bg: int,
    ledger: ObjectStateLedger | None = None,
) -> float:
    """Domain-general risk estimate in [0.0, 1.0] for an entity or cell feature.

    Combines: (1) empirical lethal/threat evidence, (2) grounded concept role priors
    and analogous hazard concepts, (3) in-context inert/quiescent sibling evidence,
    and (4) baseline uncertainty for unverified features.
    """
    th = engine.symbolic_theory
    hz = engine.hazard_tracker
    if (
        feature_id in hz.known_lethal_features
        or feature_id in engine.mobile_threat_features
        or feature_id in engine.episodic_cortex.discovered_lethal_features
        or feature_id in th.context_verified_features("lethal")
    ):
        return 1.0
    if any(c in engine.current_context_hazards for c in cells):
        return 1.0

    # Confirmed safe roles
    if (
        th.is_cell_feasible(feature_id, bg_feature=bg)
        or feature_id in th.goal_features
        or th.is_avatar_feature(feature_id)
        or any(c in engine.learned_goal_positions for c in cells)
    ):
        return 0.0

    risk = 0.0
    # 1. Explicit grounded concept role priors (color-agnostic learned/SNN or SGE bridge)
    role_priors: dict[int, dict[str, float]] = engine.working_memory.concept_role_priors
    rp = role_priors.get(int(feature_id))
    if rp:
        haz_p = float(rp.get("hazard", 0.0))
        safe_p = max(
            float(rp.get("goal", 0.0)),
            float(rp.get("actuator", 0.0)),
            float(rp.get("cargo", 0.0)),
            float(rp.get("walkable", 0.0)),
        )
        if haz_p > safe_p:
            risk = max(risk, min(0.95, haz_p))

    # 2. Structural analogy against stored role-annotated GroundedConcepts
    analog_risk = _analogous_concept_hazard_risk(engine, feature_id, cells, curr_grid, bg)
    if analog_risk > 0.0:
        risk = max(risk, analog_risk)

    # 3. In-context sibling inert / quiescent evidence
    if ledger is not None:
        if feature_id in ledger.interacted_features and feature_id not in ledger.effective_features:
            risk = max(risk, 0.45)
    if feature_id in engine.working_memory.quiescent_features:
        risk = max(risk, 0.55)

    # 4. Baseline unverified uncertainty
    if engine.is_feature_unverified(feature_id, bg):
        risk = max(risk, 0.15)
    if ledger is not None and ledger.is_in_newly_reachable(cells):
        risk *= 0.35
    return float(min(1.0, max(0.0, risk)))


def compute_stance_priority(
    engine: Any,
    curr_grid: np.ndarray,
    H: int,
    W: int,
    bg: int,
    avatar_pos: tuple[int, int],
    p: tuple[int, int],
    satisfied_when: dict[tuple[int, int], dict[str, Any]],
    newly_unlocked: set[tuple[int, int]],
    role_scores_fn: Any,
) -> tuple[int, float, int]:
    """Compute epistemic priority tuple ``(tier, secondary_key, distance)`` for a goal stance.

    Ensures unlocked compartment goals and confirmed goals rank ahead of exploration,
    safe adjacent reversible probes rank ahead of unverified contact escalation, and
    lower-risk entities always precede higher-risk analogous or inert candidates.
    """
    th = engine.symbolic_theory
    ledger = engine.object_planner.ledger
    dist = abs(p[0] - avatar_pos[0]) + abs(p[1] - avatar_pos[1])
    if p == avatar_pos and p in satisfied_when and "entity_pos" in satisfied_when[p]:
        return (0, 0.0, 0)

    p_feat_cell = int(curr_grid[p[0], p[1]])
    kb_ev = th.evaluate_transition(p_feat_cell, locus=p)
    if kb_ev.risk_cost >= 40.0 and p_feat_cell not in th.goal_features:
        return (8, float(kb_ev.risk_cost), dist)
    meta = satisfied_when.get(p)
    is_inert_unverified_contact = (
        (meta is None or "entity_pos" not in meta)
        and p_feat_cell != bg
        and kb_ev.traversable
        and p_feat_cell not in th.goal_features
        and p not in engine.learned_goal_positions
        and engine.is_feature_unverified(p_feat_cell, bg)
        and any(
            0 <= p[0] + dr < H
            and 0 <= p[1] + dc < W
            and th.is_locus_actuator_probed((p[0] + dr, p[1] + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
    )

    coupling_offsets = engine.get_effector_coupling_offsets()

    def _walk_deg(pt: tuple[int, int]) -> int:
        return sum(
            1
            for dr, dc in coupling_offsets
            if 0 <= pt[0] + dr < H
            and 0 <= pt[1] + dc < W
            and (
                (pt[0] + dr, pt[1] + dc) == avatar_pos
                or th.is_cell_feasible(
                    int(curr_grid[pt[0] + dr, pt[1] + dc]),
                    locus=(pt[0] + dr, pt[1] + dc),
                    bg_feature=bg,
                    learned_barriers=engine.learned_barriers,
                    hazard_loci=engine.current_context_hazards,
                )
            )
        )

    has_inventory_or_tools = bool(
        engine.working_memory.collected_inventory_features
        or engine.working_memory.body_schema.held_tools
    )

    if newly_unlocked and (
        p in newly_unlocked
        or any(
            (p[0] + dr, p[1] + dc) in newly_unlocked
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
    ):
        target_feat = (
            int(meta["target_feature"])
            if (meta and meta.get("target_feature") is not None)
            else int(meta["feature"])
            if (meta and meta.get("feature") is not None)
            else p_feat_cell
        )
        cnt_t = int(np.count_nonzero(curr_grid == target_feat)) if target_feat != bg else 1
        rarity_pen = 0.25 if cnt_t > 1 else 0.0
        est_r = estimate_entity_risk(engine, target_feat, {p}, curr_grid, bg, ledger=ledger)
        gate_pt = meta["entity_pos"] if (meta and "entity_pos" in meta) else p
        is_unlocked_gate = (
            cnt_t == 1
            and 2 <= gate_pt[0] <= H - 3
            and 2 <= gate_pt[1] <= W - 3
            and _walk_deg(gate_pt) <= 2
        )
        if is_unlocked_gate and (
            engine.working_memory.pending_gate_check or has_inventory_or_tools
        ):
            return (1, round(rarity_pen + est_r * 0.25 - 0.30, 3), dist)
        inert_bump = 0.05 if is_inert_unverified_contact else 0.0
        return (1, round(rarity_pen + inert_bump + est_r * 0.25, 3), dist)

    if meta and "entity_pos" in meta:
        ep = meta["entity_pos"]
        raw_ef = meta.get("feature")
        ef = int(raw_ef) if raw_ef is not None else -1
        pref_score, barrier_score, hazard_score = role_scores_fn(ef) if ef >= 0 else (0.0, 0.0, 0.0)
        if ef >= 0:
            est_r = estimate_entity_risk(engine, ef, {ep}, curr_grid, bg, ledger=ledger)
            if (hazard_score >= 0.65 or est_r >= 0.60) and pref_score < 0.45:
                return (8, max(hazard_score, est_r), dist)
            if barrier_score >= 0.72 and pref_score < 0.40:
                return (7, barrier_score, dist)

            walk_n = _walk_deg(ep)
            is_interior_switch = walk_n == 4
            is_dividing_wall_gate = 2 <= ep[0] <= H - 3 and 2 <= ep[1] <= W - 3 and walk_n <= 2
            cnt = int(np.count_nonzero(curr_grid == ef))
            if cnt == 2 and is_interior_switch:
                locs = [(int(r), int(c)) for r, c in zip(*np.where(curr_grid == ef), strict=False)]
                other = locs[1] if locs[0] == ep else locs[0]
                u_vals, u_cnts = np.unique(curr_grid, return_counts=True)
                n_singletons = sum(
                    1
                    for uv, uc in zip(u_vals.tolist(), u_cnts.tolist(), strict=False)
                    if int(uc) == 1
                    and int(uv) != bg
                    and not th.is_avatar_feature(int(uv))
                    and not th.is_barrier(int(uv))
                )
                if _walk_deg(other) < 4 and n_singletons >= 2:
                    th.induce_relational_cue_pair(ef, locs)
                    return (2, -round(pref_score - est_r, 3), dist)
            if (
                cnt == 1
                and is_dividing_wall_gate
                and (engine.working_memory.pending_gate_check or has_inventory_or_tools)
            ):
                return (2, -round(pref_score - est_r, 3) - 0.35, dist)
            if cnt == 1 and not is_dividing_wall_gate:
                return (3, -round(pref_score - est_r, 3), dist)
            if pref_score >= 0.55:
                return (4, -round(pref_score - est_r, 3), dist)
            return (5 if cnt == 1 else 6, -round(pref_score - est_r, 3), dist)

    p_feat = int(curr_grid[p[0], p[1]])
    raw_tf = meta.get("target_feature") if meta else None
    target_feat = int(raw_tf) if raw_tf is not None else p_feat
    pref_score, barrier_score, hazard_score = (
        role_scores_fn(target_feat) if target_feat != bg else (0.0, 0.0, 0.0)
    )
    est_risk = (
        estimate_entity_risk(engine, target_feat, {p}, curr_grid, bg, ledger=ledger)
        if target_feat != bg
        else 0.0
    )
    p_ev = th.evaluate_transition(p_feat, locus=p)
    t_ev = th.evaluate_transition(target_feat, locus=p)
    if (
        (hazard_score >= 0.65 or est_risk >= 0.60 or t_ev.risk_cost >= 40.0)
        and pref_score < 0.45
        and target_feat not in th.goal_features
    ):
        return (8, max(hazard_score, est_risk), dist)
    if barrier_score >= 0.72 and pref_score < 0.40 and target_feat not in th.goal_features:
        return (8, barrier_score, dist)
    p_cnt = int(np.count_nonzero(curr_grid == p_feat)) if p_feat != bg else 0
    if p_ev.goal_value > 0.5 or p_feat in th.goal_features or p in engine.learned_goal_positions:
        return (1, -round(pref_score, 3), dist)
    if (
        p_feat != bg
        and p_ev.traversable
        and engine.is_feature_unverified(p_feat, bg)
        and any(
            0 <= p[0] + dr < H
            and 0 <= p[1] + dc < W
            and th.is_locus_actuator_probed((p[0] + dr, p[1] + dc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )
    ):
        if p_cnt == 1:
            is_gate = 2 <= p[0] <= H - 3 and 2 <= p[1] <= W - 3 and _walk_deg(p) <= 2
            if is_gate and (engine.working_memory.pending_gate_check or has_inventory_or_tools):
                return (2, round(est_risk - pref_score, 3) - 0.25, dist)
            return (5, round(1.0 + est_risk - pref_score, 3), dist)
        return (8, round(est_risk, 3), dist)
    if (
        p_cnt == 1
        and p_feat != bg
        and p_ev.traversable
        and (engine.working_memory.pending_gate_check or has_inventory_or_tools)
        and 2 <= p[0] <= H - 3
        and 2 <= p[1] <= W - 3
        and _walk_deg(p) <= 2
    ):
        return (2, round(est_risk - pref_score, 3) - 0.25, dist)
    if pref_score >= 0.55:
        return (4, -round(pref_score - est_risk, 3), dist)
    return (5 if p_cnt == 1 else 6, -round(pref_score - est_risk, 3), dist)


def compute_slip_adjacency_penalty(
    engine: Any,
    curr_grid: np.ndarray,
    nr: int,
    nc: int,
    tr: int,
    tc: int,
    bg: int,
    H: int,
    W: int,
    rp_map: dict[int, dict[str, float]] | None,
) -> float:
    """Return dynamic collision penalty when stepping next to lethal/unverified cells under motor slip."""
    if not any(d.slip_rate > 0.05 for d in engine.action_dynamics.values()):
        return 0.0
    for dr_s, dc_s in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        sr_s, sc_s = nr + dr_s, nc + dc_s
        if 0 <= sr_s < H and 0 <= sc_s < W:
            sv_s = int(curr_grid[sr_s, sc_s])
            if (
                sv_s in engine.hazard_tracker.known_lethal_features
                or (sr_s, sc_s) in engine.hazard_tracker.static_lethal_positions
                or (rp_map.get(sv_s, {}).get("hazard", 0.0) >= 0.65 if rp_map else False)
            ):
                return 80.0
            if (
                sv_s != bg
                and sv_s not in engine.symbolic_theory.barrier_features
                and sv_s not in engine.symbolic_theory.goal_features
                and engine.is_feature_unverified(sv_s, bg)
                and (sr_s, sc_s) != (tr, tc)
            ):
                return 55.0
    return 0.0
