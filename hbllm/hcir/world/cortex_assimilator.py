"""Epistemic Feedback Assimilator Faculty.

Assimilates sensory feedback (pixel diffs, kinetic flow, reward signals)
into motor dynamics, avatar identity consensus, and empirical neuro-symbolic theory.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.cortex_causal import ActionAffordance
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.spatiotemporal_tracker import MorphologicalEntity

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class EpistemicPhase(StrEnum):
    """Cognitive lifecycle phases for autonomous exploration and problem solving."""

    MOTOR_GROUNDING = "motor_grounding"  # Self-identification & basic action calibration
    EPISTEMIC_EXPLORATION = (
        "epistemic_exploration"  # Trial-and-error hypothesis testing on unknown objects
    )
    MENTAL_SIMULATION = "mental_simulation"  # Forward planning in mind
    EXPLOITATION = "exploitation"  # Direct execution of verified solution
    REPLANNING = "replanning"  # Recovering from unexpected contradiction


class EpistemicFeedbackAssimilator:
    """Assimilates environmental feedback into motor dynamics and neuro-symbolic theory."""

    @staticmethod
    def _record_safe_traversal(
        engine: AutonomousEpistemicEngine,
        landing_cells: list[tuple[int, int]],
        delta: tuple[int, int],
        H: int,
        W: int,
    ) -> None:
        """Mark features on the swept trajectory of a survived displacement as safe.

        For each landing cell, walk back along the displacement vector to the
        origin. Every cell crossed (in the pre-move frame) was traversed without
        dying, which is direct experiential evidence of safety.
        """
        if engine.prev_grid is None:
            return
        dr, dc = delta
        steps = max(abs(dr), abs(dc))
        sr = (dr > 0) - (dr < 0)
        sc = (dc > 0) - (dc < 0)
        av = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        lethal = engine.hazard_tracker.known_lethal_features
        for cr, cc in landing_cells:
            for k in range(0, max(steps, 1)):
                r, c = cr - sr * k, cc - sc * k
                if not (0 <= r < H and 0 <= c < W):
                    continue
                wf = int(engine.prev_grid[r, c])
                if wf in av or wf in lethal:
                    continue
                engine.symbolic_theory.induce_walkable(wf)
                engine.verified_safe_features.add(wf)

    @staticmethod
    def _commit_avatar_identity(
        engine: AutonomousEpistemicEngine,
        pairs: list[tuple[SpatialEntity, SpatialEntity]],
        delta: tuple[int, int],
        action: int | None,
        H: int,
        W: int,
        calibrate_dynamics: bool = True,
    ) -> None:
        """Commit an avatar identity from entity pairs sharing a displacement delta.

        This is the shared commitment step used by both fresh-start (single-frame
        heuristic) and controllability-based (multi-frame correlation) identification.
        """
        dr, dc = delta

        def _adjacent(a: SpatialEntity, b: SpatialEntity) -> bool:
            b1, b2 = a.bounding_box, b.bounding_box
            r_gap = max(0, b1[0] - b2[1] - 1, b2[0] - b1[1] - 1)
            c_gap = max(0, b1[2] - b2[3] - 1, b2[2] - b1[3] - 1)
            return r_gap <= 1 and c_gap <= 1

        comp_pairs: list[tuple[SpatialEntity, SpatialEntity]] = [pairs[0]]
        for pe, ce in pairs[1:]:
            if any(_adjacent(ce, c_ce) for _, c_ce in comp_pairs):
                comp_pairs.append((pe, ce))

        engine.avatar_features = {ce.feature_id for _, ce in comp_pairs}
        engine.avatar_feature = next(iter(engine.avatar_features))
        engine.avatar_size = sum(ce.area for _, ce in comp_pairs)
        all_cells = [
            cell for _, ce in comp_pairs for cell in ce.properties.get("cells", [ce.grid_pos])
        ]
        engine.avatar_pos = (
            int(round(sum(c[0] for c in all_cells) / len(all_cells))),
            int(round(sum(c[1] for c in all_cells) / len(all_cells))),
        )
        # Features the avatar just traversed are verified safe
        EpistemicFeedbackAssimilator._record_safe_traversal(engine, all_cells, delta, H, W)
        if hasattr(engine, "entity_action_failed_positions"):
            engine.entity_action_failed_positions.clear()
        if hasattr(engine, "immobile_entity_actions"):
            engine.immobile_entity_actions.clear()

        if calibrate_dynamics and action is not None:
            engine.action_dynamics[action] = ActionDynamicsModel(
                action_id=action,
                delta_r=dr,
                delta_c=dc,
                confidence=0.7,
                probes_tested=1,
            )
            if action in engine.action_affordances:
                engine.action_affordances[action].is_displacement = dr != 0 or dc != 0
                engine.action_affordances[action].delta = (dr, dc)
            else:
                engine.action_affordances[action] = ActionAffordance(
                    action_id=action,
                    is_displacement=(dr != 0 or dc != 0),
                    delta=(dr, dc),
                )

        logger.info(
            "AutonomousEpistemicEngine: Avatar identified (feats=%s, size=%d). "
            "Action %s -> delta=(%d, %d), calibrate=%s",
            engine.avatar_features,
            engine.avatar_size,
            action,
            dr,
            dc,
            calibrate_dynamics,
        )

    @staticmethod
    def assimilate(
        engine: AutonomousEpistemicEngine,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action: Any | None = None,
        action_data: dict[str, Any] | None = None,
    ) -> None:
        """Assimilate sensory feedback from previous action into empirical world models."""
        if action is not None:
            engine.last_action = action
            if not hasattr(engine, "recent_actions"):
                engine.recent_actions = deque(maxlen=16)
            engine.recent_actions.append(action)
        if action_data is not None:
            engine.last_action_data = action_data

        if engine.prev_grid is None or engine.last_action is None:
            return

        engine._feedback_assimilated = True
        prev_avatar_pos = engine.avatar_pos

        curr_grid = engine.normalize_sensory_input(curr_grid)
        H, W = curr_grid.shape

        if not is_win:
            walkable_feats = (
                engine.symbolic_theory.walkable_features | engine.verified_safe_features
            )
            engine.hazard_tracker.record_frame(
                engine.step_counter,
                curr_grid,
                background_feature=engine.bg_feature,
                avatar_features=engine.avatar_features,
                walkable_features=walkable_feats,
            )
            # Faculty E: Basal Ganglia & Cerebellar Predictive Phase Entrainment
            engine.cerebellar_clock.record_step(engine.step_counter, curr_grid, is_reset=is_lost)
            if engine.cerebellar_clock.detected_macro_period is not None:
                engine.hazard_tracker.environmental_period = max(
                    engine.hazard_tracker.environmental_period,
                    engine.cerebellar_clock.detected_macro_period,
                )

        diff = engine.compute_frame_diff(engine.prev_grid, curr_grid)
        action = engine.last_action
        action_data = engine.last_action_data

        # Faculty: Anterior Mid-Cingulate & Habenula Episodic IOR Transition Tracing
        engine.working_memory.habenular_ior.record_step(
            step=engine.step_counter,
            avatar_pos=prev_avatar_pos,
            action=action,
            action_data=action_data,
        )

        aff = engine.action_affordances.get(action)
        if aff is not None:
            is_effector_action = aff.requires_spatial_target
        else:
            is_effector_action = bool(
                action is not None
                and (
                    engine.is_spatial_effector(action)
                    or (
                        action_data
                        and isinstance(action_data, dict)
                        and any(k in action_data for k in ("x", "y", "col", "row"))
                    )
                )
            )

        is_focus_switch = engine.is_focus_switch_action(action)
        if is_effector_action or is_focus_switch:
            if hasattr(engine, "entity_action_failed_positions"):
                engine.entity_action_failed_positions.clear()
            if hasattr(engine, "immobile_entity_actions"):
                engine.immobile_entity_actions.clear()
            if is_focus_switch:
                engine.avatar_features.clear()
                engine.avatar_feature = None
                engine.avatar_pos = None
                engine.mental_plan.clear()

        if is_effector_action and action not in engine.action_affordances:
            engine.action_affordances[action] = ActionAffordance(
                action_id=action,
                requires_spatial_target=True,
            )

        act_dyn = engine.action_dynamics.get(action) if not is_effector_action else None
        is_known_displacement = bool(
            not is_effector_action and act_dyn is not None and act_dyn.is_displacement_action()
        )

        target_coord: tuple[int, int] | None = None
        if action_data and isinstance(action_data, dict):
            tc = action_data.get("x", action_data.get("col", action_data.get("c")))
            tr = action_data.get("y", action_data.get("row", action_data.get("r")))
            if tc is not None and tr is not None:
                try:
                    target_coord = (int(tr), int(tc))
                except (ValueError, TypeError):
                    pass

        if diff.changed_pixel_count == 0:
            engine.consecutive_quiescent_actions += 1
            engine.consecutive_effective_clicks = 0
            engine.last_effective_click_coord = None
            engine.active_goal_converging_coord = None
            engine.consecutive_goal_converging_clicks = 0
            if target_coord is not None:
                engine.quiescent_click_targets.add(target_coord)
                engine.effective_click_targets.discard(target_coord)
                tr, tc = target_coord
                if 0 <= tr < H and 0 <= tc < W:
                    engine.working_memory.visuospatial.record_probe(
                        pos=target_coord,
                        feature_id=int(curr_grid[tr, tc]),
                        step=engine.step_counter,
                    )
            if action is not None:
                prev_inh = engine.inhibited_actions.get(action, 0)
                engine.inhibited_actions[action] = max(prev_inh + 2, 3)

            if is_effector_action and not is_win and not is_lost:
                return
        else:
            engine.consecutive_quiescent_actions = 0
            if target_coord is not None:
                tr, tc = target_coord
                if 0 <= tr < H and 0 <= tc < W:
                    engine.working_memory.visuospatial.record_probe(
                        pos=target_coord,
                        feature_id=int(curr_grid[tr, tc]),
                        step=engine.step_counter,
                    )
                if getattr(engine, "last_effective_click_coord", None) == target_coord:
                    engine.consecutive_effective_clicks = (
                        getattr(engine, "consecutive_effective_clicks", 0) + 1
                    )
                else:
                    engine.last_effective_click_coord = target_coord
                    engine.consecutive_effective_clicks = 1
                engine.effective_click_targets.add(target_coord)
                engine.quiescent_click_targets.discard(target_coord)
                if engine.prev_grid is not None:
                    r, c = target_coord
                    if 0 <= r < engine.prev_grid.shape[0] and 0 <= c < engine.prev_grid.shape[1]:
                        target_feat = int(engine.prev_grid[r, c])
                        engine.effective_features.add(target_feat)
                if diff.changed_mask is not None:
                    mut_cells = [(int(r), int(c)) for r, c in zip(*np.where(diff.changed_mask))]
                    engine.click_affordances[target_coord] = mut_cells
                    engine.working_memory.visuospatial.record_diff_stencil(
                        target_coord, set(mut_cells)
                    )

                if is_effector_action and diff.changed_pixel_count > 0:
                    engine.avatar_pos = target_coord
                    clicked_feats = {
                        int(f)
                        for f in np.unique(
                            curr_grid[
                                max(0, tr - 2) : min(H, tr + 3), max(0, tc - 2) : min(W, tc + 3)
                            ]
                        )
                        if int(f) != engine.bg_feature
                    }
                    if clicked_feats:
                        engine.avatar_features = clicked_feats
                        engine.avatar_feature = next(iter(clicked_feats))
                        from scipy.ndimage import label

                        mask_c = np.isin(curr_grid, list(clicked_feats))
                        lab_c, _ = label(mask_c)
                        if 0 <= tr < H and 0 <= tc < W and lab_c[tr, tc] > 0:
                            comp_c = np.argwhere(lab_c == lab_c[tr, tc])
                            engine.avatar_size = len(comp_c)
                            engine.avatar_pos = (
                                int(round(float(np.mean(comp_c[:, 0])))),
                                int(round(float(np.mean(comp_c[:, 1])))),
                            )
                    engine.mental_plan.clear()
                    engine.phase = EpistemicPhase.MENTAL_SIMULATION

                # Numerical / cardinality constraint discovery (e.g. Minesweeper / local count grids):
                tr, tc = target_coord
                if 0 <= tr < curr_grid.shape[0] and 0 <= tc < curr_grid.shape[1]:
                    new_feat = int(curr_grid[tr, tc])
                    if 0 <= new_feat <= 8 and new_feat != engine.bg_feature:
                        new_safe, new_hazards = (
                            engine.working_memory.register_cardinality_constraint(
                                center=target_coord,
                                count=new_feat,
                                grid_shape=curr_grid.shape,
                                radius=1,
                            )
                        )
                        for hz in new_hazards:
                            engine.hazard_tracker.register_lethal_feature(
                                int(curr_grid[hz[0], hz[1]])
                            )
                            engine.learned_barriers.add(hz)

            # Faculty: Ventromedial Prefrontal Cortex (vmPFC) Remote Causal Attribution
            loci: list[tuple[int, int]] = []
            if target_coord is not None:
                loci.append(target_coord)
            if engine.avatar_pos is not None:
                loci.append(engine.avatar_pos)
            if prev_avatar_pos is not None and prev_avatar_pos != engine.avatar_pos:
                loci.append(prev_avatar_pos)

            if engine.prev_grid is not None:
                for locus in set(loci):
                    confirmed_affs = engine.working_memory.remote_causal.record_transition(
                        prev_grid=engine.prev_grid,
                        curr_grid=curr_grid,
                        action_pos=locus,
                        background_feature=engine.bg_feature,
                        avatar_features=engine.avatar_features,
                    )
                    for aff in confirmed_affs:
                        if (
                            aff.mutated_remote_feature == engine.bg_feature
                            or engine.symbolic_theory.is_walkable(aff.mutated_remote_feature)
                        ):
                            engine.learned_barriers.discard(aff.remote_pos)
                        engine.working_memory.activated_triggers.add(aff.trigger_pos)

            # Dorsal Visual Stream (V4/MT): Object motion tracking & Teleological Distance Gradient
            if diff.changed_mask is not None and engine.prev_grid is not None:
                bg = engine.bg_feature if engine.bg_feature is not None else 0
                disappeared_pts = np.argwhere(diff.changed_mask & (engine.prev_grid != bg))
                appeared_pts = np.argwhere(diff.changed_mask & (curr_grid != bg))
                if 1 <= len(disappeared_pts) <= 36 and 1 <= len(appeared_pts) <= 36:
                    p_old = np.mean(disappeared_pts, axis=0)
                    p_new = np.mean(appeared_pts, axis=0)
                    goal_positions: list[tuple[int, int]] = list(engine.learned_goal_positions)
                    for g in PerceptionEngine.detect_structural_goals(curr_grid, bg=bg):
                        if "position" in g:
                            goal_positions.append(g["position"])
                    for gf in engine.learned_goal_features:
                        g_pts = np.argwhere(curr_grid == gf)
                        if len(g_pts) > 0:
                            goal_positions.append(
                                (int(np.mean(g_pts[:, 0])), int(np.mean(g_pts[:, 1])))
                            )
                    if goal_positions and target_coord is not None:
                        min_d_old = min(
                            abs(p_old[0] - gp[0]) + abs(p_old[1] - gp[1]) for gp in goal_positions
                        )
                        min_d_new = min(
                            abs(p_new[0] - gp[0]) + abs(p_new[1] - gp[1]) for gp in goal_positions
                        )
                        if min_d_new < min_d_old:
                            engine.active_goal_converging_coord = target_coord
                            engine.consecutive_goal_converging_clicks = (
                                getattr(engine, "consecutive_goal_converging_clicks", 0) + 1
                            )
                        elif min_d_new > min_d_old:
                            if (
                                getattr(engine, "active_goal_converging_coord", None)
                                == target_coord
                            ):
                                engine.active_goal_converging_coord = None
                                engine.consecutive_goal_converging_clicks = 0

        # ── Intuitive Physics Transition Tracking ────────────────────────────
        engine.physics_engine.record_transition(
            prev_pos=prev_avatar_pos,
            curr_pos=engine.avatar_pos,
            is_displacement_action=is_known_displacement,
            commanded_delta=act_dyn.get_displacement() if act_dyn else (0, 0),
        )

        # ── A. Motor Calibration & Proprioception ────────────────────────────
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        prev_entities = engine.extract_entities(engine.prev_grid, bg)
        curr_entities = engine.extract_entities(curr_grid, bg)

        unmatched_prev: list[SpatialEntity] = []
        unmatched_curr: dict[int, SpatialEntity] = {id(ce): ce for ce in curr_entities}

        for pe in prev_entities:
            stat_match = None
            for ce_id, ce in unmatched_curr.items():
                if (
                    pe.feature_id == ce.feature_id
                    and pe.area == ce.area
                    and pe.grid_pos == ce.grid_pos
                ):
                    stat_match = ce_id
                    break
            if stat_match is not None:
                del unmatched_curr[stat_match]
            else:
                unmatched_prev.append(pe)

        moved_entities: list[tuple[SpatialEntity, SpatialEntity, tuple[int, int]]] = []
        for pe in unmatched_prev:
            best_ce_id = None
            min_dist = float("inf")
            best_delta = (0, 0)
            for ce_id, ce in unmatched_curr.items():
                if pe.feature_id == ce.feature_id and pe.area == ce.area and pe.area <= 144:
                    dr = int(round(ce.centroid[0] - pe.centroid[0]))
                    dc = int(round(ce.centroid[1] - pe.centroid[1]))
                    dist = abs(dr) + abs(dc)
                    is_valid_displacement = (0 < dist <= 16 and (dr == 0 or dc == 0)) or (
                        0 < dist <= 8 and abs(dr) <= 4 and abs(dc) <= 4
                    )
                    if is_valid_displacement:
                        if dist < min_dist:
                            min_dist = dist
                            best_ce_id = ce_id
                            best_delta = (dr, dc)
            if best_ce_id is not None:
                matched_ce = unmatched_curr.pop(best_ce_id)
                moved_entities.append((pe, matched_ce, best_delta))

        # Faculty: Spatiotemporal Lifecycle & Directed Causal Lineage DAG (W014, W059)
        if hasattr(engine, "deformation_tracker") and engine.deformation_tracker is not None:
            morph_prev = [
                MorphologicalEntity(
                    entity_id=pe.id or f"prev_{pe.feature_id}_{pe.grid_pos[0]}_{pe.grid_pos[1]}",
                    feature_id=pe.feature_id,
                    cells=set(pe.properties.get("cells", [pe.grid_pos])),
                    centroid=pe.centroid,
                    bounding_box=pe.bounding_box,
                )
                for pe in prev_entities
            ]
            morph_curr = [
                MorphologicalEntity(
                    entity_id=ce.id or f"curr_{ce.feature_id}_{ce.grid_pos[0]}_{ce.grid_pos[1]}",
                    feature_id=ce.feature_id,
                    cells=set(ce.properties.get("cells", [ce.grid_pos])),
                    centroid=ce.centroid,
                    bounding_box=ce.bounding_box,
                )
                for ce in curr_entities
            ]
            lifecycle_res = engine.deformation_tracker.track_lifecycle(
                morph_prev, morph_curr, conservation_required=False
            )
            engine.latest_lifecycle_result = lifecycle_res
            if hasattr(engine, "entity_lineage_graph") and engine.entity_lineage_graph is not None:
                engine.entity_lineage_graph.update(lifecycle_res.lineage_graph)

        # Faculty D: Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring)
        # Only evaluate bilateral pairing if entities belong to controllable agent features,
        # never autonomous mobile threats, lethal hazards, or background void.
        if len(moved_entities) >= 2:
            candidate_bilat = [
                (pe.grid_pos, delta, ce.feature_id)
                for pe, ce, delta in moved_entities
                if ce.area <= 64
                and ce.feature_id not in engine.mobile_threat_features
                and ce.feature_id not in engine.hazard_tracker.known_lethal_features
                and ce.feature_id != bg
                and ce.feature_id != engine.bg_feature
                and ce.feature_id != 0
            ]
            if len(candidate_bilat) >= 2:
                bilat_state = engine.bilateral_integrator.detect_bilateral_pairing(
                    candidate_bilat, (H, W)
                )
                if bilat_state is not None:
                    engine.bilateral_state = bilat_state
        elif engine.bilateral_state is not None:
            f_list = list(engine.bilateral_state.features)
            if f_list:
                f1 = f_list[0]
                f2 = f_list[1] if len(f_list) > 1 else f_list[0]
                pts1 = np.argwhere(curr_grid == f1)
                pts2 = np.argwhere(curr_grid == f2)
                if len(pts1) > 0 and len(pts2) > 0:
                    engine.bilateral_state.pos1 = (
                        int(np.mean(pts1[:, 0])),
                        int(np.mean(pts1[:, 1])),
                    )
                    engine.bilateral_state.pos2 = (
                        int(np.mean(pts2[:, 0])),
                        int(np.mean(pts2[:, 1])),
                    )

        def _are_adjacent(e1: SpatialEntity, e2: SpatialEntity) -> bool:
            bb1 = e1.bounding_box
            bb2 = e2.bounding_box
            r_gap = max(0, bb1[0] - bb2[1] - 1, bb2[0] - bb1[1] - 1)
            c_gap = max(0, bb1[2] - bb2[3] - 1, bb2[2] - bb1[3] - 1)
            return r_gap <= 1 and c_gap <= 1

        if (
            not is_win
            and not is_effector_action
            and not engine.avatar_features
            and engine.avatar_feature is None
        ):
            if moved_entities:
                delta_groups: dict[tuple[int, int], list[tuple[SpatialEntity, SpatialEntity]]] = {}
                for pe, ce, delta in moved_entities:
                    delta_groups.setdefault(delta, []).append((pe, ce))

                has_retained_dynamics = bool(engine.action_dynamics)

                if not has_retained_dynamics:
                    # ── Fresh start (no prior knowledge) ──────────────────────
                    # No dynamics to cross-reference. Use the original heuristic:
                    # the largest group of entities sharing the same displacement
                    # is most likely the avatar (first level, typically no enemies).
                    best_delta, pairs = max(delta_groups.items(), key=lambda item: len(item[1]))
                    EpistemicFeedbackAssimilator._commit_avatar_identity(
                        engine,
                        pairs,
                        best_delta,
                        action,
                        H,
                        W,
                        calibrate_dynamics=True,
                    )
                else:
                    # ── Controllability Test (retained dynamics) ───────────────
                    # The human brain identifies its avatar not from a single
                    # observation but from AGENCY: "I pressed up and THIS thing
                    # moved up. I pressed right and THIS SAME thing moved right."
                    #
                    # Accumulate (action, observed_delta) evidence per feature.
                    # Commit only when ONE feature consistently matches the
                    # expected displacement across 2+ different actions.
                    # This naturally disambiguates the avatar from enemies that
                    # move independently of player input.
                    for delta_val, group in delta_groups.items():
                        for _, ce in group:
                            feat = ce.feature_id
                            if feat != bg and feat != engine.bg_feature:
                                engine._avatar_controllability_evidence.setdefault(feat, []).append(
                                    (action, delta_val)
                                )

                    # Check if any feature has demonstrated controllability:
                    # its observed delta matches the expected action delta across
                    # at least 2 DIFFERENT displacement actions.
                    best_candidate: int | None = None
                    best_score = 0
                    for feat, observations in engine._avatar_controllability_evidence.items():
                        confirmed_actions: set[int] = set()
                        for obs_action, obs_delta in observations:
                            dyn = engine.action_dynamics.get(obs_action)
                            if dyn is not None and dyn.is_displacement_action():
                                expected = dyn.get_displacement()
                                if obs_delta == expected:
                                    confirmed_actions.add(obs_action)
                        if len(confirmed_actions) > best_score:
                            best_score = len(confirmed_actions)
                            best_candidate = feat

                    if best_score >= 2 and best_candidate is not None:
                        # Found a feature that consistently responds to our
                        # actions — this is the avatar. Retrieve its last
                        # observation to commit identity.
                        last_obs = engine._avatar_controllability_evidence[best_candidate][-1]
                        obs_action, obs_delta = last_obs
                        matching_pairs = delta_groups.get(obs_delta, [])
                        # Filter to only the confirmed feature
                        avatar_pairs = [
                            (pe, ce) for pe, ce in matching_pairs if ce.feature_id == best_candidate
                        ]
                        if not avatar_pairs:
                            # Feature was confirmed by prior frame, find it now
                            for dg_pairs in delta_groups.values():
                                for pe, ce in dg_pairs:
                                    if ce.feature_id == best_candidate:
                                        avatar_pairs.append((pe, ce))
                        if avatar_pairs:
                            commit_delta = obs_delta
                            EpistemicFeedbackAssimilator._commit_avatar_identity(
                                engine,
                                avatar_pairs,
                                commit_delta,
                                action,
                                H,
                                W,
                                calibrate_dynamics=False,  # keep retained dynamics
                            )
                            engine._avatar_controllability_evidence.clear()
                            logger.info(
                                "AutonomousEpistemicEngine: Avatar confirmed via controllability "
                                "test — feature %s responded to %d different actions.",
                                best_candidate,
                                best_score,
                            )
                    else:
                        logger.debug(
                            "AutonomousEpistemicEngine: Avatar identification deferred — "
                            "accumulating controllability evidence (best_score=%d, candidates=%d)",
                            best_score,
                            len(engine._avatar_controllability_evidence),
                        )

        # Faculty C: Dorsal Visual Stream (Area MT/V5) Kinetic Figure-Ground Segregation
        cmd_delta = act_dyn.get_displacement() if act_dyn else None
        kinetic_res = engine.dorsal_kinetic_stream.segregate_motion(
            prev_grid=engine.prev_grid,
            curr_grid=curr_grid,
            commanded_delta=cmd_delta,
            background_feature=bg,
            known_avatar_features=engine.avatar_features,
        )
        if kinetic_res.external_agents and engine.prev_grid is not None and action is not None:
            for ext in kinetic_res.external_agents:
                if ext.velocity != (0.0, 0.0):
                    valid_threat_feats = {
                        f
                        for f in ext.features
                        if f != bg
                        and f not in engine.avatar_features
                        and (engine.avatar_feature is None or f != engine.avatar_feature)
                    }
                    engine.mobile_threat_features.update(valid_threat_feats)

        # Premotor Collision Cones: Update forward space-time trajectories
        static_walls = set(engine.learned_barriers)
        for r_w in range(H):
            for c_w in range(W):
                if int(curr_grid[r_w, c_w]) in engine.symbolic_theory.barrier_features:
                    static_walls.add((r_w, c_w))
        engine.collision_cones.update_trajectories(
            kinetic_entities=kinetic_res.external_agents,
            static_barriers=static_walls,
            grid_shape=(H, W),
            horizon=12,
        )

        if (
            not is_win
            and not is_effector_action
            and kinetic_res.self_avatar is not None
            and kinetic_res.self_avatar.confidence >= 0.70
            and cmd_delta is not None
            and (cmd_delta[0] != 0 or cmd_delta[1] != 0)
        ):
            valid_feats = {
                f
                for f in kinetic_res.self_avatar.features
                if f != bg and f != engine.bg_feature and f != 0
            }
            if valid_feats:
                engine.avatar_features = valid_feats
                engine.avatar_feature = next(iter(valid_feats))
                engine.avatar_size = kinetic_res.self_avatar.area
                engine.avatar_pos = (
                    int(round(kinetic_res.self_avatar.centroid[0])),
                    int(round(kinetic_res.self_avatar.centroid[1])),
                )
                if kinetic_res.raw_delta is not None and action is not None:
                    dr, dc = kinetic_res.raw_delta
                    EpistemicFeedbackAssimilator._record_safe_traversal(
                        engine, kinetic_res.self_avatar.cells, (dr, dc), H, W
                    )
                    if action not in engine.action_dynamics:
                        engine.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr,
                            delta_c=dc,
                            confidence=0.75,
                            probes_tested=1,
                        )
                    else:
                        engine.action_dynamics[action].update_from_trial((dr, dc), success=True)
                logger.info(
                    "AutonomousEpistemicEngine: Avatar segregated/re-grounded via MT/V5 kinetic stream (feats=%s, size=%d, pos=%s)",
                    engine.avatar_features,
                    engine.avatar_size,
                    engine.avatar_pos,
                )

        elif not is_effector_action:
            known_av_feats = engine.avatar_features or (
                {engine.avatar_feature} if engine.avatar_feature is not None else set()
            )
            av_moved = [item for item in moved_entities if item[1].feature_id in known_av_feats]
            if (
                not is_win
                and not is_lost
                and not av_moved
                and not is_effector_action
                and moved_entities
                and (action in engine.action_dynamics or not action_data)
            ):
                valid_moved = [
                    (pe, ce, delta)
                    for pe, ce, delta in moved_entities
                    if ce.feature_id != bg
                    and ce.feature_id != engine.bg_feature
                    and ce.feature_id != 0
                    and ce.area <= max(49, int(H * W * 0.15))
                ]
                if valid_moved:
                    delta_groups_remap: dict[
                        tuple[int, int], list[tuple[SpatialEntity, SpatialEntity]]
                    ] = {}
                    for pe, ce, delta in valid_moved:
                        delta_groups_remap.setdefault(delta, []).append((pe, ce))
                    best_delta, pairs = max(
                        delta_groups_remap.items(), key=lambda item: len(item[1])
                    )
                    comp_pairs = [pairs[0]]
                    for pe, ce in pairs[1:]:
                        if any(_are_adjacent(ce, c_ce) for _, c_ce in comp_pairs):
                            comp_pairs.append((pe, ce))
                    engine.avatar_features = {
                        ce.feature_id
                        for _, ce in comp_pairs
                        if ce.feature_id != bg
                        and ce.feature_id != engine.bg_feature
                        and ce.feature_id != 0
                    }
                    if engine.avatar_features:
                        engine.avatar_feature = next(iter(engine.avatar_features))
                    engine.avatar_size = sum(ce.area for _, ce in comp_pairs)
                    known_av_feats = engine.avatar_features
                    av_moved = [
                        item for item in moved_entities if item[1].feature_id in known_av_feats
                    ]
                    if getattr(engine, "last_non_displacement_action", None) is not None:
                        engine.mark_focus_switch_action(engine.last_non_displacement_action)
                        engine.last_non_displacement_action = None

            if av_moved:
                if len(av_moved) > 1 and prev_avatar_pos is not None:
                    av_moved.sort(
                        key=lambda item: (
                            abs(item[0].grid_pos[0] - prev_avatar_pos[0])
                            + abs(item[0].grid_pos[1] - prev_avatar_pos[1])
                        )
                    )
                    primary_ce = av_moved[0][1]
                    filtered_av_moved = [av_moved[0]]
                    for item in av_moved[1:]:
                        if _are_adjacent(item[1], primary_ce):
                            filtered_av_moved.append(item)
                    av_moved = filtered_av_moved

                av_pe, av_ce, (dr, dc) = av_moved[0]
                if prev_avatar_pos is not None:
                    dist_to_prev = abs(av_pe.grid_pos[0] - prev_avatar_pos[0]) + abs(
                        av_pe.grid_pos[1] - prev_avatar_pos[1]
                    )
                    if dist_to_prev > max(abs(dr), abs(dc)) + 3:
                        if getattr(engine, "last_non_displacement_action", None) is not None:
                            engine.mark_focus_switch_action(engine.last_non_displacement_action)
                            engine.last_non_displacement_action = None
                    else:
                        engine.last_non_displacement_action = None
                all_cells = [
                    cell
                    for _, ce, _ in av_moved
                    for cell in ce.properties.get("cells", [ce.grid_pos])
                ]
                engine.avatar_pos = (
                    int(round(sum(c[0] for c in all_cells) / len(all_cells))),
                    int(round(sum(c[1] for c in all_cells) / len(all_cells))),
                )
                if not is_win:
                    EpistemicFeedbackAssimilator._record_safe_traversal(
                        engine, all_cells, (dr, dc), H, W
                    )

                if not is_effector_action:
                    if action not in engine.action_dynamics:
                        engine.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr,
                            delta_c=dc,
                            confidence=0.6,
                            probes_tested=1,
                        )
                    else:
                        engine.action_dynamics[action].update_from_trial(
                            (dr, dc), success=True, learning_rate=0.5
                        )
                    if action in engine.action_affordances:
                        engine.action_affordances[action].is_displacement = dr != 0 or dc != 0
                        engine.action_affordances[action].delta = (dr, dc)
                    else:
                        engine.action_affordances[action] = ActionAffordance(
                            action_id=action,
                            is_displacement=(dr != 0 or dc != 0),
                            delta=(dr, dc),
                        )

                if not is_win:
                    for pe, ce, delta in moved_entities:
                        if ce.feature_id in known_av_feats:
                            continue
                        # Pushing requires physical contact: entity must be adjacent and ahead in motion direction
                        is_contact = _are_adjacent(pe, av_pe)
                        is_ahead = False
                        if dr != 0 and dc == 0:
                            is_ahead = (pe.centroid[0] - av_pe.centroid[0]) * dr > 0 and abs(
                                pe.centroid[1] - av_pe.centroid[1]
                            ) <= 3
                        elif dc != 0 and dr == 0:
                            is_ahead = (pe.centroid[1] - av_pe.centroid[1]) * dc > 0 and abs(
                                pe.centroid[0] - av_pe.centroid[0]
                            ) <= 3
                        else:
                            is_ahead = (pe.centroid[0] - av_pe.centroid[0]) * dr + (
                                pe.centroid[1] - av_pe.centroid[1]
                            ) * dc > 0

                        is_pushed = is_contact and is_ahead and (delta == (dr, dc))
                        if is_pushed:
                            engine.symbolic_theory.induce_cargo(ce.feature_id)
                            logger.info(
                                "AutonomousEpistemicEngine: Discovered PUSHABLE CARGO (feat=%d, size=%d)",
                                ce.feature_id,
                                ce.area,
                            )
                        elif delta != (0, 0):
                            # Autonomous dynamic entity (moved under its own agency) — not passive cargo
                            engine.symbolic_theory.cargo_features.discard(ce.feature_id)
                        elif delta == (dr, dc) and any(
                            _are_adjacent(ce, c_ce) for _, c_ce, _ in av_moved
                        ):
                            engine.avatar_features.add(ce.feature_id)
                            engine.avatar_size += ce.area
                            logger.info(
                                "AutonomousEpistemicEngine: Discovered compound avatar component (feat=%d, size=%d)",
                                ce.feature_id,
                                ce.area,
                            )
        # ── Proprioceptive Efference Copy Verification & Obstacle Grounding ──
        avatar_moved = bool(
            engine.avatar_pos is not None
            and prev_avatar_pos is not None
            and engine.avatar_pos != prev_avatar_pos
        )
        if not is_effector_action and not is_known_displacement and not avatar_moved:
            engine.last_non_displacement_action = action

        if is_known_displacement and prev_avatar_pos is not None:
            # 1. Physical resistance detection (Avatar did not move upon directional motor command)
            if not avatar_moved and not is_win:
                if act_dyn is not None and act_dyn.is_displacement_action():
                    # Motor command failed to displace avatar! Ground obstacle cell regardless of other background motion
                    exp_dr, exp_dc = act_dyn.get_displacement()
                    step_r = int(np.sign(exp_dr))
                    step_c = int(np.sign(exp_dc))
                    dist = max(abs(exp_dr), abs(exp_dc))

                    if dist > 0 and prev_avatar_pos is not None:
                        av_feats = engine.avatar_features or (
                            {engine.avatar_feature} if engine.avatar_feature is not None else set()
                        )
                        # HCIR Attractor Invariant: Gather candidate goal features
                        cand_goals = engine.detect_structural_goals(curr_grid)
                        cand_goal_feats = {
                            int(g["feature"])
                            for g in cand_goals
                            if "feature" in g and g["feature"] is not None
                        }
                        cand_goal_positions = {
                            g["position"]
                            for g in cand_goals
                            if "position" in g and g["position"] is not None
                        }
                        engine.symbolic_theory.candidate_goal_features.update(cand_goal_feats)

                        for k in range(1, dist + 1):
                            kr = prev_avatar_pos[0] + step_r * k
                            kc = prev_avatar_pos[1] + step_c * k
                            if 0 <= kr < H and 0 <= kc < W:
                                val = (
                                    int(engine.prev_grid[kr, kc])
                                    if engine.prev_grid is not None
                                    else int(curr_grid[kr, kc])
                                )
                                # HCIR Attractor Invariant: Candidate goals are NEVER barriers
                                # Proprioceptive obstacle grounding: any non-walkable feature blocking movement is an empirical barrier
                                if (
                                    val != bg
                                    and val != engine.bg_feature
                                    and val not in av_feats
                                    and val not in engine.learned_goal_features
                                    and val not in cand_goal_feats
                                    and (kr, kc) not in cand_goal_positions
                                    and not engine.symbolic_theory.is_walkable(val)
                                ):
                                    engine.learned_barriers.add((kr, kc))
                                    if not engine.symbolic_theory.is_barrier(val):
                                        engine.symbolic_theory.induce_barrier(val)
                                        logger.info(
                                            "AutonomousEpistemicEngine: Proprioceptive obstacle grounded along ray at (%d, %d) with feature %d",
                                            kr,
                                            kc,
                                            val,
                                        )
                                        if engine.bg_feature == val:
                                            engine.bg_feature = engine.estimate_background(
                                                curr_grid
                                            )
                                    break
                        else:
                            end_r = prev_avatar_pos[0] + step_r * dist
                            end_c = prev_avatar_pos[1] + step_c * dist
                            if 0 <= end_r < H and 0 <= end_c < W:
                                engine.learned_barriers.add((end_r, end_c))

                    if engine.last_action is not None and prev_avatar_pos is not None:
                        engine.failed_transitions.add((prev_avatar_pos, engine.last_action))
                        if not hasattr(engine, "entity_action_failed_positions"):
                            engine.entity_action_failed_positions = {}
                        if not hasattr(engine, "immobile_entity_actions"):
                            engine.immobile_entity_actions = set()
                        engine.entity_action_failed_positions.setdefault(
                            engine.last_action, set()
                        ).add(prev_avatar_pos)
                        if len(engine.entity_action_failed_positions[engine.last_action]) >= 2:
                            engine.immobile_entity_actions.add(engine.last_action)
                        prev_inh = engine.inhibited_actions.get(engine.last_action, 0)
                        engine.inhibited_actions[engine.last_action] = max(prev_inh + 2, 3)
                        engine.simulation_cooldown = 0
                        engine.consecutive_simulation_failures = 0

                if prev_avatar_pos is not None and engine.avatar_pos is not None:
                    if engine.avatar_pos == prev_avatar_pos and not is_win and not is_lost:
                        engine.consecutive_stuck_steps += 1
                    else:
                        engine.consecutive_stuck_steps = 0
            else:
                engine.consecutive_stuck_steps = 0

            # 2. Predictive coding efference copy divergence check & surprise computation
            expected_dr = (
                engine.last_predicted_pos[0] - prev_avatar_pos[0]
                if engine.last_predicted_pos is not None and prev_avatar_pos is not None
                else (act_dyn.delta_r if act_dyn is not None and is_known_displacement else 0)
            )
            expected_dc = (
                engine.last_predicted_pos[1] - prev_avatar_pos[1]
                if engine.last_predicted_pos is not None and prev_avatar_pos is not None
                else (act_dyn.delta_c if act_dyn is not None and is_known_displacement else 0)
            )
            actual_dr = (
                engine.avatar_pos[0] - prev_avatar_pos[0]
                if engine.avatar_pos is not None and prev_avatar_pos is not None
                else 0
            )
            actual_dc = (
                engine.avatar_pos[1] - prev_avatar_pos[1]
                if engine.avatar_pos is not None and prev_avatar_pos is not None
                else 0
            )

            expected_state = {
                "dr": expected_dr,
                "dc": expected_dc,
                "avatar_moved": 1
                if (is_known_displacement or expected_dr != 0 or expected_dc != 0)
                else 0,
            }
            actual_state = {
                "dr": actual_dr,
                "dc": actual_dc,
                "avatar_moved": 1 if avatar_moved else 0,
            }

            confidence = 0.85 if engine.mental_plan else (0.70 if is_known_displacement else 0.40)
            salience = 1.3 if diff.changed_pixel_count > 0 else 0.8
            surprise_eval = engine.surprise_engine.evaluate_surprise(
                prediction_id=f"step_{engine.step_counter}",
                expected_state=expected_state,
                actual_state=actual_state,
                confidence=confidence,
                attention_salience=salience,
                prediction_source="motor_calibration",
                context_signature=f"act_{action}",
            )
            engine.last_surprise = surprise_eval.surprise_score
            engine.last_surprise_eval = surprise_eval

            discrepancy = False
            if not is_win:
                if (
                    not avatar_moved
                    and diff.changed_pixel_count == 0
                    and (
                        engine.last_predicted_pos is None
                        or engine.last_predicted_pos != prev_avatar_pos
                    )
                ):
                    discrepancy = True
                elif (
                    engine.last_predicted_pos is not None
                    and engine.avatar_pos is not None
                    and max(
                        abs(engine.avatar_pos[0] - engine.last_predicted_pos[0]),
                        abs(engine.avatar_pos[1] - engine.last_predicted_pos[1]),
                    )
                    > 1
                ):
                    discrepancy = True
                elif surprise_eval.is_surprising and not avatar_moved and is_known_displacement:
                    discrepancy = True

            # Involuntary Orienting Reflex: When surprising visual change occurs,
            # shift prefrontal attentional focus to the centroid of the mutation
            if (
                surprise_eval.is_surprising
                and diff.changed_mask is not None
                and np.any(diff.changed_mask)
            ):
                mut_cells = np.argwhere(diff.changed_mask)
                if len(mut_cells) > 0:
                    sal_r = int(round(float(np.mean([pt[0] for pt in mut_cells]))))
                    sal_c = int(round(float(np.mean([pt[1] for pt in mut_cells]))))
                    engine.working_memory.orient_attention((sal_r, sal_c))

            if discrepancy:
                did_displace_correctly = (expected_dr != 0 and actual_dr * expected_dr > 0) or (
                    expected_dc != 0 and actual_dc * expected_dc > 0
                )
                if (
                    engine.last_action is not None
                    and prev_avatar_pos is not None
                    and not did_displace_correctly
                ):
                    engine.failed_transitions.add((prev_avatar_pos, engine.last_action))
                    prev_inh = engine.inhibited_actions.get(engine.last_action, 0)
                    engine.inhibited_actions[engine.last_action] = max(prev_inh + 2, 3)
                if engine.mental_plan:
                    logger.debug(
                        "AutonomousEpistemicEngine: Sensory discrepancy/surprise detected (pred=%s, actual=%s, surprise=%.3f). Invalidating %d-step mental plan.",
                        engine.last_predicted_pos,
                        engine.avatar_pos,
                        surprise_eval.surprise_score,
                        len(engine.mental_plan),
                    )
                    engine.mental_plan.clear()
                    engine.phase = EpistemicPhase.REPLANNING
                    if not did_displace_correctly:
                        engine.consecutive_plan_failures += 1
                        if engine.consecutive_plan_failures >= 2:
                            engine.exploration_cooldown = 2
                            engine.consecutive_plan_failures = 0
            else:
                if avatar_moved or diff.changed_pixel_count > 0:
                    engine.consecutive_plan_failures = 0

        # ── B. Environmental Mutation Induction ──────────────────────────────
        movable_features = set(engine.symbolic_theory.cargo_features)
        movable_features.update(engine.avatar_features)
        if engine.avatar_feature is not None:
            movable_features.add(engine.avatar_feature)

        def is_hud_mutation(r: int, c: int) -> bool:
            if H >= 24:
                if engine.avatar_pos is not None:
                    if r >= H - 6 and engine.avatar_pos[0] < H - 8:
                        return True
                    if r < 3 and engine.avatar_pos[0] >= 5:
                        return True
                    min_r, max_r = min(engine.avatar_pos[0], r), max(engine.avatar_pos[0], r)
                    for div_r in range(min_r + 1, max_r):
                        div_val = int(engine.prev_grid[div_r, 0])
                        if np.all(engine.prev_grid[div_r, :] == div_val):
                            if div_r >= H - 16 or div_r <= 16:
                                return True
                    min_c, max_c = min(engine.avatar_pos[1], c), max(engine.avatar_pos[1], c)
                    for div_c in range(min_c + 1, max_c):
                        div_val = int(engine.prev_grid[0, div_c])
                        if np.all(engine.prev_grid[:, div_c] == div_val):
                            if div_c >= W - 16 or div_c <= 16:
                                return True
                else:
                    if r >= H - 6 or r < 3:
                        return True
            return False

        av_radius = max(2, int(round(math.sqrt(max(1, engine.avatar_size)))))

        def is_local_to_avatar(r: int, c: int) -> bool:
            if engine.avatar_pos is not None:
                if max(abs(r - engine.avatar_pos[0]), abs(c - engine.avatar_pos[1])) <= av_radius:
                    return True
            if prev_avatar_pos is not None:
                if max(abs(r - prev_avatar_pos[0]), abs(c - prev_avatar_pos[1])) <= av_radius:
                    return True
            return False

        distant_mutations = [
            (r, c, old_v, new_v)
            for r, c, old_v, new_v in diff.mutated_pixels
            if old_v not in movable_features
            and new_v not in movable_features
            and not is_hud_mutation(r, c)
            and not is_local_to_avatar(r, c)
        ]

        if distant_mutations and len(distant_mutations) <= 64:
            trigger_pos = (
                (int(action_data["y"]), int(action_data["x"]))
                if action_data and "x" in action_data
                else engine.avatar_pos
            )
            trigger_feat = (
                int(engine.prev_grid[trigger_pos[0], trigger_pos[1]])
                if trigger_pos and 0 <= trigger_pos[0] < H and 0 <= trigger_pos[1] < W
                else None
            )

            if (
                trigger_feat is None
                or trigger_feat == bg
                or trigger_feat == engine.bg_feature
                or engine.symbolic_theory.is_walkable(trigger_feat)
            ):
                trigger_feat = None

            existing_mut = next(
                (
                    m
                    for m in engine.state_mutations
                    if m.trigger_pos == trigger_pos
                    and m.trigger_feature == trigger_feat
                    and m.prior_value == distant_mutations[0][2]
                ),
                None,
            )
            if existing_mut is not None:
                existing_mut.record_observation(distant_mutations[0][3])
            else:
                mutation_model = StateMutationModel(
                    trigger_type="CONTACT" if is_known_displacement else "ACTION",
                    trigger_pos=trigger_pos,
                    trigger_feature=trigger_feat,
                    mutation_type="ENVIRONMENTAL_TOGGLE",
                    prior_value=distant_mutations[0][2],
                    posterior_value=distant_mutations[0][3],
                    confidence=0.8,
                    occurrences=1,
                    metadata={"action_id": action},
                )
                engine.state_mutations.append(mutation_model)
                engine.exhausted_candidate_goals.clear()
                logger.info(
                    "AutonomousEpistemicEngine: Induced StateMutationModel! Trigger %s at %s changed %d distant pixels.",
                    mutation_model.trigger_type,
                    trigger_pos,
                    len(distant_mutations),
                )

        # ── C. Win / Loss Feedback Assimilation ──────────────────────────────
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        if is_win:
            win_target = None
            if prev_avatar_pos is not None:
                dr, dc = 0, 0
                if action in engine.action_dynamics:
                    dr, dc = engine.action_dynamics[action].get_displacement()
                elif action_data and "x" in action_data and "y" in action_data:
                    win_target = (int(action_data["y"]), int(action_data["x"]))
                if win_target is None:
                    tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                    if 0 <= tr < H and 0 <= tc < W:
                        win_target = (tr, tc)
            elif engine.current_simulated_goal is not None:
                win_target = engine.current_simulated_goal
            elif engine.avatar_pos is not None:
                win_target = engine.avatar_pos
            elif action_data and "x" in action_data and "y" in action_data:
                win_target = (int(action_data["y"]), int(action_data["x"]))

            if win_target is not None:
                engine.learned_goal_positions.add(win_target)
                if (
                    engine.prev_grid is not None
                    and 0 <= win_target[0] < H
                    and 0 <= win_target[1] < W
                ):
                    goal_feat = int(engine.prev_grid[win_target[0], win_target[1]])
                    is_distant_target = (
                        prev_avatar_pos is None
                        or abs(win_target[0] - prev_avatar_pos[0]) > 0
                        or abs(win_target[1] - prev_avatar_pos[1]) > 0
                    )
                    if (
                        goal_feat != bg
                        and goal_feat != engine.bg_feature
                        and (goal_feat not in av_feats or is_distant_target)
                        and not engine.symbolic_theory.is_barrier(goal_feat)
                    ):
                        engine.symbolic_theory.induce_goal(goal_feat)
                        logger.info(
                            "AutonomousEpistemicEngine: Grounded invariant WIN GOAL FEATURE %d at %s",
                            goal_feat,
                            win_target,
                        )

                for pe in prev_entities:
                    is_distant_pe = (
                        prev_avatar_pos is None
                        or abs(pe.grid_pos[0] - prev_avatar_pos[0]) > 2
                        or abs(pe.grid_pos[1] - prev_avatar_pos[1]) > 2
                    )
                    if (
                        (pe.feature_id not in av_feats or is_distant_pe)
                        and pe.feature_id != bg
                        and pe.feature_id != engine.bg_feature
                        and not engine.symbolic_theory.is_barrier(pe.feature_id)
                        and 1 <= pe.area <= 64
                    ):
                        pe_cells = pe.properties.get("cells", [pe.grid_pos])
                        if win_target in pe_cells or pe.grid_pos == win_target:
                            engine.symbolic_theory.induce_goal(pe.feature_id)
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded invariant WIN GOAL ENTITY feature %d (area=%d)",
                                pe.feature_id,
                                pe.area,
                            )

            if engine.current_simulated_goal is not None and engine.prev_grid is not None:
                sr, sc = engine.current_simulated_goal
                if 0 <= sr < H and 0 <= sc < W:
                    sim_feat = int(engine.prev_grid[sr, sc])
                    if (
                        sim_feat != bg
                        and sim_feat != engine.bg_feature
                        and sim_feat not in av_feats
                        and not engine.symbolic_theory.is_barrier(sim_feat)
                    ):
                        engine.symbolic_theory.induce_goal(sim_feat)

            if engine.active_hypothesis:
                engine.active_hypothesis.confirmed = True
                engine.active_hypothesis.confidence = 1.0

            # Level transition: cleanly reset episodic spatial memory for the next level
            engine.reset_episode(
                retain_dynamics=True, is_new_level=True, level=engine.current_level + 1
            )
            return
        elif engine.avatar_pos is not None and not is_lost:
            engine.exhausted_candidate_goals.add(engine.avatar_pos)

        if is_lost:
            # Faculty: Lateral Habenula & aMCC Fatal Prefix Backpropagation & Negative Valence
            engine.working_memory.habenular_ior.record_catastrophe(
                final_step=engine.step_counter,
                is_lost=True,
            )
            # Faculty: Anterior Cingulate Cortex Conflict & Frustration Monitor
            if hasattr(engine.working_memory, "acc_conflict"):
                actor = prev_avatar_pos or engine.avatar_pos
                if actor is not None:
                    engine.working_memory.acc_conflict.register_death_event(
                        actor_pos=(float(actor[0]), float(actor[1])),
                        step=engine.step_counter,
                        last_action=action,
                    )
            # Faculty: Hippocampal Sharp-Wave Ripple (SWR) Backward Negative Replay
            if hasattr(engine, "episodic_cortex"):
                engine.episodic_cortex.trigger_sharp_wave_ripple_replay(
                    is_lost=True,
                    is_win=False,
                    prev_grid=engine.prev_grid,
                    curr_grid=curr_grid,
                    bg_feature=engine.bg_feature,
                    avatar_features=av_feats,
                    is_walkable_fn=engine.symbolic_theory.is_walkable,
                    is_barrier_fn=engine.symbolic_theory.is_barrier,
                )

            if prev_avatar_pos is not None and action is not None:
                engine.failed_transitions.add((prev_avatar_pos, action))
                logger.info(
                    "AutonomousEpistemicEngine: Grounded lethal failed transition (%s, %s)",
                    prev_avatar_pos,
                    action,
                )

            target_pos = None
            if prev_avatar_pos is not None:
                dr, dc = 0, 0
                if action in engine.action_dynamics:
                    dr, dc = engine.action_dynamics[action].get_displacement()
                elif action_data and "x" in action_data and "y" in action_data:
                    target_pos = (int(action_data["y"]), int(action_data["x"]))
                if target_pos is None:
                    tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                    if 0 <= tr < H and 0 <= tc < W:
                        target_pos = (tr, tc)
            elif engine.avatar_pos is not None:
                target_pos = engine.avatar_pos
            elif action_data and "x" in action_data and "y" in action_data:
                target_pos = (int(action_data["y"]), int(action_data["x"]))

            if target_pos is not None:
                if (
                    engine.prev_grid is not None
                    and 0 <= target_pos[0] < H
                    and 0 <= target_pos[1] < W
                ):
                    # Check for lethal hazard features in a 5x5 window around target_pos
                    r0 = max(0, target_pos[0] - 2)
                    r1 = min(H, target_pos[0] + 3)
                    c0 = max(0, target_pos[1] - 2)
                    c1 = min(W, target_pos[1] + 3)
                    candidate_cells = set(np.unique(curr_grid[r0:r1, c0:c1])).union(
                        set(np.unique(engine.prev_grid[r0:r1, c0:c1]))
                    )
                    # Also inspect unobstructed cardinal line-of-sight rays for remote turrets/projectiles
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        curr_r, curr_c = target_pos[0] + dr, target_pos[1] + dc
                        while 0 <= curr_r < H and 0 <= curr_c < W:
                            val = int(engine.prev_grid[curr_r, curr_c])
                            if engine.symbolic_theory.is_barrier(val):
                                break
                            if (
                                val != engine.bg_feature
                                and val not in av_feats
                                and not engine.symbolic_theory.is_walkable(val)
                            ):
                                candidate_cells.add(val)
                            curr_r += dr
                            curr_c += dc
                    for cand_feat in candidate_cells:
                        cand_feat_int = int(cand_feat)
                        if (
                            cand_feat_int != engine.bg_feature
                            and cand_feat_int not in av_feats
                            and not engine.symbolic_theory.is_walkable(cand_feat_int)
                            and int(np.sum(engine.prev_grid == cand_feat_int)) < 40
                        ):
                            engine.hazard_tracker.register_lethal_feature(cand_feat_int)
                            if int(np.sum(engine.prev_grid == cand_feat_int)) >= 25:
                                engine.symbolic_theory.induce_barrier(cand_feat_int)
                            engine.symbolic_theory.cargo_features.discard(cand_feat_int)
                            engine.symbolic_theory.goal_features.discard(cand_feat_int)
                            engine.symbolic_theory.candidate_goal_features.discard(cand_feat_int)
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded lethal feature %d at barrier %s",
                                cand_feat_int,
                                target_pos,
                            )
                    engine.exhausted_candidate_goals.add(target_pos)

                    # Trajectory-level aversive conditioning:
                    # The destination cell target_pos resulted in catastrophic loss/death.
                    # Ground it as a static lethal position and barrier so neither mental simulation
                    # nor exploratory probes step into this coordinate again!
                    engine.hazard_tracker.register_lethal_position(target_pos)
                    engine.learned_barriers.add(target_pos)
                    engine.level_learned_barriers.setdefault(engine.current_level, set()).add(
                        target_pos
                    )
                    engine.level_lethal_positions.setdefault(engine.current_level, set()).add(
                        target_pos
                    )

                    # Blacklist all cardinal incoming moves into target_pos from immediate neighbors
                    for m_dr, m_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nbr = (target_pos[0] - m_dr, target_pos[1] - m_dc)
                        if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                            for cand_a, dyn in engine.action_dynamics.items():
                                if dyn.is_displacement_action() and dyn.get_displacement() == (
                                    m_dr,
                                    m_dc,
                                ):
                                    engine.failed_transitions.add((nbr, cand_a))

            if action is not None:
                engine.inhibited_actions[action] = 5

            if engine.active_hypothesis:
                engine.active_hypothesis.confidence = 0.0

        # Prefrontal item acquisition
        if engine.avatar_pos is not None:
            for pe in prev_entities:
                if (
                    pe.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE)
                    and pe.feature_id not in av_feats
                    and pe.feature_id != 0
                    and not engine.symbolic_theory.is_walkable(pe.feature_id)
                ):
                    dist_to_av = abs(pe.grid_pos[0] - engine.avatar_pos[0]) + abs(
                        pe.grid_pos[1] - engine.avatar_pos[1]
                    )
                    if dist_to_av <= 1:
                        engine.working_memory.acquire_item(
                            feature_id=pe.feature_id,
                            role=pe.role.value,
                            step=engine.step_counter,
                            position=pe.grid_pos,
                        )

        # Faculty: Hippocampal CA1/CA3 Experiential Trace Buffer
        if hasattr(engine, "episodic_cortex"):
            engine.episodic_cortex.record_transition(
                step=engine.step_counter,
                avatar_pos=engine.avatar_pos,
                action=action,
                action_data=action_data,
                observation_diff=diff,
                reward=1.0 if is_win else (-1.0 if is_lost else 0.0),
                is_terminal=is_win or is_lost,
                is_lost=is_lost,
                is_win=is_win,
                grid_snapshot=curr_grid,
            )

        # Faculty: vmPFC Action-Effect Interventional Binding (Pearl's do(a))
        if hasattr(engine, "causal_cortex") and engine.prev_grid is not None:
            engine.causal_cortex.bind_intervention(
                action=action,
                origin_pos=prev_avatar_pos,
                target_pos=engine.avatar_pos,
                obs_diff=diff,
                prev_grid=engine.prev_grid,
                curr_grid=curr_grid,
                bg_feature=engine.bg_feature,
            )

        # Faculty: Prefrontal Inductive Hypothesis Testing & Relational Rule Discovery
        if hasattr(engine.working_memory, "hypothesis_engine") and engine.prev_grid is not None:
            engine.working_memory.hypothesis_engine.observe_transition(
                prev_grid=engine.prev_grid,
                action=action,
                curr_grid=curr_grid,
                prev_avatar_pos=prev_avatar_pos,
                curr_avatar_pos=engine.avatar_pos,
                is_dead=is_lost,
                is_won=is_win,
                background_feature=engine.bg_feature,
            )
