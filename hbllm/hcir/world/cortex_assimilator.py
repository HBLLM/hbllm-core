"""Epistemic Feedback Assimilator Faculty.

Assimilates sensory feedback (pixel diffs, kinetic flow, reward signals)
into motor dynamics, avatar identity consensus, and empirical neuro-symbolic theory.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
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
    EPISTEMIC_PROBE = (
        "epistemic_probe"  # Active epistemic probing upon conflict / unknown discovery
    )
    MENTAL_SIMULATION = "mental_simulation"  # Forward planning in mind
    EXPLOITATION = "exploitation"  # Direct execution of verified solution
    EXPLOITING = "exploitation"  # Direct execution alias
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
                if wf in av or wf in lethal or wf in engine.mobile_threat_features:
                    continue
                engine.symbolic_theory.induce_walkable(wf)
                engine.verified_safe_features.add(wf)

    @staticmethod
    def _synchronize_lattice_stride(engine: AutonomousEpistemicEngine, stride: int) -> None:
        """Propagate discovered lattice quantum to uncalibrated cardinal action priors."""
        if stride <= 1 or not hasattr(engine, "action_dynamics"):
            return
        for other_act, other_dyn in engine.action_dynamics.items():
            probes = getattr(other_dyn, "probes_tested", 0)
            if probes == 0 or (
                probes <= 2 and max(abs(other_dyn.delta_r), abs(other_dyn.delta_c)) < stride
            ):
                if other_dyn.delta_r != 0 and other_dyn.delta_c == 0:
                    sign = 1 if other_dyn.delta_r > 0 else -1
                    other_dyn.delta_r = sign * stride
                    other_dyn.displacement = (float(other_dyn.delta_r), 0.0)
                    if (
                        hasattr(engine, "action_affordances")
                        and other_act in engine.action_affordances
                    ):
                        engine.action_affordances[other_act].delta = (other_dyn.delta_r, 0)
                elif other_dyn.delta_c != 0 and other_dyn.delta_r == 0:
                    sign = 1 if other_dyn.delta_c > 0 else -1
                    other_dyn.delta_c = sign * stride
                    other_dyn.displacement = (0.0, float(other_dyn.delta_c))
                    if (
                        hasattr(engine, "action_affordances")
                        and other_act in engine.action_affordances
                    ):
                        engine.action_affordances[other_act].delta = (0, other_dyn.delta_c)

    @staticmethod
    def _is_edge_hud(entity: SpatialEntity, H: int, W: int) -> bool:
        """Check if an entity is a peripheral gauge / status meter along the outer edge."""
        cells = entity.properties.get("cells", [entity.grid_pos])
        if not cells:
            return False
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        return (min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)) or (
            min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4)
        )

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
        if abs(dc) >= 2 * abs(dr):
            dr = 0
        elif abs(dr) >= 2 * abs(dc):
            dc = 0

        def _adjacent(a: SpatialEntity, b: SpatialEntity) -> bool:
            b1, b2 = a.bounding_box, b.bounding_box
            r_gap = max(0, b1[0] - b2[1] - 1, b2[0] - b1[1] - 1)
            c_gap = max(0, b1[2] - b2[3] - 1, b2[2] - b1[3] - 1)
            return r_gap <= 1 and c_gap <= 1

        bg_val = engine.bg_feature or 0
        valid_initial_pairs = [(pe, ce) for pe, ce in pairs if ce.feature_id not in (0, bg_val)]
        if not valid_initial_pairs:
            return

        comp_pairs: list[tuple[SpatialEntity, SpatialEntity]] = [valid_initial_pairs[0]]
        for pe, ce in valid_initial_pairs[1:]:
            if any(_adjacent(ce, c_ce) for _, c_ce in comp_pairs):
                comp_pairs.append((pe, ce))

        new_av_feats = {ce.feature_id for _, ce in comp_pairs if ce.feature_id not in (0, bg_val)}
        if not new_av_feats:
            return
        if new_av_feats != engine.avatar_features:
            engine.mental_plan.clear()
        engine.avatar_features = new_av_feats
        feat_areas: dict[int, int] = defaultdict(int)
        for _, ce in comp_pairs:
            feat_areas[ce.feature_id] += ce.area
        sorted_feats = sorted(
            engine.avatar_features,
            key=lambda f: feat_areas[f] if f != 0 else 0,
            reverse=True,
        )
        engine.avatar_feature = (
            sorted_feats[0] if sorted_feats else next(iter(engine.avatar_features))
        )
        engine.mobile_threat_features.difference_update(engine.avatar_features)
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
        engine.entity_action_failed_positions.clear()
        engine.immobile_entity_actions.clear()

        if calibrate_dynamics and action is not None:
            engine.action_dynamics[action] = ActionDynamicsModel(
                action_id=action,
                delta_r=dr,
                delta_c=dc,
                confidence=0.5,
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
            if dr != 0 or dc != 0:
                EpistemicFeedbackAssimilator._synchronize_lattice_stride(
                    engine, max(abs(dr), abs(dc))
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
            walkable_feats = engine.symbolic_theory.walkable_features
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
            engine.entity_action_failed_positions.clear()
            engine.immobile_entity_actions.clear()
            if is_focus_switch:
                engine.avatar_features.clear()
                # Preserve prior avatar feature for re-grounding recovery
                # in case the focus switch was actually an INTERACT action
                if engine.avatar_feature is not None:
                    engine.prior_avatar_feature = engine.avatar_feature
                engine.avatar_feature = None
                engine.avatar_pos = None
                engine.avatar_size = 0
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

        # Active Manipulandum Effector Binding (Modal Select-Then-Direct):
        if target_coord is not None:
            engine.working_memory.last_effector_target_coord = target_coord
            tr_t, tc_t = target_coord
            if 0 <= tr_t < H and 0 <= tc_t < W:
                clicked_feat = int(curr_grid[tr_t, tc_t])
                bg_val = engine.bg_feature if engine.bg_feature is not None else 0
                ambient_bg = {0, bg_val}
                if engine.bg_feature is not None:
                    ambient_bg.add(engine.bg_feature)
                if (
                    clicked_feat not in ambient_bg
                    and not engine.symbolic_theory.is_barrier(clicked_feat)
                    and clicked_feat not in engine.hazard_tracker.known_lethal_features
                ):
                    manip_id = f"manip_{clicked_feat}_{tr_t}_{tc_t}"
                    engine.working_memory.last_effector_target_entity_id = manip_id
                    engine.working_memory.body_schema.bind_manipulandum(
                        entity_id=manip_id,
                        feature_id=clicked_feat,
                        pos=(tr_t, tc_t),
                        cells=[(tr_t, tc_t)],
                        step=engine.step_counter,
                    )
                    has_disp = any(
                        engine.is_displacement_action(a) for a in available_actions
                    ) or any(
                        dyn.is_displacement_action() for dyn in engine.action_dynamics.values()
                    )
                    if has_disp and engine.avatar_pos is None:
                        engine.avatar_pos = (tr_t, tc_t)
                        engine.avatar_features = {clicked_feat}
                        engine.avatar_feature = clicked_feat

        # Empirical State-Reversion / Undo Action Discovery:
        # If executing this action restored the grid to the previous-previous state,
        # ground this action as an UNDO/REVERT affordance and inhibit it during forward progress.
        is_interaction_act = (
            action in getattr(engine.symbolic_theory, "interaction_actions", set())
            or action in getattr(engine, "interaction_actions", set())
            or (aff is not None and getattr(aff, "is_interactive", False))
            or (aff is not None and getattr(aff, "is_manipulation", False))
            or engine.working_memory.body_schema.has_manipulandum()
        )
        if (
            action is not None
            and hasattr(engine, "grid_history")
            and len(engine.grid_history) >= 2
            and not is_effector_action
            and not is_interaction_act
            and not engine.is_displacement_action(action)
        ):
            if np.array_equal(curr_grid, engine.grid_history[-2]):
                engine.symbolic_theory.induce_undo_action(action)
                engine.undo_actions.add(action)
                if action in engine.action_affordances:
                    engine.action_affordances[action].is_undo = True
                engine.inhibited_actions[action] = 999999
                logger.info(
                    "AutonomousEpistemicEngine: Grounded empirical UNDO / STATE_REVERSION affordance for action %s",
                    action,
                )
        elif is_interaction_act and action is not None:
            engine.undo_actions.discard(action)
            if hasattr(engine.symbolic_theory, "undo_actions"):
                engine.symbolic_theory.undo_actions.discard(action)
            if action in engine.inhibited_actions and engine.inhibited_actions[action] >= 999999:
                engine.inhibited_actions.pop(action, None)

        if diff.changed_pixel_count == 0:
            engine.consecutive_quiescent_actions += 1
            engine.consecutive_effective_clicks = 0
            engine.last_effective_click_coord = None
            engine.active_goal_converging_coord = None
            engine.consecutive_goal_converging_clicks = 0
            if getattr(engine, "active_causal_actuator_coord", None) == target_coord:
                engine.active_causal_actuator_coord = None
                engine.consecutive_causal_probes = 0
            if target_coord is not None:
                engine.quiescent_click_targets.add(target_coord)
                engine.effective_click_targets.discard(target_coord)
                tr, tc = target_coord
                if 0 <= tr < H and 0 <= tc < W:
                    probed_feat = int(curr_grid[tr, tc])
                    engine.working_memory.visuospatial.record_probe(
                        pos=target_coord,
                        feature_id=probed_feat,
                        step=engine.step_counter,
                    )
                    # Categorical Causal Lifting: If clicking background produced 0 diff, background is inert
                    if probed_feat == engine.bg_feature or probed_feat == 0:
                        engine.quiescent_features.add(probed_feat)
                    else:
                        q_count = engine.working_memory.feature_quiescent_counts
                        q_count[probed_feat] = q_count.get(probed_feat, 0) + 1
                        if (
                            q_count[probed_feat] >= 2
                            and probed_feat not in engine.effective_features
                        ):
                            engine.quiescent_features.add(probed_feat)
            if action is not None:
                prev_inh = engine.inhibited_actions.get(action, 0)
                inh_steps = 4 if engine.consecutive_quiescent_actions >= 2 else 3
                engine.inhibited_actions[action] = max(prev_inh + 2, inh_steps)
                if prev_avatar_pos is not None:
                    engine.failed_transitions.add((prev_avatar_pos, action))

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
                if engine.last_effective_click_coord == target_coord:
                    engine.consecutive_effective_clicks += 1
                else:
                    engine.last_effective_click_coord = target_coord
                    engine.consecutive_effective_clicks = 1
                engine.effective_click_targets.add(target_coord)
                engine.quiescent_click_targets.discard(target_coord)
                # Active Causal Empowerment: Bind this coordinate as the active causal actuator
                if getattr(engine, "active_causal_actuator_coord", None) == target_coord:
                    engine.consecutive_causal_probes = (
                        getattr(engine, "consecutive_causal_probes", 0) + 1
                    )
                    if engine.consecutive_causal_probes >= getattr(
                        engine, "causal_actuator_streak_limit", 4
                    ):
                        engine.active_causal_actuator_coord = None
                        engine.consecutive_causal_probes = 0
                else:
                    engine.active_causal_actuator_coord = target_coord
                    engine.consecutive_causal_probes = 1
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

                has_displacement = any(
                    engine.is_displacement_action(a) for a in available_actions
                ) or any(dyn.is_displacement_action() for dyn in engine.action_dynamics.values())

                if is_effector_action and diff.changed_pixel_count > 0:
                    # Active Inference: Discover effector kinetics mode (attraction/pull vs repulsion/push)
                    if diff.changed_mask is not None and engine.prev_grid is not None:
                        bg_val = engine.bg_feature or 0
                        vanished_mask = (engine.prev_grid != bg_val) & (curr_grid == bg_val)
                        appeared_mask = (curr_grid != bg_val) & (engine.prev_grid == bg_val)
                        v_rows, v_cols = np.where(vanished_mask)
                        a_rows, a_cols = np.where(appeared_mask)
                        if len(v_rows) > 0 and len(a_rows) > 0:
                            prev_cr, prev_cc = float(np.mean(v_rows)), float(np.mean(v_cols))
                            curr_cr, curr_cc = float(np.mean(a_rows)), float(np.mean(a_cols))
                            disp_r, disp_c = curr_cr - prev_cr, curr_cc - prev_cc
                            off_r, off_c = float(tr) - prev_cr, float(tc) - prev_cc
                            dot = off_r * disp_r + off_c * disp_c
                            if dot > 0.5:
                                engine.effector_kinetics_mode = 1
                            elif dot < -0.5:
                                engine.effector_kinetics_mode = -1

                    # Only assign avatar position/features if the environment possesses displacement dynamics
                    # (embodied agent with teleportation/movement), NOT in pure allocentric/click puzzle environments.
                    if has_displacement:
                        known_avs = engine.avatar_features or (
                            {engine.avatar_feature} if engine.avatar_feature is not None else set()
                        )
                        can_relocate = False
                        if known_avs:
                            nearby = curr_grid[
                                max(0, tr - 2) : min(H, tr + 3), max(0, tc - 2) : min(W, tc + 3)
                            ]
                            if any(int(f) in known_avs for f in np.unique(nearby)):
                                can_relocate = True
                        elif engine.avatar_pos is None:
                            can_relocate = True

                        if can_relocate:
                            engine.avatar_pos = target_coord
                            ambient_bg = {0, engine.bg_feature or 0}
                            clicked_feats = {
                                int(f)
                                for f in np.unique(
                                    curr_grid[
                                        max(0, tr - 2) : min(H, tr + 3),
                                        max(0, tc - 2) : min(W, tc + 3),
                                    ]
                                )
                                if int(f) not in ambient_bg
                            }
                            if clicked_feats and not known_avs:
                                engine.avatar_features = clicked_feats
                                engine.avatar_feature = next(iter(clicked_feats))
                            from scipy.ndimage import label

                            active_feats = known_avs or clicked_feats
                            if active_feats:
                                mask_c = np.isin(curr_grid, list(active_feats))
                                lab_c, _ = label(mask_c)
                                if 0 <= tr < H and 0 <= tc < W and lab_c[tr, tc] > 0:
                                    comp_c = np.argwhere(lab_c == lab_c[tr, tc])
                                    engine.avatar_size = len(comp_c)
                                    engine.avatar_pos = (
                                        int(round(float(np.mean(comp_c[:, 0])))),
                                        int(round(float(np.mean(comp_c[:, 1])))),
                                    )
                            engine.mental_plan.clear()
                            engine.transition_to_phase(
                                EpistemicPhase.MENTAL_SIMULATION,
                                reason="avatar_relocation_assimilated",
                            )

                # Numerical / neighborhood cardinality constraint discovery:
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
                            engine.record_learned_barrier(hz)

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
                            engine.consecutive_goal_converging_clicks += 1
                        elif min_d_new > min_d_old:
                            if engine.active_goal_converging_coord == target_coord:
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
            if EpistemicFeedbackAssimilator._is_edge_hud(pe, H, W):
                continue
            best_ce_id = None
            min_dist = float("inf")
            best_delta = (0, 0)
            for ce_id, ce in unmatched_curr.items():
                if EpistemicFeedbackAssimilator._is_edge_hud(ce, H, W):
                    continue
                if pe.feature_id == ce.feature_id and pe.area == ce.area and pe.area <= 144:
                    dr = int(round(ce.centroid[0] - pe.centroid[0]))
                    dc = int(round(ce.centroid[1] - pe.centroid[1]))
                    dist = abs(dr) + abs(dc)
                    is_valid_displacement = 0 < dist <= 8 and (
                        dr == 0 or dc == 0 or (abs(dr) <= 4 and abs(dc) <= 4)
                    )
                    if is_valid_displacement:
                        if dist < min_dist:
                            min_dist = dist
                            best_ce_id = ce_id
                            best_delta = (dr, dc)
            if best_ce_id is not None:
                matched_ce = unmatched_curr.pop(best_ce_id)
                moved_entities.append((pe, matched_ce, best_delta))

        # Faculty: Active Manipulandum Dynamic Tracking
        manip = engine.working_memory.body_schema.active_manipulandum
        manip_moved = False
        if manip is not None and moved_entities:
            for pe_m, ce_m, delta_m in moved_entities:
                if (
                    ce_m.feature_id == manip.feature_id
                    or pe_m.grid_pos == manip.pos
                    or (
                        abs(pe_m.grid_pos[0] - manip.pos[0]) <= 1
                        and abs(pe_m.grid_pos[1] - manip.pos[1]) <= 1
                    )
                ):
                    dr_m, dc_m = delta_m
                    engine.working_memory.body_schema.update_manipulandum_pos(dr_m, dc_m)
                    manip_moved = True
                    if getattr(manip, "role", "") != "cargo":
                        if engine.avatar_pos is not None:
                            engine.avatar_pos = (
                                engine.avatar_pos[0] + dr_m,
                                engine.avatar_pos[1] + dc_m,
                            )
                        else:
                            engine.avatar_pos = manip.pos
                    if action is not None:
                        prev_tested = getattr(
                            engine.action_dynamics.get(action), "probes_tested", 0
                        )
                        engine.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr_m,
                            delta_c=dc_m,
                            confidence=0.8,
                            probes_tested=max(1, prev_tested + 1),
                        )
                        if action in engine.action_affordances:
                            engine.action_affordances[action].is_displacement = (
                                dr_m != 0 or dc_m != 0
                            )
                            engine.action_affordances[action].delta = (dr_m, dc_m)
                    logger.info(
                        "AutonomousEpistemicEngine: Active manipulandum translated by (%d, %d) via action %s",
                        dr_m,
                        dc_m,
                        action,
                    )
                    break

        # Faculty: Empirical Dial / State-Cycler Affordance Induction (Norman 1988 Parameter Control)
        if (
            action is not None
            and not is_effector_action
            and not engine.is_displacement_action(action)
        ):
            if 1 <= len(moved_entities) <= 2 and diff.changed_pixel_count <= 48:
                deltas = {m[2] for m in moved_entities}
                if len(deltas) == 1:
                    dr_sc, dc_sc = next(iter(deltas))
                    ce_sc = moved_entities[0][1]
                    if ce_sc.area <= 48 and (
                        (dr_sc == 0 and dc_sc != 0) or (dr_sc != 0 and dc_sc == 0)
                    ):
                        if action not in engine.value_mutator_actions:
                            engine.symbolic_theory.induce_index_selector(action)
                            engine.index_selector_actions.add(action)
                            if action in engine.action_affordances:
                                engine.action_affordances[action].is_index_selector = True
            elif len(moved_entities) == 0 and 1 <= diff.changed_pixel_count <= 16:
                # Check if a single focal indicator/marker transferred between entities
                is_marker_transfer = False
                if engine.prev_grid is not None and 2 <= diff.changed_pixel_count <= 8:
                    prev_g = engine.prev_grid
                    curr_g = curr_grid
                    diff_mask = prev_g != curr_g
                    diff_coords = np.argwhere(diff_mask)
                    from hbllm.hcir.world.cortex_motor import MotorCortexEffector

                    non_bg_diffs = [
                        (int(r), int(c))
                        for r, c in diff_coords
                        if not MotorCortexEffector.is_peripheral_frame_cell(
                            int(r), int(c), curr_g.shape[0], curr_g.shape[1], curr_g
                        )
                    ]
                    if len(non_bg_diffs) == 2:
                        c1, c2 = non_bg_diffs[0], non_bg_diffs[1]
                        v1_prev, v1_curr = int(prev_g[c1]), int(curr_g[c1])
                        v2_prev, v2_curr = int(prev_g[c2]), int(curr_g[c2])
                        if (v1_curr == v2_prev and v1_prev != bg and v2_curr != bg) or (
                            v2_curr == v1_prev and v2_prev != bg and v1_curr != bg
                        ):
                            is_marker_transfer = True

                if is_marker_transfer:
                    engine.mark_focus_switch_action(action)
                    engine.working_memory.focus_switch_actions.add(action)
                    # The new focus locus is the coordinate where the marker appeared in curr_g
                    new_locus = None
                    if curr_g[c1] == 0 or (v1_curr != v1_prev and v1_curr == v2_prev):
                        new_locus = c1
                    elif curr_g[c2] == 0 or (v2_curr != v2_prev and v2_curr == v1_prev):
                        new_locus = c2
                    else:
                        new_locus = c2

                    new_feat = None
                    if new_locus is not None:
                        lr, lc = new_locus
                        for dr in range(-2, 3):
                            for dc in range(-2, 3):
                                nr, nc = lr + dr, lc + dc
                                if 0 <= nr < H and 0 <= nc < W:
                                    val = int(curr_g[nr, nc])
                                    if val not in (bg, engine.bg_feature, 0):
                                        new_feat = val
                                        break
                            if new_feat is not None:
                                break

                    old_feat = (
                        getattr(engine, "prior_avatar_feature", None) or engine.avatar_feature
                    )
                    if new_feat is not None:
                        engine.prior_avatar_feature = old_feat
                        engine.avatar_features = {new_feat}
                        engine.avatar_feature = new_feat
                        engine.avatar_pos = None
                        engine.avatar_size = 0
                        engine._update_avatar_position_from_grid(
                            curr_g, {new_feat}, preferred_pos=new_locus
                        )
                    else:
                        engine.avatar_features.clear()
                        engine.avatar_feature = None
                        engine.avatar_pos = None
                    engine.mental_plan.clear()
                elif action not in engine.index_selector_actions:
                    engine.symbolic_theory.induce_value_mutator(action)
                    engine.value_mutator_actions.add(action)
                    if action in engine.action_affordances:
                        engine.action_affordances[action].is_value_mutator = True
                    if engine.prev_grid is not None:
                        diff_mask = engine.prev_grid != curr_grid
                        if np.any(diff_mask):
                            curr_sig = hash(curr_grid[diff_mask].tobytes())
                            prev_sig = hash(engine.prev_grid[diff_mask].tobytes())
                            engine.symbolic_theory.record_state_mutation(
                                action, curr_sig, prior_state_sig=prev_sig
                            )

            if (
                action is not None
                and action in engine.value_mutator_actions
                and engine.prev_grid is not None
            ):
                diff_mask = engine.prev_grid != curr_grid
                if np.any(diff_mask):
                    curr_sig = hash(curr_grid[diff_mask].tobytes())
                    prev_sig = hash(engine.prev_grid[diff_mask].tobytes())
                    engine.symbolic_theory.record_state_mutation(
                        action, curr_sig, prior_state_sig=prev_sig
                    )

        # Faculty: Spatiotemporal Lifecycle & Directed Causal Lineage DAG (W014, W059)
        if engine.deformation_tracker is not None:
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
            if engine.entity_lineage_graph is not None:
                engine.entity_lineage_graph.update(lifecycle_res.lineage_graph)

        # Faculty D: Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring)
        # Only evaluate bilateral pairing if entities belong to controllable agent features,
        # never autonomous mobile threats, lethal hazards, or background void.
        if engine.bilateral_state is None and len(moved_entities) >= 2:
            cargos_known = engine.symbolic_theory.cargo_features | getattr(
                engine, "learned_cargo_features", set()
            )
            recs_known = engine.symbolic_theory.receptacle_features
            active_m_id = (
                engine.working_memory.body_schema.active_manipulandum.entity_id
                if engine.working_memory.body_schema.has_manipulandum()
                and engine.working_memory.body_schema.active_manipulandum is not None
                else None
            )
            candidate_bilat = [
                (ce.grid_pos, delta, ce.feature_id)
                for pe, ce, delta in moved_entities
                if ce.area <= 64
                and ce.feature_id not in engine.mobile_threat_features
                and ce.feature_id not in engine.hazard_tracker.known_lethal_features
                and ce.feature_id not in cargos_known
                and ce.feature_id not in recs_known
                and getattr(ce, "id", getattr(ce, "entity_id", None)) != active_m_id
                and not engine.symbolic_theory.has_affordance(ce, "can_pickup")
                and not engine.symbolic_theory.has_affordance(ce, "can_carry")
                and not engine.symbolic_theory.has_affordance(ce, "cargo")
                and not engine.symbolic_theory.has_affordance(ce, "receptacle")
                and ce.feature_id != bg
                and ce.feature_id != engine.bg_feature
                and (ce.feature_id != 0 or (bg != 0 and engine.bg_feature != 0))
            ]
            if len(candidate_bilat) >= 2:
                bilat_state = engine.bilateral_integrator.detect_bilateral_pairing(
                    candidate_bilat, (H, W)
                )
                if bilat_state is not None:
                    engine.bilateral_state = bilat_state
        if engine.bilateral_state is not None:
            from scipy.ndimage import label

            f_list = list(engine.bilateral_state.features)
            if f_list:
                if engine.bilateral_state.quad_positions:
                    prev_positions = list(engine.bilateral_state.quad_positions)
                    all_centroids: list[tuple[int, int]] = []
                    for feat in f_list:
                        mask = curr_grid == feat
                        lab, num_comp = label(mask)
                        for comp_i in range(1, num_comp + 1):
                            pts = np.argwhere(lab == comp_i)
                            if len(pts) > 0:
                                cr = int(round(float(np.mean(pts[:, 0]))))
                                cc = int(round(float(np.mean(pts[:, 1]))))
                                all_centroids.append((cr, cc))
                    if len(all_centroids) >= len(prev_positions):
                        if len(all_centroids) > 8:
                            all_centroids.sort(
                                key=lambda c: min(
                                    abs(c[0] - p[0]) + abs(c[1] - p[1]) for p in prev_positions
                                )
                            )
                            all_centroids = all_centroids[:8]
                        import itertools

                        best_perm = None
                        min_tot_dist = float("inf")
                        for chosen_cands in itertools.permutations(
                            all_centroids, len(prev_positions)
                        ):
                            d = sum(
                                abs(p[0] - c[0]) + abs(p[1] - c[1])
                                for p, c in zip(prev_positions, chosen_cands)
                            )
                            if d < min_tot_dist:
                                min_tot_dist = d
                                best_perm = list(chosen_cands)
                        if best_perm is not None:
                            engine.bilateral_state.quad_positions = best_perm
                            engine.bilateral_state.pos1 = best_perm[0]
                            engine.bilateral_state.pos2 = best_perm[1]
                else:
                    f1 = f_list[0]
                    f2 = f_list[1] if len(f_list) > 1 else f_list[0]
                    prev_p1 = engine.bilateral_state.pos1
                    prev_p2 = engine.bilateral_state.pos2

                    if f1 == f2:
                        mask = curr_grid == f1
                        lab, num_comp = label(mask)
                        centroids: list[tuple[int, int]] = []
                        for comp_i in range(1, num_comp + 1):
                            pts = np.argwhere(lab == comp_i)
                            if len(pts) > 0:
                                cr = int(round(float(np.mean(pts[:, 0]))))
                                cc = int(round(float(np.mean(pts[:, 1]))))
                                centroids.append((cr, cc))
                        if len(centroids) >= 2:
                            best_pair = None
                            min_d = float("inf")
                            for i, c1 in enumerate(centroids):
                                for j, c2 in enumerate(centroids):
                                    if i != j:
                                        d = (
                                            abs(c1[0] - prev_p1[0])
                                            + abs(c1[1] - prev_p1[1])
                                            + abs(c2[0] - prev_p2[0])
                                            + abs(c2[1] - prev_p2[1])
                                        )
                                        if d < min_d:
                                            min_d = d
                                            best_pair = (c1, c2)
                            if best_pair is not None:
                                engine.bilateral_state.pos1 = best_pair[0]
                                engine.bilateral_state.pos2 = best_pair[1]
                        elif len(centroids) == 1:
                            single_centroid = centroids[0]
                            d1 = abs(single_centroid[0] - prev_p1[0]) + abs(
                                single_centroid[1] - prev_p1[1]
                            )
                            d2 = abs(single_centroid[0] - prev_p2[0]) + abs(
                                single_centroid[1] - prev_p2[1]
                            )
                            if d1 <= d2:
                                engine.bilateral_state.pos1 = single_centroid
                            else:
                                engine.bilateral_state.pos2 = single_centroid
                    else:
                        for feat, attr in [(f1, "pos1"), (f2, "pos2")]:
                            prev_pos = getattr(engine.bilateral_state, attr)
                            mask = curr_grid == feat
                            lab, num_comp = label(mask)
                            best_c = None
                            min_d = float("inf")
                            for comp_i in range(1, num_comp + 1):
                                pts = np.argwhere(lab == comp_i)
                                if len(pts) > 0:
                                    cr = int(round(float(np.mean(pts[:, 0]))))
                                    cc = int(round(float(np.mean(pts[:, 1]))))
                                    d = abs(cr - prev_pos[0]) + abs(cc - prev_pos[1])
                                    if d < min_d:
                                        min_d = d
                                        best_c = (cr, cc)
                            if best_c is not None:
                                setattr(engine.bilateral_state, attr, best_c)

        def _are_adjacent(e1: SpatialEntity, e2: SpatialEntity) -> bool:
            bb1 = e1.bounding_box
            bb2 = e2.bounding_box
            r_gap = max(0, bb1[0] - bb2[1] - 1, bb2[0] - bb1[1] - 1)
            c_gap = max(0, bb1[2] - bb2[3] - 1, bb2[2] - bb1[3] - 1)
            stride_tol = max(2, getattr(engine, "lattice_stride", 1) or 1)
            return r_gap <= stride_tol and c_gap <= stride_tol

        if (
            not is_win
            and not is_effector_action
            and not engine.avatar_features
            and engine.avatar_feature is None
        ):
            if moved_entities:
                delta_groups: dict[tuple[int, int], list[tuple[SpatialEntity, SpatialEntity]]] = {}
                for pe, ce, delta in moved_entities:
                    if ce.feature_id not in (0, bg, engine.bg_feature):
                        # Locality prior: an embodied avatar cannot teleport across > 8 cells in a single frame
                        if 0 < max(abs(delta[0]), abs(delta[1])) <= 8:
                            delta_groups.setdefault(delta, []).append((pe, ce))

                has_retained_dynamics = any(
                    dyn.probes_tested > 0 for dyn in engine.action_dynamics.values()
                )

                if delta_groups and not has_retained_dynamics:
                    # ── Fresh start (no prior knowledge) ──────────────────────
                    # Prefer groups with cardinal displacement and sensible locomotion stride (1..4)
                    def _delta_priority(
                        item: tuple[tuple[int, int], list[Any]],
                    ) -> tuple[int, int, int]:
                        d, pair_list = item
                        is_cardinal = 1 if (d[0] == 0 or d[1] == 0) else 0
                        stride_score = 1 if max(abs(d[0]), abs(d[1])) <= 4 else 0
                        return (len(pair_list), is_cardinal, stride_score)

                    best_delta, pairs = max(delta_groups.items(), key=_delta_priority)
                    EpistemicFeedbackAssimilator._commit_avatar_identity(
                        engine,
                        pairs,
                        best_delta,
                        action,
                        H,
                        W,
                        calibrate_dynamics=True,
                    )
                elif delta_groups:
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
                                if dyn.probes_tested > 0:
                                    is_match = obs_delta == expected
                                else:
                                    is_match = (
                                        (np.sign(obs_delta[0]) == np.sign(expected[0]))
                                        and (np.sign(obs_delta[1]) == np.sign(expected[1]))
                                        and (obs_delta != (0, 0))
                                    )
                                if is_match:
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

        # Differential Kinetic Motion Identification (Area MT/V5 Centroid Segregation):
        # Fallback when connected components fail to extract moving agent (e.g. avatar touches same-color walls)
        if (
            not is_win
            and not is_effector_action
            and not engine.avatar_features
            and engine.avatar_feature is None
            and diff.changed_pixel_count > 0
            and engine.prev_grid is not None
            and 1 <= diff.changed_pixel_count <= 64
        ):
            bg_val = engine.bg_feature if engine.bg_feature is not None else bg
            diff_mask = engine.prev_grid != curr_grid
            diff_coords = np.argwhere(diff_mask)
            # Universal physical invariant: spatial locomotion over a surface requires
            # covering and uncovering ambient surface/background cells. If zero background/floor
            # cells were involved in the difference, the change is entirely an internal
            # state/value mutation of a foreground object (e.g. Norman dial/parameter), NOT spatial locomotion.
            prev_diff_colors = set(engine.prev_grid[diff_mask].flatten())
            curr_diff_colors = set(curr_grid[diff_mask].flatten())
            ambient_bg = {0, bg_val}
            if engine.bg_feature is not None:
                ambient_bg.add(engine.bg_feature)
            touches_bg = bool((prev_diff_colors | curr_diff_colors) & ambient_bg)
            if not touches_bg:
                if action is not None and action not in engine.index_selector_actions:
                    engine.symbolic_theory.induce_value_mutator(action)
                    engine.value_mutator_actions.add(action)
                    if engine.prev_grid is not None and np.any(diff_mask):
                        curr_sig = hash(curr_grid[diff_mask].tobytes())
                        prev_sig = hash(engine.prev_grid[diff_mask].tobytes())
                        engine.symbolic_theory.record_state_mutation(
                            action, curr_sig, prior_state_sig=prev_sig
                        )
                return

            feats_in_diff = {
                int(f) for f in (prev_diff_colors | curr_diff_colors) if int(f) not in ambient_bg
            }
            best_feat: int | None = None
            diff_best_delta: tuple[int, int] | None = None
            best_pos: tuple[int, int] | None = None
            best_size: int = 0
            for f in feats_in_diff:
                c_prev = [
                    (int(r), int(c)) for r, c in diff_coords if int(engine.prev_grid[r, c]) == f
                ]
                c_curr = [(int(r), int(c)) for r, c in diff_coords if int(curr_grid[r, c]) == f]
                if c_prev and c_curr:
                    min_r_prev, max_r_prev = min(r for r, _ in c_prev), max(r for r, _ in c_prev)
                    min_c_prev, max_c_prev = min(c for _, c in c_prev), max(c for _, c in c_prev)
                    min_r_curr, max_r_curr = min(r for r, _ in c_curr), max(r for r, _ in c_curr)
                    min_c_curr, max_c_curr = min(c for _, c in c_curr), max(c for _, c in c_curr)
                    bbox_dr = min_r_curr - min_r_prev
                    bbox_dc = min_c_curr - min_c_prev
                    # Geometric rigid-body expansion check:
                    # If an entity translates by (dr, dc) != (0, 0), the total envelope containing
                    # both old and new positions must be strictly larger than the entity.
                    span_r = max(max_r_prev, max_r_curr) - min(min_r_prev, min_r_curr) + 1
                    span_c = max(max_c_prev, max_c_curr) - min(min_c_prev, min_c_curr) + 1
                    entity_h = max_r_prev - min_r_prev + 1
                    entity_w = max_c_prev - min_c_prev + 1
                    if span_r <= entity_h and span_c <= entity_w:
                        if action is not None and action not in engine.index_selector_actions:
                            engine.symbolic_theory.induce_value_mutator(action)
                            engine.value_mutator_actions.add(action)
                            if engine.prev_grid is not None and np.any(diff_mask):
                                curr_sig = hash(curr_grid[diff_mask].tobytes())
                                prev_sig = hash(engine.prev_grid[diff_mask].tobytes())
                                engine.symbolic_theory.record_state_mutation(
                                    action, curr_sig, prior_state_sig=prev_sig
                                )
                        continue
                    if bbox_dr == 0 and bbox_dc == 0:
                        if action is not None and action not in engine.index_selector_actions:
                            engine.symbolic_theory.induce_value_mutator(action)
                            engine.value_mutator_actions.add(action)
                            if engine.prev_grid is not None and np.any(diff_mask):
                                curr_sig = hash(curr_grid[diff_mask].tobytes())
                                prev_sig = hash(engine.prev_grid[diff_mask].tobytes())
                                engine.symbolic_theory.record_state_mutation(
                                    action, curr_sig, prior_state_sig=prev_sig
                                )
                        continue

                    dr = float(np.mean([r for r, _ in c_curr])) - float(
                        np.mean([r for r, _ in c_prev])
                    )
                    dc = float(np.mean([c for _, c in c_curr])) - float(
                        np.mean([c for _, c in c_prev])
                    )
                    k_dr = int(round(dr))
                    k_dc = int(round(dc))
                    if abs(k_dc) >= 2 * abs(k_dr):
                        k_dr = 0
                    elif abs(k_dr) >= 2 * abs(k_dc):
                        k_dc = 0
                    if (k_dr != 0 or k_dc != 0) and abs(k_dr) <= 8 and abs(k_dc) <= 8:
                        is_concordant = True
                        if action == 1 and k_dr >= 0:
                            is_concordant = False
                        elif action == 2 and k_dr <= 0:
                            is_concordant = False
                        elif action == 3 and k_dc >= 0:
                            is_concordant = False
                        elif action == 4 and k_dc <= 0:
                            is_concordant = False
                        if is_concordant:
                            best_feat = f
                            diff_best_delta = (k_dr, k_dc)
                            best_pos = (
                                int(round(float(np.mean([r for r, _ in c_curr])))),
                                int(round(float(np.mean([c for _, c in c_curr])))),
                            )
                            best_size = len(c_curr)
                            break
            if best_feat is not None and diff_best_delta is not None and best_pos is not None:
                engine.avatar_features = {best_feat}
                engine.avatar_feature = best_feat
                engine.avatar_pos = best_pos
                engine.avatar_size = best_size
                if action is not None:
                    engine.action_dynamics[action] = ActionDynamicsModel(
                        action_id=action,
                        delta_r=diff_best_delta[0],
                        delta_c=diff_best_delta[1],
                        confidence=0.8,
                        probes_tested=1,
                    )
                    if action in engine.action_affordances:
                        engine.action_affordances[action].is_displacement = True
                        engine.action_affordances[action].delta = best_delta
                logger.info(
                    "AutonomousEpistemicEngine: Avatar identified via differential kinetic stream — feature %s at pos %s (delta: %d, %d)",
                    best_feat,
                    best_pos,
                    best_delta[0],
                    best_delta[1],
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
        if (
            not is_win
            and not is_lost
            and kinetic_res.external_agents
            and engine.prev_grid is not None
            and action is not None
        ):
            self_feats = (
                set(kinetic_res.self_avatar.features)
                if kinetic_res.self_avatar is not None
                else set()
            )
            for ext in kinetic_res.external_agents:
                if ext.velocity != (0.0, 0.0):
                    cargos_known_threat = engine.symbolic_theory.cargo_features | getattr(
                        engine, "learned_cargo_features", set()
                    )
                    valid_threat_feats = {
                        f
                        for f in ext.features
                        if f != 0
                        and f != bg
                        and f != engine.bg_feature
                        and f not in engine.avatar_features
                        and (engine.avatar_feature is None or f != engine.avatar_feature)
                        and f not in self_feats
                        and f not in cargos_known_threat
                    }
                    for f in valid_threat_feats:
                        engine.symbolic_theory.walkable_features.discard(f)
                    engine.mobile_threat_features.update(valid_threat_feats)
                    engine.mobile_threat_features.discard(0)

        # Premotor Collision Cones: Update forward space-time trajectories
        if not is_win:
            static_walls = set(engine.learned_barriers)
            for r_w in range(H):
                for c_w in range(W):
                    if int(curr_grid[r_w, c_w]) in engine.symbolic_theory.barrier_features:
                        static_walls.add((r_w, c_w))
            cargos_known_t = engine.symbolic_theory.cargo_features | getattr(
                engine, "learned_cargo_features", set()
            )
            av_feats_t = engine.avatar_features or (
                {engine.avatar_feature} if engine.avatar_feature is not None else set()
            )
            threat_agents = [
                e
                for e in kinetic_res.external_agents
                if (
                    not e.features
                    or not any(f in cargos_known_t or f in av_feats_t for f in e.features)
                )
            ]
            engine.collision_cones.update_trajectories(
                kinetic_entities=threat_agents,
                static_barriers=static_walls,
                grid_shape=(H, W),
                horizon=12,
                oriented_threats=getattr(engine, "_prev_oriented_threats", None),
                mobile_threat_features=engine.mobile_threat_features,
                stride=getattr(engine, "spatial_lattice_stride", 1),
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
                if f != 0 and f != bg and f != engine.bg_feature
            }
            if valid_feats:
                if not engine.avatar_features:
                    engine.avatar_features = valid_feats
                    sorted_vf = sorted(
                        valid_feats,
                        key=lambda feat: (
                            kinetic_res.self_avatar.feature_distribution.get(feat, 0)
                            if hasattr(kinetic_res.self_avatar, "feature_distribution")
                            else 0
                        ),
                        reverse=True,
                    )
                    engine.avatar_feature = sorted_vf[0] if sorted_vf else next(iter(valid_feats))
                    engine.mobile_threat_features.difference_update(valid_feats)
                    engine.avatar_size = kinetic_res.self_avatar.area
                    engine.avatar_pos = (
                        int(round(kinetic_res.self_avatar.centroid[0])),
                        int(round(kinetic_res.self_avatar.centroid[1])),
                    )
                elif valid_feats & engine.avatar_features:
                    # Confirmed body schema: retain established avatar features against contamination
                    matching_cells = [
                        (r, c)
                        for r, c in kinetic_res.self_avatar.cells
                        if int(curr_grid[r, c]) in engine.avatar_features
                    ]
                    if matching_cells:
                        engine.avatar_pos = (
                            int(round(float(np.mean([r for r, _ in matching_cells])))),
                            int(round(float(np.mean([c for _, c in matching_cells])))),
                        )
                        engine.avatar_size = len(matching_cells)
                    else:
                        engine.avatar_pos = (
                            int(round(kinetic_res.self_avatar.centroid[0])),
                            int(round(kinetic_res.self_avatar.centroid[1])),
                        )
                if kinetic_res.raw_delta is not None and action is not None:
                    dr, dc = kinetic_res.raw_delta
                    if abs(dc) >= 2 * abs(dr):
                        dr = 0
                    elif abs(dr) >= 2 * abs(dc):
                        dc = 0
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
                    if dr != 0 or dc != 0:
                        EpistemicFeedbackAssimilator._synchronize_lattice_stride(
                            engine, max(abs(dr), abs(dc))
                        )
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
            # Check if known avatar entity is still visible on the grid
            avatar_still_present = (
                any(ce.feature_id in known_av_feats for ce in curr_entities)
                if known_av_feats
                else False
            )
            if avatar_still_present and not av_moved:
                # Avatar is present but stationary (e.g., bumped a wall or barrier)
                if prev_avatar_pos is not None:
                    engine.avatar_pos = prev_avatar_pos
                else:
                    av_components = [
                        ce
                        for ce in curr_entities
                        if ce.feature_id in known_av_feats
                        and not EpistemicFeedbackAssimilator._is_edge_hud(ce, H, W)
                    ]
                    if av_components:
                        all_cells = [
                            cell
                            for ce in av_components
                            for cell in ce.properties.get("cells", [ce.grid_pos])
                        ]
                        if all_cells:
                            engine.avatar_pos = (
                                int(round(float(np.mean([r for r, _ in all_cells])))),
                                int(round(float(np.mean([c for _, c in all_cells])))),
                            )
            if (
                not is_win
                and not is_lost
                and not av_moved
                and not avatar_still_present
                and not is_effector_action
                and moved_entities
                and (action in engine.action_dynamics or not action_data)
            ):
                valid_moved = [
                    (pe, ce, delta)
                    for pe, ce, delta in moved_entities
                    if ce.feature_id != bg
                    and ce.feature_id != engine.bg_feature
                    and (ce.feature_id != 0 or (bg != 0 and engine.bg_feature != 0))
                    and ce.area <= max(49, int(H * W * 0.15))
                    and not EpistemicFeedbackAssimilator._is_edge_hud(ce, H, W)
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
                    candidate_feats = {
                        ce.feature_id
                        for _, ce in comp_pairs
                        if ce.feature_id != 0
                        and ce.feature_id != bg
                        and ce.feature_id != engine.bg_feature
                    }
                    if candidate_feats:
                        if not engine.avatar_features:
                            engine.avatar_features = candidate_feats
                            engine.avatar_feature = next(iter(candidate_feats))
                            engine.avatar_size = sum(ce.area for _, ce in comp_pairs)
                        elif candidate_feats & engine.avatar_features:
                            engine.avatar_size = sum(
                                ce.area
                                for _, ce in comp_pairs
                                if ce.feature_id in engine.avatar_features
                            )
                    known_av_feats = engine.avatar_features
                    av_moved = [
                        item for item in moved_entities if item[1].feature_id in known_av_feats
                    ]
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
                if abs(dc) >= 2 * abs(dr):
                    dr = 0
                elif abs(dr) >= 2 * abs(dc):
                    dc = 0
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

                if not is_effector_action and not is_win and not is_lost:
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
                    if dr != 0 or dc != 0:
                        EpistemicFeedbackAssimilator._synchronize_lattice_stride(
                            engine, max(abs(dr), abs(dc))
                        )

                if not is_win:
                    manip_cargo_synced = False
                    if engine.working_memory.body_schema.has_manipulandum():
                        active_m = engine.working_memory.body_schema.active_manipulandum
                        cargos_known = engine.symbolic_theory.cargo_features | getattr(
                            engine, "learned_cargo_features", set()
                        )
                        for pe, ce, delta in moved_entities:
                            if ce.feature_id in known_av_feats:
                                continue
                            if (
                                active_m is not None
                                and (
                                    ce.feature_id == active_m.feature_id
                                    or ce.feature_id in cargos_known
                                    or engine.symbolic_theory.has_affordance(ce, "can_pickup")
                                )
                                and delta == (dr, dc)
                            ):
                                manip_cargo_synced = True
                                active_m.pos = ce.grid_pos
                                active_m.cells = ce.properties.get("cells", [ce.grid_pos])
                                break

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
                        is_towed = is_contact and not is_ahead and (delta == (dr, dc))

                        if (
                            (is_pushed or is_towed)
                            and ce.feature_id != 0
                            and ce.feature_id != bg
                            and ce.feature_id != engine.bg_feature
                        ):
                            # Composite sprite guard: if this entity's bounding box
                            # is contained within (or overlaps with) another moved
                            # entity whose feature is already known cargo and that
                            # moved identically, then this entity is a sub-component
                            # of the same composite sprite — do NOT induce it as
                            # standalone cargo.  This prevents e.g. a 4×4 crate's
                            # inner-color from being inducted as a separate cargo type
                            # when its border-color was already inducted.
                            _existing_cargo = engine.symbolic_theory.cargo_features
                            if ce.feature_id not in _existing_cargo:
                                ce_bb = ce.bounding_box
                                is_composite_sub = False
                                if ce_bb is not None:
                                    for pe2, ce2, delta2 in moved_entities:
                                        if ce2.feature_id == ce.feature_id:
                                            continue
                                        if ce2.feature_id not in _existing_cargo:
                                            continue
                                        if delta2 != delta:
                                            continue
                                        ce2_bb = ce2.bounding_box
                                        if ce2_bb is None:
                                            continue
                                        # Check if ce's bounding box is contained
                                        # within ce2's bounding box (with 1-cell
                                        # tolerance for stride alignment).
                                        if (
                                            ce_bb[0] >= ce2_bb[0] - 1
                                            and ce_bb[1] <= ce2_bb[1] + 1
                                            and ce_bb[2] >= ce2_bb[2] - 1
                                            and ce_bb[3] <= ce2_bb[3] + 1
                                        ):
                                            is_composite_sub = True
                                            break
                                if is_composite_sub:
                                    logger.debug(
                                        "AutonomousEpistemicEngine: Skipping composite sub-feature %d "
                                        "(bbox contained within known cargo feature)",
                                        ce.feature_id,
                                    )
                                    continue

                            engine.symbolic_theory.induce_cargo(ce.feature_id)
                            logger.info(
                                "AutonomousEpistemicEngine: Discovered %s CARGO (feat=%d, size=%d)",
                                "TOWED" if is_towed else "PUSHABLE",
                                ce.feature_id,
                                ce.area,
                            )
                            if is_towed:
                                engine.working_memory.body_schema.bind_manipulandum(
                                    entity_id=f"cargo_{ce.feature_id}_{ce.grid_pos[0]}_{ce.grid_pos[1]}",
                                    feature_id=ce.feature_id,
                                    pos=ce.grid_pos,
                                    cells=ce.properties.get("cells", [ce.grid_pos]),
                                    step=engine.step_counter,
                                    role="cargo",
                                )
                                manip_cargo_synced = True
                        elif delta == (dr, dc) and any(
                            _are_adjacent(ce, c_ce) for _, c_ce, _ in av_moved
                        ):
                            # Ensure ce is an organic compound avatar part, not environmental cargo or goal drop zone
                            if (
                                ce.feature_id != 0
                                and ce.feature_id != bg
                                and ce.feature_id != engine.bg_feature
                                and ce.feature_id not in engine.symbolic_theory.cargo_features
                                and ce.feature_id not in engine.symbolic_theory.goal_features
                                and ce.feature_id not in engine.symbolic_theory.receptacle_features
                            ):
                                engine.avatar_features.add(ce.feature_id)
                                engine.avatar_size += ce.area
                                logger.info(
                                    "AutonomousEpistemicEngine: Discovered compound avatar component (feat=%d, size=%d)",
                                    ce.feature_id,
                                    ce.area,
                                )
                        elif delta != (0, 0) and delta != (dr, dc):
                            # Autonomous dynamic entity (moved under its own agency) — not passive cargo
                            engine.symbolic_theory.cargo_features.discard(ce.feature_id)

                    if (
                        not manip_cargo_synced
                        and (dr != 0 or dc != 0)
                        and engine.working_memory.body_schema.has_manipulandum()
                        and getattr(
                            engine.working_memory.body_schema.active_manipulandum,
                            "role",
                            "",
                        )
                        == "cargo"
                    ):
                        # Avatar moved independently without cargo — cargo was decoupled or released
                        engine.working_memory.body_schema.unbind_manipulandum()
        # ── Proprioceptive Efference Copy Verification & Obstacle Grounding ──
        avatar_physically_displaced = bool(
            engine.avatar_pos is not None
            and prev_avatar_pos is not None
            and engine.avatar_pos != prev_avatar_pos
        )
        avatar_moved = bool(avatar_physically_displaced or manip_moved)
        if not is_effector_action and not is_known_displacement and not avatar_physically_displaced:
            engine.last_non_displacement_action = action
            is_interaction_act = bool(
                action in getattr(engine.symbolic_theory, "interaction_actions", set())
                or action == getattr(engine.symbolic_theory, "primary_interaction_action", None)
                or action == getattr(engine, "interaction_action", None)
                or action == 5
            )
            if is_interaction_act:
                if engine.working_memory.body_schema.has_manipulandum():
                    engine.working_memory.body_schema.unbind_manipulandum()
                    logger.info(
                        "CortexAssimilator: Unbound active manipulandum upon interaction action %s",
                        action,
                    )
                else:
                    stride = max(1, getattr(engine, "lattice_stride", 1) or 1)
                    av_p = engine.avatar_pos
                    if av_p is not None:
                        cargos_known = (
                            engine.symbolic_theory.cargo_features | engine.learned_cargo_features
                        )
                        for e in curr_entities:
                            if (
                                e.feature_id in cargos_known
                                or engine.symbolic_theory.has_affordance(e, "can_pickup")
                                or getattr(e, "role", "") == EntityRole.MANIPULABLE
                            ):
                                er, ec = e.grid_pos
                                if abs(er - av_p[0]) + abs(ec - av_p[1]) <= max(2, stride * 2):
                                    engine.working_memory.body_schema.bind_manipulandum(
                                        entity_id=f"cargo_{e.feature_id}_{er}_{ec}",
                                        feature_id=e.feature_id,
                                        pos=e.grid_pos,
                                        cells=e.properties.get("cells", [e.grid_pos]),
                                        step=engine.step_counter,
                                        role="cargo",
                                    )
                                    logger.info(
                                        "CortexAssimilator: Bound adjacent entity %s at %s as cargo upon interaction action %s",
                                        e.id,
                                        e.grid_pos,
                                        action,
                                    )
                                    break

        if is_known_displacement and prev_avatar_pos is not None:
            # 1. Physical resistance detection (Avatar did not move upon directional motor command)
            if not avatar_moved and not is_win:
                is_intentional_bump = (
                    engine.last_predicted_pos is not None
                    and engine.last_predicted_pos == prev_avatar_pos
                )
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
                                # Proprioceptive obstacle grounding: any feature blocking movement that is not
                                # verified safe/traversable is an empirical barrier
                                if (
                                    val not in av_feats
                                    and val not in engine.learned_goal_features
                                    and val not in cand_goal_feats
                                    and (kr, kc) not in cand_goal_positions
                                ):
                                    engine.record_learned_barrier((kr, kc))
                                    if (
                                        val in engine.symbolic_theory.walkable_features
                                        and val not in engine.verified_safe_features
                                    ):
                                        engine.symbolic_theory.walkable_features.discard(val)
                                    if val not in engine.verified_safe_features:
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
                                engine.record_learned_barrier((end_r, end_c))

                    if is_intentional_bump:
                        engine.working_memory.update_salience_baseline(engine)
                    elif engine.last_action is not None and prev_avatar_pos is not None:
                        engine.failed_transitions.add((prev_avatar_pos, engine.last_action))
                        is_known_disp = engine.is_displacement_action(engine.last_action)
                        if not is_known_disp:
                            engine.entity_action_failed_positions.setdefault(
                                engine.last_action, set()
                            ).add(prev_avatar_pos)
                            if len(engine.entity_action_failed_positions[engine.last_action]) >= 2:
                                engine.immobile_entity_actions.add(engine.last_action)
                            prev_inh = engine.inhibited_actions.get(engine.last_action, 0)
                            engine.inhibited_actions[engine.last_action] = max(prev_inh + 2, 3)
                        engine.simulation_cooldown = 0
                        engine.consecutive_simulation_failures = 0

            if diff.changed_pixel_count == 0 and not is_win and not is_lost:
                engine.consecutive_stuck_steps += 1
                if engine.last_action is not None:
                    recent_acts = list(engine.recent_actions)[-3:]
                    if len(recent_acts) >= 3 and all(a == engine.last_action for a in recent_acts):
                        engine.inhibited_actions[engine.last_action] = max(
                            engine.inhibited_actions.get(engine.last_action, 0) + 4, 6
                        )
                    if hasattr(engine, "working_memory") and hasattr(
                        engine.working_memory, "habenular_ior"
                    ):
                        engine.working_memory.habenular_ior.record_ineffectual_action(
                            avatar_pos=engine.avatar_pos or prev_avatar_pos,
                            action=engine.last_action,
                            action_data=engine.last_action_data,
                            streak=engine.consecutive_stuck_steps,
                        )
            elif prev_avatar_pos is not None and engine.avatar_pos is not None:
                if engine.avatar_pos == prev_avatar_pos and not is_win and not is_lost:
                    engine.consecutive_stuck_steps += 1
                    if engine.last_action is not None:
                        if hasattr(engine, "working_memory") and hasattr(
                            engine.working_memory, "habenular_ior"
                        ):
                            engine.working_memory.habenular_ior.record_ineffectual_action(
                                avatar_pos=engine.avatar_pos,
                                action=engine.last_action,
                                action_data=engine.last_action_data,
                                streak=engine.consecutive_stuck_steps,
                            )
                else:
                    engine.consecutive_stuck_steps = 0
            else:
                engine.consecutive_stuck_steps = 0

            # Avatar Agency Falsification Watchdog:
            # If an entity was committed as an avatar, but after taking multiple actions
            # it remains completely immobile (consecutive_stuck_steps >= 6) while diffs occur or actions fail,
            # the avatar hypothesis is falsified. Drop the stale avatar identity to allow re-induction.
            if (
                engine.avatar_features
                and engine.consecutive_stuck_steps >= 6
                and not is_win
                and not is_lost
            ):
                logger.info(
                    "AutonomousEpistemicEngine: Avatar agency falsified after %d consecutive immobile steps. "
                    "Invalidating stale avatar hypothesis %s at %s.",
                    engine.consecutive_stuck_steps,
                    engine.avatar_features,
                    engine.avatar_pos,
                )
                engine.avatar_features.clear()
                engine.avatar_feature = None
                engine.avatar_pos = None
                engine.avatar_size = 0
                engine.prior_avatar_feature = None
                engine.consecutive_stuck_steps = 0
                engine.mental_plan.clear()
                engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION

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
                "avatar_moved": 1 if (expected_dr != 0 or expected_dc != 0) else 0,
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
                elif engine.last_predicted_pos is not None and engine.avatar_pos is not None:
                    stride = max(1, getattr(engine, "lattice_stride", 1) or 1)
                    max_tol = 1 if stride > 2 else 0
                    dr_err = abs(engine.avatar_pos[0] - engine.last_predicted_pos[0])
                    dc_err = abs(engine.avatar_pos[1] - engine.last_predicted_pos[1])
                    if dr_err <= max_tol and dc_err <= max_tol:
                        pos_mismatch = False
                    else:
                        pos_mismatch = True

                    if pos_mismatch or (
                        not avatar_moved
                        and is_known_displacement
                        and (
                            engine.last_predicted_pos is None
                            or engine.last_predicted_pos != prev_avatar_pos
                        )
                    ):
                        discrepancy = True
                elif (
                    surprise_eval.is_surprising
                    and not avatar_moved
                    and is_known_displacement
                    and (
                        engine.last_predicted_pos is None
                        or engine.last_predicted_pos != prev_avatar_pos
                    )
                ):
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
                    engine.inhibited_actions[engine.last_action] = max(prev_inh + 2, 4)
                    if expected_dr != 0 or expected_dc != 0:
                        blocked_r = prev_avatar_pos[0] + expected_dr
                        blocked_c = prev_avatar_pos[1] + expected_dc
                        if (
                            0 <= blocked_r < curr_grid.shape[0]
                            and 0 <= blocked_c < curr_grid.shape[1]
                        ):
                            engine.record_learned_barrier((blocked_r, blocked_c))
                if engine.mental_plan:
                    logger.debug(
                        "AutonomousEpistemicEngine: Sensory discrepancy/surprise detected (pred=%s, actual=%s, surprise=%.3f). Invalidating %d-step mental plan.",
                        engine.last_predicted_pos,
                        engine.avatar_pos,
                        surprise_eval.surprise_score,
                        len(engine.mental_plan),
                    )
                    engine.mental_plan.clear()
                    engine.transition_to_phase(
                        EpistemicPhase.REPLANNING,
                        reason="motor_surprise_plan_divergence",
                    )
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
            """Perimeter status bar / HUD isolation: detects mutations in edge bands or behind solid non-background divider lines."""
            if H >= 24 and engine.prev_grid is not None:
                if engine.avatar_pos is not None:
                    if r >= H - 6 and engine.avatar_pos[0] < H - 8:
                        return True
                    if r < 3 and engine.avatar_pos[0] >= 5:
                        return True
                    min_r = max(0, min(engine.avatar_pos[0], r))
                    max_r = min(engine.prev_grid.shape[0] - 1, max(engine.avatar_pos[0], r))
                    for div_r in range(min_r + 1, max_r):
                        if 0 <= div_r < engine.prev_grid.shape[0]:
                            div_val = int(engine.prev_grid[div_r, 0])
                            if div_val != bg and np.all(engine.prev_grid[div_r, :] == div_val):
                                if div_r >= H - 16 or div_r <= 16:
                                    return True
                    min_c = max(0, min(engine.avatar_pos[1], c))
                    max_c = min(engine.prev_grid.shape[1] - 1, max(engine.avatar_pos[1], c))
                    for div_c in range(min_c + 1, max_c):
                        if 0 <= div_c < engine.prev_grid.shape[1]:
                            div_val = int(engine.prev_grid[0, div_c])
                            if div_val != bg and np.all(engine.prev_grid[:, div_c] == div_val):
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

        # Faculty: Telic Depletion Goal Induction
        # When an interventional action clears pixels of a non-background, non-avatar feature,
        # hypothesize that feature as an active depletion objective (e.g. board clearing / matching).
        if diff.mutated_pixels and not is_lost and not is_win and engine.prev_grid is not None:
            cargos_known = engine.symbolic_theory.cargo_features | getattr(
                engine, "learned_cargo_features", set()
            )
            for r, c, old_v, new_v in diff.mutated_pixels:
                if (
                    new_v == bg
                    and old_v != bg
                    and old_v not in av_feats
                    and old_v not in cargos_known
                    and not engine.symbolic_theory.is_barrier(old_v)
                    and old_v not in engine.mobile_threat_features
                    and old_v not in engine.hazard_tracker.known_lethal_features
                ):
                    prev_count = int(np.count_nonzero(engine.prev_grid == old_v))
                    curr_count = int(np.count_nonzero(curr_grid == old_v))
                    if curr_count < prev_count and prev_count <= 64:
                        if prev_avatar_pos is not None:
                            dist = abs(r - prev_avatar_pos[0]) + abs(c - prev_avatar_pos[1])
                            if dist <= 3 or engine.is_spatial_effector(action):
                                engine.symbolic_theory.induce_depletion_goal(old_v)
                                break
                        elif engine.is_spatial_effector(action):
                            engine.symbolic_theory.induce_depletion_goal(old_v)
                            break

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
                engine.record_learned_goal(win_target)
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
                    feat_extent = int(np.count_nonzero(engine.prev_grid == goal_feat))
                    is_substrate = (
                        goal_feat in engine.symbolic_theory.walkable_features or feat_extent > 64
                    )
                    if (
                        goal_feat != bg
                        and goal_feat != engine.bg_feature
                        and (goal_feat not in av_feats or is_distant_target)
                        and not engine.symbolic_theory.is_barrier(goal_feat)
                        and not is_substrate
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
                    sim_extent = int(np.count_nonzero(engine.prev_grid == sim_feat))
                    if (
                        sim_feat != bg
                        and sim_feat != engine.bg_feature
                        and sim_feat not in av_feats
                        and not engine.symbolic_theory.is_barrier(sim_feat)
                        and not (
                            sim_feat in engine.symbolic_theory.walkable_features or sim_extent > 64
                        )
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
            actor = prev_avatar_pos or engine.avatar_pos
            if actor is not None:
                engine.working_memory.acc_conflict.register_death_event(
                    actor_pos=(float(actor[0]), float(actor[1])),
                    step=engine.step_counter,
                    last_action=action,
                )

            # Faculty: Prefrontal Metacognitive Conflict-Driven Schema Revision
            if engine.working_memory.active_schema is not None:
                invalidated = engine.working_memory.handle_schema_failure(actor)
                if invalidated or engine.working_memory.acc_conflict.is_frustrated():
                    logger.warning(
                        "AutonomousEpistemicEngine: Active schema invalidated due to conflict/catastrophe. Shifting to EPISTEMIC_PROBE."
                    )
                    engine.transition_to_phase(
                        EpistemicPhase.EPISTEMIC_PROBE,
                        reason="metacognitive_schema_conflict_invalidation",
                    )
            # Faculty: Hippocampal Sharp-Wave Ripple (SWR) Backward Negative Replay
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
                    engine.record_learned_barrier(target_pos)
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
                    and (pe.feature_id != 0 or (bg != 0 and engine.bg_feature != 0))
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

        # Tool-Barrier Resonance Credit Assignment
        if (
            engine.prev_grid is not None
            and engine.working_memory.held_items
            and diff.changed_mask is not None
        ):
            changed_coords = np.argwhere(diff.changed_mask)
            for mr, mc in changed_coords:
                orig_val = int(engine.prev_grid[mr, mc])
                post_val = int(curr_grid[mr, mc])
                if (
                    orig_val in engine.symbolic_theory.barrier_features
                    or (int(mr), int(mc)) in engine.learned_barriers
                ) and (
                    post_val == engine.bg_feature
                    or post_val == 0
                    or engine.symbolic_theory.is_walkable(post_val)
                ):
                    for h_item in engine.working_memory.held_items:
                        engine.working_memory.register_tool_unlock(h_item.feature_id, orig_val)

        # Faculty: Hippocampal CA1/CA3 Experiential Trace Buffer
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
        if engine.prev_grid is not None:
            action_locus: tuple[int, int] | None = None
            if action_data and isinstance(action_data, dict):
                if "y" in action_data and "x" in action_data:
                    action_locus = (int(action_data["y"]), int(action_data["x"]))
                elif "r" in action_data and "c" in action_data:
                    action_locus = (int(action_data["r"]), int(action_data["c"]))
                elif "target" in action_data and isinstance(action_data["target"], (list, tuple)):
                    action_locus = (int(action_data["target"][0]), int(action_data["target"][1]))
            elif engine.avatar_pos is not None:
                action_locus = engine.avatar_pos

            engine.causal_cortex.bind_intervention(
                action=action,
                origin_pos=prev_avatar_pos,
                target_pos=engine.avatar_pos,
                obs_diff=diff,
                prev_grid=engine.prev_grid,
                curr_grid=curr_grid,
                bg_feature=engine.bg_feature,
                action_locus=action_locus,
                avatar_features=engine.avatar_features,
            )

        # Faculty: Prefrontal Inductive Hypothesis Testing & Relational Rule Discovery
        if engine.prev_grid is not None:
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

        # Enforce avatar/threat non-intersection invariant
        if engine.avatar_features:
            engine.mobile_threat_features.difference_update(engine.avatar_features)
        if engine.avatar_feature is not None:
            engine.mobile_threat_features.discard(engine.avatar_feature)

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export feedback assimilator capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="epistemic_feedback_assimilator",
            operator="==",
            value="ACTIVE",
            relation="motor_dynamics_assimilation",
            target_entity="cortex_assimilator",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering feedback assimilator."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="feedback_assimilator",
                node_data={
                    "faculty": "epistemic_feedback_assimilator",
                    "status": "ACTIVE",
                },
            )
        ]
