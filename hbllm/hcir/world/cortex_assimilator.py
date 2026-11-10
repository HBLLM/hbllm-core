"""Epistemic Feedback Assimilator Faculty.

Assimilates sensory feedback (pixel diffs, kinetic flow, reward signals)
into motor dynamics, avatar identity consensus, and empirical neuro-symbolic theory.
"""

from __future__ import annotations


def get_sub_attr(target: Any, attr: str, default: Any = None) -> Any:
    return getattr(target, attr, default) if target is not None else default


import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_causal import ActionAffordance
from hbllm.hcir.world.cortex_perception import FrameDiffType, PerceptionEngine
from hbllm.hcir.world.cortical_stage import StageResult, WorkingMemoryDelta
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.spatiotemporal_tracker import MorphologicalEntity

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext

from hbllm.hcir.world.percept import EpistemicPhase, TerminalSignal

logger = logging.getLogger(__name__)


class EpistemicFeedbackAssimilator:
    """Assimilates environmental feedback into motor dynamics and neuro-symbolic theory."""

    @staticmethod
    def _record_safe_traversal(
        view: Any,
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
        if view.prev_grid is None:
            return
        dr, dc = delta
        steps = max(abs(dr), abs(dc))
        sr = (dr > 0) - (dr < 0)
        sc = (dc > 0) - (dc < 0)
        av = view.avatar_features or (
            {view.avatar_feature} if view.avatar_feature is not None else set()
        )
        lethal = view.hazard_tracker.known_lethal_features
        for cr, cc in landing_cells:
            for k in range(0, max(steps, 1)):
                r, c = cr - sr * k, cc - sc * k
                if not (0 <= r < H and 0 <= c < W):
                    continue
                wf = int(view.prev_grid[r, c])
                if wf in av or wf in lethal:
                    continue
                view.symbolic_theory.induce_walkable(wf, evidence="traversal")

    @staticmethod
    def _synchronize_lattice_stride(view: Any, stride: int) -> None:
        """Propagate discovered lattice quantum to uncalibrated cardinal action priors."""
        if stride <= 1 or not hasattr(view, "action_dynamics"):
            return
        view.state.spatial_lattice_stride = stride
        for other_act, other_dyn in view.action_dynamics.items():
            if getattr(other_dyn, "probes_tested", 0) == 0:
                if abs(other_dyn.delta_r) == 1 and other_dyn.delta_c == 0:
                    other_dyn.delta_r *= stride
                    other_dyn.displacement = (float(other_dyn.delta_r), 0.0)
                    if hasattr(view, "action_affordances") and other_act in view.action_affordances:
                        view.action_affordances[other_act].delta = (other_dyn.delta_r, 0)
                elif abs(other_dyn.delta_c) == 1 and other_dyn.delta_r == 0:
                    other_dyn.delta_c *= stride
                    other_dyn.displacement = (0.0, float(other_dyn.delta_c))
                    if hasattr(view, "action_affordances") and other_act in view.action_affordances:
                        view.action_affordances[other_act].delta = (0, other_dyn.delta_c)

    @staticmethod
    def _commit_avatar_identity(
        view: Any,
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

        comp_pairs: list[tuple[SpatialEntity, SpatialEntity]] = [pairs[0]]
        for pe, ce in pairs[1:]:
            if any(_adjacent(ce, c_ce) for _, c_ce in comp_pairs):
                comp_pairs.append((pe, ce))

        view.state.avatar_features = {ce.feature_id for _, ce in comp_pairs}
        if view.state.avatar_features:
            view.state.avatar_feature = next(iter(view.state.avatar_features))
        view.mobile_threat_features.difference_update(view.avatar_features)
        view.hazard_tracker.known_lethal_features.difference_update(view.avatar_features)
        if hasattr(view, "episodic_cortex"):
            view.episodic_cortex.discovered_lethal_features.difference_update(view.avatar_features)
        view.state.avatar_size = sum(ce.area for _, ce in comp_pairs)
        all_cells = [
            cell for _, ce in comp_pairs for cell in ce.properties.get("cells", [ce.grid_pos])
        ]
        if all_cells:
            view.state.avatar_pos = (
                int(round(sum(c[0] for c in all_cells) / len(all_cells))),
                int(round(sum(c[1] for c in all_cells) / len(all_cells))),
            )
            view.state.avatar_h = max(c[0] for c in all_cells) - min(c[0] for c in all_cells) + 1
            view.state.avatar_w = max(c[1] for c in all_cells) - min(c[1] for c in all_cells) + 1
        # Features the avatar just traversed are verified safe
        EpistemicFeedbackAssimilator._record_safe_traversal(view, all_cells, delta, H, W)
        view.entity_action_failed_positions.clear()
        view.immobile_entity_actions.clear()

        if calibrate_dynamics and action is not None:
            view.action_dynamics[action] = ActionDynamicsModel(
                action_id=action,
                delta_r=dr,
                delta_c=dc,
                confidence=0.5,
                probes_tested=1,
            )
            if action in view.action_affordances:
                view.action_affordances[action].is_displacement = dr != 0 or dc != 0
                view.action_affordances[action].delta = (dr, dc)
            else:
                view.action_affordances[action] = ActionAffordance(
                    action_id=action,
                    is_displacement=(dr != 0 or dc != 0),
                    delta=(dr, dc),
                )
            if dr != 0 or dc != 0:
                EpistemicFeedbackAssimilator._synchronize_lattice_stride(
                    view, max(abs(dr), abs(dc))
                )

        logger.info(
            "AutonomousEpistemicEngine: Avatar identified (feats=%s, size=%d). "
            "Action %s -> delta=(%d, %d), calibrate=%s",
            view.avatar_features,
            view.avatar_size,
            action,
            dr,
            dc,
            calibrate_dynamics,
        )

    @staticmethod
    def assimilate(
        view: Any,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: TerminalSignal | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action: Any | None = None,
        action_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Assimilate sensory feedback from previous action into empirical world models."""
        is_success = is_success or terminal == TerminalSignal.SUCCESS
        is_failure = is_failure or terminal in (
            TerminalSignal.FAILURE,
            TerminalSignal.TIMEOUT,
        )
        if action is not None:
            view.state.last_action = action
            view.recent_actions.append(action)
        if action_data is not None:
            view.state.last_action_data = action_data

        if view.prev_grid is None or view.last_action is None:
            return

        view.state._feedback_assimilated = True
        prev_avatar_pos = view.avatar_pos

        curr_grid = view.normalize_sensory_input(curr_grid)
        if view.prev_grid is not None and view.prev_grid.shape != curr_grid.shape:
            # Lattice extent changed (new level/episode): prior-lattice
            # diffs are incomparable — start the episode's diff chain fresh.
            view.apply_wm_delta(WorkingMemoryDelta("prev_grid", None))
            return
        H, W = curr_grid.shape

        if not is_success:
            walkable_feats = view.symbolic_theory.walkable_features
            view.hazard_tracker.record_frame(
                view.step_counter,
                curr_grid,
                background_feature=view.bg_feature,
                avatar_features=view.avatar_features,
                walkable_features=walkable_feats,
            )
            # Faculty E: Basal Ganglia & Cerebellar Predictive Phase Entrainment
            view.cerebellar_clock.record_step(view.step_counter, curr_grid, is_reset=is_failure)
            if view.cerebellar_clock.detected_macro_period is not None:
                view.hazard_tracker.environmental_period = max(
                    view.hazard_tracker.environmental_period,
                    view.cerebellar_clock.detected_macro_period,
                )

        diff = view.compute_frame_diff(view.prev_grid, curr_grid)
        action = view.last_action
        action_data = view.last_action_data

        # WP0.3: classify the observed effect kind for the motor posterior.
        if action is not None and hasattr(view, "record_action_effect"):
            _is_eff = bool(
                (
                    action_data
                    and isinstance(action_data, dict)
                    and any(k in action_data for k in ("x", "y", "col", "row"))
                )
                or (hasattr(view, "is_spatial_effector") and view.is_spatial_effector(action))
            )
            if _is_eff:
                _kind = "point_effect"
            elif diff.changed_pixel_count == 0:
                _kind = "no_op"
            else:
                _av_prev = getattr(view, "prev_avatar_pos", None)
                _av_cur = getattr(view, "avatar_pos", None)
                _kind = (
                    "translate"
                    if (_av_prev is not None and _av_cur is not None and _av_prev != _av_cur)
                    else "state_change"
                )
            view.record_action_effect(action, _kind)

        aff = view.action_affordances.get(action)

        if aff is not None:
            is_effector_action = aff.requires_spatial_target
        else:
            is_effector_action = bool(
                action is not None
                and (
                    view.is_spatial_effector(action)
                    or (
                        action_data
                        and isinstance(action_data, dict)
                        and any(k in action_data for k in ("x", "y", "col", "row"))
                    )
                )
            )

        is_focus_switch = view.is_focus_switch_action(action)
        if is_effector_action or is_focus_switch:
            view.entity_action_failed_positions.clear()
            view.immobile_entity_actions.clear()
            if is_focus_switch:
                view.inhibited_actions[action] = max(view.inhibited_actions.get(action, 0), 8)
                view.avatar_features.clear()
                # Preserve prior avatar feature for re-grounding recovery
                # in case the focus switch was actually an INTERACT action
                if view.avatar_feature is not None:
                    view.state.prior_avatar_feature = view.avatar_feature
                view.state.avatar_feature = None
                view.state.avatar_pos = None
                view.mental_plan.clear()

        if is_effector_action and action not in view.action_affordances:
            view.action_affordances[action] = ActionAffordance(
                action_id=action,
                requires_spatial_target=True,
            )

        act_dyn = view.action_dynamics.get(action) if not is_effector_action else None
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
            view.working_memory.last_effector_target_coord = target_coord
            tr_t, tc_t = target_coord
            if 0 <= tr_t < H and 0 <= tc_t < W:
                actuated_feat = int(curr_grid[tr_t, tc_t])
                bg_val = view.bg_feature if view.bg_feature is not None else 0
                if (
                    actuated_feat != bg_val
                    and not view.symbolic_theory.is_barrier(actuated_feat)
                    and actuated_feat not in view.hazard_tracker.known_lethal_features
                ):
                    manip_id = f"manip_{actuated_feat}_{tr_t}_{tc_t}"
                    view.working_memory.last_effector_target_entity_id = manip_id
                    view.working_memory.body_schema.bind_manipulandum(
                        entity_id=manip_id,
                        feature_id=actuated_feat,
                        pos=(tr_t, tc_t),
                        cells=[(tr_t, tc_t)],
                        step=view.step_counter,
                    )

        # Empirical State-Reversion / Undo Action Discovery:
        # If executing this action restored the grid to the previous-previous state,
        # ground this action as an UNDO/REVERT affordance and inhibit it during forward progress.
        if (
            action is not None
            and hasattr(view, "grid_history")
            and len(view.grid_history) >= 2
            and not is_effector_action
            and not view.is_displacement_action(action)
        ):
            if np.array_equal(curr_grid, view.grid_history[-2]):
                view.symbolic_theory.induce_undo_action(action)
                view.undo_actions.add(action)
                if action in view.action_affordances:
                    view.action_affordances[action].is_undo = True
                view.inhibited_actions[action] = 999999
                logger.info(
                    "AutonomousEpistemicEngine: Grounded empirical UNDO / STATE_REVERSION affordance for action %s",
                    action,
                )

        if diff.changed_pixel_count == 0:
            view.consecutive_quiescent_actions += 1
            view.state.consecutive_effective_actuations = 0
            view.state.last_effective_actuation_coord = None
            view.state.active_goal_converging_coord = None
            view.state.consecutive_goal_converging_actuations = 0
            if get_engine_state_attr(view, "active_causal_actuator_coord", None) == target_coord:
                view.state.active_causal_actuator_coord = None
                view.state.consecutive_causal_probes = 0
            if target_coord is not None:
                view.quiescent_actuation_targets.add(target_coord)
                view.effective_actuation_targets.discard(target_coord)
                tr, tc = target_coord
                if 0 <= tr < H and 0 <= tc < W:
                    probed_feat = int(curr_grid[tr, tc])
                    view.working_memory.visuospatial.record_probe(
                        pos=target_coord,
                        feature_id=probed_feat,
                        step=view.step_counter,
                    )
                    # Categorical Causal Lifting: If actuating background produced 0 diff, background is inert
                    if probed_feat == view.bg_feature:
                        view.quiescent_features.add(probed_feat)
                    else:
                        q_count = get_engine_state_attr(view, "feature_quiescent_counts", None)
                        if q_count is None:
                            q_count = get_engine_state_attr(view, "_feature_quiescent_counts", None)
                        if q_count is None:
                            q_count = {}
                            view.state.feature_quiescent_counts = q_count
                            view.state._feature_quiescent_counts = q_count
                        q_count[probed_feat] = q_count.get(probed_feat, 0) + 1
                        if q_count[probed_feat] >= 2 and probed_feat not in view.effective_features:
                            view.quiescent_features.add(probed_feat)
            if action is not None and not is_effector_action:
                prev_inh = view.inhibited_actions.get(action, 0)
                inh_steps = 4 if view.consecutive_quiescent_actions >= 2 else 3
                view.inhibited_actions[action] = max(prev_inh + 2, inh_steps)
                if prev_avatar_pos is not None:
                    view.failed_transitions.add((prev_avatar_pos, action))

            if is_effector_action and target_coord is not None:
                view.quiescent_actuation_targets.add(target_coord)
                if hasattr(view, "working_memory") and hasattr(
                    view.working_memory, "habenular_ior"
                ):
                    tok = (action, target_coord[0], target_coord[1])
                    view.working_memory.habenular_ior.active_inhibitions[(view.avatar_pos, tok)] = (
                        20
                    )

            if is_effector_action and not is_success and not is_failure:
                return
        else:
            view.state.consecutive_quiescent_actions = 0
            if target_coord is not None:
                tr, tc = target_coord
                if 0 <= tr < H and 0 <= tc < W:
                    view.working_memory.visuospatial.record_probe(
                        pos=target_coord,
                        feature_id=int(curr_grid[tr, tc]),
                        step=view.step_counter,
                    )
                if view.last_effective_actuation_coord == target_coord:
                    view.consecutive_effective_actuations += 1
                else:
                    view.state.last_effective_actuation_coord = target_coord
                    view.state.consecutive_effective_actuations = 1
                view.effective_actuation_targets.add(target_coord)
                view.quiescent_actuation_targets.discard(target_coord)
                # Active Causal Empowerment: Bind this coordinate as the active causal actuator
                if (
                    get_engine_state_attr(view, "active_causal_actuator_coord", None)
                    == target_coord
                ):
                    view.state.consecutive_causal_probes = (
                        get_engine_state_attr(view, "consecutive_causal_probes", 0) + 1
                    )
                    if view.consecutive_causal_probes >= getattr(
                        view, "causal_actuator_streak_limit", 4
                    ):
                        view.state.active_causal_actuator_coord = None
                        view.state.consecutive_causal_probes = 0
                else:
                    view.state.active_causal_actuator_coord = target_coord
                    view.state.consecutive_causal_probes = 1
                if view.prev_grid is not None:
                    r, c = target_coord
                    if 0 <= r < view.prev_grid.shape[0] and 0 <= c < view.prev_grid.shape[1]:
                        target_feat = int(view.prev_grid[r, c])
                        view.effective_features.add(target_feat)
                if diff.changed_mask is not None:
                    mut_cells = [(int(r), int(c)) for r, c in zip(*np.where(diff.changed_mask))]
                    view.actuation_affordances[target_coord] = mut_cells
                    view.working_memory.visuospatial.record_diff_stencil(
                        target_coord, set(mut_cells)
                    )

                if is_effector_action and diff.changed_pixel_count > 0:
                    if not view.has_displacement_capabilities():
                        view.state.avatar_pos = target_coord

                    # Active Inference: Discover effector kinetics mode (attraction/pull vs repulsion/push)
                    if diff.changed_mask is not None and view.prev_grid is not None:
                        bg_val = view.bg_feature or 0
                        vanished_mask = (view.prev_grid != bg_val) & (curr_grid == bg_val)
                        appeared_mask = (curr_grid != bg_val) & (view.prev_grid == bg_val)
                        v_rows, v_cols = np.where(vanished_mask)
                        a_rows, a_cols = np.where(appeared_mask)
                        if len(v_rows) > 0 and len(a_rows) > 0:
                            prev_cr, prev_cc = float(np.mean(v_rows)), float(np.mean(v_cols))
                            curr_cr, curr_cc = float(np.mean(a_rows)), float(np.mean(a_cols))
                            disp_r, disp_c = curr_cr - prev_cr, curr_cc - prev_cc
                            off_r, off_c = float(tr) - prev_cr, float(tc) - prev_cc
                            dot = off_r * disp_r + off_c * disp_c
                            if dot > 0.5:
                                view.state.effector_kinetics_mode = 1
                            elif dot < -0.5:
                                view.state.effector_kinetics_mode = -1

                    if not view.has_displacement_capabilities():
                        actuated_feats = {
                            int(f)
                            for f in np.unique(
                                curr_grid[
                                    max(0, tr - 2) : min(H, tr + 3), max(0, tc - 2) : min(W, tc + 3)
                                ]
                            )
                            if int(f) != view.bg_feature
                        }
                        if actuated_feats:
                            view.state.avatar_features = actuated_feats
                            view.state.avatar_feature = next(iter(actuated_feats))
                            from scipy.ndimage import label

                            mask_c = np.isin(curr_grid, list(actuated_feats))
                            lab_c, _ = label(mask_c)
                            if 0 <= tr < H and 0 <= tc < W and lab_c[tr, tc] > 0:
                                comp_c = np.argwhere(lab_c == lab_c[tr, tc])
                                view.state.avatar_size = len(comp_c)
                                view.state.avatar_pos = (
                                    int(round(float(np.mean(comp_c[:, 0])))),
                                    int(round(float(np.mean(comp_c[:, 1])))),
                                )
                        view.mental_plan.clear()
                        view.transition_to_phase(
                            EpistemicPhase.MENTAL_SIMULATION,
                            reason="avatar_relocation_assimilated",
                        )

                # Numerical / neighborhood cardinality constraint discovery:
                if get_sub_attr(view.working_memory, "has_cardinality_hypothesis", False):
                    tr, tc = target_coord
                    if 0 <= tr < curr_grid.shape[0] and 0 <= tc < curr_grid.shape[1]:
                        new_feat = int(curr_grid[tr, tc])
                        if 0 <= new_feat <= 8 and new_feat != view.bg_feature:
                            new_safe, new_hazards = (
                                view.working_memory.register_cardinality_constraint(
                                    center=target_coord,
                                    count=new_feat,
                                    grid_shape=curr_grid.shape,
                                    radius=1,
                                )
                            )
                            for hz in new_hazards:
                                view.hazard_tracker.register_lethal_feature(
                                    int(curr_grid[hz[0], hz[1]])
                                )
                                view.record_learned_barrier(hz)

            # Faculty: Ventromedial Prefrontal Cortex (vmPFC) Remote Causal Attribution
            loci: list[tuple[int, int]] = []
            if target_coord is not None:
                loci.append(target_coord)
            if view.avatar_pos is not None:
                loci.append(view.avatar_pos)
            if prev_avatar_pos is not None and prev_avatar_pos != view.avatar_pos:
                loci.append(prev_avatar_pos)

            if view.prev_grid is not None:
                for locus in set(loci):
                    confirmed_affs = view.working_memory.remote_causal.record_transition(
                        prev_grid=view.prev_grid,
                        curr_grid=curr_grid,
                        action_pos=locus,
                        background_feature=view.bg_feature,
                        avatar_features=view.avatar_features,
                    )
                    for aff in confirmed_affs:
                        if (
                            aff.mutated_remote_feature == view.bg_feature
                            or view.symbolic_theory.is_walkable(aff.mutated_remote_feature)
                        ):
                            view.learned_barriers.discard(aff.remote_pos)
                        view.working_memory.activated_triggers.add(aff.trigger_pos)

            # Dorsal Visual Stream (V4/MT): Object motion tracking & Teleological Distance Gradient
            if diff.changed_mask is not None and view.prev_grid is not None:
                bg = view.bg_feature if view.bg_feature is not None else 0
                disappeared_pts = np.argwhere(diff.changed_mask & (view.prev_grid != bg))
                appeared_pts = np.argwhere(diff.changed_mask & (curr_grid != bg))
                if 1 <= len(disappeared_pts) <= 256 and 1 <= len(appeared_pts) <= 256:
                    p_old = np.mean(disappeared_pts, axis=0)
                    p_new = np.mean(appeared_pts, axis=0)
                    goal_positions: list[tuple[int, int]] = list(view.learned_goal_positions)
                    for g in PerceptionEngine.detect_structural_goals(curr_grid, bg=bg):
                        if "position" in g:
                            goal_positions.append(g["position"])
                    for gf in view.learned_goal_features:
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
                            view.state.active_goal_converging_coord = target_coord
                            view.consecutive_goal_converging_actuations += 1
                        elif min_d_new > min_d_old:
                            if view.active_goal_converging_coord == target_coord:
                                view.state.active_goal_converging_coord = None
                                view.state.consecutive_goal_converging_actuations = 0
                            if target_coord is not None:
                                view.entity_visit_counts[
                                    f"actuate_{target_coord[0]}_{target_coord[1]}"
                                ] = (
                                    view.entity_visit_counts.get(
                                        f"actuate_{target_coord[0]}_{target_coord[1]}", 0
                                    )
                                    + 5
                                )

        # ── Intuitive Physics Transition Tracking ────────────────────────────
        view.physics_engine.record_transition(
            prev_pos=prev_avatar_pos,
            curr_pos=view.avatar_pos,
            is_displacement_action=is_known_displacement,
            commanded_delta=act_dyn.get_displacement() if act_dyn else (0, 0),
        )

        # ── A. Motor Calibration & Proprioception ────────────────────────────
        H, W = curr_grid.shape
        bg = view.estimate_background(curr_grid)
        prev_entities = view.extract_entities(view.prev_grid, bg)
        curr_entities = view.extract_entities(curr_grid, bg)

        unmatched_prev: list[SpatialEntity] = []
        unmatched_curr: dict[int, SpatialEntity] = {id(ce): ce for ce in curr_entities}

        # Per-key buckets preserve dict-order tie-breaking of the original
        # first-match scan while avoiding the O(prev x curr) pairwise sweep.
        _static_bucket: dict[tuple[Any, Any, Any], list[int]] = {}
        for ce_id, ce in unmatched_curr.items():
            _static_bucket.setdefault((ce.feature_id, ce.area, ce.grid_pos), []).append(ce_id)

        for pe in prev_entities:
            stat_ids = _static_bucket.get((pe.feature_id, pe.area, pe.grid_pos))
            if stat_ids:
                stat_match = stat_ids.pop(0)
                del unmatched_curr[stat_match]
            else:
                unmatched_prev.append(pe)

        _move_bucket: dict[tuple[Any, Any], list[int]] = {}
        for ce_id, ce in unmatched_curr.items():
            _move_bucket.setdefault((ce.feature_id, ce.area), []).append(ce_id)

        moved_entities: list[tuple[SpatialEntity, SpatialEntity, tuple[int, int]]] = []
        for pe in unmatched_prev:
            pe_fid = pe.feature_id
            pe_area = pe.area
            best_ce_id = None
            min_dist = float("inf")
            best_delta = (0, 0)
            if pe_area <= 144:
                pe_cr, pe_cc = pe.centroid
                for ce_id in _move_bucket.get((pe_fid, pe_area), ()):
                    ce = unmatched_curr.get(ce_id)
                    if ce is None:
                        continue
                    dr = int(round(ce.centroid[0] - pe_cr))
                    dc = int(round(ce.centroid[1] - pe_cc))
                    dist = abs(dr) + abs(dc)
                    is_valid_displacement = (0 < dist < max(H, W) and (dr == 0 or dc == 0)) or (
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

        # Faculty: Active Manipulandum Dynamic Tracking
        manip = view.working_memory.body_schema.active_manipulandum
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
                    view.working_memory.body_schema.update_manipulandum_pos(dr_m, dc_m)
                    manip_moved = True
                    if view.avatar_pos is not None:
                        view.state.avatar_pos = (
                            view.avatar_pos[0] + dr_m,
                            view.avatar_pos[1] + dc_m,
                        )
                    else:
                        view.state.avatar_pos = manip.pos
                    if action is not None:
                        prev_tested = getattr(view.action_dynamics.get(action), "probes_tested", 0)
                        view.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr_m,
                            delta_c=dc_m,
                            confidence=0.8,
                            probes_tested=max(1, prev_tested + 1),
                        )
                        if action in view.action_affordances:
                            view.action_affordances[action].is_displacement = dr_m != 0 or dc_m != 0
                            view.action_affordances[action].delta = (dr_m, dc_m)
                    logger.info(
                        "AutonomousEpistemicEngine: Active manipulandum translated by (%d, %d) via action %s",
                        dr_m,
                        dc_m,
                        action,
                    )
                    break

        # Faculty: Empirical Dial / State-Cycler Affordance Induction (Norman 1988 Parameter Control)
        if action is not None and not is_effector_action:
            if len(moved_entities) == 1:
                pe_sc, ce_sc, (dr_sc, dc_sc) = moved_entities[0]
                if (dr_sc == 0 and dc_sc != 0) or (dr_sc != 0 and dc_sc == 0):
                    view.symbolic_theory.induce_index_selector(action)
                    view.index_selector_actions.add(action)
                    if action in view.action_affordances:
                        view.action_affordances[action].is_index_selector = True
            elif len(moved_entities) == 0 and (
                diff.diff_type in (FrameDiffType.IN_PLACE_MUTATION, FrameDiffType.INDEX_CYCLE)
                or getattr(diff, "is_in_place_mutation", False)
                or (
                    diff.changed_pixel_count > 0
                    and not getattr(diff, "is_global_transition", False)
                )
            ):
                view.symbolic_theory.induce_value_mutator(action)
                view.value_mutator_actions.add(action)
                if action in view.action_affordances:
                    view.action_affordances[action].is_value_mutator = True

        # Faculty: Spatiotemporal Lifecycle & Directed Causal Lineage DAG (W014, W059)
        # WP0: run only when the frame diff intersects tracked entity masks —
        # Euler/fission/fusion can only fire on cells that actually moved.
        _tracked_cells: set[tuple[int, int]] = set()
        for _e in prev_entities:
            _tracked_cells.update(_e.properties.get("cells", [_e.grid_pos]))
        for _e in curr_entities:
            _tracked_cells.update(_e.properties.get("cells", [_e.grid_pos]))
        _diff_cells: set[tuple[int, int]] = set(_tracked_cells)
        if view.prev_grid is not None and view.prev_grid.shape == curr_grid.shape:
            _diff_cells = {
                (int(r), int(c))
                for r, c in np.argwhere(np.asarray(view.prev_grid) != np.asarray(curr_grid))
            }
        if view.deformation_tracker is not None and (
            not _tracked_cells or (_diff_cells & _tracked_cells)
        ):
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
            lifecycle_res = view.deformation_tracker.track_lifecycle(
                morph_prev, morph_curr, conservation_required=False
            )
            view.state.latest_lifecycle_result = lifecycle_res
            if view.entity_lineage_graph is not None:
                view.entity_lineage_graph.update(lifecycle_res.lineage_graph)

        # Faculty D: Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring)
        # Only evaluate bilateral pairing if entities belong to controllable agent features,
        # never autonomous mobile threats, lethal hazards, or background void.
        if view.bilateral_state is None and len(moved_entities) >= 2:
            av_feats = view.avatar_features or (
                {view.avatar_feature} if view.avatar_feature is not None else set()
            )
            candidate_bilat = [
                (ce.grid_pos, delta, ce.feature_id)
                for pe, ce, delta in moved_entities
                if ce.area <= 64
                and (
                    (
                        not av_feats
                        and not get_engine_state_attr(view, "_avatar_controllability_evidence", {})
                    )
                    or ce.feature_id in av_feats
                    or len(
                        get_engine_state_attr(view, "_avatar_controllability_evidence", {}).get(
                            ce.feature_id, []
                        )
                    )
                    > 0
                )
                and ce.feature_id not in view.mobile_threat_features
                and ce.feature_id not in view.hazard_tracker.known_lethal_features
                and ce.feature_id != bg
                and ce.feature_id != view.bg_feature
            ]
            if len(candidate_bilat) >= 2:
                bilat_state = view.bilateral_integrator.detect_bilateral_pairing(
                    candidate_bilat, (H, W)
                )
                if bilat_state is not None:
                    view.state.bilateral_state = bilat_state
        if view.bilateral_state is not None:
            from scipy.ndimage import label

            f_list = list(view.bilateral_state.features)
            if f_list:
                if view.bilateral_state.quad_positions:
                    prev_positions = list(view.bilateral_state.quad_positions)
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
                            view.bilateral_state.quad_positions = best_perm
                            view.bilateral_state.pos1 = best_perm[0]
                            view.bilateral_state.pos2 = best_perm[1]
                else:
                    f1 = f_list[0]
                    f2 = f_list[1] if len(f_list) > 1 else f_list[0]
                    prev_p1 = view.bilateral_state.pos1
                    prev_p2 = view.bilateral_state.pos2

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
                                view.bilateral_state.pos1 = best_pair[0]
                                view.bilateral_state.pos2 = best_pair[1]
                        elif len(centroids) == 1:
                            single_centroid = centroids[0]
                            d1 = abs(single_centroid[0] - prev_p1[0]) + abs(
                                single_centroid[1] - prev_p1[1]
                            )
                            d2 = abs(single_centroid[0] - prev_p2[0]) + abs(
                                single_centroid[1] - prev_p2[1]
                            )
                            if d1 <= d2:
                                view.bilateral_state.pos1 = single_centroid
                            else:
                                view.bilateral_state.pos2 = single_centroid
                    else:
                        for feat, attr in [(f1, "pos1"), (f2, "pos2")]:
                            prev_pos = get_sub_attr(view.bilateral_state, attr)
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
                                setattr(view.bilateral_state, attr, best_c)

        def _are_adjacent(e1: SpatialEntity, e2: SpatialEntity) -> bool:
            bb1 = e1.bounding_box
            bb2 = e2.bounding_box
            r_gap = max(0, bb1[0] - bb2[1] - 1, bb2[0] - bb1[1] - 1)
            c_gap = max(0, bb1[2] - bb2[3] - 1, bb2[2] - bb1[3] - 1)
            return r_gap <= 1 and c_gap <= 1

        if (
            not is_success
            and not is_effector_action
            and not view.avatar_features
            and view.avatar_feature is None
        ):
            if moved_entities:
                delta_groups: dict[tuple[int, int], list[tuple[SpatialEntity, SpatialEntity]]] = {}
                for pe, ce, delta in moved_entities:
                    delta_groups.setdefault(delta, []).append((pe, ce))

                has_retained_dynamics = any(
                    dyn.probes_tested > 0 for dyn in view.action_dynamics.values()
                )

                if not has_retained_dynamics:
                    # ── Fresh start (no prior knowledge) ──────────────────────
                    # No dynamics to cross-reference. Use the original heuristic:
                    # the largest group of entities sharing the same displacement
                    # is most likely the avatar (first level, typically no enemies).
                    best_delta, pairs = max(delta_groups.items(), key=lambda item: len(item[1]))
                    EpistemicFeedbackAssimilator._commit_avatar_identity(
                        view,
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
                            if feat != bg and feat != view.bg_feature:
                                view._avatar_controllability_evidence.setdefault(feat, []).append(
                                    (action, delta_val)
                                )

                    # Check if any feature has demonstrated controllability:
                    # its observed delta matches the expected action delta across
                    # at least 2 DIFFERENT displacement actions.
                    best_candidate: int | None = None
                    best_score = 0
                    for feat, observations in view._avatar_controllability_evidence.items():
                        confirmed_actions: set[int] = set()
                        for obs_action, obs_delta in observations:
                            dyn = view.action_dynamics.get(obs_action)
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
                        last_obs = view._avatar_controllability_evidence[best_candidate][-1]
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
                                view,
                                avatar_pairs,
                                commit_delta,
                                action,
                                H,
                                W,
                                calibrate_dynamics=False,  # keep retained dynamics
                            )
                            view._avatar_controllability_evidence.clear()
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
                            len(view._avatar_controllability_evidence),
                        )

        # Faculty C: Dorsal Visual Stream (Area MT/V5) Kinetic Figure-Ground Segregation
        cmd_delta = act_dyn.get_displacement() if act_dyn else None
        kinetic_res = view.dorsal_kinetic_stream.segregate_motion(
            prev_grid=view.prev_grid,
            curr_grid=curr_grid,
            commanded_delta=cmd_delta,
            background_feature=bg,
            known_avatar_features=view.avatar_features,
        )
        if kinetic_res.external_agents and view.prev_grid is not None and action is not None:
            self_feats = (
                set(kinetic_res.self_avatar.features)
                if kinetic_res.self_avatar is not None
                else set()
            )
            for ext in kinetic_res.external_agents:
                if ext.velocity != (0.0, 0.0):
                    valid_threat_feats = {
                        f
                        for f in ext.features
                        if f != bg
                        and f != view.bg_feature
                        and f not in view.avatar_features
                        and (view.avatar_feature is None or f != view.avatar_feature)
                        and f not in self_feats
                    }
                    view.mobile_threat_features.update(valid_threat_feats)

        # Premotor Collision Cones: Update forward space-time trajectories
        static_walls = set(view.learned_barriers)
        for r_w in range(H):
            for c_w in range(W):
                c_val = int(curr_grid[r_w, c_w])
                if c_val in view.symbolic_theory.barrier_features:
                    static_walls.add((r_w, c_w))
                elif (
                    len(view.symbolic_theory.walkable_features) > 0
                    and view.bg_feature is not None
                    and view.bg_feature not in view.symbolic_theory.walkable_features
                    and c_val == view.bg_feature
                ):
                    static_walls.add((r_w, c_w))

        # Dynamic entities cannot be static barriers to themselves
        for ext_entity in kinetic_res.external_agents:
            if ext_entity.cells:
                static_walls.difference_update(ext_entity.cells)
            if ext_entity.centroid is not None:
                cr_i, cc_i = int(round(ext_entity.centroid[0])), int(round(ext_entity.centroid[1]))
                static_walls.discard((cr_i, cc_i))
        view.collision_cones.update_trajectories(
            kinetic_entities=kinetic_res.external_agents,
            static_barriers=static_walls,
            grid_shape=(H, W),
            horizon=12,
            oriented_threats=get_engine_state_attr(view, "_prev_oriented_threats", None),
            mobile_threat_features=view.mobile_threat_features,
            stride=get_engine_state_attr(view, "spatial_lattice_stride", 1),
        )

        if (
            not is_success
            and not is_effector_action
            and kinetic_res.self_avatar is not None
            and kinetic_res.self_avatar.confidence >= 0.70
            and cmd_delta is not None
            and (cmd_delta[0] != 0 or cmd_delta[1] != 0)
        ):
            valid_feats = {
                f for f in kinetic_res.self_avatar.features if f != bg and f != view.bg_feature
            }
            if valid_feats:
                view.state.avatar_features = valid_feats
                view.state.avatar_feature = next(iter(valid_feats))
                view.mobile_threat_features.difference_update(valid_feats)
                for vf in valid_feats:
                    view.symbolic_theory.induce_avatar(vf, evidence="kinetic_stream")
                cells_to_purge = set(kinetic_res.self_avatar.cells)
                if kinetic_res.raw_delta is not None:
                    dr_p, dc_p = kinetic_res.raw_delta
                    cells_to_purge.update(
                        (r - dr_p, c - dc_p) for r, c in kinetic_res.self_avatar.cells
                    )
                view.learned_goal_positions.difference_update(cells_to_purge)
                view.state.avatar_size = kinetic_res.self_avatar.area
                view.state.avatar_pos = (
                    int(round(kinetic_res.self_avatar.centroid[0])),
                    int(round(kinetic_res.self_avatar.centroid[1])),
                )
                if kinetic_res.self_avatar.cells:
                    ac = kinetic_res.self_avatar.cells
                    view.state.avatar_h = max(c[0] for c in ac) - min(c[0] for c in ac) + 1
                    view.state.avatar_w = max(c[1] for c in ac) - min(c[1] for c in ac) + 1
                if kinetic_res.raw_delta is not None and action is not None:
                    dr, dc = kinetic_res.raw_delta
                    if abs(dc) >= 2 * abs(dr):
                        dr = 0
                    elif abs(dr) >= 2 * abs(dc):
                        dc = 0
                    EpistemicFeedbackAssimilator._record_safe_traversal(
                        view, kinetic_res.self_avatar.cells, (dr, dc), H, W
                    )
                    if (
                        action not in view.action_dynamics
                        or not view.action_dynamics[action].is_displacement_action()
                    ):
                        dyn = view.action_dynamics.get(action)
                        prev_p = getattr(dyn, "probes_tested", 0) if dyn is not None else 0
                        view.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr,
                            delta_c=dc,
                            confidence=0.8,
                            probes_tested=max(1, prev_p + 1),
                        )
                    else:
                        view.action_dynamics[action].update_from_trial((dr, dc), success=True)
                    if dr != 0 or dc != 0:
                        EpistemicFeedbackAssimilator._synchronize_lattice_stride(
                            view, max(abs(dr), abs(dc))
                        )
                logger.info(
                    "AutonomousEpistemicEngine: Avatar segregated/re-grounded via MT/V5 kinetic stream (feats=%s, size=%d, pos=%s)",
                    view.avatar_features,
                    view.avatar_size,
                    view.avatar_pos,
                )

        elif not is_effector_action:
            known_av_feats = view.avatar_features or (
                {view.avatar_feature} if view.avatar_feature is not None else set()
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
                for ce in curr_entities:
                    if ce.feature_id in known_av_feats:
                        view.state.avatar_pos = ce.grid_pos
                        break
            if (
                not is_success
                and not is_failure
                and not av_moved
                and not avatar_still_present
                and not is_effector_action
                and moved_entities
                and (action in view.action_dynamics or not action_data)
            ):
                valid_moved = [
                    (pe, ce, delta)
                    for pe, ce, delta in moved_entities
                    if ce.feature_id != bg
                    and ce.feature_id != view.bg_feature
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
                    view.state.avatar_features = {
                        ce.feature_id
                        for _, ce in comp_pairs
                        if ce.feature_id != bg and ce.feature_id != view.bg_feature
                    }
                    if view.state.avatar_features:
                        view.state.avatar_feature = next(iter(view.state.avatar_features))
                    view.state.avatar_size = sum(ce.area for _, ce in comp_pairs)
                    known_av_feats = view.avatar_features
                    av_moved = [
                        item for item in moved_entities if item[1].feature_id in known_av_feats
                    ]
                    view.state.last_non_displacement_action = None

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
                view.state.last_non_displacement_action = None
                all_cells = [
                    cell
                    for _, ce, _ in av_moved
                    for cell in ce.properties.get("cells", [ce.grid_pos])
                ]
                if all_cells:
                    view.state.avatar_pos = (
                        int(round(sum(c[0] for c in all_cells) / len(all_cells))),
                        int(round(sum(c[1] for c in all_cells) / len(all_cells))),
                    )
                    view.state.avatar_h = (
                        max(c[0] for c in all_cells) - min(c[0] for c in all_cells) + 1
                    )
                    view.state.avatar_w = (
                        max(c[1] for c in all_cells) - min(c[1] for c in all_cells) + 1
                    )
                if not is_success:
                    EpistemicFeedbackAssimilator._record_safe_traversal(
                        view, all_cells, (dr, dc), H, W
                    )

                if not is_effector_action:
                    if action not in view.action_dynamics:
                        view.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr,
                            delta_c=dc,
                            confidence=0.6,
                            probes_tested=1,
                        )
                    else:
                        view.action_dynamics[action].update_from_trial(
                            (dr, dc), success=True, learning_rate=0.5
                        )
                    if action in view.action_affordances:
                        view.action_affordances[action].is_displacement = dr != 0 or dc != 0
                        view.action_affordances[action].delta = (dr, dc)
                    else:
                        view.action_affordances[action] = ActionAffordance(
                            action_id=action,
                            is_displacement=(dr != 0 or dc != 0),
                            delta=(dr, dc),
                        )
                    if dr != 0 or dc != 0:
                        EpistemicFeedbackAssimilator._synchronize_lattice_stride(
                            view, max(abs(dr), abs(dc))
                        )

                if not is_success:
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
                            view.symbolic_theory.induce_cargo(ce.feature_id)
                            logger.info(
                                "AutonomousEpistemicEngine: Discovered PUSHABLE CARGO (feat=%d, size=%d)",
                                ce.feature_id,
                                ce.area,
                            )
                        elif delta != (0, 0):
                            # Autonomous dynamic entity (moved under its own agency) — not passive cargo
                            view.symbolic_theory.cargo_features.discard(ce.feature_id)
                        elif delta == (dr, dc) and any(
                            _are_adjacent(ce, c_ce) for _, c_ce, _ in av_moved
                        ):
                            view.avatar_features.add(ce.feature_id)
                            view.avatar_size += ce.area
                            view.symbolic_theory.cargo_features.discard(ce.feature_id)
                            logger.info(
                                "AutonomousEpistemicEngine: Discovered compound avatar component (feat=%d, size=%d)",
                                ce.feature_id,
                                ce.area,
                            )
        # ── Proprioceptive Efference Copy Verification & Obstacle Grounding ──
        avatar_moved = bool(
            (
                view.avatar_pos is not None
                and prev_avatar_pos is not None
                and view.avatar_pos != prev_avatar_pos
            )
            or manip_moved
        )
        if not is_effector_action and not is_known_displacement and not avatar_moved:
            view.state.last_non_displacement_action = action

        if view.avatar_pos is not None:
            view.learned_goal_positions.discard(view.avatar_pos)
            if prev_avatar_pos is not None:
                view.learned_goal_positions.discard(prev_avatar_pos)
            view.learned_goal_positions.difference_update(view.recent_positions)
            if view.avatar_features:
                for af in view.avatar_features:
                    view.symbolic_theory.induce_avatar(af, evidence="motor_grounding")
                view.hazard_tracker.known_lethal_features.difference_update(view.avatar_features)
                if hasattr(view, "episodic_cortex"):
                    view.episodic_cortex.discovered_lethal_features.difference_update(
                        view.avatar_features
                    )
            if view.avatar_feature is not None:
                view.symbolic_theory.induce_avatar(view.avatar_feature, evidence="motor_grounding")
                view.hazard_tracker.known_lethal_features.discard(view.avatar_feature)
                if hasattr(view, "episodic_cortex"):
                    view.episodic_cortex.discovered_lethal_features.discard(view.avatar_feature)

        if is_known_displacement and prev_avatar_pos is not None:
            # 1. Physical resistance detection (Avatar did not move upon directional motor command)
            if not avatar_moved and not is_success:
                if act_dyn is not None and act_dyn.is_displacement_action():
                    # Motor command failed to displace avatar! Ground obstacle cell regardless of other background motion
                    exp_dr, exp_dc = act_dyn.get_displacement()
                    step_r = int(np.sign(exp_dr))
                    step_c = int(np.sign(exp_dc))
                    dist = max(abs(exp_dr), abs(exp_dc))

                    if dist > 0 and prev_avatar_pos is not None:
                        av_feats = view.avatar_features or (
                            {view.avatar_feature} if view.avatar_feature is not None else set()
                        )
                        av_rad_r = max(0, (get_engine_state_attr(view, "avatar_h", 1) or 1) // 2)
                        av_rad_c = max(0, (get_engine_state_attr(view, "avatar_w", 1) or 1) // 2)

                        # Proprioceptive obstacle grounding: only the immediate leading edge cell
                        # can physically impede motor displacement. Distant features across void are NEVER obstacles.
                        adj_r = prev_avatar_pos[0] + step_r * (av_rad_r + 1)
                        adj_c = prev_avatar_pos[1] + step_c * (av_rad_c + 1)
                        if 0 <= adj_r < H and 0 <= adj_c < W:
                            val = (
                                int(view.prev_grid[adj_r, adj_c])
                                if view.prev_grid is not None
                                else int(curr_grid[adj_r, adj_c])
                            )
                            is_confirmed_goal = (
                                val in view.symbolic_theory.goal_features
                                or (adj_r, adj_c) in view.learned_goal_positions
                            )
                            is_avatar = (
                                val in av_feats
                                or val in view.avatar_features
                                or val == view.avatar_feature
                            )
                            if (
                                not is_avatar
                                and not is_confirmed_goal
                                and val not in view.symbolic_theory.walkable_features
                            ):
                                # Multi-cell leading edge recording
                                if step_c != 0:
                                    for r_off in range(-av_rad_r, av_rad_r + 1):
                                        edge_r = prev_avatar_pos[0] + r_off
                                        if 0 <= edge_r < H and 0 <= adj_c < W:
                                            view.record_learned_barrier((edge_r, adj_c))
                                elif step_r != 0:
                                    for c_off in range(-av_rad_c, av_rad_c + 1):
                                        edge_c = prev_avatar_pos[1] + c_off
                                        if 0 <= adj_r < H and 0 <= edge_c < W:
                                            view.record_learned_barrier((adj_r, edge_c))
                                else:
                                    view.record_learned_barrier((adj_r, adj_c))

                                if val not in view.verified_safe_features:
                                    view.symbolic_theory.retract_walkable(
                                        val, reason="collision_obstacle"
                                    )
                                    if not view.symbolic_theory.is_barrier(val):
                                        view.symbolic_theory.induce_barrier(
                                            val, evidence="collision"
                                        )
                                        logger.info(
                                            "AutonomousEpistemicEngine: Proprioceptive obstacle grounded at leading edge (%d, %d) with feature %d",
                                            adj_r,
                                            adj_c,
                                            val,
                                        )
                                        if view.bg_feature == val or bg == val:
                                            view.state.bg_feature = view.estimate_background(
                                                curr_grid
                                            )
                        else:
                            end_r = prev_avatar_pos[0] + step_r * dist
                            end_c = prev_avatar_pos[1] + step_c * dist
                            if 0 <= end_r < H and 0 <= end_c < W:
                                view.record_learned_barrier((end_r, end_c))

                    if view.last_action is not None and prev_avatar_pos is not None:
                        view.failed_transitions.add((prev_avatar_pos, view.last_action))
                        view.entity_action_failed_positions.setdefault(view.last_action, set()).add(
                            prev_avatar_pos
                        )
                        if len(view.entity_action_failed_positions[view.last_action]) >= 2:
                            view.immobile_entity_actions.add(view.last_action)
                        prev_inh = view.inhibited_actions.get(view.last_action, 0)
                        view.inhibited_actions[view.last_action] = max(prev_inh + 2, 3)
                        view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))
                        view.apply_wm_delta(
                            WorkingMemoryDelta("consecutive_simulation_failures", 0)
                        )

            if diff.changed_pixel_count == 0 and not is_success and not is_failure:
                view.consecutive_stuck_steps += 1
                if view.last_action is not None:
                    prev_inh = view.inhibited_actions.get(view.last_action, 0)
                    view.inhibited_actions[view.last_action] = max(prev_inh + 3, 5)
                    if view.consecutive_stuck_steps >= 2:
                        view.inhibited_actions[view.last_action] = max(prev_inh + 6, 8)
                    if prev_avatar_pos is not None:
                        view.failed_transitions.add((prev_avatar_pos, view.last_action))
                        view.working_memory.habenular_ior.failed_transitions.add(
                            (prev_avatar_pos, view.last_action)
                        )
            elif prev_avatar_pos is not None and view.avatar_pos is not None:
                if view.avatar_pos == prev_avatar_pos and not is_success and not is_failure:
                    view.consecutive_stuck_steps += 1
                    if view.last_action is not None:
                        prev_inh = view.inhibited_actions.get(view.last_action, 0)
                        view.inhibited_actions[view.last_action] = max(prev_inh + 2, 4)
                        if view.is_displacement_action(view.last_action):
                            view.failed_transitions.add((prev_avatar_pos, view.last_action))
                            view.working_memory.habenular_ior.failed_transitions.add(
                                (prev_avatar_pos, view.last_action)
                            )
                else:
                    view.state.consecutive_stuck_steps = 0
            else:
                view.state.consecutive_stuck_steps = 0

            # 2. Predictive coding efference copy divergence check & surprise computation
            expected_dr = (
                view.last_predicted_pos[0] - prev_avatar_pos[0]
                if view.last_predicted_pos is not None and prev_avatar_pos is not None
                else (act_dyn.delta_r if act_dyn is not None and is_known_displacement else 0)
            )
            expected_dc = (
                view.last_predicted_pos[1] - prev_avatar_pos[1]
                if view.last_predicted_pos is not None and prev_avatar_pos is not None
                else (act_dyn.delta_c if act_dyn is not None and is_known_displacement else 0)
            )
            actual_dr = (
                view.avatar_pos[0] - prev_avatar_pos[0]
                if view.avatar_pos is not None and prev_avatar_pos is not None
                else 0
            )
            actual_dc = (
                view.avatar_pos[1] - prev_avatar_pos[1]
                if view.avatar_pos is not None and prev_avatar_pos is not None
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

            confidence = 0.85 if view.mental_plan else (0.70 if is_known_displacement else 0.40)
            salience = 1.3 if diff.changed_pixel_count > 0 else 0.8
            surprise_eval = view.surprise_engine.evaluate_surprise(
                prediction_id=f"step_{view.step_counter}",
                expected_state=expected_state,
                actual_state=actual_state,
                confidence=confidence,
                attention_salience=salience,
                prediction_source="motor_calibration",
                context_signature=f"act_{action}",
            )
            view.state.last_surprise = surprise_eval.surprise_score
            view.state.last_surprise_eval = surprise_eval

            discrepancy = False
            if not is_success:
                if (
                    not avatar_moved
                    and diff.changed_pixel_count == 0
                    and (
                        view.last_predicted_pos is None
                        or view.last_predicted_pos != prev_avatar_pos
                    )
                ):
                    discrepancy = True
                elif (
                    view.last_predicted_pos is not None
                    and view.avatar_pos is not None
                    and max(
                        abs(view.avatar_pos[0] - view.last_predicted_pos[0]),
                        abs(view.avatar_pos[1] - view.last_predicted_pos[1]),
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
                    view.working_memory.orient_attention((sal_r, sal_c))

            if discrepancy:
                did_displace_correctly = (expected_dr != 0 and actual_dr * expected_dr > 0) or (
                    expected_dc != 0 and actual_dc * expected_dc > 0
                )
                if (
                    view.last_action is not None
                    and prev_avatar_pos is not None
                    and not did_displace_correctly
                ):
                    view.failed_transitions.add((prev_avatar_pos, view.last_action))
                    prev_inh = view.inhibited_actions.get(view.last_action, 0)
                    view.inhibited_actions[view.last_action] = max(prev_inh + 2, 3)
                if view.mental_plan:
                    logger.debug(
                        "AutonomousEpistemicEngine: Sensory discrepancy/surprise detected (pred=%s, actual=%s, surprise=%.3f). Invalidating %d-step mental plan.",
                        view.last_predicted_pos,
                        view.avatar_pos,
                        surprise_eval.surprise_score,
                        len(view.mental_plan),
                    )
                    view.mental_plan.clear()
                    view.transition_to_phase(
                        EpistemicPhase.REPLANNING,
                        reason="motor_surprise_plan_divergence",
                    )
                    if not did_displace_correctly:
                        view.consecutive_plan_failures += 1
                        if view.consecutive_plan_failures >= 2:
                            view.apply_wm_delta(WorkingMemoryDelta("exploration_cooldown", 2))
                            view.apply_wm_delta(WorkingMemoryDelta("consecutive_plan_failures", 0))
            else:
                if avatar_moved or diff.changed_pixel_count > 0:
                    view.apply_wm_delta(WorkingMemoryDelta("consecutive_plan_failures", 0))

        # ── B. Environmental Mutation Induction ──────────────────────────────
        # Per-call invariants resolved once: the owned-subsystem reads below are
        # re-evaluated per mutated pixel otherwise (journaled-view overhead).
        _sym_theory = view.symbolic_theory
        _bg_feature = view.bg_feature
        _prev_grid = view.prev_grid
        _avatar_pos_now = view.avatar_pos
        _hazard_tracker = view.hazard_tracker

        movable_features = set(_sym_theory.cargo_features)
        movable_features.update(view.avatar_features)
        if view.avatar_feature is not None:
            movable_features.add(view.avatar_feature)
        if _hazard_tracker is not None and hasattr(_hazard_tracker, "mobile_threat_features"):
            movable_features.update(_hazard_tracker.mobile_threat_features)

        # Divider-line scan is invariant for the whole assimilate call:
        # compute each row's dominant value/count once (lazily), then reuse.
        _divider_rows: np.ndarray | None = None

        def _hud_dividers() -> np.ndarray:
            nonlocal _divider_rows
            rows = _prev_grid
            n_vals = int(rows.max()) + 1
            counts = np.stack([np.bincount(row.astype(np.int64), minlength=n_vals) for row in rows])
            dom_vals = np.argmax(counts, axis=1)
            dom_counts = counts[np.arange(rows.shape[0]), dom_vals]
            _divider_rows = (dom_vals != bg) & (dom_counts >= int(rows.shape[1] * 0.85))
            return _divider_rows

        # The HUD predicate depends only on the mutated row: the divider-slice
        # test is precomputed once per row instead of per mutated pixel.
        _hud_row_flags: np.ndarray | None = None

        def _hud_flags() -> np.ndarray:
            nonlocal _hud_row_flags
            pH, _ = _prev_grid.shape
            max_hud_depth = max(16, pH // 4)
            div_rows = _divider_rows if _divider_rows is not None else _hud_dividers()
            flags = np.zeros(pH, dtype=bool)
            # Top peripheral band partitioned by a solid non-background divider
            # line spanning >= 85% width.
            for r in range(min(max_hud_depth + 1, pH)):
                if div_rows[r + 1 : min(max_hud_depth + 1, pH)].any():
                    flags[r] = True
            # Bottom peripheral band, symmetric to the top band.
            for r in range(max(0, pH - 1 - max_hud_depth), pH):
                if not flags[r] and div_rows[max(0, pH - 1 - max_hud_depth) : r].any():
                    flags[r] = True
            _hud_row_flags = flags
            return flags

        def is_hud_mutation(r: int, c: int) -> bool:
            """Perimeter status bar / HUD isolation: detects mutations in peripheral bands separated by solid divider lines."""
            if _prev_grid is None:
                return False
            pH, pW = _prev_grid.shape
            if pH < 24 or pW < 24:
                return False
            flags = _hud_row_flags if _hud_row_flags is not None else _hud_flags()
            return bool(flags[r])

        av_radius = max(2, int(round(math.sqrt(max(1, view.avatar_size)))))

        def is_local_to_avatar(r: int, c: int) -> bool:
            if _avatar_pos_now is not None:
                if max(abs(r - _avatar_pos_now[0]), abs(c - _avatar_pos_now[1])) <= av_radius:
                    return True
            if prev_avatar_pos is not None:
                if max(abs(r - prev_avatar_pos[0]), abs(c - prev_avatar_pos[1])) <= av_radius:
                    return True
            return False

        # Autonomous Dynamic Entities & Rigid Mass Conservation Detection:
        # If pixels of a feature appear and disappear elsewhere in balanced numbers,
        # it is a translating dynamic entity (patroller/hazard), NOT an environmental toggle.
        lost_counts: dict[int, int] = {}
        gained_counts: dict[int, int] = {}
        for r, c, old_v, new_v in diff.mutated_pixels:
            if not is_hud_mutation(r, c) and not is_local_to_avatar(r, c):
                if old_v != bg and old_v != _bg_feature:
                    lost_counts[old_v] = lost_counts.get(old_v, 0) + 1
                if new_v != bg and new_v != _bg_feature:
                    gained_counts[new_v] = gained_counts.get(new_v, 0) + 1

        for feat, count in lost_counts.items():
            if feat in gained_counts and abs(count - gained_counts[feat]) <= max(
                2, int(count * 0.25)
            ):
                movable_features.add(feat)
                if _hazard_tracker is not None and hasattr(
                    _hazard_tracker, "mobile_threat_features"
                ):
                    _hazard_tracker.mobile_threat_features.add(feat)

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
                else _avatar_pos_now
            )
            trigger_feat = (
                int(_prev_grid[trigger_pos[0], trigger_pos[1]])
                if trigger_pos and 0 <= trigger_pos[0] < H and 0 <= trigger_pos[1] < W
                else None
            )

            if trigger_feat is None or trigger_feat == bg or trigger_feat == _bg_feature:
                trigger_feat = None

            if trigger_pos is not None or trigger_feat is not None:
                existing_mut = next(
                    (
                        m
                        for m in view.state_mutations
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
                    view.state_mutations.append(mutation_model)
                    view.exhausted_candidate_goals.clear()
                    logger.info(
                        "AutonomousEpistemicEngine: Induced StateMutationModel! Trigger %s at %s changed %d distant pixels.",
                        mutation_model.trigger_type,
                        trigger_pos,
                        len(distant_mutations),
                    )

        # ── C. Win / Loss Feedback Assimilation ──────────────────────────────
        av_feats = view.avatar_features or (
            {view.avatar_feature} if view.avatar_feature is not None else set()
        )

        # Faculty: Telic Depletion Goal Induction
        # When an interventional non-locomotion action clears pixels of a non-background, non-avatar feature,
        # hypothesize that feature as an active depletion objective (e.g. board clearing / matching).
        is_locomotion = False
        if action in view.action_dynamics and view.action_dynamics[action].is_displacement_action():
            is_locomotion = True
        elif view.is_displacement_action(action):
            is_locomotion = True

        if (
            diff.mutated_pixels
            and not is_failure
            and not is_locomotion
            and view.is_motor_grounded()
        ):
            for r, c, old_v, new_v in diff.mutated_pixels:
                if (
                    new_v == bg
                    and old_v != bg
                    and old_v not in av_feats
                    and old_v != view.avatar_feature
                    and not _sym_theory.is_barrier(old_v)
                ):
                    _sym_theory.induce_depletion_goal(old_v, evidence="pixel_depletion_heuristic")
                    break

        if is_success:
            win_target = None
            if view.avatar_pos is not None and (
                prev_avatar_pos is None or view.avatar_pos != prev_avatar_pos
            ):
                win_target = view.avatar_pos
            elif prev_avatar_pos is not None:
                dr, dc = 0, 0
                if action in view.action_dynamics:
                    dr, dc = view.action_dynamics[action].get_displacement()
                elif action_data and "x" in action_data and "y" in action_data:
                    win_target = (int(action_data["y"]), int(action_data["x"]))
                if win_target is None:
                    tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                    if 0 <= tr < H and 0 <= tc < W:
                        win_target = (tr, tc)
            elif view.current_simulated_goal is not None:
                win_target = view.current_simulated_goal
            elif view.avatar_pos is not None:
                win_target = view.avatar_pos
            elif action_data and "x" in action_data and "y" in action_data:
                win_target = (int(action_data["y"]), int(action_data["x"]))

            if win_target is not None:
                view.record_learned_goal(win_target)
                if _prev_grid is not None and 0 <= win_target[0] < H and 0 <= win_target[1] < W:
                    goal_feat = int(_prev_grid[win_target[0], win_target[1]])
                    is_distant_target = (
                        prev_avatar_pos is None
                        or abs(win_target[0] - prev_avatar_pos[0]) > 0
                        or abs(win_target[1] - prev_avatar_pos[1]) > 0
                    )
                    feat_extent = int(np.count_nonzero(_prev_grid == goal_feat))
                    is_substrate = goal_feat in _sym_theory.walkable_features or feat_extent > 64
                    if (
                        goal_feat != bg
                        and goal_feat != _bg_feature
                        and (goal_feat not in av_feats or is_distant_target)
                        and not _sym_theory.is_barrier(goal_feat)
                        and not is_substrate
                    ):
                        _sym_theory.induce_goal(goal_feat, evidence=TerminalSignal.SUCCESS)
                        logger.info(
                            "AutonomousEpistemicEngine: Grounded invariant WIN GOAL FEATURE %d at %s",
                            goal_feat,
                            win_target,
                        )

                max_entity_area = max(64, int(H * W * 0.15))
                for pe in prev_entities:
                    is_distant_pe = (
                        prev_avatar_pos is None
                        or abs(pe.grid_pos[0] - prev_avatar_pos[0]) > 2
                        or abs(pe.grid_pos[1] - prev_avatar_pos[1]) > 2
                    )
                    if (
                        (pe.feature_id not in av_feats or is_distant_pe)
                        and pe.feature_id != bg
                        and pe.feature_id != _bg_feature
                        and not _sym_theory.is_barrier(pe.feature_id)
                        and 1 <= pe.area <= max_entity_area
                    ):
                        pe_cells = pe.properties.get("cells", [pe.grid_pos])
                        if win_target in pe_cells or pe.grid_pos == win_target:
                            _sym_theory.induce_goal(pe.feature_id, evidence=TerminalSignal.SUCCESS)
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded invariant WIN GOAL ENTITY feature %d (area=%d)",
                                pe.feature_id,
                                pe.area,
                            )

            if view.current_simulated_goal is not None and _prev_grid is not None:
                sr, sc = view.current_simulated_goal
                if 0 <= sr < H and 0 <= sc < W:
                    sim_feat = int(_prev_grid[sr, sc])
                    sim_extent = int(np.count_nonzero(_prev_grid == sim_feat))
                    max_sim_extent = max(64, int(H * W * 0.15))
                    if (
                        sim_feat != bg
                        and sim_feat != _bg_feature
                        and sim_feat not in av_feats
                        and not _sym_theory.is_barrier(sim_feat)
                        and not (
                            sim_feat in _sym_theory.walkable_features or sim_extent > max_sim_extent
                        )
                    ):
                        _sym_theory.induce_goal(sim_feat, evidence=TerminalSignal.SUCCESS)

            if view.active_hypothesis:
                view.active_hypothesis.confirmed = True
                view.active_hypothesis.confidence = 1.0

            # Level transition: cleanly reset episodic spatial memory for the next level
            view.reset_episode(
                retain_dynamics=True, is_new_level=True, level=view.current_level + 1
            )
            return
        elif view.avatar_pos is not None and not is_failure:
            view.exhausted_candidate_goals.add(view.avatar_pos)

        if is_failure:
            # Faculty: Lateral Habenula & aMCC Fatal Prefix Backpropagation & Negative Valence
            view.working_memory.habenular_ior.record_catastrophe(
                final_step=view.step_counter,
                is_fatal=True,
            )
            # Faculty: Anterior Cingulate Cortex Conflict & Frustration Monitor
            actor = prev_avatar_pos or view.avatar_pos
            if actor is not None:
                view.working_memory.acc_conflict.register_death_event(
                    actor_pos=(float(actor[0]), float(actor[1])),
                    step=view.step_counter,
                    last_action=action,
                )

            # Faculty: Prefrontal Metacognitive Conflict-Driven Schema Revision
            if view.working_memory.active_schema is not None:
                invalidated = view.working_memory.handle_schema_failure(actor)
                if invalidated or view.working_memory.acc_conflict.is_frustrated():
                    logger.warning(
                        "AutonomousEpistemicEngine: Active schema invalidated due to conflict/catastrophe. Shifting to EPISTEMIC_PROBE."
                    )
                    view.transition_to_phase(
                        EpistemicPhase.EPISTEMIC_PROBE,
                        reason="metacognitive_schema_conflict_invalidation",
                    )
            # Faculty: Hippocampal Sharp-Wave Ripple (SWR) Backward Negative Replay
            view.episodic_cortex.trigger_sharp_wave_ripple_replay(
                is_failure=True,
                is_success=False,
                prev_grid=view.prev_grid,
                curr_grid=curr_grid,
                bg_feature=view.bg_feature,
                avatar_features=av_feats,
                is_walkable_fn=view.symbolic_theory.is_walkable,
                is_barrier_fn=view.symbolic_theory.is_barrier,
            )

            if prev_avatar_pos is not None and action is not None:
                view.failed_transitions.add((prev_avatar_pos, action))
                logger.info(
                    "AutonomousEpistemicEngine: Grounded lethal failed transition (%s, %s)",
                    prev_avatar_pos,
                    action,
                )

            target_pos = None
            if prev_avatar_pos is not None:
                dr, dc = 0, 0
                if action in view.action_dynamics:
                    dr, dc = view.action_dynamics[action].get_displacement()
                elif action_data and "x" in action_data and "y" in action_data:
                    target_pos = (int(action_data["y"]), int(action_data["x"]))
                if target_pos is None:
                    tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                    if 0 <= tr < H and 0 <= tc < W:
                        target_pos = (tr, tc)
            elif view.avatar_pos is not None:
                target_pos = view.avatar_pos
            elif action_data and "x" in action_data and "y" in action_data:
                target_pos = (int(action_data["y"]), int(action_data["x"]))

            if target_pos is not None:
                if _prev_grid is not None and 0 <= target_pos[0] < H and 0 <= target_pos[1] < W:
                    # Check for lethal hazard features in a 5x5 window around target_pos
                    r0 = max(0, target_pos[0] - 2)
                    r1 = min(H, target_pos[0] + 3)
                    c0 = max(0, target_pos[1] - 2)
                    c1 = min(W, target_pos[1] + 3)
                    candidate_cells = set(np.unique(curr_grid[r0:r1, c0:c1])).union(
                        set(np.unique(_prev_grid[r0:r1, c0:c1]))
                    )
                    # Also inspect unobstructed cardinal line-of-sight rays for remote turrets/projectiles
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        curr_r, curr_c = target_pos[0] + dr, target_pos[1] + dc
                        while 0 <= curr_r < H and 0 <= curr_c < W:
                            val = int(_prev_grid[curr_r, curr_c])
                            if _sym_theory.is_barrier(val):
                                break
                            if (
                                val != _bg_feature
                                and val not in av_feats
                                and not _sym_theory.is_walkable(val)
                            ):
                                candidate_cells.add(val)
                            curr_r += dr
                            curr_c += dc
                    total_cells = H * W
                    for cand_feat in candidate_cells:
                        cand_feat_int = int(cand_feat)
                        feat_count = int(np.sum(_prev_grid == cand_feat_int))
                        if (
                            cand_feat_int != _bg_feature
                            and cand_feat_int not in av_feats
                            and not _sym_theory.is_walkable(cand_feat_int)
                            and feat_count <= max(64, int(total_cells * 0.40))
                        ):
                            _hazard_tracker.register_lethal_feature(cand_feat_int)
                            _sym_theory.induce_barrier(cand_feat_int, evidence="lethal_hazard")
                            _sym_theory.retract_cargo(cand_feat_int, reason="lethal_hazard")
                            _sym_theory.retract_goal(cand_feat_int, reason="lethal_hazard")
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded lethal feature %d at barrier %s",
                                cand_feat_int,
                                target_pos,
                            )
                    view.exhausted_candidate_goals.add(target_pos)

                    # Trajectory-level aversive conditioning:
                    # The destination cell target_pos resulted in catastrophic loss/death.
                    # Ground it as a static lethal position and barrier so neither mental simulation
                    # nor exploratory probes step into this coordinate again!
                    _hazard_tracker.register_lethal_position(target_pos)
                    view.record_learned_barrier(target_pos)
                    view.level_learned_barriers.setdefault(view.current_level, set()).add(
                        target_pos
                    )
                    view.level_lethal_positions.setdefault(view.current_level, set()).add(
                        target_pos
                    )

                    # Blacklist all cardinal incoming moves into target_pos from immediate neighbors
                    for m_dr, m_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nbr = (target_pos[0] - m_dr, target_pos[1] - m_dc)
                        if 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                            for cand_a, dyn in view.action_dynamics.items():
                                if dyn.is_displacement_action() and dyn.get_displacement() == (
                                    m_dr,
                                    m_dc,
                                ):
                                    view.failed_transitions.add((nbr, cand_a))

            if action is not None:
                view.inhibited_actions[action] = 5

            if view.active_hypothesis:
                view.active_hypothesis.confidence = 0.0

        # Prefrontal item acquisition
        if view.avatar_pos is not None:
            for pe in prev_entities:
                if (
                    pe.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE)
                    and pe.feature_id not in av_feats
                    and pe.feature_id != bg
                    and pe.feature_id != _bg_feature
                    and not _sym_theory.is_walkable(pe.feature_id)
                ):
                    dist_to_av = abs(pe.grid_pos[0] - view.avatar_pos[0]) + abs(
                        pe.grid_pos[1] - view.avatar_pos[1]
                    )
                    if dist_to_av <= 1:
                        view.working_memory.acquire_item(
                            feature_id=pe.feature_id,
                            role=pe.role.value,
                            step=view.step_counter,
                            position=pe.grid_pos,
                        )

        # Tool-Barrier Resonance Credit Assignment
        if (
            _prev_grid is not None
            and view.working_memory.held_items
            and diff.changed_mask is not None
        ):
            changed_coords = np.argwhere(diff.changed_mask)
            _barrier_feats = _sym_theory.barrier_features
            _learned_barriers = view.learned_barriers
            for mr, mc in changed_coords:
                orig_val = int(_prev_grid[mr, mc])
                post_val = int(curr_grid[mr, mc])
                if (orig_val in _barrier_feats or (int(mr), int(mc)) in _learned_barriers) and (
                    post_val == _bg_feature or _sym_theory.is_walkable(post_val)
                ):
                    for h_item in view.working_memory.held_items:
                        view.working_memory.register_tool_unlock(h_item.feature_id, orig_val)

        # Faculty: Hippocampal CA1/CA3 Experiential Trace Buffer
        view.episodic_cortex.record_transition(
            step=view.step_counter,
            avatar_pos=view.avatar_pos,
            action=action,
            action_data=action_data,
            observation_diff=diff,
            reward=1.0 if is_success else (-1.0 if is_failure else 0.0),
            is_terminal=is_success or is_failure,
            terminal=terminal,
            is_failure=is_failure,
            is_success=is_success,
            grid_snapshot=curr_grid,
        )

        # Faculty: vmPFC Action-Effect Interventional Binding (Pearl's do(a))
        if view.prev_grid is not None:
            action_locus: tuple[int, int] | None = None
            if action_data and isinstance(action_data, dict):
                if "y" in action_data and "x" in action_data:
                    action_locus = (int(action_data["y"]), int(action_data["x"]))
                elif "r" in action_data and "c" in action_data:
                    action_locus = (int(action_data["r"]), int(action_data["c"]))
                elif "target" in action_data and isinstance(action_data["target"], (list, tuple)):
                    action_locus = (int(action_data["target"][0]), int(action_data["target"][1]))
            elif view.avatar_pos is not None:
                action_locus = view.avatar_pos

            view.causal_cortex.bind_intervention(
                action=action,
                origin_pos=prev_avatar_pos,
                target_pos=view.avatar_pos,
                obs_diff=diff,
                prev_grid=view.prev_grid,
                curr_grid=curr_grid,
                bg_feature=view.bg_feature,
                action_locus=action_locus,
                avatar_features=view.avatar_features,
            )

        # Faculty: Prefrontal Inductive Hypothesis Testing & Relational Rule Discovery
        if view.prev_grid is not None:
            view.working_memory.hypothesis_engine.observe_transition(
                prev_grid=view.prev_grid,
                action=action,
                curr_grid=curr_grid,
                prev_avatar_pos=prev_avatar_pos,
                curr_avatar_pos=view.avatar_pos,
                avatar_h=getattr(view.state, "avatar_h", 1) or 1,
                avatar_w=getattr(view.state, "avatar_w", 1) or 1,
                is_dead=is_failure,
                is_won=is_success,
                background_feature=view.bg_feature,
            )

        # Enforce avatar/threat non-intersection invariant
        if view.avatar_features:
            view.mobile_threat_features.difference_update(view.avatar_features)
        if view.avatar_feature is not None:
            view.mobile_threat_features.discard(view.avatar_feature)

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


@dataclass
class AssimilationStage:
    """Cortical Stage 2: Assimilation & Surprise Induction.

    Binds motor diffs to actions; updates EpistemicPhase; detects high surprise
    to invalidate stale causal hypotheses. Emits transactions updating
    dynamic theories and working memory.
    """

    name: str = "assimilation_stage"
    engine: Any | None = None
    _current_journal: Any = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        engine = getattr(self, "engine", None)
        if engine is not None:
            # WP1: real assimilation — the full prepare() preamble (terminal
            # resolution, step_counter, normalize, strict-budget, bg estimate,
            # assimilate_feedback) runs here; DeliberationStage then calls
            # deliberate() starting from the already-assimilated state.
            from hbllm.hcir.world.engine_state import (
                FacultyWriteJournal,
                JournaledEngineView,
            )

            journal = FacultyWriteJournal(author="assimilation_stage")
            self._current_journal = journal
            view = JournaledEngineView(engine, journal)
            try:
                g, term = engine.deliberation_engine.prepare(
                    view,
                    ctx.signals.get("_curr_grid"),
                    ctx.signals.get("_available_actions", []),
                    terminal=ctx.signals.get("_terminal"),
                    is_success=ctx.signals.get("_is_success", False),
                    is_failure=ctx.signals.get("_is_failure", False),
                    action_schemas=ctx.signals.get("_action_schemas"),
                    **ctx.signals.get("_kwargs", {}),
                )
            except Exception:
                engine.rollback_faculty_journal(journal)
                raise
            finally:
                self._current_journal = None
            return StageResult(
                author=self.name,
                journal=journal,
                wm_deltas=list(journal.deltas),
                signals={"_curr_grid": g, "_terminal": term},
            )
        ops: list[TransactionOperation] = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="assimilation_state",
                node_data={
                    "stage": self.name,
                    "phase": str(ctx.phase),
                    "step": ctx.step_counter,
                },
            )
        ]
        signals = {
            "assimilated_phase": str(ctx.phase),
            "step": ctx.step_counter,
        }
        return StageResult(
            author=self.name,
            operations=ops,
            signals=signals,
            proposed_action=None,
        )
