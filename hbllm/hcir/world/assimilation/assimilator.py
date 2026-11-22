"""Epistemic Feedback Assimilator Faculty.

Assimilates sensory feedback (pixel diffs, kinetic flow, reward signals)
into motor dynamics, avatar identity consensus, and empirical neuro-symbolic theory.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import get_sub_attr
from hbllm.hcir.world.cortex_causal import ActionAffordance
from hbllm.hcir.world.cortex_perception import FrameDiffType, PerceptionEngine
from hbllm.hcir.world.cortical_stage import WorkingMemoryDelta
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.percept import EpistemicPhase, TerminalSignal

if TYPE_CHECKING:
    pass

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
        *,
        is_success: bool = False,
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
        bg_val = view.bg_feature
        th = view.symbolic_theory
        wm = view.working_memory
        lethal = view.hazard_tracker.known_lethal_features
        for cr, cc in landing_cells:
            for k in range(0, max(steps, 1)):
                r, c = cr - sr * k, cc - sc * k
                if not (0 <= r < H and 0 <= c < W):
                    continue
                if k == 0:
                    wm.visited_cells.add((r, c))
                wf = int(view.prev_grid[r, c])
                if wf in av:
                    continue
                if k == 0 and wf != bg_val and not th.is_walkable(wf):
                    th.mark_locus_explored((r, c), reason="traversal")
                    view.exhausted_candidate_goals.add((r, c))
                # Detect cross-context label contradiction when safely traversing prior beliefs
                if th.has_prior_context_transition and wf != bg_val:
                    prior_lethal_falsified = wf in lethal and not th.is_verified_in_context(
                        "lethal", wf
                    )
                    prior_goal_falsified = (
                        not is_success
                        and wf in th.goal_features
                        and not th.is_verified_in_context("goal", wf)
                        and not th.cargo_features
                        and not th.receptacle_features
                        and not view.learned_cargo_features
                    )
                    prior_barrier_falsified = (
                        wf in th.barrier_features
                        and not th.is_verified_in_context("barrier", wf)
                        and wf not in th.conditional_affordances
                    )
                    if prior_lethal_falsified or prior_goal_falsified or prior_barrier_falsified:
                        from hbllm.hcir.world.cortex_causal import (
                            invalidate_stale_context_ontology,
                        )

                        reason = (
                            f"safe_traversal_over_prior_lethal:{wf}"
                            if prior_lethal_falsified
                            else f"non_terminal_traversal_over_prior_goal:{wf}"
                            if prior_goal_falsified
                            else f"safe_traversal_over_prior_barrier:{wf}"
                        )
                        invalidate_stale_context_ontology(
                            th,
                            hazard_tracker=view.hazard_tracker,
                            episodic_cortex=view.episodic_cortex,
                            working_memory=wm,
                            engine_or_view=view,
                            bg_feature=bg_val,
                            preserve_current_context=True,
                            reason=reason,
                        )
                        lethal = view.hazard_tracker.known_lethal_features
                if (
                    wf in lethal
                    or wf in view.mobile_threat_features
                    or wf in view.hazard_tracker.mobile_entities
                ):
                    continue
                th.induce_walkable(wf, evidence="traversal")

    @staticmethod
    def _synchronize_lattice_stride(view: Any, stride: int) -> None:
        """Propagate discovered lattice quantum to uncalibrated cardinal action priors."""
        if stride <= 1:
            return
        view.state.spatial_lattice_stride = stride
        for other_act, other_dyn in view.action_dynamics.items():
            if other_dyn.probes_tested == 0:
                if abs(other_dyn.delta_r) == 1 and other_dyn.delta_c == 0:
                    other_dyn.delta_r *= stride
                    other_dyn.displacement = (float(other_dyn.delta_r), 0.0)
                    if other_act in view.action_affordances:
                        view.action_affordances[other_act].delta = (other_dyn.delta_r, 0)
                elif abs(other_dyn.delta_c) == 1 and other_dyn.delta_r == 0:
                    other_dyn.delta_c *= stride
                    other_dyn.displacement = (0.0, float(other_dyn.delta_c))
                    if other_act in view.action_affordances:
                        view.action_affordances[other_act].delta = (0, other_dyn.delta_c)

    @staticmethod
    def _assimilate_cargo_pushing(
        view: Any,
        moved_entities: list[tuple[SpatialEntity, SpatialEntity, tuple[int, int]]],
        prev_avatar_pos: tuple[int, int] | None,
        dr: int,
        dc: int,
        is_success: bool,
    ) -> None:
        """Induce pushable cargo affordance when avatar displacement moves an adjacent entity."""
        if is_success or prev_avatar_pos is None or (dr == 0 and dc == 0):
            return
        known_av_feats = view.avatar_features or (
            {view.avatar_feature} if view.avatar_feature is not None else set()
        )
        av_r, av_c = prev_avatar_pos
        for pe, ce, delta in moved_entities:
            if ce.feature_id in known_av_feats:
                continue
            pe_r, pe_c = pe.grid_pos
            # Pushing requires direct physical contact at the avatar's leading step target
            pe_cells = pe.properties.get("cells") if pe.properties else None
            lead_pos = (av_r + dr, av_c + dc)
            is_contact = (
                lead_pos in pe_cells if pe_cells else (pe_r == lead_pos[0] and pe_c == lead_pos[1])
            )
            is_pushed = is_contact and (delta == (dr, dc))
            if is_pushed:
                view.symbolic_theory.induce_cargo(ce.feature_id, evidence="contact_displacement")
                active_arb = view.active_arbitrated_goal
                if active_arb is not None:
                    arb_meta = active_arb.metadata or {}
                    if (
                        arb_meta.get("feature_id") == ce.feature_id
                        or active_arb.goal_type == "epistemic_exploration"
                    ):
                        view.apply_wm_delta(WorkingMemoryDelta("active_arbitrated_goal", None))
                logger.info(
                    "AutonomousEpistemicEngine: Discovered PUSHABLE CARGO (feat=%d, size=%d)",
                    ce.feature_id,
                    ce.area,
                )
            elif delta != (0, 0):
                view.symbolic_theory.retract_cargo(ce.feature_id, reason="autonomous_dynamic")
                view.learned_cargo_features.discard(ce.feature_id)

    @staticmethod
    def _commit_avatar_identity(
        view: Any,
        pairs: list[tuple[SpatialEntity, SpatialEntity]],
        delta: tuple[int, int],
        action: int | None,
        H: int,
        W: int,
        calibrate_dynamics: bool = True,
        is_failure: bool = False,
        is_success: bool = False,
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
            if a.area == 1 and b.area == 1:
                return abs(a.grid_pos[0] - b.grid_pos[0]) + abs(a.grid_pos[1] - b.grid_pos[1]) == 1
            b1, b2 = a.bounding_box, b.bounding_box
            r_gap = max(0, b1[0] - b2[1] - 1, b2[0] - b1[1] - 1)
            c_gap = max(0, b1[2] - b2[3] - 1, b2[2] - b1[3] - 1)
            return (b1[0] <= b2[1] and b2[0] <= b1[1] and c_gap == 0) or (
                b1[2] <= b2[3] and b2[2] <= b1[3] and r_gap == 0
            )

        comp_pairs: list[tuple[SpatialEntity, SpatialEntity]] = [pairs[0]]
        for pe, ce in pairs[1:]:
            if any(_adjacent(pe, c_pe) and _adjacent(ce, c_ce) for c_pe, c_ce in comp_pairs):
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
        # Features the avatar just traversed are verified safe (only on non-fatal steps)
        if not is_failure:
            EpistemicFeedbackAssimilator._record_safe_traversal(
                view, all_cells, delta, H, W, is_success=is_success
            )
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
            # Lattice extent changed (new context/episode): prior-lattice
            # diffs are incomparable — start the episode's diff chain fresh.
            view.apply_wm_delta(WorkingMemoryDelta("prev_grid", None))
            return
        H, W = curr_grid.shape

        if not is_success:
            walkable_feats = view.symbolic_theory.walkable_features
            cargo_feats = view.symbolic_theory.cargo_features | view.learned_cargo_features
            av_feats_frame = set(view.avatar_features or ())
            if view.avatar_feature is not None:
                av_feats_frame.add(view.avatar_feature)
            view.hazard_tracker.record_frame(
                view.step_counter,
                curr_grid,
                background_feature=view.bg_feature,
                avatar_features=av_feats_frame,
                walkable_features=walkable_feats,
                cargo_features=cargo_feats,
            )
            for mob_f in view.hazard_tracker.mobile_entities:
                if mob_f != view.bg_feature and mob_f not in cargo_feats:
                    view.symbolic_theory.retract_walkable(mob_f, reason="autonomous_mobile")
                    view.symbolic_theory.retract_candidate_goal(mob_f, reason="autonomous_mobile")
                    if mob_f != view.avatar_feature:
                        view.avatar_features.discard(mob_f)

        diff = view.compute_frame_diff(view.prev_grid, curr_grid)
        action = view.last_action
        action_data = view.last_action_data

        # Track visited cells in PrefrontalWorkingMemory (P2-9)
        if view.avatar_pos is not None:
            view.working_memory.visited_cells.add(view.avatar_pos)

        # Epistemic Goal Lifecycle Evaluation (P2-8):
        # A curiosity goal is satisfied when:
        # 1. Target feature is absent at target coordinate (consumed / collected / changed)
        # 2. Or target coordinate was visited without surprise >= max_visits
        satisfied_when_map = getattr(view.working_memory, "active_epistemic_conditions", {})
        goals_to_retire: set[tuple[int, int]] = set()
        for target_pos, cond in list(satisfied_when_map.items()):
            tr, tc = cond.get("entity_pos", target_pos)
            expected_feat = cond.get("feature")
            max_visits = cond.get("max_visits", 2)

            # Condition 1: Target feature absent at target coordinate
            if expected_feat is not None and 0 <= tr < H and 0 <= tc < W:
                curr_cell_val = int(curr_grid[tr, tc])
                if curr_cell_val != expected_feat:
                    view.symbolic_theory.mark_feature_explored(
                        expected_feat, reason="target_absent_or_consumed"
                    )
                    view.symbolic_theory.mark_locus_explored(target_pos, reason="feature_consumed")
                    goals_to_retire.add(target_pos)
                    logger.info(
                        "AutonomousEpistemicEngine: Epistemic goal satisfied — feature %s absent at %s (now %s)",
                        expected_feat,
                        target_pos,
                        curr_cell_val,
                    )

            # Condition 2: Avatar reached target coordinate
            if view.avatar_pos == target_pos:
                visits = cond.get("visits", 0) + 1
                cond["visits"] = visits
                if visits >= max_visits:
                    entity_pos_cond = cond.get("entity_pos")
                    is_adjacent_stance = (
                        entity_pos_cond is not None and entity_pos_cond != target_pos
                    )
                    if expected_feat is not None and not is_adjacent_stance:
                        view.symbolic_theory.mark_feature_explored(
                            expected_feat, reason="visited_without_surprise"
                        )
                    view.symbolic_theory.mark_locus_explored(target_pos, reason="visited_target")
                    if is_adjacent_stance:
                        view.exhausted_candidate_goals.add(target_pos)
                    goals_to_retire.add(target_pos)
                    logger.info(
                        "AutonomousEpistemicEngine: Epistemic goal satisfied — reached %s %d times",
                        target_pos,
                        visits,
                    )

        for gp in goals_to_retire:
            cond = satisfied_when_map.pop(gp, None)
            view.exhausted_candidate_goals.add(gp)
            if hasattr(view, "goals") and gp in view.goals:
                view.goals.remove(gp)
            active_arb = getattr(view, "active_arbitrated_goal", None)
            if active_arb is not None:
                arb_tgt = active_arb.target
                if arb_tgt == gp or (cond and cond.get("entity_pos") == arb_tgt):
                    setattr(view, "active_arbitrated_goal", None)
            if view.mental_plan:
                last_step = view.mental_plan[-1]
                if getattr(last_step, "predicted_avatar_pos", None) == gp:
                    view.mental_plan.clear()

        # Classify the observed effect kind for the motor posterior.
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
                except (ValueError, TypeError) as exc:
                    logger.debug(
                        "Non-integer action target coord ignored (%r, %r): %s", tr, tc, exc
                    )

        _wm_exog_early = view.working_memory
        _bg_exog_early = view.bg_feature if view.bg_feature is not None else 0
        if diff.mutated_pixels and view.prev_grid is not None:
            av_set_exog = set(view.avatar_features) if view.avatar_features else set()
            if view.avatar_feature is not None:
                av_set_exog.add(view.avatar_feature)
            av_appeared = (
                [
                    (int(r), int(c))
                    for r, c, _, new_v in diff.mutated_pixels
                    if int(new_v) in av_set_exog
                    and not _wm_exog_early.exogenous_tracker.is_partitioned_or_enclosing(
                        (int(r), int(c)), view.prev_grid, bg=_bg_exog_early
                    )
                ]
                if av_set_exog
                else []
            )
            if av_appeared:
                est_av_now: tuple[int, int] | None = (
                    int(round(sum(p[0] for p in av_appeared) / len(av_appeared))),
                    int(round(sum(p[1] for p in av_appeared) / len(av_appeared))),
                )
            else:
                est_av_now = view.avatar_pos
            pred_av_pos: tuple[int, int] | None = None
            if view.avatar_pos is not None and action in view.action_deltas:
                dr_p, dc_p = view.action_deltas[action]
                pred_av_pos = (view.avatar_pos[0] + dr_p, view.avatar_pos[1] + dc_p)
            _wm_exog_early.exogenous_tracker.observe_transition(
                [(int(r), int(c)) for r, c, _, _ in diff.mutated_pixels],
                action,
                avatar_locus=est_av_now,
                target_locus=target_coord,
                extra_local_loci=(view.avatar_pos, pred_av_pos),
            )

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

        # Empirical State-Reversion / Undo Action Discovery (P2-7):
        # An action is an undo candidate ONLY if it reverts a non-locomotion state change
        # (e.g. object transform, switch, paint, or cargo state), never reversible locomotion.
        if (
            action is not None
            and len(view.grid_history) >= 2
            and not is_effector_action
            and not view.is_displacement_action(action)
            and action not in getattr(view, "action_deltas", {})
        ):
            prev_prev_grid = view.grid_history[-2]
            prev_grid = view.prev_grid
            av_feats = set(view.avatar_features)
            if view.avatar_feature is not None:
                av_feats.add(view.avatar_feature)

            # Check if avatar moved in current transition
            av_mask_curr = (
                np.isin(curr_grid, list(av_feats))
                if av_feats
                else np.zeros_like(curr_grid, dtype=bool)
            )
            av_mask_prev = (
                np.isin(prev_grid, list(av_feats))
                if av_feats
                else np.zeros_like(prev_grid, dtype=bool)
            )
            avatar_displaced = bool(
                np.any((curr_grid != prev_grid) & (av_mask_curr | av_mask_prev))
            )

            # Check if non-avatar cells changed in prior step and were reverted in this step
            prior_diff = prev_grid != prev_prev_grid
            prior_coords = np.argwhere(prior_diff)
            non_av_changed_prior = any(
                int(prev_grid[r, c]) not in av_feats and int(prev_prev_grid[r, c]) not in av_feats
                for r, c in prior_coords
            )
            curr_diff = curr_grid != prev_grid
            curr_coords = np.argwhere(curr_diff)
            non_av_changed_curr = any(
                int(curr_grid[r, c]) not in av_feats and int(prev_grid[r, c]) not in av_feats
                for r, c in curr_coords
            )

            if (
                not avatar_displaced
                and non_av_changed_prior
                and non_av_changed_curr
                and np.array_equal(curr_grid, prev_prev_grid)
            ):
                view.symbolic_theory.induce_undo_action(action)
                view.undo_actions.add(action)
                if action in view.action_affordances:
                    view.action_affordances[action].is_undo = True
                view.apply_wm_delta(WorkingMemoryDelta("inhibit_action", (action, 4)))
                logger.info(
                    "AutonomousEpistemicEngine: Grounded empirical UNDO / STATE_REVERSION affordance for action %s",
                    action,
                )

        continuous_moved = bool(getattr(view.working_memory, "last_continuous_moved", False))
        if continuous_moved:
            view.state.consecutive_quiescent_actions = 0
        elif diff.changed_pixel_count == 0:
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
                tok = (action, target_coord[0], target_coord[1])
                view.working_memory.habenular_ior.active_inhibitions[(view.avatar_pos, tok)] = 20

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
                    has_known_avatar = view.avatar_feature is not None or bool(view.avatar_features)
                    if not view.has_displacement_capabilities() and not has_known_avatar:
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

                    if not view.has_displacement_capabilities() and not has_known_avatar:
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

            # Faculty: Ventromedial Prefrontal Cortex (vmPFC) Remote Causal Attribution
            loci: list[tuple[int, int]] = []
            if target_coord is not None:
                loci.append(target_coord)
            else:
                if view.avatar_pos is not None:
                    loci.append(view.avatar_pos)
                if prev_avatar_pos is not None and prev_avatar_pos != view.avatar_pos:
                    loci.append(prev_avatar_pos)

                agent_ref = prev_avatar_pos or view.avatar_pos
                if agent_ref is not None and view.prev_grid is not None:
                    ar, ac = agent_ref
                    H_p, W_p = view.prev_grid.shape
                    mutated_adj: list[tuple[int, int]] = []
                    nonbarrier_adj: list[tuple[int, int]] = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = ar + dr, ac + dc
                        if 0 <= nr < H_p and 0 <= nc < W_p:
                            f_adj = int(view.prev_grid[nr, nc])
                            if (
                                f_adj != view.bg_feature
                                and (
                                    view.avatar_features is None
                                    or f_adj not in view.avatar_features
                                )
                                and f_adj != view.avatar_feature
                                and not view.symbolic_theory.is_walkable(f_adj)
                            ):
                                if int(curr_grid[nr, nc]) != f_adj:
                                    mutated_adj.append((nr, nc))
                                elif (
                                    not is_known_displacement
                                    and not view.symbolic_theory.is_barrier(f_adj)
                                ):
                                    nonbarrier_adj.append((nr, nc))
                    loci.extend(mutated_adj if mutated_adj else nonbarrier_adj)

            if view.prev_grid is not None:
                av_rc = set(view.avatar_features or ())
                if view.avatar_feature is not None:
                    av_rc.add(view.avatar_feature)
                for locus in dict.fromkeys(loci):
                    lr_l, lc_l = locus
                    if not (0 <= lr_l < H and 0 <= lc_l < W):
                        continue
                    if _wm_exog_early.is_exogenous_locus(
                        locus, lattice=view.prev_grid, bg=_bg_exog_early, include_outer_edge=False
                    ):
                        continue
                    locus_feat = int(view.prev_grid[lr_l, lc_l])
                    if view.symbolic_theory.is_barrier(
                        locus_feat
                    ) or view.symbolic_theory.is_walkable(locus_feat):
                        continue
                    confirmed_affs = view.working_memory.remote_causal.record_transition(
                        prev_grid=view.prev_grid,
                        curr_grid=curr_grid,
                        action_pos=locus,
                        background_feature=view.bg_feature,
                        avatar_features=av_rc,
                    )
                    for aff in confirmed_affs:
                        if _wm_exog_early.is_exogenous_locus(
                            aff.remote_pos,
                            lattice=view.prev_grid,
                            bg=_bg_exog_early,
                            include_outer_edge=False,
                        ):
                            continue
                        if (
                            aff.mutated_remote_feature == view.bg_feature
                            or view.symbolic_theory.is_walkable(aff.mutated_remote_feature)
                        ):
                            view.learned_barriers.discard(aff.remote_pos)
                            view.failed_transitions.clear()
                            view.recent_positions.clear()
                            if (
                                hasattr(view, "acc_conflict_monitor")
                                and view.acc_conflict_monitor is not None
                            ):
                                if hasattr(view.acc_conflict_monitor, "action_history"):
                                    view.acc_conflict_monitor.action_history.clear()
                                if hasattr(view.acc_conflict_monitor, "position_history"):
                                    view.acc_conflict_monitor.position_history.clear()
                            view.mental_plan.clear()
                            for dr_u, dc_u in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                nr_u, nc_u = aff.remote_pos[0] + dr_u, aff.remote_pos[1] + dc_u
                                if 0 <= nr_u < H and 0 <= nc_u < W:
                                    if _wm_exog_early.is_exogenous_locus(
                                        (nr_u, nc_u),
                                        lattice=curr_grid,
                                        bg=_bg_exog_early,
                                        include_outer_edge=False,
                                    ):
                                        continue
                                    nf_u = int(curr_grid[nr_u, nc_u])
                                    if (
                                        nf_u != view.bg_feature
                                        and (
                                            view.avatar_features is None
                                            or nf_u not in view.avatar_features
                                        )
                                        and nf_u != view.avatar_feature
                                        and int(np.count_nonzero(curr_grid == nf_u)) <= 2
                                    ):
                                        view.learned_barriers.discard((nr_u, nc_u))
                                        view.symbolic_theory.retract_barrier(
                                            nf_u, reason="unlocked_by_remote_door"
                                        )
                                        view.symbolic_theory.unmark_locus_explored((nr_u, nc_u))
                                        view.exhausted_candidate_goals.discard((nr_u, nc_u))
                                        view.record_learned_goal((nr_u, nc_u))
                                        view.symbolic_theory.induce_candidate_goal(
                                            nf_u, evidence="adjacent_to_opened_door"
                                        )
                            if view.subgoal_stack.has_pending_goals:
                                view.subgoal_stack.pop_subgoal()
                        view.working_memory.activated_triggers.add(aff.trigger_pos)
                        if (
                            aff.trigger_feature is not None
                            and (
                                view.avatar_features is None
                                or aff.trigger_feature not in view.avatar_features
                            )
                            and aff.original_remote_feature != view.bg_feature
                            and not view.symbolic_theory.is_walkable(aff.original_remote_feature)
                        ):
                            view.symbolic_theory.induce_conditional_affordance(
                                feature=aff.original_remote_feature,
                                condition=CausalPredicate(
                                    variable="trigger",
                                    operator="==",
                                    value=aff.trigger_feature,
                                ),
                                evidence=f"remote_causal:{aff.trigger_pos}->{aff.remote_pos}",
                                prerequisite_feature=aff.trigger_feature,
                            )

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

        # Faculty D: Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring)
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
            if e1.area == 1 and e2.area == 1:
                return (
                    abs(e1.grid_pos[0] - e2.grid_pos[0]) + abs(e1.grid_pos[1] - e2.grid_pos[1]) == 1
                )
            bb1 = e1.bounding_box
            bb2 = e2.bounding_box
            r_gap = max(0, bb1[0] - bb2[1] - 1, bb2[0] - bb1[1] - 1)
            c_gap = max(0, bb1[2] - bb2[3] - 1, bb2[2] - bb1[3] - 1)
            return (bb1[0] <= bb2[1] and bb2[0] <= bb1[1] and c_gap == 0) or (
                bb1[2] <= bb2[3] and bb2[2] <= bb1[3] and r_gap == 0
            )

        av_moved: list[Any] = []
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
                    best_delta, pairs = max(delta_groups.items(), key=lambda item: len(item[1]))
                    single_rigid_group = len(delta_groups) == 1
                    if single_rigid_group and len(pairs) > 1:
                        comp_check = [pairs[0]]
                        for pe_c, ce_c in pairs[1:]:
                            if any(
                                _are_adjacent(pe_c, p_k) and _are_adjacent(ce_c, c_k)
                                for p_k, c_k in comp_check
                            ):
                                comp_check.append((pe_c, ce_c))
                        single_rigid_group = len(comp_check) == len(pairs)

                    if single_rigid_group and not view._avatar_controllability_evidence:
                        EpistemicFeedbackAssimilator._commit_avatar_identity(
                            view,
                            pairs,
                            best_delta,
                            action,
                            H,
                            W,
                            calibrate_dynamics=True,
                            is_failure=is_failure,
                            is_success=is_success,
                        )
                    else:
                        for delta_val, group in delta_groups.items():
                            for _, ce in group:
                                feat = ce.feature_id
                                if feat != bg and feat != view.bg_feature:
                                    view._avatar_controllability_evidence.setdefault(
                                        feat, []
                                    ).append((action, delta_val))

                        best_cand_fresh: int | None = None
                        best_tuple_fresh = (-1, -1)
                        best_act_deltas: dict[int, tuple[int, int]] = {}
                        for feat, observations in view._avatar_controllability_evidence.items():
                            act_map: dict[int, set[tuple[int, int]]] = {}
                            bounced = False
                            for obs_a, obs_d in observations:
                                if obs_a is None or obs_d == (0, 0):
                                    continue
                                prev_ds = act_map.setdefault(obs_a, set())
                                if (-obs_d[0], -obs_d[1]) in prev_ds:
                                    bounced = True
                                    break
                                prev_ds.add(obs_d)
                            if bounced or len(act_map) < 2:
                                continue
                            all_ds = {d for ds in act_map.values() for d in ds}
                            if len(all_ds) < 2:
                                continue
                            spans_2d = any(d[0] != 0 for d in all_ds) and any(
                                d[1] != 0 for d in all_ds
                            )
                            if not (spans_2d or H == 1 or W == 1 or len(act_map) >= 3):
                                continue
                            cand_key = (1 if spans_2d else 0, len(all_ds))
                            if cand_key > best_tuple_fresh:
                                best_tuple_fresh = cand_key
                                best_cand_fresh = feat
                                best_act_deltas = {
                                    oa: next(iter(ods)) for oa, ods in act_map.items()
                                }

                        if best_cand_fresh is not None:
                            last_a, last_d = view._avatar_controllability_evidence[best_cand_fresh][
                                -1
                            ]
                            avatar_pairs_f = [
                                (pe, ce)
                                for dg_p in delta_groups.values()
                                for pe, ce in dg_p
                                if ce.feature_id == best_cand_fresh
                            ]
                            if avatar_pairs_f:
                                EpistemicFeedbackAssimilator._commit_avatar_identity(
                                    view,
                                    avatar_pairs_f,
                                    last_d,
                                    last_a,
                                    H,
                                    W,
                                    calibrate_dynamics=False,
                                    is_failure=is_failure,
                                    is_success=is_success,
                                )
                                for cal_a, (cal_dr, cal_dc) in best_act_deltas.items():
                                    view.action_dynamics[cal_a] = ActionDynamicsModel(
                                        action_id=cal_a,
                                        delta_r=cal_dr,
                                        delta_c=cal_dc,
                                        confidence=0.8,
                                        probes_tested=1,
                                    )
                                    view.action_affordances[cal_a] = ActionAffordance(
                                        action_id=cal_a,
                                        is_displacement=(cal_dr != 0 or cal_dc != 0),
                                        delta=(cal_dr, cal_dc),
                                    )
                                view._avatar_controllability_evidence.clear()
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
                                is_failure=is_failure,
                                is_success=is_success,
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

        landing_cells_now: list[tuple[int, int]] = []
        if not is_effector_action:
            known_av_feats = view.avatar_features or (
                {view.avatar_feature} if view.avatar_feature is not None else set()
            )
            av_moved = [item for item in moved_entities if item[1].feature_id in known_av_feats]
            if (
                av_moved
                and not is_success
                and not is_failure
                and len(known_av_feats) == 1
                and len(view.action_dynamics) <= 3
                and not any(d.slip_rate > 0.05 for d in view.action_dynamics.values())
            ):
                _, chk_ce, chk_delta = av_moved[0]
                f_av = chk_ce.feature_id
                reversed_on_same = (
                    action in view.action_dynamics
                    and chk_delta != (0, 0)
                    and chk_delta
                    == (
                        -view.action_dynamics[action].delta_r,
                        -view.action_dynamics[action].delta_c,
                    )
                )
                drifted_on_new = (
                    action is not None
                    and action not in view.action_dynamics
                    and chk_delta != (0, 0)
                    and any(
                        d.get_displacement() == chk_delta for d in view.action_dynamics.values()
                    )
                    and any(
                        ce_o.feature_id != f_av
                        and ce_o.feature_id != bg
                        and ce_o.feature_id != view.bg_feature
                        and d_o != chk_delta
                        for _, ce_o, d_o in moved_entities
                    )
                )
                if reversed_on_same or drifted_on_new:
                    view.state.avatar_feature = None
                    view.state.avatar_features = set()
                    view.state.avatar_pos = None
                    view.action_dynamics.clear()
                    view.action_affordances.clear()
                    for _, ce_ev, d_ev in moved_entities:
                        if ce_ev.feature_id != bg and ce_ev.feature_id != view.bg_feature:
                            view._avatar_controllability_evidence.setdefault(
                                ce_ev.feature_id, []
                            ).append((action, d_ev))
                    known_av_feats = set()
                    av_moved = []
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
                    landing_cells_now = list(all_cells)
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
                    view.working_memory.visited_cells.add(view.state.avatar_pos)
                    av_cell_set = set(all_cells)
                    for t_pos in list(satisfied_when_map.keys()):
                        if t_pos == view.state.avatar_pos or t_pos in av_cell_set:
                            cond = satisfied_when_map[t_pos]
                            visits = cond.get("visits", 0) + 1
                            cond["visits"] = visits
                            if visits >= cond.get("max_visits", 2):
                                view.symbolic_theory.mark_locus_explored(
                                    t_pos, reason="visited_target"
                                )
                                view.exhausted_candidate_goals.add(t_pos)
                                satisfied_when_map.pop(t_pos, None)
                if not is_success and not is_failure:
                    EpistemicFeedbackAssimilator._record_safe_traversal(
                        view, all_cells, (dr, dc), H, W
                    )

                if not is_effector_action:
                    if dr != 0 or dc != 0:
                        view.immobile_entity_actions.discard(action)
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
                    EpistemicFeedbackAssimilator._assimilate_cargo_pushing(
                        view, moved_entities, prev_avatar_pos, dr, dc, is_success
                    )

                if not is_success:
                    for _, ce, delta in moved_entities:
                        if (
                            ce.feature_id in view.avatar_features
                            and ce.feature_id != view.avatar_feature
                            and delta != (dr, dc)
                        ):
                            view.avatar_features.discard(ce.feature_id)
                    for pe, ce, delta in moved_entities:
                        if ce.feature_id in known_av_feats:
                            continue
                        if (
                            ce.feature_id in view.symbolic_theory.cargo_features
                            or ce.feature_id in view.mobile_threat_features
                            or ce.feature_id in view.hazard_tracker.mobile_entities
                        ):
                            continue
                        if delta == (dr, dc) and any(
                            _are_adjacent(pe, p_ce) and _are_adjacent(ce, c_ce)
                            for p_ce, c_ce, _ in av_moved
                        ):
                            view.avatar_features.add(ce.feature_id)
                            view.avatar_size += ce.area
                            logger.info(
                                "AutonomousEpistemicEngine: Discovered compound avatar component (feat=%d, size=%d)",
                                ce.feature_id,
                                ce.area,
                            )
        # ── Proprioceptive Efference Copy Verification & Obstacle Grounding ──
        continuous_moved = bool(getattr(view.working_memory, "last_continuous_moved", False))
        if continuous_moved and action is not None and not is_effector_action:
            cd_x, cd_y = getattr(view.working_memory, "continuous_delta", (0.0, 0.0))
            if (abs(cd_x) > 1e-4 or abs(cd_y) > 1e-4) and action in view.action_dynamics:
                view.action_dynamics[action].record_continuous_delta(cd_x, cd_y)
        if not av_moved and continuous_moved and action is not None:
            cd_x, cd_y = getattr(view.working_memory, "continuous_delta", (0.0, 0.0))
            if abs(cd_x) > 1e-4 or abs(cd_y) > 1e-4:
                rf = getattr(view.working_memory, "reference_frame", None)
                origin_desc = (
                    getattr(rf, "origin_description", "bottom_left_metric")
                    if rf
                    else "bottom_left_metric"
                )
                y_sign = -1 if "bottom_left" in origin_desc else 1

                if abs(cd_y) >= abs(cd_x):
                    dr_c = y_sign if cd_y > 0 else -y_sign
                    dc_c = 0
                else:
                    dr_c = 0
                    dc_c = 1 if cd_x > 0 else -1

                if not is_effector_action:
                    if action not in view.action_dynamics:
                        view.action_dynamics[action] = ActionDynamicsModel(
                            action_id=action,
                            delta_r=dr_c,
                            delta_c=dc_c,
                            confidence=0.7,
                            probes_tested=1,
                        )
                    else:
                        view.action_dynamics[action].update_from_trial(
                            (dr_c, dc_c), success=True, learning_rate=0.5
                        )
                    view.action_dynamics[action].record_continuous_delta(cd_x, cd_y)
                    if action in view.action_affordances:
                        view.action_affordances[action].is_displacement = True
                        view.action_affordances[action].delta = (dr_c, dc_c)
                    else:
                        view.action_affordances[action] = ActionAffordance(
                            action_id=action,
                            is_displacement=True,
                            delta=(dr_c, dc_c),
                        )

        if continuous_moved and (view.avatar_feature is None or view.avatar_pos is None):
            c_pos = getattr(view.working_memory, "continuous_agent_pos", None)
            rf = getattr(view.working_memory, "reference_frame", None)
            if c_pos is not None and len(c_pos) >= 2:
                origin_desc = (
                    getattr(rf, "origin_description", "bottom_left_metric")
                    if rf
                    else "bottom_left_metric"
                )
                y_max = (
                    float(rf.bounds[1][1])
                    if (rf and rf.bounds and len(rf.bounds) >= 2)
                    else float(H)
                )
                if "bottom_left" in origin_desc:
                    ar_l = int(np.clip(round(y_max - 0.5 - float(c_pos[1])), 0, H - 1))
                else:
                    ar_l = int(np.clip(round(float(c_pos[1]) - 0.5), 0, H - 1))
                ac_l = int(np.clip(round(float(c_pos[0]) - 0.5), 0, W - 1))
                feat_l = int(curr_grid[ar_l, ac_l])
                if feat_l != bg and feat_l != view.bg_feature:
                    if view.avatar_feature is None:
                        view.state.avatar_features = {feat_l}
                        view.state.avatar_feature = feat_l
                        view.symbolic_theory.induce_avatar(
                            feat_l, evidence="continuous_kinematic_stream"
                        )
                    view.state.avatar_pos = (ar_l, ac_l)
        avatar_moved = bool(
            (
                view.avatar_pos is not None
                and prev_avatar_pos is not None
                and view.avatar_pos != prev_avatar_pos
            )
            or manip_moved
            or continuous_moved
        )
        if not is_effector_action and not is_known_displacement and not avatar_moved:
            view.state.last_non_displacement_action = action

        if view.avatar_pos is not None:
            if (
                getattr(view.working_memory, "continuous_agent_pos", None) is None
                and getattr(view.working_memory, "reference_frame", None) is None
                and not view.symbolic_theory.cargo_features
                and not view.symbolic_theory.receptacle_features
            ):
                confirmed_pos = set()
                if view.prev_grid is not None:
                    confirmed_feats = (
                        view.symbolic_theory.goal_features
                        | view.symbolic_theory.receptacle_features
                    )
                    if confirmed_feats:
                        confirmed_pos = {
                            (r, c)
                            for r in range(H)
                            for c in range(W)
                            if int(view.prev_grid[r, c]) in confirmed_feats
                        }
                if view.avatar_pos not in confirmed_pos:
                    view.learned_goal_positions.discard(view.avatar_pos)
                if prev_avatar_pos is not None and prev_avatar_pos not in confirmed_pos:
                    view.learned_goal_positions.discard(prev_avatar_pos)
                view.learned_goal_positions.difference_update(
                    set(view.recent_positions) - confirmed_pos
                )
            if view.avatar_features:
                for af in view.avatar_features:
                    view.symbolic_theory.induce_avatar(af, evidence="motor_grounding")
                view.hazard_tracker.known_lethal_features.difference_update(view.avatar_features)
                view.episodic_cortex.discovered_lethal_features.difference_update(
                    view.avatar_features
                )
            if view.avatar_feature is not None:
                view.symbolic_theory.induce_avatar(view.avatar_feature, evidence="motor_grounding")
                view.hazard_tracker.known_lethal_features.discard(view.avatar_feature)
                view.episodic_cortex.discovered_lethal_features.discard(view.avatar_feature)

        leading_cell_mutated = False
        possible_orthogonal_wall_slip = False
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
                        leading_cell_mutated = bool(
                            view.prev_grid is not None
                            and 0 <= adj_r < H
                            and 0 <= adj_c < W
                            and int(curr_grid[adj_r, adj_c]) != int(view.prev_grid[adj_r, adj_c])
                        )
                        if (
                            not leading_cell_mutated
                            and 0 <= adj_r < H
                            and 0 <= adj_c < W
                            and view.working_memory.consecutive_stuck_steps < 2
                            and any(d.slip_rate > 0.05 for d in view.action_dynamics.values())
                            and not view.symbolic_theory.is_barrier(int(curr_grid[adj_r, adj_c]))
                            and (adj_r, adj_c) not in view.learned_barriers
                        ):
                            pr_s, pc_s = prev_avatar_pos
                            possible_orthogonal_wall_slip = any(
                                orth_r * step_r + orth_c * step_c == 0
                                and (
                                    not (0 <= pr_s + orth_r < H and 0 <= pc_s + orth_c < W)
                                    or view.symbolic_theory.is_barrier(
                                        int(curr_grid[pr_s + orth_r, pc_s + orth_c])
                                    )
                                    or (pr_s + orth_r, pc_s + orth_c) in view.learned_barriers
                                )
                                for orth_r, orth_c in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            )
                        if leading_cell_mutated:
                            view.learned_barriers.discard((adj_r, adj_c))
                        elif (
                            0 <= adj_r < H and 0 <= adj_c < W and not possible_orthogonal_wall_slip
                        ):
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
                            if not is_avatar and is_confirmed_goal:
                                if (
                                    view.symbolic_theory.has_prior_context_transition
                                    and val in view.symbolic_theory.goal_features
                                    and not view.symbolic_theory.is_verified_in_context("goal", val)
                                    and not view.symbolic_theory.cargo_features
                                ):
                                    from hbllm.hcir.world.cortex_causal import (
                                        invalidate_stale_context_ontology,
                                    )

                                    invalidate_stale_context_ontology(
                                        view.symbolic_theory,
                                        hazard_tracker=view.hazard_tracker,
                                        episodic_cortex=view.episodic_cortex,
                                        working_memory=view.working_memory,
                                        engine_or_view=view,
                                        bg_feature=view.bg_feature,
                                        preserve_current_context=True,
                                        reason=f"prior_goal_blocked_on_collision:{val}",
                                    )
                                    view.symbolic_theory.induce_barrier(val, evidence="collision")
                                if val not in view.symbolic_theory.goal_features:
                                    view.learned_goal_positions.discard((adj_r, adj_c))
                                view.record_learned_barrier((adj_r, adj_c))
                            elif (
                                not is_avatar
                                and not is_confirmed_goal
                                and val != bg
                                and val != view.bg_feature
                                and val != view.working_memory.shroud_feature
                                and val not in view.mobile_threat_features
                                and val not in view.hazard_tracker.mobile_entities
                                and not (
                                    view.symbolic_theory.has_verified_walkable(val)
                                    and view.symbolic_theory.is_verified_in_context("walkable", val)
                                )
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

                                has_discrete_non_disp = len(available_actions or []) > 4 and any(
                                    not view.is_displacement_action(a)
                                    and not view.is_spatial_effector(a)
                                    for a in (available_actions or [])
                                )
                                view.learned_goal_positions.discard((adj_r, adj_c))
                                is_isolated_interior = (
                                    has_discrete_non_disp
                                    and 1 <= adj_r < H - 1
                                    and 1 <= adj_c < W - 1
                                    and int(np.count_nonzero(curr_grid == val)) <= 2
                                )

                                if (
                                    view.symbolic_theory.has_verified_walkable(val)
                                    and view.symbolic_theory.has_prior_context_transition
                                    and not view.symbolic_theory.is_verified_in_context(
                                        "walkable", val
                                    )
                                ):
                                    from hbllm.hcir.world.cortex_causal import (
                                        invalidate_stale_context_ontology,
                                    )

                                    invalidate_stale_context_ontology(
                                        view.symbolic_theory,
                                        hazard_tracker=view.hazard_tracker,
                                        episodic_cortex=view.episodic_cortex,
                                        working_memory=view.working_memory,
                                        engine_or_view=view,
                                        bg_feature=view.bg_feature,
                                        preserve_current_context=True,
                                        reason=f"prior_walkable_blocked_on_collision:{val}",
                                    )
                                if not (
                                    view.symbolic_theory.has_verified_walkable(val)
                                    and view.symbolic_theory.is_verified_in_context("walkable", val)
                                ):
                                    view.symbolic_theory.retract_walkable(
                                        val, reason="collision_obstacle"
                                    )
                                    if (
                                        not is_isolated_interior
                                        and not view.symbolic_theory.is_barrier(val)
                                    ):
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

                    if (
                        not leading_cell_mutated
                        and view.last_action is not None
                        and prev_avatar_pos is not None
                    ):
                        lead_mob = (
                            0 <= adj_r < H
                            and 0 <= adj_c < W
                            and (
                                int(curr_grid[adj_r, adj_c]) in view.mobile_threat_features
                                or int(curr_grid[adj_r, adj_c])
                                in view.hazard_tracker.mobile_entities
                            )
                        )
                        if not lead_mob and not possible_orthogonal_wall_slip:
                            view.failed_transitions.add((prev_avatar_pos, view.last_action))
                        view.working_memory.pending_gate_check = False
                        is_disp = view.is_displacement_action(view.last_action)
                        if not is_disp:
                            view.entity_action_failed_positions.setdefault(
                                view.last_action, set()
                            ).add(prev_avatar_pos)
                            cal_dirs_now = {
                                (
                                    (1 if d.delta_r > 0 else (-1 if d.delta_r < 0 else 0)),
                                    (1 if d.delta_c > 0 else (-1 if d.delta_c < 0 else 0)),
                                )
                                for d in view.action_dynamics.values()
                                if d.is_displacement_action()
                            }
                            cal_dirs_now.discard((0, 0))
                            pr, pc = prev_avatar_pos
                            all_cardinals_open = all(
                                0 <= pr + dr_n < H
                                and 0 <= pc + dc_n < W
                                and (
                                    int(curr_grid[pr + dr_n, pc + dc_n]) == bg
                                    or view.symbolic_theory.is_walkable(
                                        int(curr_grid[pr + dr_n, pc + dc_n])
                                    )
                                )
                                for dr_n, dc_n in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            )
                            if len(view.entity_action_failed_positions[view.last_action]) >= 2 and (
                                len(cal_dirs_now) >= 4 or all_cardinals_open
                            ):
                                view.immobile_entity_actions.add(view.last_action)
                            prev_inh = view.inhibited_actions.get(view.last_action, 0)
                            view.inhibited_actions[view.last_action] = max(prev_inh + 2, 3)
                        view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 0))
                        view.apply_wm_delta(
                            WorkingMemoryDelta("consecutive_simulation_failures", 0)
                        )

            if diff.changed_pixel_count == 0 and not is_success and not is_failure:
                if continuous_moved or leading_cell_mutated:
                    view.working_memory.consecutive_stuck_steps = 0
                else:
                    view.working_memory.consecutive_stuck_steps += 1
                    if (
                        view.last_action is not None
                        and view.is_displacement_action(view.last_action)
                        and not possible_orthogonal_wall_slip
                    ):
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
                    if continuous_moved or leading_cell_mutated:
                        view.working_memory.consecutive_stuck_steps = 0
                    else:
                        view.working_memory.consecutive_stuck_steps += 1
                        if (
                            view.last_action is not None
                            and view.is_displacement_action(view.last_action)
                            and not possible_orthogonal_wall_slip
                        ):
                            prev_inh = view.inhibited_actions.get(view.last_action, 0)
                            view.inhibited_actions[view.last_action] = max(prev_inh + 2, 4)
                            view.failed_transitions.add((prev_avatar_pos, view.last_action))
                            view.working_memory.habenular_ior.failed_transitions.add(
                                (prev_avatar_pos, view.last_action)
                            )
                else:
                    view.working_memory.consecutive_stuck_steps = 0
                    for inh_act in list(view.inhibited_actions.keys()):
                        if view.is_displacement_action(inh_act):
                            view.inhibited_actions.pop(inh_act, None)
            else:
                view.working_memory.consecutive_stuck_steps = 0

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

            view.state.last_surprise = 0.0
            view.state.last_surprise_eval = None

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
                    and view.avatar_pos != view.last_predicted_pos
                ):
                    discrepancy = True

            if discrepancy:
                did_displace_correctly = (expected_dr != 0 and actual_dr * expected_dr > 0) or (
                    expected_dc != 0 and actual_dc * expected_dc > 0
                )
                pred_open = False
                if view.last_predicted_pos is not None:
                    pr_p, pc_p = view.last_predicted_pos
                    if 0 <= pr_p < H and 0 <= pc_p < W:
                        pv_p = int(curr_grid[pr_p, pc_p])
                        pred_open = pv_p == bg or view.symbolic_theory.is_walkable(pv_p)
                is_orthogonal_slip = bool(avatar_moved and not did_displace_correctly) or bool(
                    not avatar_moved
                    and pred_open
                    and any(d.slip_rate > 0.05 for d in view.action_dynamics.values())
                )
                if continuous_moved:
                    cd = view.working_memory.continuous_delta
                    c_dr = -cd[1]
                    c_dc = cd[0]
                    if (
                        (expected_dr != 0 and c_dr * expected_dr > 0)
                        or (expected_dc != 0 and c_dc * expected_dc > 0)
                        or (
                            expected_dr == 0
                            and expected_dc == 0
                            and (abs(cd[0]) > 1e-4 or abs(cd[1]) > 1e-4)
                        )
                    ):
                        did_displace_correctly = True

                if not (continuous_moved and did_displace_correctly):
                    pred_mob = False
                    if view.last_predicted_pos is not None:
                        pr_m, pc_m = view.last_predicted_pos
                        if 0 <= pr_m < H and 0 <= pc_m < W:
                            pm_curr = int(curr_grid[pr_m, pc_m])
                            pm_prev = (
                                int(view.prev_grid[pr_m, pc_m])
                                if view.prev_grid is not None
                                else pm_curr
                            )
                            pred_mob = (
                                pm_curr != pm_prev
                                or pm_curr in view.mobile_threat_features
                                or pm_curr in view.hazard_tracker.mobile_entities
                                or pm_prev in view.mobile_threat_features
                                or pm_prev in view.hazard_tracker.mobile_entities
                            )
                    if (
                        view.last_action is not None
                        and prev_avatar_pos is not None
                        and not did_displace_correctly
                        and not is_orthogonal_slip
                        and not pred_mob
                    ):
                        view.failed_transitions.add((prev_avatar_pos, view.last_action))
                        prev_inh = view.inhibited_actions.get(view.last_action, 0)
                        view.inhibited_actions[view.last_action] = max(prev_inh + 2, 3)
                    if view.mental_plan:
                        view.mental_plan.clear()
                        view.working_memory.active_plan_step = None
                        view.transition_to_phase(
                            EpistemicPhase.REPLANNING,
                            reason="motor_surprise_plan_divergence",
                        )
                        if is_orthogonal_slip:
                            view.working_memory.simulation_cooldown = 0
                        elif not did_displace_correctly:
                            view.consecutive_plan_failures += 1
                            if view.consecutive_plan_failures >= 2:
                                view.apply_wm_delta(WorkingMemoryDelta("exploration_cooldown", 2))
                                view.apply_wm_delta(
                                    WorkingMemoryDelta("consecutive_plan_failures", 0)
                                )
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
        movable_features.update(view.mobile_threat_features)

        _wm_exog = view.working_memory

        def is_exogenous_mutation(r: int, c: int) -> bool:
            """Isolate uncontrollable exogenous / topologically partitioned mutations (I(DeltaS; A) ~ 0)."""
            return bool(
                _wm_exog.is_exogenous_locus(
                    (r, c), lattice=_prev_grid, bg=bg, include_outer_edge=False
                )
            )

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
            if not is_exogenous_mutation(r, c) and not is_local_to_avatar(r, c):
                if old_v != bg and old_v != _bg_feature:
                    lost_counts[old_v] = lost_counts.get(old_v, 0) + 1
                if new_v != bg and new_v != _bg_feature:
                    gained_counts[new_v] = gained_counts.get(new_v, 0) + 1

        for feat, count in lost_counts.items():
            if feat in gained_counts and abs(count - gained_counts[feat]) <= max(
                2, int(count * 0.25)
            ):
                movable_features.add(feat)
                if feat not in view.symbolic_theory.cargo_features:
                    view.mobile_threat_features.add(feat)

        distant_mutations = [
            (r, c, old_v, new_v)
            for r, c, old_v, new_v in diff.mutated_pixels
            if old_v not in movable_features
            and new_v not in movable_features
            and not is_exogenous_mutation(r, c)
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

            av_feats = set(view.avatar_features or ()) | (
                {view.avatar_feature} if view.avatar_feature is not None else set()
            )
            if (not action_data or "x" not in action_data) and (
                trigger_feat is None
                or trigger_feat == bg
                or trigger_feat == _bg_feature
                or trigger_feat in av_feats
            ):
                found_trigger = False
                for cr_l, cc_l in landing_cells_now:
                    if 0 <= cr_l < H and 0 <= cc_l < W:
                        cand_l = int(_prev_grid[cr_l, cc_l])
                        if (
                            cand_l != bg
                            and cand_l != _bg_feature
                            and cand_l not in av_feats
                            and not view.symbolic_theory.is_walkable(cand_l)
                        ):
                            trigger_pos = (cr_l, cc_l)
                            trigger_feat = cand_l
                            found_trigger = True
                            break
                agent_origin = prev_avatar_pos or _avatar_pos_now
                if not found_trigger and agent_origin is not None:
                    ar, ac = agent_origin
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = ar + dr, ac + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            cand_f = int(_prev_grid[nr, nc])
                            if (
                                cand_f != bg
                                and cand_f != _bg_feature
                                and cand_f not in av_feats
                                and not view.symbolic_theory.is_walkable(cand_f)
                            ):
                                trigger_pos = (nr, nc)
                                trigger_feat = cand_f
                                break

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
                    existing_mut.metadata["mutated_positions"] = [
                        (int(mr), int(mc)) for mr, mc, _, _ in distant_mutations
                    ]
                else:
                    is_action_trigger = (
                        bool(action_data and "x" in action_data)
                        or (trigger_pos != _avatar_pos_now)
                        or not is_known_displacement
                    )
                    mutation_model = StateMutationModel(
                        trigger_type="ACTION" if is_action_trigger else "CONTACT",
                        trigger_pos=trigger_pos,
                        trigger_feature=trigger_feat,
                        mutation_type="ENVIRONMENTAL_TOGGLE",
                        prior_value=distant_mutations[0][2],
                        posterior_value=distant_mutations[0][3],
                        confidence=0.8,
                        occurrences=1,
                        metadata={
                            "action_id": action,
                            "mutated_positions": [
                                (int(mr), int(mc)) for mr, mc, _, _ in distant_mutations
                            ],
                        },
                    )
                    view.state_mutations.append(mutation_model)
                    if (
                        trigger_feat is not None
                        and mutation_model.prior_value != view.bg_feature
                        and not view.symbolic_theory.is_walkable(mutation_model.prior_value)
                        and (
                            mutation_model.posterior_value == view.bg_feature
                            or view.symbolic_theory.is_walkable(mutation_model.posterior_value)
                        )
                    ):
                        view.symbolic_theory.induce_conditional_affordance(
                            feature=mutation_model.prior_value,
                            condition=CausalPredicate(
                                variable="trigger",
                                operator="==",
                                value=trigger_feat,
                            ),
                            evidence=f"state_mutation:{trigger_pos}",
                            prerequisite_feature=trigger_feat,
                        )
                    view.exhausted_candidate_goals.clear()
                    if view.avatar_pos is not None:
                        satisfied_when_map.pop(view.avatar_pos, None)
                    if view.avatar_pos is not None:
                        av_set_m = set(view.avatar_features or ())
                        if view.avatar_feature is not None:
                            av_set_m.add(view.avatar_feature)
                        if view.prev_grid is not None and view.prev_grid.shape == curr_grid.shape:
                            view.object_planner.ledger.observe_grid_connectivity(
                                view.prev_grid,
                                view.avatar_pos,
                                view.bg_feature,
                                set(view.symbolic_theory.walkable_features),
                                av_set_m,
                            )
                        view.object_planner.ledger.observe_grid_connectivity(
                            curr_grid,
                            view.avatar_pos,
                            view.bg_feature,
                            set(view.symbolic_theory.walkable_features),
                            av_set_m,
                        )
                    if trigger_pos is not None:
                        view.symbolic_theory.mark_locus_explored(
                            trigger_pos, reason="remote_mutation_triggered"
                        )
                        if action is not None and not view.is_displacement_action(action):
                            for oa in available_actions or ():
                                if not view.is_displacement_action(
                                    oa
                                ) and not view.is_spatial_effector(oa):
                                    view.tested_action_positions.add((trigger_pos, oa))
                                    view.entity_action_failed_positions.setdefault(oa, set()).add(
                                        trigger_pos
                                    )
                    view.mental_plan.clear()
                    mutated_set = {(int(mr), int(mc)) for mr, mc, _, _ in distant_mutations}
                    for mr, mc in mutated_set:
                        view.learned_barriers.discard((mr, mc))
                        view.symbolic_theory.unmark_locus_explored((mr, mc))
                        view.exhausted_candidate_goals.discard((mr, mc))
                        for dr_m, dc_m in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                            adj_m = (mr + dr_m, mc + dc_m)
                            for fp in [fp for fp in view.failed_transitions if fp[0] == adj_m]:
                                view.failed_transitions.discard(fp)
                    mutated_adj_set = {
                        (mr + dr_m, mc + dc_m)
                        for mr, mc in mutated_set
                        for dr_m, dc_m in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
                    }
                    for ent in view.extract_entities(curr_grid, view.bg_feature):
                        if (
                            ent.feature_id != view.bg_feature
                            and ent.role not in (EntityRole.AGENT, EntityRole.OBSTACLE)
                            and not _wm_exog.is_exogenous_locus(
                                ent.grid_pos, lattice=curr_grid, bg=bg, include_outer_edge=False
                            )
                        ):
                            if ent.area <= 64:
                                view.symbolic_theory.induce_candidate_goal(
                                    ent.feature_id, evidence="post_mutation_exploration"
                                )
                            ent_cells = ent.properties.get("cells", [ent.grid_pos])
                            if any(tuple(ec) in mutated_adj_set for ec in ent_cells):
                                view.symbolic_theory.unmark_locus_explored(ent.grid_pos)
                                view.exhausted_candidate_goals.discard(ent.grid_pos)
                                for ec in ent_cells:
                                    view.symbolic_theory.unmark_locus_explored(tuple(ec))
                                    view.exhausted_candidate_goals.discard(tuple(ec))
                    if view.subgoal_stack.has_pending_goals:
                        cur_sg = view.subgoal_stack.current_subgoal
                        if cur_sg is not None:
                            cur_sg.metadata["triggered"] = True
                            view.subgoal_stack.pop_subgoal()
                    logger.info(
                        "AutonomousEpistemicEngine: Induced StateMutationModel! Trigger %s at %s changed %d distant pixels.",
                        mutation_model.trigger_type,
                        trigger_pos,
                        len(distant_mutations),
                    )

        # ── B2. Inventory / Contact-Item Pickup -> Re-evaluate Singleton Gates ──
        av_feats = view.avatar_features or (
            {view.avatar_feature} if view.avatar_feature is not None else set()
        )
        if (
            not is_failure
            and not is_success
            and view.avatar_pos is not None
            and _prev_grid is not None
            and _prev_grid.shape == curr_grid.shape
        ):
            ar_now, ac_now = view.avatar_pos
            if 0 <= ar_now < H and 0 <= ac_now < W:
                stepped_val = int(_prev_grid[ar_now, ac_now])
                if (
                    stepped_val != bg
                    and stepped_val != _bg_feature
                    and stepped_val not in av_feats
                    and stepped_val != view.working_memory.shroud_feature
                    and not _sym_theory.is_barrier(stepped_val)
                    and int(np.count_nonzero(curr_grid == stepped_val))
                    < int(np.count_nonzero(_prev_grid == stepped_val))
                ):
                    view.working_memory.collected_inventory_features.add(stepped_val)
                    if (
                        stepped_val in _sym_theory.barrier_features
                        or (ar_now, ac_now) in view.learned_barriers
                    ):
                        view.working_memory.pending_gate_check = False
                    else:
                        view.working_memory.pending_gate_check = True
                    gate_pts: set[tuple[int, int]] = set()
                    for b_feat in list(_sym_theory.barrier_features):
                        b_pts = np.argwhere(curr_grid == b_feat)
                        if 1 <= len(b_pts) <= 2:
                            _sym_theory.retract_barrier(b_feat, reason="inventory_state_change")
                            _sym_theory._explored_features.discard(b_feat)
                            for br_i, bc_i in b_pts:
                                gate_pts.add((int(br_i), int(bc_i)))
                    for lb_r, lb_c in list(view.learned_barriers):
                        if 0 <= lb_r < H and 0 <= lb_c < W:
                            lb_v = int(curr_grid[lb_r, lb_c])
                            if (
                                lb_v != bg
                                and lb_v != _bg_feature
                                and lb_v != view.working_memory.shroud_feature
                                and lb_v not in _hazard_tracker.known_lethal_features
                                and int(np.count_nonzero(curr_grid == lb_v)) <= 2
                            ):
                                _sym_theory.retract_barrier(lb_v, reason="inventory_state_change")
                                _sym_theory._explored_features.discard(lb_v)
                                gate_pts.add((lb_r, lb_c))
                    for b_pt in gate_pts:
                        view.learned_barriers.discard(b_pt)
                        view.exhausted_candidate_goals.discard(b_pt)
                        _sym_theory.unmark_locus_explored(b_pt)
                        for dr_g, dc_g in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            adj_g = (b_pt[0] + dr_g, b_pt[1] + dc_g)
                            view.exhausted_candidate_goals.discard(adj_g)
                            _sym_theory.unmark_locus_explored(adj_g)
                            for fp in [fp for fp in view.failed_transitions if fp[0] == adj_g]:
                                view.failed_transitions.discard(fp)
                    if gate_pts:
                        view.mental_plan.clear()

        # ── C. Win / Loss Feedback Assimilation ──────────────────────────────

        if is_success:
            win_target = None
            cargo_win_target = None
            cargo_feat = None
            receptacle_feat = None
            is_layout_transition = bool(
                diff
                and (
                    diff.diff_type == FrameDiffType.GLOBAL_TRANSITION
                    or diff.changed_pixel_count
                    > max(4 * max(1, view.avatar_size), int(H * W * 0.08))
                )
            )
            obstacle_feats = {
                pe.feature_id for pe in prev_entities if pe.role == EntityRole.OBSTACLE
            }
            max_compact_extent = max(4, int(H * W * 0.08))

            def _is_valid_global_goal_feat(feat: int) -> bool:
                if (
                    feat == bg
                    or feat == _bg_feature
                    or feat in av_feats
                    or _sym_theory.is_barrier(feat)
                    or feat in _sym_theory.walkable_features
                    or feat in obstacle_feats
                ):
                    return False
                if (
                    _prev_grid is not None
                    and int(np.count_nonzero(_prev_grid == feat)) > max_compact_extent
                ):
                    return False
                return True

            # Detect non-ego entity displacement to goal on winning step (e.g. block push / cargo delivery)
            if not is_layout_transition and diff and diff.mutated_pixels:
                for r_m, c_m, old_m, new_m in diff.mutated_pixels:
                    new_i = int(new_m)
                    old_i = int(old_m)
                    if (
                        (r_m, c_m) != view.avatar_pos
                        and (r_m, c_m) != prev_avatar_pos
                        and new_i not in av_feats
                        and new_i != bg
                        and new_i != _bg_feature
                        and not _sym_theory.is_barrier(new_i)
                        and new_i not in _sym_theory.walkable_features
                        and (
                            _prev_grid is None
                            or int(np.count_nonzero(_prev_grid == new_i)) <= max_compact_extent
                        )
                        and not _wm_exog.is_exogenous_locus(
                            (r_m, c_m), lattice=curr_grid, bg=bg, include_outer_edge=True
                        )
                    ):
                        cargo_win_target = (r_m, c_m)
                        cargo_feat = new_i
                        if _is_valid_global_goal_feat(old_i):
                            receptacle_feat = old_i
                        break

            if cargo_win_target is not None:
                win_target = cargo_win_target
                if cargo_feat is not None:
                    _sym_theory.induce_cargo(cargo_feat, evidence="cargo_delivery_win")
                if receptacle_feat is not None:
                    _sym_theory.induce_receptacle(receptacle_feat, evidence="cargo_delivery_win")
                    _sym_theory.induce_goal(receptacle_feat, evidence=TerminalSignal.SUCCESS)
                    _sym_theory.mark_verified_in_context("goal", receptacle_feat)
                    view.learned_receptacle_positions.add(cargo_win_target)
            elif (
                not is_layout_transition
                and view.avatar_pos is not None
                and (prev_avatar_pos is None or view.avatar_pos != prev_avatar_pos)
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
                c_goal = getattr(view.working_memory, "continuous_goal_pos", None)
                if c_goal is not None and _prev_grid is not None:
                    rf = getattr(view.working_memory, "reference_frame", None)
                    origin_desc = (
                        getattr(rf, "origin_description", "bottom_left_metric")
                        if rf
                        else "bottom_left_metric"
                    )
                    y_max = (
                        float(rf.bounds[1][1])
                        if (rf and rf.bounds and len(rf.bounds) >= 2)
                        else float(H)
                    )
                    if "bottom_left" in origin_desc:
                        gr_l = int(np.clip(round(y_max - 0.5 - float(c_goal[1])), 0, H - 1))
                    else:
                        gr_l = int(np.clip(round(float(c_goal[1]) - 0.5), 0, H - 1))
                    gc_l = int(np.clip(round(float(c_goal[0]) - 0.5), 0, W - 1))
                    cand_c_feat = int(_prev_grid[gr_l, gc_l])
                    if cand_c_feat != bg and cand_c_feat != _bg_feature:
                        win_target = (gr_l, gc_l)

                view.record_learned_goal(win_target)
                if _prev_grid is not None and 0 <= win_target[0] < H and 0 <= win_target[1] < W:
                    goal_feat = int(_prev_grid[win_target[0], win_target[1]])
                    if (goal_feat == bg or goal_feat == _bg_feature) and _prev_grid is not None:
                        for dr_w in (-1, 0, 1):
                            for dc_w in (-1, 0, 1):
                                nr_w, nc_w = win_target[0] + dr_w, win_target[1] + dc_w
                                if 0 <= nr_w < H and 0 <= nc_w < W:
                                    cand_f = int(_prev_grid[nr_w, nc_w])
                                    if (
                                        cand_f != bg
                                        and cand_f != _bg_feature
                                        and cand_f not in av_feats
                                        and not _sym_theory.is_barrier(cand_f)
                                        and (
                                            cand_f in _sym_theory.candidate_goal_features
                                            or cand_f in _sym_theory.goal_features
                                        )
                                    ):
                                        goal_feat = cand_f
                                        win_target = (nr_w, nc_w)
                                        break
                            if goal_feat != bg and goal_feat != _bg_feature:
                                break
                    if _is_valid_global_goal_feat(goal_feat):
                        _sym_theory.induce_goal(goal_feat, evidence=TerminalSignal.SUCCESS)
                        _sym_theory.mark_verified_in_context("goal", goal_feat)
                        view.learned_barriers.discard(win_target)
                        logger.info(
                            "AutonomousEpistemicEngine: Grounded invariant WIN GOAL FEATURE %d at %s",
                            goal_feat,
                            win_target,
                        )

                max_entity_area = max(64, int(H * W * 0.15))
                for pe in prev_entities:
                    if (
                        pe.role not in (EntityRole.OBSTACLE, EntityRole.AGENT)
                        and _is_valid_global_goal_feat(pe.feature_id)
                        and 1 <= pe.area <= max_entity_area
                    ):
                        pe_cells = pe.properties.get("cells", [pe.grid_pos])
                        if win_target in pe_cells or pe.grid_pos == win_target:
                            _sym_theory.induce_goal(pe.feature_id, evidence=TerminalSignal.SUCCESS)
                            _sym_theory.mark_verified_in_context("goal", pe.feature_id)
                            view.learned_barriers.discard(win_target)
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded invariant WIN GOAL ENTITY feature %d (area=%d)",
                                pe.feature_id,
                                pe.area,
                            )

            if view.current_simulated_goal is not None and _prev_grid is not None:
                sr, sc = view.current_simulated_goal
                if 0 <= sr < H and 0 <= sc < W:
                    sim_feat = int(_prev_grid[sr, sc])
                    if _is_valid_global_goal_feat(sim_feat):
                        _sym_theory.induce_goal(sim_feat, evidence=TerminalSignal.SUCCESS)
                        _sym_theory.mark_verified_in_context("goal", sim_feat)

            if view.active_hypothesis:
                view.active_hypothesis.confirmed = True
                view.active_hypothesis.confidence = 1.0

            # Record the terminal winning transition before reset clears current_episode
            # so Hippocampal SWR consolidation and DreamCoder LibraryLearner capture the full trace.
            view.episodic_cortex.record_transition(
                step=view.step_counter,
                avatar_pos=view.avatar_pos,
                action=action,
                action_data=action_data,
                observation_diff=diff,
                reward=1.0,
                is_terminal=True,
                terminal=terminal,
                is_failure=False,
                is_success=True,
                grid_snapshot=curr_grid,
            )

            # Context transition: cleanly reset episodic spatial memory for the next context
            view.reset_episode(
                retain_dynamics=True,
                is_new_context=True,
                context_epoch=view.context_epoch + 1,
            )
            return
        elif view.avatar_pos is not None and not is_failure:
            if view.avatar_pos not in getattr(
                view.working_memory, "active_epistemic_conditions", {}
            ):
                view.exhausted_candidate_goals.add(view.avatar_pos)

        is_fatal = (terminal == TerminalSignal.FAILURE) or (
            is_failure and terminal != TerminalSignal.TIMEOUT
        )

        if is_fatal:
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
            # Derive death cell from the terminal lattice (curr_grid) first.
            # view.avatar_pos is NOT yet updated by _update_avatar_position_from_grid
            # (that runs in deliberate(), after prepare()/assimilate_feedback()).
            # In stochastic dynamics the avatar may have landed on a different cell
            # than predicted from displacement alone, so we scan curr_grid for the
            # avatar feature to get the actual terminal position.
            if curr_grid is not None and av_feats:
                for av_f in av_feats:
                    av_locs = np.argwhere(curr_grid == av_f)
                    if len(av_locs) > 0:
                        # Pick the nearest candidate to prev_avatar_pos (or any if unknown)
                        if prev_avatar_pos is not None:
                            dists = [
                                abs(int(r) - prev_avatar_pos[0]) + abs(int(c) - prev_avatar_pos[1])
                                for r, c in av_locs
                            ]
                            best_idx = int(np.argmin(dists))
                        else:
                            best_idx = 0
                        cand_p = (int(av_locs[best_idx][0]), int(av_locs[best_idx][1]))
                        if cand_p != prev_avatar_pos:
                            target_pos = cand_p
                            break

            if target_pos is None and prev_avatar_pos is not None:
                # Fallback: predict from action displacement (deterministic envs)
                if action_data and "x" in action_data and "y" in action_data:
                    target_pos = (int(action_data["y"]), int(action_data["x"]))
                else:
                    dr, dc = 0, 0
                    if action in view.action_dynamics:
                        dr, dc = view.action_dynamics[action].get_displacement()
                    tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                    if 0 <= tr < H and 0 <= tc < W:
                        target_pos = (tr, tc)
            elif target_pos is None:
                if view.avatar_pos is not None:
                    target_pos = view.avatar_pos
                elif action_data and "x" in action_data and "y" in action_data:
                    target_pos = (int(action_data["y"]), int(action_data["x"]))

            if target_pos is not None:
                if _prev_grid is not None and 0 <= target_pos[0] < H and 0 <= target_pos[1] < W:
                    direct_feat = int(_prev_grid[target_pos[0], target_pos[1]])
                    is_direct_contact = direct_feat != _bg_feature and direct_feat not in av_feats
                    if is_direct_contact:
                        # Direct physical contact hazard at the destination cell
                        candidate_cells = {direct_feat}
                    else:
                        # Check adjacent mobile entities first (e.g. patroller stepping onto avatar)
                        mob_cands: set[int] = set()
                        for mr_c in range(max(0, target_pos[0] - 2), min(H, target_pos[0] + 3)):
                            for mc_c in range(max(0, target_pos[1] - 2), min(W, target_pos[1] + 3)):
                                if abs(mr_c - target_pos[0]) + abs(mc_c - target_pos[1]) <= 2:
                                    mv = int(_prev_grid[mr_c, mc_c])
                                    if (
                                        mv in _hazard_tracker.mobile_entities
                                        or mv in view.mobile_threat_features
                                    ):
                                        mob_cands.add(mv)
                        if mob_cands:
                            candidate_cells = mob_cands
                            is_direct_contact = True
                        else:
                            # Remote hazard attribution (e.g. projectile or turret in line-of-sight)
                            candidate_cells = set()
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
                            and cand_feat_int != bg
                            and cand_feat_int not in av_feats
                            and (
                                is_direct_contact
                                or not (
                                    _sym_theory.is_walkable(cand_feat_int)
                                    and _sym_theory.is_verified_in_context(
                                        "walkable", cand_feat_int
                                    )
                                )
                            )
                            and feat_count <= max(64, int(total_cells * 0.40))
                        ):
                            was_prior_safe = _sym_theory.has_prior_context_transition and (
                                (
                                    cand_feat_int in _sym_theory.goal_features
                                    and not _sym_theory.is_verified_in_context(
                                        "goal", cand_feat_int
                                    )
                                )
                                or (
                                    cand_feat_int != bg
                                    and cand_feat_int != _bg_feature
                                    and cand_feat_int in _sym_theory.walkable_features
                                    and not _sym_theory.is_verified_in_context(
                                        "walkable", cand_feat_int
                                    )
                                )
                                or (
                                    cand_feat_int in _sym_theory.cargo_features
                                    and not _sym_theory.is_verified_in_context(
                                        "cargo", cand_feat_int
                                    )
                                )
                            )
                            _hazard_tracker.register_lethal_feature(cand_feat_int)
                            _sym_theory.mark_verified_in_context("lethal", cand_feat_int)
                            _sym_theory.retract_cargo(cand_feat_int, reason="lethal_hazard")
                            _sym_theory.retract_goal(cand_feat_int, reason="lethal_hazard")
                            _sym_theory.retract_walkable(cand_feat_int, reason="lethal_hazard")
                            _sym_theory.retract_conditional_affordance(
                                cand_feat_int, reason="lethal_hazard"
                            )
                            _sym_theory.induce_barrier(cand_feat_int, evidence="lethal_hazard")
                            if was_prior_safe:
                                from hbllm.hcir.world.cortex_causal import (
                                    invalidate_stale_context_ontology,
                                )

                                invalidate_stale_context_ontology(
                                    _sym_theory,
                                    hazard_tracker=_hazard_tracker,
                                    episodic_cortex=view.episodic_cortex,
                                    working_memory=view.working_memory,
                                    engine_or_view=view,
                                    bg_feature=_bg_feature,
                                    preserve_current_context=True,
                                    reason=f"prior_safe_became_lethal:{cand_feat_int}",
                                )
                            logger.info(
                                "AutonomousEpistemicEngine: Grounded lethal feature %d at barrier %s",
                                cand_feat_int,
                                target_pos,
                            )
                    view.exhausted_candidate_goals.add(target_pos)

                    # Trajectory-scale aversive conditioning:
                    # The destination cell target_pos resulted in catastrophic loss/death.
                    # Ground it as a static lethal position and barrier so neither mental simulation
                    # nor exploratory probes step into this coordinate again!
                    _hazard_tracker.register_lethal_position(target_pos)
                    view.record_learned_barrier(target_pos)
                    view.context_barriers.setdefault(view.context_epoch, set()).add(target_pos)
                    view.context_hazards.setdefault(view.context_epoch, set()).add(target_pos)

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
            cond_prereqs = {
                aff.prerequisite_feature
                for aff in _sym_theory.conditional_affordances.values()
                if aff.prerequisite_feature is not None
            }
            for pe in prev_entities:
                if (
                    (
                        pe.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE)
                        or pe.feature_id in cond_prereqs
                    )
                    and pe.feature_id not in av_feats
                    and pe.feature_id != bg
                    and pe.feature_id != _bg_feature
                ):
                    dist_to_av = abs(pe.grid_pos[0] - view.avatar_pos[0]) + abs(
                        pe.grid_pos[1] - view.avatar_pos[1]
                    )
                    if dist_to_av <= 1:
                        view.working_memory.acquire_item(
                            feature_id=pe.feature_id,
                            role=getattr(pe.role, "value", str(pe.role)),
                            step=view.step_counter,
                            position=pe.grid_pos,
                        )

        # Tool-Barrier Resonance Credit Assignment
        if _prev_grid is not None and diff.changed_mask is not None:
            changed_coords = np.argwhere(diff.changed_mask)
            _barrier_feats = _sym_theory.barrier_features
            _learned_barriers = view.learned_barriers
            for mr, mc in changed_coords:
                orig_val = int(_prev_grid[mr, mc])
                post_val = int(curr_grid[mr, mc])
                if (
                    orig_val in _barrier_feats
                    or orig_val in _sym_theory.conditional_affordances
                    or (int(mr), int(mc)) in _learned_barriers
                ) and (
                    post_val == _bg_feature
                    or _sym_theory.is_walkable(post_val)
                    or _sym_theory.is_conditionally_traversable(post_val)
                ):
                    view.learned_barriers.discard((int(mr), int(mc)))
                    stale_txs = set()
                    for p, act in view.failed_transitions:
                        d = view.action_deltas.get(act)
                        if d and (p[0] + d[0], p[1] + d[1]) == (int(mr), int(mc)):
                            stale_txs.add((p, act))
                    view.failed_transitions.difference_update(stale_txs)
                    view.working_memory.habenular_ior.failed_transitions.difference_update(
                        stale_txs
                    )
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
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export feedback assimilator capability and runtime grounding state as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "motor_dynamics_assimilation"
        if view is not None:
            grounded = len(getattr(view, "action_dynamics", {}) or {})
            phase = getattr(getattr(view, "state", None), "phase", "unknown")
            rel = f"motor_dynamics_assimilation:grounded:{grounded}:phase:{phase}"
        return CausalPredicate(
            variable="epistemic_feedback_assimilator",
            operator="==",
            value=val,
            relation=rel,
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
