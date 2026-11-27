"""Epistemic Curiosity Explorer & Active Inference Motor Probes."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.active_inference import ActionNode
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.objects.risk import compute_slip_adjacency_penalty
from hbllm.hcir.world.spatial_containment import RoomDoor

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpistemicProbeProposal:
    """Pure immutable proposal emitted by EpistemicCuriosityExplorer."""

    action: int
    action_data: dict[str, Any] | None = None
    new_probe_target: tuple[int, int] | None = None
    new_probe_id: str | None = None
    clear_active_probe: bool = False
    probed_entity_ids_to_add: tuple[str, ...] = ()
    entity_visit_counts_to_inc: tuple[str, ...] = ()
    tested_action_position: tuple[int, tuple[int, int] | None] | None = None
    tested_action: int | None = None
    topological_doorway: tuple[tuple[int, int], int] | None = None
    exploration_mode: str = ""

    def to_causal_predicate(self) -> CausalPredicate:
        """Export epistemic probe proposal as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"probe_proposal:{self.action}",
            operator="==",
            value=self.exploration_mode or "epistemic_curiosity",
            relation=f"target:{self.new_probe_target}",
            target_entity=str(self.action),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording probe proposal."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"probe_proposal_{self.action}_{self.exploration_mode}",
                node_data={
                    "action": self.action,
                    "exploration_mode": self.exploration_mode,
                    "probe_target": self.new_probe_target,
                    "probe_id": self.new_probe_id,
                },
            )
        ]


class EpistemicCuriosityExplorer:
    """Active curiosity-driven hypothesis testing and information-gain exploration."""

    @staticmethod
    def _is_ray_passable(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        r0: int,
        c0: int,
        dr: int,
        dc: int,
        bg: int,
        bg_is_impassable_void: bool,
    ) -> bool:
        """Check whether the full displacement ray from (r0, c0) to (r0+dr, c0+dc) is passable."""
        H, W = curr_grid.shape
        step_r = int(np.sign(dr))
        step_c = int(np.sign(dc))
        dist = max(abs(dr), abs(dc))
        kb = engine.symbolic_theory.transition_kb
        for k in range(1, dist + 1):
            kr = r0 + step_r * k
            kc = c0 + step_c * k
            if not (0 <= kr < H and 0 <= kc < W):
                return False
            feat = int(curr_grid[kr, kc])
            ev = engine.symbolic_theory.evaluate_transition(feat, locus=(kr, kc))
            locus_b = kb.get_belief((kr, kc), scope="locus") if kb is not None else None
            in_learned_b = (kr, kc) in engine.learned_barriers and not (
                locus_b is not None and ev.traversable and ev.confidence > 0.0
            )
            if (
                ev.risk_cost >= 40.0
                or feat in engine.hazard_tracker.known_lethal_features
                or (kr, kc) in engine.current_context_hazards
            ):
                return False
            if in_learned_b or not ev.traversable:
                if not engine.working_memory.body_schema.is_barrier_permeable(feat):
                    return False
            if (
                bg_is_impassable_void
                and feat == bg
                and (kr, kc) not in engine.learned_goal_positions
            ):
                return False
        return True

    @staticmethod
    def _collect_visited_cells(engine: AutonomousEpistemicEngine) -> set[tuple[int, int]]:
        """Collect all visited coordinates across working memory and position visit counters."""
        visited_cells = set(engine.working_memory.visited_cells)
        for c_pos, cnt in engine.position_visit_counts.items():
            if cnt > 0:
                visited_cells.add(c_pos)
        if engine.avatar_pos is not None:
            visited_cells.add(engine.avatar_pos)
        return visited_cells

    @staticmethod
    def _collect_undo_actions(engine: AutonomousEpistemicEngine) -> set[int]:
        """Collect all known undo actions across engine state, working memory, and symbolic theory."""
        undo_acts = set(engine.state.undo_actions)
        undo_acts.update(engine.working_memory.undo_actions)
        undo_acts.update(engine.symbolic_theory.undo_actions)
        return undo_acts

    @staticmethod
    def _probe_uncalibrated_action(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> EpistemicProbeProposal | None:
        """Uncalibrated actions take absolute priority for motor grounding."""
        undo_acts = EpistemicCuriosityExplorer._collect_undo_actions(engine)
        uncalibrated = [
            a
            for a in available_actions
            if not engine.is_action_calibrated(a)
            and a not in engine.immobile_entity_actions
            and engine.inhibited_actions.get(a, 0) == 0
            and a not in undo_acts
        ]
        untested_at_pos = [
            a for a in uncalibrated if (a, engine.avatar_pos) not in engine.tested_action_positions
        ]
        if untested_at_pos:
            # Agency Discovery Bootstrap: If avatar is not yet localized, prioritize displacement
            # actions over spatial effectors. Ego-motion grounding is essential to discover what
            # entity on the screen moves in contingency with motor commands.
            if engine.avatar_pos is None:
                bg = engine.estimate_background(curr_grid)
                H, W = curr_grid.shape
                vals, counts = np.unique(curr_grid, return_counts=True)
                candidates: list[tuple[int, int]] = []
                for v, c in zip(vals, counts):
                    vf = int(v)
                    ev_f = engine.symbolic_theory.evaluate_transition(vf)
                    if vf != bg and c <= 2 and ev_f.traversable and ev_f.risk_cost < 40.0:
                        coords = np.argwhere(curr_grid == vf)
                        for r, c_idx in coords:
                            candidates.append((int(r), int(c_idx)))

                disp_acts = [a for a in untested_at_pos if engine.is_displacement_action(a)]
                pool = disp_acts if disp_acts else untested_at_pos
                if candidates:
                    deltas = engine.action_deltas
                    scored: list[tuple[int, float]] = []
                    for act in pool:
                        dr, dc = deltas.get(act, (0, 0))
                        score = 0.0
                        for r, c_idx in candidates:
                            nr, nc = r + dr, c_idx + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                cell_val = int(curr_grid[nr, nc])
                                ev_c = engine.symbolic_theory.evaluate_transition(
                                    cell_val, locus=(nr, nc)
                                )
                                if cell_val == bg:
                                    score += 2.0
                                elif (
                                    cell_val in engine.hazard_tracker.known_lethal_features
                                    or (nr, nc) in engine.current_context_hazards
                                    or ev_c.risk_cost >= 40.0
                                ):
                                    score -= 10.0
                                elif not ev_c.traversable:
                                    score -= 5.0
                                else:
                                    score -= 1.0
                            else:
                                score -= 10.0
                        scored.append((act, score))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    chosen = scored[0][0]
                else:
                    chosen = pool[0]
            else:
                spatial_effs = [a for a in untested_at_pos if engine.is_spatial_effector(a)]
                if spatial_effs:
                    chosen = spatial_effs[0]
                else:
                    # Spatial kinematics: prefer testing displacement actions into OPEN SPACE
                    # rather than colliding with or displacing adjacent non-background obstacles.
                    bg = engine.estimate_background(curr_grid)
                    H, W = curr_grid.shape
                    av_r, av_c = engine.avatar_pos
                    open_space_acts: list[int] = []
                    occupied_acts: list[int] = []
                    for a in untested_at_pos:
                        dr, dc = engine.action_deltas.get(a, (0, 0))
                        tr, tc = av_r + dr, av_c + dc
                        if 0 <= tr < H and 0 <= tc < W:
                            val = int(curr_grid[tr, tc])
                            ev_t = engine.symbolic_theory.evaluate_transition(val, locus=(tr, tc))
                            if (
                                val == bg
                                or val == 0
                                or (ev_t.traversable and ev_t.confidence > 0.0)
                            ):
                                open_space_acts.append(a)
                            else:
                                occupied_acts.append(a)
                        else:
                            occupied_acts.append(a)
                    chosen = open_space_acts[0] if open_space_acts else untested_at_pos[0]
            action_data = None
            if engine.is_spatial_effector(chosen):
                action_data = engine.ground_effector_action(curr_grid, chosen)
                if action_data is None:
                    non_effs = [a for a in untested_at_pos if not engine.is_spatial_effector(a)]
                    if non_effs:
                        chosen = non_effs[0]
                    else:
                        return None
            return EpistemicProbeProposal(
                action=chosen,
                action_data=action_data,
                tested_action_position=(chosen, engine.avatar_pos),
                tested_action=chosen,
                exploration_mode="uncalibrated_grounding",
            )
        return None

    @staticmethod
    def _probe_motor_babble(
        engine: AutonomousEpistemicEngine,
        displacement_actions: list[int],
        curr_grid: np.ndarray | None = None,
    ) -> EpistemicProbeProposal | None:
        """Motor babbling for self-identification and agency discovery."""
        if displacement_actions and not engine.avatar_features and engine.avatar_feature is None:
            # If curr_grid is provided, evaluate displacement actions across candidate
            # avatar features to prioritize stepping into free background space rather
            # than stepping blindly into walls, hazards, or out of bounds.
            if curr_grid is not None and curr_grid.ndim >= 2:
                bg = engine.estimate_background(curr_grid)
                H, W = curr_grid.shape
                vals, counts = np.unique(curr_grid, return_counts=True)
                candidates: list[tuple[int, int]] = []
                for v, c in zip(vals, counts):
                    vf = int(v)
                    ev_f = engine.symbolic_theory.evaluate_transition(vf)
                    if vf != bg and c <= 2 and ev_f.traversable and ev_f.risk_cost < 40.0:
                        coords = np.argwhere(curr_grid == vf)
                        for r, c_idx in coords:
                            candidates.append((int(r), int(c_idx)))

                if candidates:
                    deltas = engine.action_deltas
                    scored_actions: list[tuple[int, float]] = []
                    for act in displacement_actions:
                        dr, dc = deltas.get(act, (0, 0))
                        score = 0.0
                        for r, c_idx in candidates:
                            nr, nc = r + dr, c_idx + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                cell_val = int(curr_grid[nr, nc])
                                ev_c = engine.symbolic_theory.evaluate_transition(
                                    cell_val, locus=(nr, nc)
                                )
                                if cell_val == bg:
                                    score += 2.0
                                elif (
                                    cell_val in engine.hazard_tracker.known_lethal_features
                                    or (nr, nc) in engine.current_context_hazards
                                    or ev_c.risk_cost >= 40.0
                                ):
                                    score -= 10.0
                                elif not ev_c.traversable:
                                    score -= 5.0
                                else:
                                    score -= 1.0
                            else:
                                score -= 10.0
                        scored_actions.append((act, score))

                    scored_actions.sort(key=lambda x: x[1], reverse=True)
                    best_act = scored_actions[0][0]
                    return EpistemicProbeProposal(
                        action=best_act,
                        exploration_mode="motor_babbling",
                    )

            babble_idx = engine.context_epistemic_probes % len(displacement_actions)
            return EpistemicProbeProposal(
                action=displacement_actions[babble_idx],
                exploration_mode="motor_babbling",
            )
        return None

    @staticmethod
    def _probe_spatial_effector(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        displacement_actions: list[int],
        spatial_effector_actions: list[int],
    ) -> EpistemicProbeProposal | None:
        """Parietal Affordance Competition (Cisek's Hypothesis): Reach gating over locomotion."""
        if not spatial_effector_actions:
            return None

        structural_goals = PerceptionEngine.detect_structural_goals(
            curr_grid,
            bg=engine.estimate_background(curr_grid),
            avatar_features=engine.avatar_features,
            avatar_feature=engine.avatar_feature,
        )
        has_optical_targets = any(
            g.get("type") in ("relational_alignment", "optical_mirror_target", "reflection_target")
            for g in structural_goals
        )
        chosen_eff = spatial_effector_actions[0]
        if engine.inhibited_actions.get(chosen_eff, 0) > 0:
            return None

        is_stuck_or_looping = (
            engine.consecutive_stuck_steps >= 2 and engine.last_action in displacement_actions
        )
        has_effective_target = any(
            p not in engine.quiescent_actuation_targets
            and engine.entity_visit_counts.get(f"actuate_{p[0]}_{p[1]}", 0) < 3
            for p in engine.effective_actuation_targets
        )
        interleaved_reach = engine.context_epistemic_probes % 6 == 0

        should_probe = (
            not displacement_actions
            or (
                engine.avatar_pos is not None
                and (
                    is_stuck_or_looping
                    or has_effective_target
                    or interleaved_reach
                    or has_optical_targets
                )
            )
            or (
                engine.avatar_pos is None
                and engine.context_epistemic_probes > 4
                and not any(a for a in displacement_actions if not engine.is_action_calibrated(a))
            )
        )
        if should_probe:
            action_data = engine.ground_effector_action(curr_grid, chosen_eff)
            if action_data is not None:
                if displacement_actions:
                    engine.inhibited_actions[chosen_eff] = max(
                        engine.inhibited_actions.get(chosen_eff, 0), 3
                    )
                return EpistemicProbeProposal(
                    action=chosen_eff,
                    action_data=action_data,
                    exploration_mode="spatial_effector",
                )
        return None

    @staticmethod
    def _probe_attractor_oscillation(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        discrete_transform_actions: list[int],
        disp_actions: list[tuple[int, tuple[int, int]]],
        spatial_effector_actions: list[int],
        H: int,
        W: int,
        bg: int,
    ) -> EpistemicProbeProposal | None:
        """Oscillation & Attractor Loop Breaking when agent is trapped in a cycle."""
        is_stuck = engine.consecutive_stuck_steps >= 2
        is_oscillating = (
            engine.avatar_pos is not None and engine.recent_positions.count(engine.avatar_pos) >= 3
        ) or is_stuck
        if not is_oscillating or engine.avatar_pos is None:
            return None

        logger.debug(
            "AutonomousEpistemicEngine: Attractor loop/stuck state detected at %s! Breaking loop...",
            engine.avatar_pos,
        )

        walkable_and_safe = engine.symbolic_theory.walkable_features
        bg_is_impassable_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
        visited_cells = EpistemicCuriosityExplorer._collect_visited_cells(engine)
        frontier_escape = EpistemicCuriosityExplorer._probe_frontier_escape_with_target(
            engine, curr_grid, disp_actions, bg, bg_is_impassable_void
        )
        if frontier_escape is not None and frontier_escape[1] not in visited_cells:
            return EpistemicProbeProposal(
                action=frontier_escape[0],
                clear_active_probe=True,
                new_probe_target=frontier_escape[1],
                new_probe_id=f"frontier_{frontier_escape[1][0]}_{frontier_escape[1][1]}",
                exploration_mode="frontier_backtrack",
            )

        uncalibrated = [
            a
            for a in available_actions
            if a not in engine.action_dynamics
            or not engine.action_dynamics[a].is_displacement_action()
        ]
        untested_here = [
            a for a in uncalibrated if (a, engine.avatar_pos) not in engine.tested_action_positions
        ]
        if untested_here:
            for act in untested_here:
                act_data = None
                if engine.is_spatial_effector(act):
                    act_data = engine.ground_effector_action(curr_grid, act)
                    if act_data is None:
                        continue
                return EpistemicProbeProposal(
                    action=act,
                    action_data=act_data,
                    clear_active_probe=True,
                    tested_action_position=(act, engine.avatar_pos),
                    exploration_mode="attractor_loop_untested",
                )

        if frontier_escape is not None:
            return EpistemicProbeProposal(
                action=frontier_escape[0],
                clear_active_probe=True,
                new_probe_target=frontier_escape[1],
                exploration_mode="frontier_backtrack",
            )

        best_act = None
        min_visits = float("inf")
        for act, (dr_cal, dc_cal) in disp_actions:
            if engine.inhibited_actions.get(act, 0) > 0:
                continue
            if (engine.avatar_pos, act) in engine.failed_transitions:
                continue
            nr = engine.avatar_pos[0] + dr_cal
            nc = engine.avatar_pos[1] + dc_cal
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            _osc_feat = int(curr_grid[nr, nc])
            ev_osc = engine.symbolic_theory.evaluate_transition(_osc_feat, locus=(nr, nc))
            osc_locus_b = engine.symbolic_theory.transition_kb.get_belief((nr, nc), scope="locus")
            in_learned_b = (nr, nc) in engine.learned_barriers and not (
                osc_locus_b is not None and ev_osc.traversable and ev_osc.confidence > 0.0
            )
            if (
                (nr, nc) in engine.hazard_tracker.static_lethal_positions
                or in_learned_b
                or not ev_osc.traversable
                or ev_osc.risk_cost >= 40.0
                or _osc_feat in engine.hazard_tracker.known_lethal_features
            ):
                continue
            uncertainty_pen = engine._epistemic_uncertainty_cost(_osc_feat, bg) * 3.0
            visits = engine.position_visit_counts.get((nr, nc), 0)
            recency_pen = 20.0 if (nr, nc) in list(engine.recent_positions)[-6:] else 0.0
            total_score = visits + recency_pen + uncertainty_pen
            if total_score < min_visits:
                min_visits = total_score
                best_act = act

        if best_act is not None:
            return EpistemicProbeProposal(
                action=best_act,
                clear_active_probe=True,
                exploration_mode="attractor_loop_least_visited",
            )

        untested_discrete = [
            a
            for a in discrete_transform_actions
            if (a, engine.avatar_pos) not in engine.tested_action_positions
        ]
        if untested_discrete and (is_stuck or is_oscillating):
            disc_act = untested_discrete[engine.context_epistemic_probes % len(untested_discrete)]
            return EpistemicProbeProposal(
                action=disc_act,
                clear_active_probe=True,
                tested_action_position=(disc_act, engine.avatar_pos),
                exploration_mode="attractor_loop_discrete",
            )

        if spatial_effector_actions and min_visits == float("inf"):
            chosen_eff = spatial_effector_actions[0]
            eff_data = engine.ground_effector_action(curr_grid, chosen_eff)
            if eff_data is not None:
                return EpistemicProbeProposal(
                    action=chosen_eff,
                    action_data=eff_data,
                    clear_active_probe=True,
                    exploration_mode="attractor_loop_effector",
                )

        undo_acts = EpistemicCuriosityExplorer._collect_undo_actions(engine)
        uninhibited_available = [
            a
            for a in available_actions
            if engine.inhibited_actions.get(a, 0) == 0
            and a not in undo_acts
            and not engine.is_spatial_effector(a)
        ]
        non_undo_available = [
            a for a in available_actions if a not in undo_acts and not engine.is_spatial_effector(a)
        ]
        fallback_action = (
            uninhibited_available[0]
            if uninhibited_available
            else (non_undo_available[0] if non_undo_available else available_actions[0])
        )
        fallback_data = None
        if engine.is_spatial_effector(fallback_action):
            fallback_data = engine.ground_effector_action(curr_grid, fallback_action)
        return EpistemicProbeProposal(
            action=fallback_action,
            action_data=fallback_data,
            clear_active_probe=True,
            exploration_mode="attractor_loop_fallback",
        )

    @staticmethod
    def _probe_looming_threat_evasion(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        disp_actions: list[tuple[int, tuple[int, int]]],
        bg: int,
        bg_is_impassable_void: bool,
    ) -> int | None:
        """Premotor Looming Hazard Reflex: Evasion when impending collision detected."""
        if engine.avatar_pos is None:
            return None

        ar, ac = engine.avatar_pos
        return None

    @staticmethod
    def _probe_active_inference(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        disp_actions: list[tuple[int, tuple[int, int]]],
        target_pos: tuple[int, int],
        bg: int,
        bg_is_impassable_void: bool,
        H: int,
        W: int,
    ) -> int | None:
        """Free Energy Active Inference Action Selection trading info gain vs risk."""
        if engine.avatar_pos is None:
            return None

        tr, tc = target_pos
        candidate_action_nodes: list[ActionNode] = []
        info_gain_map: dict[str, float] = {}
        node_to_act_map: dict[str, int] = {}

        for act, (dr_cal, dc_cal) in disp_actions:
            if (engine.avatar_pos, act) in engine.failed_transitions:
                continue
            if engine.inhibited_actions.get(act, 0) > 0:
                continue
            if engine.working_memory.habenular_ior.is_action_inhibited(engine.avatar_pos, act):
                continue
            ar, ac = engine.avatar_pos
            nr = ar + dr_cal
            nc = ac + dc_cal
            if not EpistemicCuriosityExplorer._is_ray_passable(
                engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
            ):
                continue

            cell_feat = int(curr_grid[nr, nc])
            rp_map = engine.working_memory.concept_role_priors
            hz_prior = rp_map.get(cell_feat, {}).get("hazard", 0.0) if rp_map else 0.0
            if hz_prior >= 0.65 and (nr, nc) != (tr, tc):
                continue

            # OFC Deadlock Pruning
            if cell_feat in engine.learned_cargo_features:
                pushed_r, pushed_c = nr + dr_cal, nc + dc_cal
                dl_res = CounterfactualDeadlockDetector.evaluate_deadlock(
                    (pushed_r, pushed_c),
                    set(engine.learned_goal_positions),
                    engine.learned_barriers,
                    {(pushed_r, pushed_c)},
                    (H, W),
                )
                if dl_res.is_deadlock:
                    continue

            visit_count = engine.position_visit_counts.get((nr, nc), 0)
            info_gain = 1.0 / (1.0 + float(visit_count))
            if (nr, nc) == (tr, tc):
                info_gain += 0.50

            uncertainty_pen = (
                0.0 if (nr, nc) == (tr, tc) else engine._epistemic_uncertainty_cost(cell_feat, bg)
            )
            risk_factor = min(1.0, uncertainty_pen / 40.0)

            rec_positions = list(engine.recent_positions)
            cycle_pen = 0.0
            if rec_positions:
                if len(rec_positions) >= 2 and (nr, nc) == rec_positions[-2]:
                    cycle_pen += 50.0  # Immediate 2-step reversal
                elif len(rec_positions) >= 4 and (nr, nc) in rec_positions[-4:]:
                    cycle_pen += 30.0  # 4-step limit cycle
                elif (nr, nc) in rec_positions:
                    cycle_pen += 10.0

            repulsion_penalty = engine.working_memory.habenular_ior.get_repulsion_penalty(
                engine.avatar_pos, act
            )
            mob_avoid = engine.hazard_tracker.get_mobile_avoidance_cells(
                curr_grid, set(engine.symbolic_theory.barrier_features), bg
            )
            dyn_risk_pen = (
                120.0 if (nr, nc) in mob_avoid else 0.0
            ) + compute_slip_adjacency_penalty(engine, curr_grid, nr, nc, tr, tc, bg, H, W, rp_map)
            dist_to_target = abs(nr - tr) + abs(nc - tc)
            estimated_cost = float(dist_to_target) + cycle_pen + repulsion_penalty + dyn_risk_pen

            act_id = f"act_{act}_{nr}_{nc}"
            node = ActionNode(
                id=act_id,
                intent=f"move_{dr_cal}_{dc_cal}",
                risk_factor=risk_factor,
                estimated_cost=int(round(estimated_cost)),
            )
            candidate_action_nodes.append(node)
            info_gain_map[act_id] = info_gain
            node_to_act_map[act_id] = act

        if candidate_action_nodes:
            from hbllm.hcir.world.active_inference import ActiveInferenceEngine

            eval_results = ActiveInferenceEngine().evaluate_candidates(
                candidate_action_nodes,
                information_gain_map=info_gain_map,
            )
            if eval_results:
                return node_to_act_map[eval_results[0].action.id]

        # Metacognitive Refractory Relaxation
        sorted_disp = sorted(
            disp_actions,
            key=lambda x: (
                engine.working_memory.habenular_ior.get_repulsion_penalty(engine.avatar_pos, x[0]),
                engine.inhibited_actions.get(x[0], 0),
            ),
        )
        ar, ac = engine.avatar_pos
        rec_positions = list(engine.recent_positions)
        for act, (dr_cal, dc_cal) in sorted_disp:
            if (engine.avatar_pos, act) in engine.failed_transitions:
                continue
            if engine.working_memory.habenular_ior.is_action_inhibited(engine.avatar_pos, act):
                continue
            nr = ar + dr_cal
            nc = ac + dc_cal
            if rec_positions and (
                (len(rec_positions) >= 2 and (nr, nc) == rec_positions[-2])
                or (len(rec_positions) >= 4 and (nr, nc) in rec_positions[-4:])
            ):
                continue
            if not EpistemicCuriosityExplorer._is_ray_passable(
                engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
            ):
                continue
            return act

        return None

    @staticmethod
    def _probe_frontier_escape_with_target(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        disp_actions: list[tuple[int, tuple[int, int]]],
        bg: int,
        bg_is_impassable_void: bool,
    ) -> tuple[int, tuple[int, int]] | None:
        """Frontier Backtracking: BFS shortest path to nearest unexhausted walkable frontier cell."""
        if not disp_actions or engine.avatar_pos is None:
            return None

        visited_cells = EpistemicCuriosityExplorer._collect_visited_cells(engine)

        bfs_q: deque[tuple[tuple[int, int], list[Any]]] = deque([(engine.avatar_pos, [])])
        bfs_visited = {engine.avatar_pos}
        best_frontier_path: list[Any] | None = None
        best_frontier_target: tuple[int, int] | None = None
        least_visited_path: list[Any] | None = None
        least_visited_target: tuple[int, int] | None = None
        least_visits = float("inf")

        while bfs_q and len(bfs_visited) < 300:
            p, path = bfs_q.popleft()
            p_visits = engine.position_visit_counts.get(p, 0)
            if p != engine.avatar_pos and p_visits < least_visits:
                least_visits = p_visits
                least_visited_path = path
                least_visited_target = p

            for act, (dr_cal, dc_cal) in disp_actions:
                nbr = (p[0] + dr_cal, p[1] + dc_cal)
                nr, nc = nbr
                if (nr, nc) in bfs_visited:
                    continue
                if not EpistemicCuriosityExplorer._is_ray_passable(
                    engine, curr_grid, p[0], p[1], dr_cal, dc_cal, bg, bg_is_impassable_void
                ):
                    continue
                if p == engine.avatar_pos and (
                    (p, act) in engine.failed_transitions
                    or engine.working_memory.habenular_ior.is_action_inhibited(p, act)
                ):
                    continue
                if nbr not in visited_cells:
                    best_frontier_path = path + [act]
                    best_frontier_target = nbr
                    break
                bfs_visited.add(nbr)
                bfs_q.append((nbr, path + [act]))

            if best_frontier_path is not None:
                break

        if best_frontier_path and best_frontier_target is not None:
            return best_frontier_path[0], best_frontier_target
        if least_visited_path and least_visited_target is not None:
            return least_visited_path[0], least_visited_target

        return None

    @classmethod
    def compute_epistemic_probe(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> EpistemicProbeProposal:
        """Pure reasoning calculation producing an EpistemicProbeProposal."""
        H, W = curr_grid.shape
        bg = (
            engine.bg_feature
            if get_engine_state_attr(engine, "bg_feature", None) is not None
            else engine.estimate_background(curr_grid)
        )
        walkable_and_safe = engine.symbolic_theory.walkable_features
        bg_is_impassable_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
        entities = engine.extract_entities(curr_grid, bg)

        # 1. Uncalibrated actions
        uncalibrated_prop = cls._probe_uncalibrated_action(engine, curr_grid, available_actions)
        if uncalibrated_prop is not None:
            return uncalibrated_prop

        # 2. Dynamic Affordance Partitioning
        displacement_actions = [a for a in available_actions if engine.is_displacement_action(a)]
        spatial_effector_actions = [a for a in available_actions if engine.is_spatial_effector(a)]
        discrete_transform_actions = [
            a
            for a in available_actions
            if a not in displacement_actions and a not in spatial_effector_actions
        ]

        # 2b. Motor babbling for self-identification
        babble_prop = cls._probe_motor_babble(engine, displacement_actions, curr_grid=curr_grid)
        if babble_prop is not None:
            return babble_prop

        # A. Non-displacement or Spatial Effector Environments
        effector_prop = cls._probe_spatial_effector(
            engine, curr_grid, displacement_actions, spatial_effector_actions
        )
        if effector_prop is not None:
            return effector_prop

        # 3. Spatial Movement Curiosity with Target Commitment & Loop Breaking
        if engine.avatar_pos is not None:
            probed_ids_to_add: list[str] = []
            for e in entities:
                if (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                    <= 1
                ):
                    probed_ids_to_add.append(e.id)

            clear_active_probe = False
            if engine.active_probe_target is not None:
                dist_to_target = abs(engine.active_probe_target[0] - engine.avatar_pos[0]) + abs(
                    engine.active_probe_target[1] - engine.avatar_pos[1]
                )
                if dist_to_target <= 1 or engine.probe_target_steps > 12:
                    if engine.active_probe_id:
                        probed_ids_to_add.append(engine.active_probe_id)
                    clear_active_probe = True

            disp_actions = [
                (act, engine.action_dynamics[act].get_displacement())
                for act in displacement_actions
                if act in engine.action_dynamics
            ]

            # Attractor oscillation breaking
            osc_prop = cls._probe_attractor_oscillation(
                engine,
                curr_grid,
                available_actions,
                discrete_transform_actions,
                disp_actions,
                spatial_effector_actions,
                H,
                W,
                bg,
            )
            if osc_prop is not None:
                return osc_prop

            candidate_entities_eval: list[tuple[SpatialEntity, Any]] = []
            for e in entities:
                if e.id in engine.probed_entity_ids:
                    continue
                if e.feature_id in engine.hazard_tracker.known_lethal_features:
                    continue
                if not (
                    e.feature_id not in engine.hazard_tracker.mobile_entities
                    or e.feature_id in engine.symbolic_theory.cargo_features
                    or e.feature_id in engine.learned_cargo_features
                ):
                    continue
                if e.feature_id == engine.working_memory.shroud_feature:
                    continue
                if (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                ) <= 1:
                    continue
                ev_ent = engine.symbolic_theory.evaluate_transition(e.feature_id, locus=e.grid_pos)
                if not (ev_ent.traversable or ev_ent.confidence == 0.0) or ev_ent.risk_cost >= 40.0:
                    continue
                candidate_entities_eval.append((e, ev_ent))

            new_probe_target: tuple[int, int] | None = None
            new_probe_id: str | None = None
            entity_visits_to_inc: list[str] = []
            doorway_reg: tuple[tuple[int, int], int] | None = None

            if engine.active_probe_target is None and candidate_entities_eval:
                scored: list[tuple[SpatialEntity, float]] = []
                for e, ev_ent in candidate_entities_eval:
                    dist = abs(e.grid_pos[0] - engine.avatar_pos[0]) + abs(
                        e.grid_pos[1] - engine.avatar_pos[1]
                    )
                    visits = engine.entity_visit_counts.get(e.id, 0)
                    # Friston Expected Free Energy (EFE): Epistemic Information Gain from TransitionKB
                    f_id = e.feature_id
                    affordance_uncertainty = 1.0 + 1.6 * float(ev_ent.info_gain)
                    if not ev_ent.mutates_state and not any(
                        m.trigger_feature == f_id
                        for m in get_engine_state_attr(engine, "state_mutations", [])
                    ):
                        affordance_uncertainty += 1.0
                    if f_id in engine.symbolic_theory.candidate_goal_features:
                        affordance_uncertainty += 1.5

                    info_val = (12.0 * affordance_uncertainty) / (visits + 1.0)
                    cost = dist + 1.0
                    scored.append((e, info_val / cost))
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen = scored[0][0]
                new_probe_target = chosen.grid_pos
                new_probe_id = chosen.id
                entity_visits_to_inc.append(chosen.id)

            # Room Topology Doorway Macro-Subgoal Selection
            if (
                engine.active_probe_target is None
                and new_probe_target is None
                and engine.topology_doors
                and engine.current_room_id is not None
            ):
                connecting_doors: list[tuple[RoomDoor, int, float]] = []
                for door in engine.topology_doors:
                    if engine.current_room_id in door.connects_rooms:
                        nbr_rooms = [r for r in door.connects_rooms if r != engine.current_room_id]
                        if not nbr_rooms:
                            continue
                        nbr_room = nbr_rooms[0]
                        nbr_cells = engine.topology_rooms.get(nbr_room, [])
                        unvisited_count = sum(
                            1
                            for cell in nbr_cells
                            if engine.position_visit_counts.get(cell, 0) == 0
                        )
                        dist_to_door = abs(door.door_coord[0] - engine.avatar_pos[0]) + abs(
                            door.door_coord[1] - engine.avatar_pos[1]
                        )
                        door_score = float(unvisited_count) / (dist_to_door + 1.0)
                        connecting_doors.append((door, nbr_room, door_score))

                if connecting_doors:
                    connecting_doors.sort(key=lambda item: item[2], reverse=True)
                    best_door, target_room, score = connecting_doors[0]
                    if score > 0.1:
                        new_probe_target = best_door.door_coord
                        new_probe_id = f"door_{best_door.door_coord[0]}_{best_door.door_coord[1]}"
                        doorway_reg = (best_door.door_coord, target_room)

            target_pos = new_probe_target or engine.active_probe_target or (H // 2, W // 2)

            if not disp_actions and displacement_actions:
                uncal = [act for act in displacement_actions if act not in engine.action_dynamics]
                if uncal:
                    return EpistemicProbeProposal(
                        action=uncal[0],
                        clear_active_probe=clear_active_probe,
                        new_probe_target=new_probe_target,
                        new_probe_id=new_probe_id,
                        probed_entity_ids_to_add=tuple(probed_ids_to_add),
                        entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                        topological_doorway=doorway_reg,
                        exploration_mode="uncalibrated_displacement",
                    )

            # Looming threat evasion
            evasion_act = cls._probe_looming_threat_evasion(
                engine, curr_grid, disp_actions, bg, bg_is_impassable_void
            )
            if evasion_act is not None:
                return EpistemicProbeProposal(
                    action=evasion_act,
                    clear_active_probe=clear_active_probe,
                    new_probe_target=new_probe_target,
                    new_probe_id=new_probe_id,
                    probed_entity_ids_to_add=tuple(probed_ids_to_add),
                    entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                    topological_doorway=doorway_reg,
                    exploration_mode="looming_threat_evasion",
                )

            # When exploring without an active probe target, when greedy progress toward a
            # distant target is blocked by walls, or when revisiting cells, prioritize Frontier
            # Backtracking (BFS to nearest unvisited cell) over naive Manhattan steps.
            visited_cells = EpistemicCuriosityExplorer._collect_visited_cells(engine)
            has_explicit_probe_target = bool(new_probe_target or engine.active_probe_target)
            tr, tc = target_pos
            ar, ac = engine.avatar_pos
            greedy_unvisited_progress = any(
                cls._is_ray_passable(
                    engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
                )
                and (engine.avatar_pos, act) not in engine.failed_transitions
                and not engine.working_memory.habenular_ior.is_action_inhibited(
                    engine.avatar_pos, act
                )
                and (ar + dr_cal, ac + dc_cal) not in visited_cells
                and (abs(ar + dr_cal - tr) + abs(ac + dc_cal - tc)) < (abs(ar - tr) + abs(ac - tc))
                for act, (dr_cal, dc_cal) in disp_actions
            )
            esc_res = cls._probe_frontier_escape_with_target(
                engine, curr_grid, disp_actions, bg, bg_is_impassable_void
            )
            if esc_res is not None and (
                not has_explicit_probe_target
                or not greedy_unvisited_progress
                or engine.recent_positions.count(engine.avatar_pos) >= 2
                or engine.probe_target_steps >= 2
            ):
                esc_act, esc_target = esc_res
                commit_target = (
                    esc_target
                    if esc_target not in visited_cells
                    else (new_probe_target or esc_target)
                )
                return EpistemicProbeProposal(
                    action=esc_act,
                    clear_active_probe=clear_active_probe,
                    new_probe_target=commit_target,
                    new_probe_id=new_probe_id or f"frontier_{commit_target[0]}_{commit_target[1]}",
                    probed_entity_ids_to_add=tuple(probed_ids_to_add),
                    entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                    topological_doorway=doorway_reg,
                    exploration_mode="frontier_backtrack",
                )

            # Free Energy Active Inference Action Selection
            inf_act = cls._probe_active_inference(
                engine, curr_grid, disp_actions, target_pos, bg, bg_is_impassable_void, H, W
            )
            if inf_act is not None:
                return EpistemicProbeProposal(
                    action=inf_act,
                    clear_active_probe=clear_active_probe,
                    new_probe_target=new_probe_target,
                    new_probe_id=new_probe_id,
                    probed_entity_ids_to_add=tuple(probed_ids_to_add),
                    entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                    topological_doorway=doorway_reg,
                    exploration_mode="active_inference",
                )

            # Frontier Backtracking fallback if active inference was blocked
            if esc_res is not None:
                esc_act, esc_target = esc_res
                return EpistemicProbeProposal(
                    action=esc_act,
                    clear_active_probe=clear_active_probe,
                    new_probe_target=new_probe_target or esc_target,
                    new_probe_id=new_probe_id,
                    probed_entity_ids_to_add=tuple(probed_ids_to_add),
                    entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                    topological_doorway=doorway_reg,
                    exploration_mode="frontier_backtrack",
                )

            if spatial_effector_actions:
                eff_candidates = [
                    a for a in spatial_effector_actions if engine.inhibited_actions.get(a, 0) == 0
                ]
                if eff_candidates:
                    chosen_eff = eff_candidates[0]
                    eff_data = engine.ground_effector_action(curr_grid, chosen_eff)
                    if eff_data is not None:
                        return EpistemicProbeProposal(
                            action=chosen_eff,
                            action_data=eff_data,
                            clear_active_probe=clear_active_probe,
                            new_probe_target=new_probe_target,
                            new_probe_id=new_probe_id,
                            probed_entity_ids_to_add=tuple(probed_ids_to_add),
                            entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                            topological_doorway=doorway_reg,
                            exploration_mode="spatial_effector_fallback",
                        )
            if discrete_transform_actions:
                uninhibited_discrete = [
                    a
                    for a in discrete_transform_actions
                    if engine.inhibited_actions.get(a, 0) == 0
                    and (a, engine.avatar_pos) not in engine.tested_action_positions
                ]
                if uninhibited_discrete:
                    disc_act = uninhibited_discrete[engine.step_counter % len(uninhibited_discrete)]
                    return EpistemicProbeProposal(
                        action=disc_act,
                        clear_active_probe=clear_active_probe,
                        new_probe_target=new_probe_target,
                        new_probe_id=new_probe_id,
                        probed_entity_ids_to_add=tuple(probed_ids_to_add),
                        entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                        tested_action_position=(disc_act, engine.avatar_pos),
                        topological_doorway=doorway_reg,
                        exploration_mode="discrete_transform_round_robin",
                    )

            uncal = [a for a in available_actions if not engine.is_displacement_action(a)]
            untested_here = [
                a for a in uncal if (a, engine.avatar_pos) not in engine.tested_action_positions
            ]
            if untested_here:
                for act in untested_here:
                    act_data = None
                    if engine.is_spatial_effector(act):
                        act_data = engine.ground_effector_action(curr_grid, act)
                        if act_data is None:
                            continue
                    return EpistemicProbeProposal(
                        action=act,
                        action_data=act_data,
                        clear_active_probe=clear_active_probe,
                        new_probe_target=new_probe_target,
                        new_probe_id=new_probe_id,
                        probed_entity_ids_to_add=tuple(probed_ids_to_add),
                        entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                        tested_action_position=(act, engine.avatar_pos),
                        topological_doorway=doorway_reg,
                        exploration_mode="untested_non_displacement",
                    )

            uninhibited_acts = [
                a
                for a in available_actions
                if engine.inhibited_actions.get(a, 0) == 0
                and not engine.working_memory.habenular_ior.is_action_inhibited(
                    engine.avatar_pos, a
                )
            ]
            cand_pool = (
                uninhibited_acts
                if uninhibited_acts
                else [a for a in available_actions if engine.inhibited_actions.get(a, 0) == 0]
                or list(available_actions)
            )
            # Metacognitive Entropy & Habenular Regret Minimization:
            # Prioritize least repulsed actions, breaking ties with least recently executed actions,
            # while heavily penalizing cyclic revisits of recently visited coordinates.
            recent_rev = list(reversed(engine.recent_actions))

            def _selection_key(a: int) -> tuple[float, int, int]:
                rep = engine.working_memory.habenular_ior.get_repulsion_penalty(
                    engine.avatar_pos, a
                )
                cycle_pen = 0.0
                visit_pen = 0
                momentum_bonus = 0.0

                if engine.avatar_pos is not None and a in engine.action_deltas:
                    dr, dc = engine.action_deltas[a]
                    target_pos = (engine.avatar_pos[0] + dr, engine.avatar_pos[1] + dc)

                    rec_positions = list(engine.recent_positions)
                    if rec_positions:
                        if len(rec_positions) >= 2 and target_pos == rec_positions[-2]:
                            cycle_pen += 50.0  # Immediate 2-step ping-pong reversal
                        elif len(rec_positions) >= 4 and target_pos in rec_positions[-4:]:
                            cycle_pen += 30.0  # 3 or 4-step limit cycle
                        elif target_pos in rec_positions:
                            cycle_pen += 10.0

                    if target_pos in engine.learned_barriers:
                        cycle_pen += 100.0

                    visit_pen = engine.position_visit_counts.get(target_pos, 0)

                    # Momentum bonus: maintain current forward direction if unblocked
                    if a == engine.last_action and engine.consecutive_stuck_steps == 0:
                        momentum_bonus = -1.0

                try:
                    rec = -recent_rev.index(a)
                except ValueError:
                    rec = -999
                total_rep = rep + cycle_pen + momentum_bonus
                return (total_rep, visit_pen, rec)

            cand_pool.sort(key=_selection_key)
            chosen_fallback = cand_pool[0]
            return EpistemicProbeProposal(
                action=chosen_fallback,
                clear_active_probe=clear_active_probe,
                new_probe_target=new_probe_target,
                new_probe_id=new_probe_id,
                probed_entity_ids_to_add=tuple(probed_ids_to_add),
                entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                topological_doorway=doorway_reg,
                exploration_mode="entropy_lru_fallback",
            )

        # 4. Fallback probe
        eff_fallbacks = [
            a for a in spatial_effector_actions if engine.inhibited_actions.get(a, 0) == 0
        ]
        if eff_fallbacks:
            chosen_eff = eff_fallbacks[0]
            eff_data = engine.ground_effector_action(curr_grid, chosen_eff)
            if eff_data is not None:
                return EpistemicProbeProposal(
                    action=chosen_eff,
                    action_data=eff_data,
                    exploration_mode="effector_global_fallback",
                )
        uninhibited_global = [
            a
            for a in available_actions
            if engine.inhibited_actions.get(a, 0) == 0
            and not engine.working_memory.habenular_ior.is_action_inhibited(engine.avatar_pos, a)
        ]
        if not uninhibited_global:
            uninhibited_global = [
                a for a in available_actions if engine.inhibited_actions.get(a, 0) == 0
            ]
        recent_rev = list(reversed(engine.recent_actions))

        def _global_key(a: int) -> tuple[float, int, int]:
            rep = engine.working_memory.habenular_ior.get_repulsion_penalty(engine.avatar_pos, a)
            cycle_pen = 0.0
            visit_pen = 0
            momentum_bonus = 0.0

            if engine.avatar_pos is not None and a in engine.action_deltas:
                dr, dc = engine.action_deltas[a]
                target_pos = (engine.avatar_pos[0] + dr, engine.avatar_pos[1] + dc)

                rec_positions = list(engine.recent_positions)
                if rec_positions:
                    if len(rec_positions) >= 2 and target_pos == rec_positions[-2]:
                        cycle_pen += 50.0
                    elif len(rec_positions) >= 4 and target_pos in rec_positions[-4:]:
                        cycle_pen += 30.0
                    elif target_pos in rec_positions:
                        cycle_pen += 10.0

                if target_pos in engine.learned_barriers:
                    cycle_pen += 100.0

                visit_pen = engine.position_visit_counts.get(target_pos, 0)
                if a == engine.last_action and engine.consecutive_stuck_steps == 0:
                    momentum_bonus = -1.0

            freq = list(engine.recent_actions).count(a)
            try:
                rec = -recent_rev.index(a)
            except ValueError:
                rec = -999
            return (rep + cycle_pen + momentum_bonus, freq + visit_pen, rec)

        if uninhibited_global:
            uninhibited_global.sort(key=_global_key)
            fallback_global = uninhibited_global[0]
        else:
            cand = list(available_actions)
            cand.sort(key=_global_key)
            fallback_global = cand[0] if cand else available_actions[0]

        fallback_data = None
        if engine.is_spatial_effector(fallback_global):
            fallback_data = engine.ground_effector_action(curr_grid, fallback_global)
            if fallback_data is None:
                non_effs = [a for a in available_actions if not engine.is_spatial_effector(a)]
                if non_effs:
                    fallback_global = non_effs[0]

        return EpistemicProbeProposal(
            action=fallback_global,
            action_data=fallback_data,
            exploration_mode="global_fallback",
        )

    @classmethod
    def proposal_to_stage_result(
        cls,
        proposal: EpistemicProbeProposal,
        author: str = "cortex_motor",
    ) -> StageResult:
        """Convert EpistemicProbeProposal into a pure StageResult proposing state transactions (WP1)."""
        ops: list[TransactionOperation] = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_data={
                    "id": f"epistemic_probe:{proposal.action}",
                    "type": "ObservationNode",
                    "properties": {
                        "action": proposal.action,
                        "action_data": proposal.action_data,
                        "new_probe_target": list(proposal.new_probe_target)
                        if proposal.new_probe_target
                        else None,
                        "new_probe_id": proposal.new_probe_id,
                        "tested_action": proposal.tested_action,
                        "tested_action_position": (
                            [
                                proposal.tested_action_position[0],
                                list(proposal.tested_action_position[1]),
                            ]
                            if proposal.tested_action_position
                            and proposal.tested_action_position[1]
                            else None
                        ),
                    },
                },
            )
        ]
        return StageResult(
            author=author,
            operations=ops,
            signals={"probe_proposal": proposal},
            proposed_action=proposal.action,
        )

    @classmethod
    def apply_proposal_to_engine(
        cls,
        view: Any,
        delta: Any,
        proposal: EpistemicProbeProposal,
    ) -> None:
        """Atomically commit EpistemicProbeProposal mutations via StateDelta (WP1)."""
        delta.set("context_epistemic_probes", view.context_epistemic_probes + 1)
        delta.set("total_epistemic_probes", view.total_epistemic_probes + 1)

        if view.avatar_pos is not None:
            delta.mutate("recent_positions", view.recent_positions).append(view.avatar_pos)
            pvc = delta.mutate("position_visit_counts", view.position_visit_counts)
            pvc[view.avatar_pos] = pvc.get(view.avatar_pos, 0) + 1
            if view.active_probe_target is not None:
                delta.set("probe_target_steps", view.probe_target_steps + 1)

        if proposal.clear_active_probe:
            delta.set("active_probe_target", None)
            delta.set("active_probe_id", None)
            delta.set("probe_target_steps", 0)

        if proposal.new_probe_target is not None:
            delta.set("active_probe_target", proposal.new_probe_target)
            delta.set("active_probe_id", proposal.new_probe_id)
            delta.set("probe_target_steps", 0)

        for eid in proposal.probed_entity_ids_to_add:
            delta.mutate("probed_entity_ids", view.probed_entity_ids).add(eid)

        for eid in proposal.entity_visit_counts_to_inc:
            evc = delta.mutate("entity_visit_counts", view.entity_visit_counts)
            evc[eid] = evc.get(eid, 0) + 1

        if proposal.tested_action_position is not None:
            delta.mutate("tested_action_positions", view.tested_action_positions).add(
                proposal.tested_action_position
            )

        if proposal.tested_action is not None:
            delta.mutate("tested_actions", view.tested_actions).add(proposal.tested_action)

        if proposal.topological_doorway is not None:
            door_coord, target_room = proposal.topological_doorway
            view.working_memory.register_topological_doorway(door_coord, target_room)

    @classmethod
    def plan_epistemic_probe(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        """Compute candidate proposal and commit it atomically to engine state."""
        proposal = cls.compute_epistemic_probe(
            engine=engine,
            curr_grid=curr_grid,
            available_actions=available_actions,
        )
        from hbllm.hcir.world.engine_state import ReadOnlyEngineView, StateDelta

        delta = StateDelta()
        cls.apply_proposal_to_engine(
            view=ReadOnlyEngineView(engine), delta=delta, proposal=proposal
        )
        engine.apply_delta(delta, author="cortex_motor")
        return proposal.action, proposal.action_data
