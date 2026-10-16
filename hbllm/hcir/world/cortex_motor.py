from __future__ import annotations

"""Basal Ganglia & Anterior Cingulate Cortex (ACC) Motor Arbitration Faculty.

Biologically modeled on mammalian striatal action selection, dopamine RPE gating,
and ACC conflict monitoring:
1. Basal Ganglia Active Inference:
   - Evaluates candidate motor commands by trading off pragmatic utility (goal approach)
     and epistemic value (information gain / uncertainty reduction).
2. Motor Grounding & Agency Discovery:
   - Motor babbling correlates intentional action efference copies with sensory visual feedback.
   - Distinguishes self-avatar displacements from external environmental dynamics.
3. ACC Conflict Monitor & Limit-Cycle Breaker:
   - Detects motor deadlocks (e.g. 2-cycle or 3-cycle oscillations) and forces orthogonal exploration.
4. Epistemic Curiosity Explorer:
   - Systematically probes uncalibrated actions, unvisited spatial frontiers, and affordance panels.
   - Emits pure EpistemicProbeProposals to maintain HCIR Reasoning and State Ownership Invariants.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.active_inference import ActionNode
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_causal import ActionAffordance
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.inferotemporal_segmentation import InferotemporalSegmentationEngine
from hbllm.hcir.world.parietal_coordinate_transform import HeadingOrientation
from hbllm.hcir.world.spatial_containment import RoomDoor

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


@dataclass
class ACCConflictRecord:
    """Record of cognitive conflict or motor deadlock detected by the ACC."""

    step: int
    cycle_period: int
    cyclic_actions: list[Any]
    suppressed_actions: list[Any] = field(default_factory=list)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export ACC conflict record as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"acc_conflict:{self.step}",
            operator="==",
            value=self.cycle_period,
            relation=f"cyclic:{len(self.cyclic_actions)}",
            target_entity=f"suppressed:{len(self.suppressed_actions)}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording ACC conflict."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"acc_conflict_{self.step}",
                node_data={
                    "step": self.step,
                    "cycle_period": self.cycle_period,
                    "cyclic_actions": [str(a) for a in self.cyclic_actions],
                    "suppressed_actions": [str(a) for a in self.suppressed_actions],
                },
            )
        ]


class ACCConflictMonitor:
    """Anterior Cingulate Cortex conflict detector and limit-cycle oscillation breaker."""

    def __init__(self, history_window: int = 16, oscillation_threshold: int = 3) -> None:
        self.history_window = history_window
        self.oscillation_threshold = oscillation_threshold
        self.action_history: deque[Any] = deque(maxlen=history_window)
        self.position_history: deque[tuple[int, int]] = deque(maxlen=history_window)
        self.conflict_records: list[ACCConflictRecord] = []

    def record_step(self, action: Any, avatar_pos: tuple[int, int] | None) -> None:
        """Record an executed action and resulting position."""
        self.action_history.append(action)
        if avatar_pos is not None:
            self.position_history.append(avatar_pos)

    @staticmethod
    def get_orthogonal_actions(
        available_actions: Sequence[Any],
        cyclic_actions: Sequence[Any],
        action_dynamics: dict[Any, Any],
    ) -> list[Any]:
        """Compute actions orthogonal or complementary to the oscillating action axis.

        If cycling on vertical displacement (e.g. up/down, dr!=0, dc==0),
        prioritizes horizontal displacements (dr==0, dc!=0).
        If cycling on horizontal displacement (dr==0, dc!=0),
        prioritizes vertical displacements (dr!=0, dc==0).
        Falls back to any available non-cyclic actions.
        """
        cyclic_set = set(cyclic_actions)
        non_cyclic = [a for a in available_actions if a not in cyclic_set]
        if not non_cyclic:
            return list(available_actions)

        # Compute aggregate direction vector of cyclic actions
        total_abs_dr = 0
        total_abs_dc = 0
        for a in cyclic_actions:
            dyn = action_dynamics.get(a)
            if dyn is not None:
                dr, dc = dyn.get_displacement()
                total_abs_dr += abs(dr)
                total_abs_dc += abs(dc)

        orthogonal_candidates: list[Any] = []
        other_displacement_candidates: list[Any] = []
        other_candidates: list[Any] = []

        for a in non_cyclic:
            dyn = action_dynamics.get(a)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                if total_abs_dr > total_abs_dc:
                    # Cyclic axis is primarily vertical -> orthogonal is horizontal
                    if abs(dc) > abs(dr):
                        orthogonal_candidates.append(a)
                    else:
                        other_displacement_candidates.append(a)
                elif total_abs_dc > total_abs_dr:
                    # Cyclic axis is primarily horizontal -> orthogonal is vertical
                    if abs(dr) > abs(dc):
                        orthogonal_candidates.append(a)
                    else:
                        other_displacement_candidates.append(a)
                else:
                    # Balanced or diagonal -> any displacement candidate
                    other_displacement_candidates.append(a)
            else:
                other_candidates.append(a)

        return orthogonal_candidates + other_displacement_candidates + other_candidates

    def detect_oscillation(self) -> tuple[bool, list[Any]]:
        """Detect 2-cycle or 3-cycle limit cycles in recent actions or positions.

        Returns:
            (is_oscillating, cyclic_actions_to_suppress)
        """
        acts = list(self.action_history)
        cyclic_acts: list[Any] = []
        if len(acts) >= 4:
            # Check 2-cycle: e.g. [A, B, A, B]
            if acts[-1] == acts[-3] and acts[-2] == acts[-4] and acts[-1] != acts[-2]:
                cyclic_acts = [acts[-1], acts[-2]]

        if not cyclic_acts and len(acts) >= 6:
            # Check 3-cycle: e.g. [A, B, C, A, B, C]
            if acts[-1] == acts[-4] and acts[-2] == acts[-5] and acts[-3] == acts[-6]:
                cyclic_acts = [acts[-1], acts[-2], acts[-3]]

        if cyclic_acts:
            pos_list = list(self.position_history)
            cycle_len = len(cyclic_acts)
            if len(pos_list) >= 2 * cycle_len:
                # In embodied/spatial navigation, an action repetition is only an oscillation
                # if the organism returns to the same location (zero net displacement).
                # If there is net spatial progress, it is corridor/staircase traversal, not a limit cycle.
                if pos_list[-1] != pos_list[-1 - cycle_len]:
                    cyclic_acts = []

        # Check position bounce: avatar stuck between two coordinates
        if not cyclic_acts:
            pos_list = list(self.position_history)
            if len(pos_list) >= 4:
                if (
                    pos_list[-1] == pos_list[-3]
                    and pos_list[-2] == pos_list[-4]
                    and pos_list[-1] != pos_list[-2]
                ):
                    cyclic_acts = list(acts[-2:])

        if cyclic_acts:
            rec = ACCConflictRecord(
                step=len(self.action_history),
                cycle_period=len(cyclic_acts),
                cyclic_actions=cyclic_acts,
                suppressed_actions=cyclic_acts,
            )
            self.conflict_records.append(rec)
            return True, cyclic_acts

        return False, []

    def reset(self) -> None:
        """Clear conflict history."""
        self.action_history.clear()
        self.position_history.clear()
        self.conflict_records.clear()

    def to_causal_predicate(self) -> CausalPredicate:
        """Export ACC conflict monitor state as a declarative causal predicate."""
        return CausalPredicate(
            variable="acc_conflict_monitor",
            operator="==",
            value=len(self.conflict_records),
            relation=f"window:{self.history_window}",
            target_entity="anterior_cingulate",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all ACC conflict records as declarative causal predicates."""
        preds = [self.to_causal_predicate()]
        for rec in self.conflict_records:
            preds.append(rec.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting ACC conflict monitor state."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="acc_conflict_summary",
                node_data={
                    "conflict_count": len(self.conflict_records),
                    "history_window": self.history_window,
                    "oscillation_threshold": self.oscillation_threshold,
                },
            )
        ]
        for rec in self.conflict_records:
            ops.extend(rec.to_transaction_operations())
        return ops


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
        for k in range(1, dist + 1):
            kr = r0 + step_r * k
            kc = c0 + step_c * k
            if not (0 <= kr < H and 0 <= kc < W):
                return False
            feat = int(curr_grid[kr, kc])
            if (kr, kc) in engine.learned_barriers or engine.symbolic_theory.is_barrier(feat):
                if not engine.working_memory.body_schema.is_barrier_permeable(feat):
                    return False
            if (
                bg_is_impassable_void
                and feat == bg
                and (kr, kc) not in engine.learned_goal_positions
            ):
                return False
            if feat in engine.hazard_tracker.known_lethal_features:
                return False
        return True

    @staticmethod
    def _probe_uncalibrated_action(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> EpistemicProbeProposal | None:
        """Uncalibrated actions take absolute priority for motor grounding."""
        undo_acts = set(getattr(engine, "undo_actions", set()))
        if hasattr(engine, "working_memory") and hasattr(engine.working_memory, "undo_actions"):
            undo_acts.update(engine.working_memory.undo_actions)
        if hasattr(engine, "symbolic_theory") and hasattr(engine.symbolic_theory, "undo_actions"):
            undo_acts.update(engine.symbolic_theory.undo_actions)
        if 7 in available_actions and len(available_actions) > 1:
            undo_acts.add(7)
        uncalibrated = [
            a
            for a in available_actions
            if not engine.is_action_calibrated(a)
            and engine.inhibited_actions.get(a, 0) == 0
            and a not in undo_acts
        ]
        untested_at_pos = [
            a for a in uncalibrated if (a, engine.avatar_pos) not in engine.tested_action_positions
        ]
        if untested_at_pos:
            spatial_effs = [a for a in untested_at_pos if engine.is_spatial_effector(a)]
            chosen = spatial_effs[0] if spatial_effs else untested_at_pos[0]
            action_data = None
            if engine.is_spatial_effector(chosen):
                action_data = engine.ground_effector_action(curr_grid, chosen)
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
    ) -> EpistemicProbeProposal | None:
        """Motor babbling for self-identification and agency discovery."""
        if displacement_actions and not engine.avatar_features and engine.avatar_feature is None:
            babble_idx = engine.level_epistemic_probes % len(displacement_actions)
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
        is_stuck_or_looping = engine.consecutive_stuck_steps >= 2 or (
            engine.avatar_pos is not None and engine.recent_positions.count(engine.avatar_pos) >= 2
        )
        has_effective_target = any(
            p not in engine.quiescent_click_targets
            and engine.entity_visit_counts.get(f"click_{p[0]}_{p[1]}", 0) < 3
            for p in engine.effective_click_targets
        )
        interleaved_reach = engine.level_epistemic_probes % 4 == 0

        should_probe = (
            not displacement_actions
            or (engine.avatar_pos is None and engine.level_epistemic_probes > 4)
            or is_stuck_or_looping
            or has_effective_target
            or interleaved_reach
            or has_optical_targets
        )
        if should_probe:
            chosen_eff = spatial_effector_actions[0]
            action_data = engine.ground_effector_action(curr_grid, chosen_eff)
            if action_data is not None:
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
            act = untested_here[0]
            return EpistemicProbeProposal(
                action=act,
                clear_active_probe=True,
                tested_action_position=(act, engine.avatar_pos),
                exploration_mode="attractor_loop_untested",
            )

        if discrete_transform_actions and (is_stuck or is_oscillating):
            disc_act = discrete_transform_actions[
                engine.level_epistemic_probes % len(discrete_transform_actions)
            ]
            return EpistemicProbeProposal(
                action=disc_act,
                clear_active_probe=True,
                exploration_mode="attractor_loop_discrete",
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
            if (
                (nr, nc) in engine.hazard_tracker.static_lethal_positions
                or (nr, nc) in engine.learned_barriers
                or engine.symbolic_theory.is_barrier(int(curr_grid[nr, nc]))
            ):
                continue
            _osc_feat = int(curr_grid[nr, nc])
            if _osc_feat in engine.hazard_tracker.known_lethal_features:
                continue
            uncertainty_pen = engine._epistemic_uncertainty_cost(_osc_feat, bg)
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

        if spatial_effector_actions and min_visits == float("inf"):
            chosen_eff = spatial_effector_actions[0]
            return EpistemicProbeProposal(
                action=chosen_eff,
                action_data=engine.ground_effector_action(curr_grid, chosen_eff),
                clear_active_probe=True,
                exploration_mode="attractor_loop_effector",
            )

        undo_acts = set(getattr(engine, "undo_actions", set()))
        if hasattr(engine, "working_memory") and hasattr(engine.working_memory, "undo_actions"):
            undo_acts.update(engine.working_memory.undo_actions)
        if hasattr(engine, "symbolic_theory") and hasattr(engine.symbolic_theory, "undo_actions"):
            undo_acts.update(engine.symbolic_theory.undo_actions)
        if 7 in available_actions and len(available_actions) > 1:
            undo_acts.add(7)
        uninhibited_available = [
            a
            for a in available_actions
            if engine.inhibited_actions.get(a, 0) == 0 and a not in undo_acts
        ]
        non_undo_available = [a for a in available_actions if a not in undo_acts]
        fallback_action = (
            uninhibited_available[0]
            if uninhibited_available
            else (non_undo_available[0] if non_undo_available else available_actions[0])
        )
        return EpistemicProbeProposal(
            action=fallback_action,
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
        is_looming_threat = any(
            engine.collision_cones.is_collision_hazard(ar, ac, t_future) for t_future in (1, 2)
        )
        if not is_looming_threat:
            return None

        for act, (dr_cal, dc_cal) in disp_actions:
            nr, nc = ar + dr_cal, ac + dc_cal
            if not EpistemicCuriosityExplorer._is_ray_passable(
                engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
            ):
                continue
            if any(engine.collision_cones.is_collision_hazard(nr, nc, t_f) for t_f in (1, 2)):
                continue
            if (engine.avatar_pos, act) in engine.failed_transitions:
                continue
            return act

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

            # Premotor Collision Cones
            if any(engine.collision_cones.is_collision_hazard(nr, nc, t_f) for t_f in (1, 2)):
                continue

            cell_feat = int(curr_grid[nr, nc])

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

            dist_to_target = abs(tr - nr) + abs(tc - nc)
            recency_penalty = 10.0 if (nr, nc) in list(engine.recent_positions)[-4:] else 0.0
            repulsion_penalty = engine.working_memory.habenular_ior.get_repulsion_penalty(
                engine.avatar_pos, act
            )
            estimated_cost = float(dist_to_target) + recency_penalty + repulsion_penalty

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
            eval_results = engine.active_inference.evaluate_candidates(
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
        for act, (dr_cal, dc_cal) in sorted_disp:
            if (engine.avatar_pos, act) in engine.failed_transitions:
                continue
            if engine.working_memory.habenular_ior.is_action_inhibited(engine.avatar_pos, act):
                continue
            nr = ar + dr_cal
            nc = ac + dc_cal
            if not EpistemicCuriosityExplorer._is_ray_passable(
                engine, curr_grid, ar, ac, dr_cal, dc_cal, bg, bg_is_impassable_void
            ):
                continue
            if any(engine.collision_cones.is_collision_hazard(nr, nc, t_f) for t_f in (1, 2)):
                continue
            return act

        return None

    @staticmethod
    def _probe_frontier_escape(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        disp_actions: list[tuple[int, tuple[int, int]]],
        bg: int,
        bg_is_impassable_void: bool,
    ) -> int | None:
        """Frontier Backtracking: BFS shortest path to nearest unexhausted walkable frontier cell."""
        if not disp_actions or engine.avatar_pos is None:
            return None

        bfs_q: deque[tuple[tuple[int, int], list[Any]]] = deque([(engine.avatar_pos, [])])
        bfs_visited = {engine.avatar_pos}
        best_frontier_path: list[Any] | None = None
        least_visited_path: list[Any] | None = None
        least_visits = float("inf")

        while bfs_q and len(bfs_visited) < 300:
            p, path = bfs_q.popleft()
            p_visits = engine.position_visit_counts.get(p, 0)
            if p != engine.avatar_pos and p_visits < least_visits:
                least_visits = p_visits
                least_visited_path = path

            has_unvisited_neighbor = False
            for act, (dr_cal, dc_cal) in disp_actions:
                nbr = (p[0] + dr_cal, p[1] + dc_cal)
                if EpistemicCuriosityExplorer._is_ray_passable(
                    engine, curr_grid, p[0], p[1], dr_cal, dc_cal, bg, bg_is_impassable_void
                ):
                    if engine.position_visit_counts.get(nbr, 0) == 0:
                        has_unvisited_neighbor = True
                        break
            if has_unvisited_neighbor and path:
                best_frontier_path = path
                break

            for act, (dr_cal, dc_cal) in disp_actions:
                nr = p[0] + dr_cal
                nc = p[1] + dc_cal
                if (nr, nc) in bfs_visited:
                    continue
                if not EpistemicCuriosityExplorer._is_ray_passable(
                    engine, curr_grid, p[0], p[1], dr_cal, dc_cal, bg, bg_is_impassable_void
                ):
                    continue
                step_idx = len(path) + 1
                if engine.collision_cones.is_collision_hazard(nr, nc, step_idx):
                    continue
                if p == engine.avatar_pos and (
                    (p, act) in engine.failed_transitions
                    or engine.working_memory.habenular_ior.is_action_inhibited(p, act)
                ):
                    continue
                bfs_visited.add((nr, nc))
                bfs_q.append(((nr, nc), path + [act]))

        chosen_escape = best_frontier_path or least_visited_path
        if chosen_escape:
            return chosen_escape[0]

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
        bg = engine.estimate_background(curr_grid)
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
        babble_prop = cls._probe_motor_babble(engine, displacement_actions)
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

            # Update room topology
            engine.update_room_topology(curr_grid)

            candidate_entities = [
                e
                for e in entities
                if e.id not in engine.probed_entity_ids
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and e.feature_id not in engine.hazard_tracker.known_lethal_features
                and (
                    abs(e.grid_pos[0] - engine.avatar_pos[0])
                    + abs(e.grid_pos[1] - engine.avatar_pos[1])
                )
                > 1
            ]

            new_probe_target: tuple[int, int] | None = None
            new_probe_id: str | None = None
            entity_visits_to_inc: list[str] = []
            doorway_reg: tuple[tuple[int, int], int] | None = None

            if engine.active_probe_target is None and candidate_entities:
                scored: list[tuple[SpatialEntity, float]] = []
                current_room_cells = (
                    set(engine.topology_rooms.get(engine.current_room_id, []))
                    if engine.current_room_id is not None
                    else set()
                )
                for e in candidate_entities:
                    dist = abs(e.grid_pos[0] - engine.avatar_pos[0]) + abs(
                        e.grid_pos[1] - engine.avatar_pos[1]
                    )
                    visits = engine.entity_visit_counts.get(e.id, 0)
                    info_val = 10.0 / (visits + 1.0)
                    if current_room_cells and e.grid_pos in current_room_cells:
                        info_val *= 1.5
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

            # Frontier Backtracking
            esc_act = cls._probe_frontier_escape(
                engine, curr_grid, disp_actions, bg, bg_is_impassable_void
            )
            if esc_act is not None:
                return EpistemicProbeProposal(
                    action=esc_act,
                    clear_active_probe=clear_active_probe,
                    new_probe_target=new_probe_target,
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
            elif discrete_transform_actions:
                disc_act = discrete_transform_actions[
                    engine.step_counter % len(discrete_transform_actions)
                ]
                return EpistemicProbeProposal(
                    action=disc_act,
                    clear_active_probe=clear_active_probe,
                    new_probe_target=new_probe_target,
                    new_probe_id=new_probe_id,
                    probed_entity_ids_to_add=tuple(probed_ids_to_add),
                    entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                    topological_doorway=doorway_reg,
                    exploration_mode="discrete_transform_round_robin",
                )

            uncal = [a for a in available_actions if not engine.is_displacement_action(a)]
            untested_here = [
                a for a in uncal if (a, engine.avatar_pos) not in engine.tested_action_positions
            ]
            if untested_here:
                act = untested_here[0]
                return EpistemicProbeProposal(
                    action=act,
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
            # Prioritize least repulsed actions, breaking ties with least recently executed actions
            recent_rev = list(reversed(engine.recent_actions))

            def _selection_key(a: int) -> tuple[float, int]:
                rep = engine.working_memory.habenular_ior.get_repulsion_penalty(
                    engine.avatar_pos, a
                )
                try:
                    rec = -recent_rev.index(a)
                except ValueError:
                    rec = -999
                return (rep, rec)

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
            freq = list(engine.recent_actions).count(a)
            try:
                rec = -recent_rev.index(a)
            except ValueError:
                rec = -999
            return (rep, freq, rec)

        if uninhibited_global:
            uninhibited_global.sort(key=_global_key)
            fallback_global = uninhibited_global[0]
        else:
            cand = list(available_actions)
            cand.sort(key=_global_key)
            fallback_global = cand[0] if cand else available_actions[0]

        return EpistemicProbeProposal(
            action=fallback_global,
            exploration_mode="global_fallback",
        )

    @classmethod
    def apply_proposal_to_engine(
        cls,
        engine: AutonomousEpistemicEngine,
        proposal: EpistemicProbeProposal,
    ) -> None:
        """Atomically commit EpistemicProbeProposal mutations to the engine."""
        engine.level_epistemic_probes += 1
        engine.total_epistemic_probes += 1

        if engine.avatar_pos is not None:
            engine.recent_positions.append(engine.avatar_pos)
            engine.position_visit_counts[engine.avatar_pos] = (
                engine.position_visit_counts.get(engine.avatar_pos, 0) + 1
            )
            if engine.active_probe_target is not None:
                engine.probe_target_steps += 1

        if proposal.clear_active_probe:
            engine.active_probe_target = None
            engine.active_probe_id = None
            engine.probe_target_steps = 0

        if proposal.new_probe_target is not None:
            engine.active_probe_target = proposal.new_probe_target
            engine.active_probe_id = proposal.new_probe_id
            engine.probe_target_steps = 0

        for eid in proposal.probed_entity_ids_to_add:
            engine.probed_entity_ids.add(eid)

        for eid in proposal.entity_visit_counts_to_inc:
            engine.entity_visit_counts[eid] = engine.entity_visit_counts.get(eid, 0) + 1

        if proposal.tested_action_position is not None:
            engine.tested_action_positions.add(proposal.tested_action_position)

        if proposal.tested_action is not None:
            engine.tested_actions.add(proposal.tested_action)

        if proposal.topological_doorway is not None:
            door_coord, target_room = proposal.topological_doorway
            engine.working_memory.register_topological_doorway(door_coord, target_room)

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
        cls.apply_proposal_to_engine(engine=engine, proposal=proposal)
        return proposal.action, proposal.action_data


class EffectorTargetScorer(ABC):
    """Abstract interface for external or domain-specific target candidate scoring.

    Allows modular game-specific or benchmark-specific solvers (e.g. ARC-AGI-3 adapters)
    to register candidate scoring strategies without coupling the core cognitive architecture.
    """

    @abstractmethod
    def score_targets(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        entities: Sequence[SpatialEntity],
        bg: int,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        """Score candidate coordinates for allocentric effector targeting."""
        ...


class MotorCortexEffector:
    """Allocentric spatial effector grounding and action affordance arbitration."""

    _registered_target_scorers: list[EffectorTargetScorer] = []

    @classmethod
    def register_target_scorer(cls, scorer: EffectorTargetScorer) -> None:
        """Register an external or modular domain target candidate scorer."""
        if scorer not in cls._registered_target_scorers:
            cls._registered_target_scorers.append(scorer)

    @classmethod
    def unregister_target_scorer(cls, scorer: EffectorTargetScorer) -> None:
        """Unregister an external target candidate scorer."""
        if scorer in cls._registered_target_scorers:
            cls._registered_target_scorers.remove(scorer)

    @classmethod
    def clear_target_scorers(cls) -> None:
        """Clear all registered target scorers."""
        cls._registered_target_scorers.clear()

    @staticmethod
    def _probe_looming_threat_evasion(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        disp_actions: list[tuple[int, tuple[int, int]]],
        bg: int,
        bg_is_impassable_void: bool,
    ) -> int | None:
        """Forwarding alias to EpistemicCuriosityExplorer._probe_looming_threat_evasion."""
        return EpistemicCuriosityExplorer._probe_looming_threat_evasion(
            engine, curr_grid, disp_actions, bg, bg_is_impassable_void
        )

    @staticmethod
    def parse_action_spec(spec: Any) -> ActionAffordance:
        """Parse external action specification into an ActionAffordance descriptor."""
        if isinstance(spec, ActionAffordance):
            return spec
        if isinstance(spec, dict):
            aid = spec.get("action_id", spec.get("id", spec.get("name", 0)))
            name = str(spec.get("name", aid))
            params = spec.get("parameters") or {}
            param_keys = tuple(params.keys()) if isinstance(params, dict) else ("x", "y")
            req_spatial = bool(
                isinstance(params, dict)
                and any(
                    k in params
                    for k in (
                        "x",
                        "y",
                        "col",
                        "row",
                        "lat",
                        "lon",
                        "azimuth",
                        "elevation",
                        "distance",
                        "c",
                        "r",
                    )
                )
            )
            name_upper = name.upper()
            is_disp = any(
                k in name_upper
                for k in (
                    "MOVE",
                    "UP",
                    "DOWN",
                    "LEFT",
                    "RIGHT",
                    "NORTH",
                    "SOUTH",
                    "EAST",
                    "WEST",
                    "STEP",
                    "WALK",
                )
            )
            meta: dict[str, Any] = {}
            if "metadata" in spec and isinstance(spec["metadata"], dict):
                meta.update(spec["metadata"])
            if "valid_targets" in spec:
                meta["valid_targets"] = spec["valid_targets"]
            return ActionAffordance(
                action_id=aid,
                name=name,
                is_displacement=is_disp,
                requires_spatial_target=req_spatial,
                target_param_keys=param_keys if param_keys else ("x", "y"),
                metadata=meta,
            )
        return ActionAffordance(action_id=spec, name=str(spec))

    @staticmethod
    def infer_motor_step_size(
        available_actions: Sequence[Any], action_dynamics: dict[Any, Any]
    ) -> int:
        """Largest calibrated displacement quantum (the world's 'cell' size)."""
        step = 1
        for act in available_actions:
            dyn = action_dynamics.get(act)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                step = max(step, abs(dr), abs(dc))
        return step

    @staticmethod
    def is_peripheral_frame_cell(r: int, c: int, H: int, W: int, grid: np.ndarray) -> bool:
        """True if (r, c) is on an outer grid frame, border line, or peripheral HUD status banner."""
        # 1. Outer 1-pixel perimeter boundary checks
        if r == 0 or r == H - 1 or c == 0 or c == W - 1:
            return True

        # 2. Universal Top HUD / Status Banner Detection:
        # A top HUD banner is a multi-row band across the top (r <= 16, H >= 24)
        # separated from the gameplay grid by a distinct divider line or having a distinct
        # banner background color different from the main gameplay field.
        if r <= 16 and H >= 24:
            for div_r in range(min(16, H // 3), 2, -1):
                line = grid[div_r, :]
                line_counts: np.ndarray = np.bincount(line.astype(int), minlength=16)
                if len(line_counts) > 0 and np.max(line_counts) >= int(W * 0.85):
                    line_color = int(np.argmax(line_counts))
                    banner_region = grid[:div_r, :]
                    field_region = grid[div_r + 1 :, :]
                    bc_banner: np.ndarray = np.bincount(
                        banner_region.flatten().astype(int), minlength=16
                    )
                    bc_field: np.ndarray = np.bincount(
                        field_region.flatten().astype(int), minlength=16
                    )
                    if len(bc_banner) > 0 and len(bc_field) > 0:
                        banner_dom = int(np.argmax(bc_banner))
                        field_dom = int(np.argmax(bc_field))
                        has_distinct_bg = (banner_dom != field_dom) and (
                            (bc_banner[banner_dom] / banner_region.size) >= 0.85
                        )
                        has_divider_line = (line_color != field_dom) and (
                            line_counts[line_color] >= int(W * 0.85)
                        )
                        if has_distinct_bg or has_divider_line:
                            if r <= div_r:
                                return True
                            break

        # 3. Bottom HUD / Status Banner Detection
        if r >= H - 16 and H >= 24:
            for div_r in range(H - 4, max(H - 18, H - H // 3), -1):
                line = grid[div_r, :]
                line_counts_bot: np.ndarray = np.bincount(line.astype(int), minlength=16)
                if len(line_counts_bot) > 0 and np.max(line_counts_bot) >= int(W * 0.85):
                    line_color = int(np.argmax(line_counts_bot))
                    banner_region = grid[div_r + 1 :, :]
                    field_region = grid[:div_r, :]
                    bc_banner_bot: np.ndarray = np.bincount(
                        banner_region.flatten().astype(int), minlength=16
                    )
                    bc_field_bot: np.ndarray = np.bincount(
                        field_region.flatten().astype(int), minlength=16
                    )
                    if len(bc_banner_bot) > 0 and len(bc_field_bot) > 0:
                        banner_dom = int(np.argmax(bc_banner_bot))
                        field_dom = int(np.argmax(bc_field_bot))
                        has_distinct_bg = (banner_dom != field_dom) and (
                            (bc_banner_bot[banner_dom] / banner_region.size) >= 0.85
                        )
                        has_divider_line = (line_color != field_dom) and (
                            line_counts_bot[line_color] >= int(W * 0.85)
                        )
                        if has_distinct_bg or has_divider_line:
                            if r >= div_r:
                                return True
                            break

        return False

    @classmethod
    def ground_effector_action(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        action: Any = None,
    ) -> dict[str, Any] | None:
        """Spatially ground an allocentric effector command onto salient affordances."""
        curr_grid = engine.normalize_sensory_input(curr_grid)
        H, W = curr_grid.shape
        aff = engine.action_affordances.get(action)
        param_keys = aff.target_param_keys if aff else ("x", "y")

        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        click_candidates: list[tuple[int, int, float]] = []

        # 1. Registered domain / adapter target scorers (e.g. ARC-AGI-3 plugin scorers)
        for scorer in cls._registered_target_scorers:
            try:
                click_candidates.extend(scorer.score_targets(engine, curr_grid, entities, bg, H, W))
            except Exception as e:
                logger.debug("Registered target scorer %s failed: %s", scorer, e)

        for scorer in getattr(engine, "target_scorers", []):
            try:
                click_candidates.extend(scorer.score_targets(engine, curr_grid, entities, bg, H, W))
            except Exception as e:
                logger.debug("Engine-local target scorer %s failed: %s", scorer, e)

        # 2. Relational Structural Affordances (Optical Targets, Receptacles, Alignment Sockets)
        structural_goals = PerceptionEngine.detect_structural_goals(
            curr_grid,
            bg=bg,
            avatar_features=av_feats,
            avatar_feature=engine.avatar_feature,
        )
        for sg in structural_goals:
            pos = sg.get("position")
            if pos is not None and isinstance(pos, (tuple, list)) and len(pos) >= 2:
                sr, sc = int(pos[0]), int(pos[1])
                if not cls.is_peripheral_frame_cell(sr, sc, H, W, curr_grid):
                    vis = engine.entity_visit_counts.get(f"click_{sr}_{sc}", 0)
                    is_mp = sg.get("is_midpoint", False)
                    # Midpoints in void space without an avatar are weak speculative heuristics
                    base_sg_score = 70.0 if (is_mp and engine.avatar_pos is None) else 180.0
                    click_candidates.append((sr, sc, base_sg_score - float(vis) * 15.0))

        # 3. Modal Manipulandum Destination Affordances:
        # If an external manipulandum is currently focused, prioritize destination sockets/receptacles
        if engine.working_memory.body_schema.has_manipulandum():
            man = engine.working_memory.body_schema.active_manipulandum
            if man is not None:
                for r_feat in engine.symbolic_theory.receptacle_features:
                    cells = np.argwhere(curr_grid == r_feat)
                    for cr, cc in cells[:8]:
                        vis = engine.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                        click_candidates.append((int(cr), int(cc), 220.0 - float(vis) * 10.0))

        # 4. Confirmed Causal Actuators (Empirical Active Inference):
        # Targets that previously produced substantial state diffs have proven causal power
        for eff_r, eff_c in engine.effective_click_targets:
            if not cls.is_peripheral_frame_cell(eff_r, eff_c, H, W, curr_grid):
                vis = engine.entity_visit_counts.get(f"click_{eff_r}_{eff_c}", 0)
                click_candidates.append((eff_r, eff_c, 260.0 - float(vis) * 10.0))

        # 5. Salient Physical Entities & Perimeter Control Panels:
        # Physical interactive entities are evaluated as viable candidates alongside structural goals
        for e in entities:
            if (
                e.feature_id == bg
                or e.feature_id in av_feats
                or e.area > 120
                or e.feature_id in engine.quiescent_features
            ):
                continue
            cr, cc = e.grid_pos
            if (
                cr,
                cc,
            ) in engine.quiescent_click_targets or MotorCortexEffector.is_peripheral_frame_cell(
                cr, cc, H, W, curr_grid
            ):
                continue
            visit_count = engine.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
            saliency = 100.0 / math.log2(2 + max(1, e.area))
            affordance_bonus = 140.0 if (cr, cc) in engine.effective_click_targets else 0.0
            lateral_actuator_bonus = (
                100.0
                if ((cc <= 10 or cc >= W - 11) and (4 <= cr <= H - 5) and 6 <= e.area <= 64)
                else 0.0
            )
            feat_bias = 40.0 if e.feature_id in engine.effective_features else 0.0
            telic_bonus = (
                150.0
                if (
                    engine.symbolic_theory.telic_goal_type == "deplete"
                    and e.feature_id == engine.symbolic_theory.telic_target_feature
                )
                else 0.0
            )
            ior_mult = 10.0 if (cr, cc) in engine.effective_click_targets else 30.0
            ior_penalty = float(visit_count) * ior_mult
            haben_pen = 0.0
            if hasattr(engine, "working_memory") and hasattr(
                engine.working_memory, "habenular_ior"
            ):
                if engine.working_memory.habenular_ior.is_action_inhibited(
                    engine.avatar_pos, action, action_data={"x": cc, "y": cr}
                ):
                    continue
                haben_pen = engine.working_memory.habenular_ior.get_repulsion_penalty(
                    engine.avatar_pos, action, action_data={"x": cc, "y": cr}
                )
            cand_score = (
                saliency
                + affordance_bonus
                + lateral_actuator_bonus
                + feat_bias
                + telic_bonus
                - ior_penalty
                - haben_pen
            )
            click_candidates.append((cr, cc, cand_score))

        # 7. Inferotemporal Affordance Centroid Segmentation
        if not click_candidates:
            it_anchors = InferotemporalSegmentationEngine.extract_affordance_anchors(
                grid=curr_grid,
                background_feature=bg,
                avatar_features=av_feats,
                quiescent_coords=engine.quiescent_click_targets,
                effective_coords=engine.effective_click_targets,
                quiescent_features=engine.quiescent_features,
                effective_features=engine.effective_features,
                visit_counts=engine.entity_visit_counts,
            )
            for ar, ac, score in it_anchors:
                if (
                    (ar, ac) in engine.quiescent_click_targets
                    or (ar, ac) == engine.avatar_pos
                    or int(curr_grid[ar, ac]) in av_feats
                    or int(curr_grid[ar, ac]) in engine.quiescent_features
                    or cls.is_peripheral_frame_cell(ar, ac, H, W, curr_grid)
                ):
                    continue
                click_candidates.append((ar, ac, score))

        # 8. Saccadic Attention Fixations
        if not click_candidates:
            fixations = engine.saccadic_attention.extract_fixations(
                grid=curr_grid,
                prev_grid=engine.prev_grid,
                background_feature=bg,
                top_k=24,
            )
            for f in fixations:
                cr, cc = f.r, f.c
                feat = int(curr_grid[cr, cc])
                if (
                    (cr, cc) in engine.quiescent_click_targets
                    or feat == bg
                    or feat in av_feats
                    or feat in engine.quiescent_features
                    or (cr, cc) == engine.avatar_pos
                    or cls.is_peripheral_frame_cell(cr, cc, H, W, curr_grid)
                ):
                    continue
                visit_count = engine.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                affordance_bonus = 60.0 if (cr, cc) in engine.effective_click_targets else 0.0
                feat_bias = 40.0 if feat in engine.effective_features else 0.0
                cand_score = (
                    f.salience
                    + affordance_bonus
                    + feat_bias
                    - float(visit_count) * (0.1 if affordance_bonus > 0 else 0.35)
                )
                click_candidates.append((cr, cc, cand_score))

        # 9. Fallback: unvisited non-background cells
        if not click_candidates:
            step_sz = (
                getattr(getattr(engine, "spatial_tracker", None), "inferred_step_size", None)
                or cls.infer_motor_step_size(
                    list(engine.action_dynamics.keys()), engine.action_dynamics
                )
                or 1
            )
            seen_cand_cells: set[tuple[int, int]] = set()
            non_bg = np.argwhere(curr_grid != bg)
            for r, c in non_bg:
                cr, cc = int(r), int(c)
                if step_sz > 1:
                    # Quantize to the lattice cell centroid to hit multi-pixel blocks accurately
                    cr = (cr // step_sz) * step_sz + step_sz // 2
                    cc = (cc // step_sz) * step_sz + step_sz // 2
                    cr = min(max(0, cr), H - 1)
                    cc = min(max(0, cc), W - 1)
                if (cr, cc) in seen_cand_cells:
                    continue
                seen_cand_cells.add((cr, cc))
                feat = int(curr_grid[cr, cc])
                if (
                    (cr, cc) in engine.quiescent_click_targets
                    or feat in av_feats
                    or feat == bg
                    or feat == engine.bg_feature
                    or (feat == 0 and bg != 0)
                    or feat in engine.quiescent_features
                    or (cr, cc) == engine.avatar_pos
                    or cls.is_peripheral_frame_cell(cr, cc, H, W, curr_grid)
                ):
                    continue
                visit_count = engine.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                feat_bias = 40.0 if feat in engine.effective_features else 0.0
                cand_score = 20.0 + feat_bias - float(visit_count) * 2.0
                click_candidates.append((cr, cc, cand_score))

        # 10. Fallback: re-probe confirmed effective targets
        if not click_candidates and engine.effective_click_targets:
            eff_sorted = sorted(
                engine.effective_click_targets,
                key=lambda p: engine.entity_visit_counts.get(f"click_{p[0]}_{p[1]}", 0),
            )
            click_candidates.append((eff_sorted[0][0], eff_sorted[0][1], 50.0))

        if click_candidates:
            click_candidates.sort(key=lambda x: x[2], reverse=True)
            best_r, best_c, _ = click_candidates[0]
            engine.entity_visit_counts[f"click_{best_r}_{best_c}"] = (
                engine.entity_visit_counts.get(f"click_{best_r}_{best_c}", 0) + 1
            )
            target_r, target_c = int(best_r), int(best_c)
            engine.last_effector_target_step[(target_r, target_c)] = engine.step_counter

            # Parietal Effector Repertoire Grounding:
            # If the action affordance defines discrete valid targets, snap the chosen target
            # to the nearest valid physical affordance point that is not known to be quiescent.
            valid_targets = aff.metadata.get("valid_targets") if aff else None
            if valid_targets and isinstance(valid_targets, (list, tuple)):
                best_score = float("inf")
                snapped_target: dict[str, Any] | None = None
                for vt in valid_targets:
                    if isinstance(vt, dict):
                        raw_vx = (
                            vt.get("x")
                            if vt.get("x") is not None
                            else vt.get("c", vt.get("col", 0))
                        )
                        raw_vy = (
                            vt.get("y")
                            if vt.get("y") is not None
                            else vt.get("r", vt.get("row", 0))
                        )
                        vx = int(raw_vx) if raw_vx is not None else 0
                        vy = int(raw_vy) if raw_vy is not None else 0
                        if (
                            not (0 <= vx <= 63 and 0 <= vy <= 63)
                            or (vy, vx) in engine.quiescent_click_targets
                        ):
                            continue
                        if hasattr(engine, "working_memory") and hasattr(
                            engine.working_memory, "habenular_ior"
                        ):
                            if engine.working_memory.habenular_ior.is_action_inhibited(
                                engine.avatar_pos, action, action_data={"x": vx, "y": vy}
                            ):
                                continue
                        elif hasattr(engine, "habenular_ior"):
                            if engine.habenular_ior.is_action_inhibited(
                                engine.avatar_pos, action, action_data={"x": vx, "y": vy}
                            ):
                                continue
                        haben_pen = 0.0
                        if hasattr(engine, "working_memory") and hasattr(
                            engine.working_memory, "habenular_ior"
                        ):
                            haben_pen = engine.working_memory.habenular_ior.get_repulsion_penalty(
                                engine.avatar_pos, action, action_data={"x": vx, "y": vy}
                            )
                        vis = engine.entity_visit_counts.get(f"click_{vy}_{vx}", 0)
                        # Distance penalization with inhibition-of-return visit penalty and Habenular repulsion
                        d = (
                            float(abs(vx - target_c) + abs(vy - target_r))
                            + float(vis) * 25.0
                            + haben_pen
                        )
                        if d < best_score:
                            best_score = d
                            snapped_target = vt

                # If all un-quiescent candidates were exhausted, select least-visited un-quiescent valid target
                if snapped_target is None and valid_targets:
                    least_visited_vt: dict[str, Any] | None = None
                    min_vis = float("inf")
                    for vt in valid_targets:
                        if isinstance(vt, dict):
                            raw_vx = vt.get("x", vt.get("c", 0))
                            raw_vy = vt.get("y", vt.get("r", 0))
                            vx = int(raw_vx) if raw_vx is not None else 0
                            vy = int(raw_vy) if raw_vy is not None else 0
                            if (
                                not (0 <= vx <= 63 and 0 <= vy <= 63)
                                or (vy, vx) in engine.quiescent_click_targets
                            ):
                                continue
                            vis = engine.entity_visit_counts.get(f"click_{vy}_{vx}", 0)
                            if vis < min_vis:
                                min_vis = vis
                                least_visited_vt = vt
                    snapped_target = least_visited_vt

                if snapped_target is not None:
                    # Update visit count for snapped coordinate
                    svx = int(snapped_target.get("x", snapped_target.get("c", 0)))
                    svy = int(snapped_target.get("y", snapped_target.get("r", 0)))
                    engine.entity_visit_counts[f"click_{svy}_{svx}"] = (
                        engine.entity_visit_counts.get(f"click_{svy}_{svx}", 0) + 1
                    )
                    snapped = dict(snapped_target)
                    for k in ("x", "c", "col", "column", "azimuth"):
                        if k in snapped and isinstance(snapped[k], (int, float)):
                            snapped[k] = max(0, min(63, int(snapped[k])))
                    for k in ("y", "r", "row", "elevation", "distance"):
                        if k in snapped and isinstance(snapped[k], (int, float)):
                            snapped[k] = max(0, min(63, int(snapped[k])))
                    return snapped
                elif valid_targets:
                    # All explicit valid targets are quiescent / exhausted: inhibit action
                    if action is not None:
                        engine.inhibited_actions[action] = max(
                            engine.inhibited_actions.get(action, 0), 6
                        )
                    return None

            coords: dict[str, Any] = {}
            for k in param_keys:
                if k in ("x", "col", "c", "column", "azimuth"):
                    coords[k] = max(0, min(63, target_c))
                elif k in ("y", "row", "r", "elevation", "distance"):
                    coords[k] = max(0, min(63, target_r))
                else:
                    coords[k] = 0
            return coords

        # Robust Spatial Grounding Invariant: On featureless/blank boards with no entities,
        # ground onto center of field to enable exploratory tactile motor babbling.
        if len(entities) == 0 and not engine.quiescent_click_targets:
            best_r, best_c = H // 2, W // 2
            engine.entity_visit_counts[f"click_{best_r}_{best_c}"] = (
                engine.entity_visit_counts.get(f"click_{best_r}_{best_c}", 0) + 1
            )
            engine.last_effector_target_step[(best_r, best_c)] = engine.step_counter
            coords_fallback: dict[str, Any] = {}
            for k in param_keys:
                if k in ("x", "col", "c", "column", "azimuth"):
                    coords_fallback[k] = max(0, min(63, best_c))
                elif k in ("y", "row", "r", "elevation", "distance"):
                    coords_fallback[k] = max(0, min(63, best_r))
                else:
                    coords_fallback[k] = 0
            return coords_fallback

        # Ventral Stream Invariant: Never click featureless void background cells or border lines.
        # When all candidates are exhausted and proven quiescent, inhibit click action
        # so deliberation and exploration switch to locomotion / directional actions.
        if action is not None:
            engine.inhibited_actions[action] = max(engine.inhibited_actions.get(action, 0), 6)
        return None

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export motor cortex effector capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="motor_cortex_effector",
            operator="==",
            value="ACTIVE",
            relation="allocentric_spatial_grounding",
            target_entity="cortex_motor",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering motor cortex effector."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="motor_cortex_effector",
                node_data={
                    "faculty": "motor_cortex_effector",
                    "status": "ACTIVE",
                },
            )
        ]


class EgocentricMotorIntent(StrEnum):
    """Viewpoint-relative embodied motor action intention."""

    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    TURN_AROUND = "TURN_AROUND"


@dataclass
class EgocentricDispatchResult:
    """Outcome of converting an egocentric motor intent into physical allocentric action."""

    intent: EgocentricMotorIntent | str
    chosen_action_id: Any | None
    resulting_heading: HeadingOrientation
    target_displacement: tuple[int, int]
    confidence: float = 1.0

    def to_causal_predicate(self) -> CausalPredicate:
        """Export egocentric dispatch as a formal declarative causal predicate."""
        return CausalPredicate(
            variable=f"ego_dispatch:{self.intent}",
            operator="==",
            value=str(self.chosen_action_id),
            relation=f"heading:{self.resulting_heading.value}",
            target_entity="cortex_motor",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording egocentric dispatch."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"ego_dispatch_{self.intent}_{self.chosen_action_id}",
                node_data={
                    "intent": str(self.intent),
                    "action_id": self.chosen_action_id,
                    "heading": str(self.resulting_heading),
                    "displacement": list(self.target_displacement),
                    "confidence": self.confidence,
                },
            )
        ]


class ParietalMotorDispatcher:
    """Coordinates egocentric action intentions with allocentric primitive motor effectors."""

    _HEADING_ROTATIONS_CW = {
        HeadingOrientation.NORTH: HeadingOrientation.EAST,
        HeadingOrientation.EAST: HeadingOrientation.SOUTH,
        HeadingOrientation.SOUTH: HeadingOrientation.WEST,
        HeadingOrientation.WEST: HeadingOrientation.NORTH,
    }

    _HEADING_ROTATIONS_CCW = {
        HeadingOrientation.NORTH: HeadingOrientation.WEST,
        HeadingOrientation.WEST: HeadingOrientation.SOUTH,
        HeadingOrientation.SOUTH: HeadingOrientation.EAST,
        HeadingOrientation.EAST: HeadingOrientation.NORTH,
    }

    @classmethod
    def dispatch_egocentric_intent(
        cls,
        intent: EgocentricMotorIntent | str,
        current_heading: HeadingOrientation | str,
        action_deltas: dict[Any, tuple[int, int]],
        available_actions: Sequence[Any] | None = None,
    ) -> EgocentricDispatchResult:
        """Translate egocentric intention into primitive action ID and updated heading."""
        if isinstance(intent, str):
            intent = EgocentricMotorIntent(intent.upper())
        if isinstance(current_heading, str):
            current_heading = HeadingOrientation(current_heading.upper())

        available = (
            set(available_actions) if available_actions is not None else set(action_deltas.keys())
        )

        # 1. Rotational intents
        if intent == EgocentricMotorIntent.TURN_LEFT:
            new_hdg = cls._HEADING_ROTATIONS_CCW[current_heading]
            return EgocentricDispatchResult(
                intent=intent,
                chosen_action_id=None,
                resulting_heading=new_hdg,
                target_displacement=(0, 0),
            )
        elif intent == EgocentricMotorIntent.TURN_RIGHT:
            new_hdg = cls._HEADING_ROTATIONS_CW[current_heading]
            return EgocentricDispatchResult(
                intent=intent,
                chosen_action_id=None,
                resulting_heading=new_hdg,
                target_displacement=(0, 0),
            )
        elif intent == EgocentricMotorIntent.TURN_AROUND:
            new_hdg = cls._HEADING_ROTATIONS_CW[cls._HEADING_ROTATIONS_CW[current_heading]]
            return EgocentricDispatchResult(
                intent=intent,
                chosen_action_id=None,
                resulting_heading=new_hdg,
                target_displacement=(0, 0),
            )

        # 2. Translational intents
        fwd_r, fwd_c = current_heading.unit_vector
        lat_r, lat_c = current_heading.right_vector

        if intent == EgocentricMotorIntent.MOVE_FORWARD:
            target_disp = (fwd_r, fwd_c)
        elif intent == EgocentricMotorIntent.MOVE_BACKWARD:
            target_disp = (-fwd_r, -fwd_c)
        elif intent == EgocentricMotorIntent.MOVE_RIGHT:
            target_disp = (lat_r, lat_c)
        elif intent == EgocentricMotorIntent.MOVE_LEFT:
            target_disp = (-lat_r, -lat_c)
        else:
            target_disp = (0, 0)

        matched_act = None
        for act, delta in action_deltas.items():
            if act in available and delta == target_disp:
                matched_act = act
                break

        return EgocentricDispatchResult(
            intent=intent,
            chosen_action_id=matched_act,
            resulting_heading=current_heading,
            target_displacement=target_disp,
            confidence=1.0 if matched_act is not None else 0.0,
        )

    @classmethod
    def resolve_tool_effector_targets(
        cls,
        avatar_pos: tuple[int, int],
        current_heading: HeadingOrientation | str,
        body_schema: Any | None = None,
    ) -> set[tuple[int, int]]:
        """Compute the spatial extent of the avatar and incorporated tools in the heading direction."""
        if isinstance(current_heading, str):
            current_heading = HeadingOrientation(current_heading.upper())
        targets: set[tuple[int, int]] = {avatar_pos}
        fwd_r, fwd_c = current_heading.unit_vector
        targets.add((avatar_pos[0] + fwd_r, avatar_pos[1] + fwd_c))
        if body_schema is not None and hasattr(body_schema, "held_tools"):
            for t in body_schema.held_tools:
                rel = getattr(t, "relative_offset", (0, 0))
                if rel != (0, 0):
                    targets.add((avatar_pos[0] + rel[0], avatar_pos[1] + rel[1]))
                else:
                    # Default tool reach extends 1 cell further along forward heading
                    targets.add((avatar_pos[0] + 2 * fwd_r, avatar_pos[1] + 2 * fwd_c))
        return targets
