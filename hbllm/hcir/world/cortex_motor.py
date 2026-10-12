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
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.active_inference import ActionNode
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_causal import ActionAffordance
from hbllm.hcir.world.cortex_perception import EntityRole, PerceptionEngine
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.inferotemporal_segmentation import InferotemporalSegmentationEngine
from hbllm.hcir.world.spatial_containment import RoomDoor
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer

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
        uncalibrated = [a for a in available_actions if not engine.is_action_calibrated(a)]
        untested_at_pos = [
            a for a in uncalibrated if (a, engine.avatar_pos) not in engine.tested_action_positions
        ]
        if untested_at_pos:
            chosen = untested_at_pos[0]
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

        uninhibited_available = [
            a for a in available_actions if engine.inhibited_actions.get(a, 0) == 0
        ]
        fallback_action = (
            uninhibited_available[0] if uninhibited_available else available_actions[0]
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
            estimated_cost = float(dist_to_target) + recency_penalty

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
        sorted_disp = sorted(disp_actions, key=lambda x: engine.inhibited_actions.get(x[0], 0))
        ar, ac = engine.avatar_pos
        for act, (dr_cal, dc_cal) in sorted_disp:
            if (engine.avatar_pos, act) in engine.failed_transitions:
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
                if p == engine.avatar_pos and (p, act) in engine.failed_transitions:
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
        bg_is_impassable_void = (
            len(engine.symbolic_theory.walkable_features) > 0
            and bg not in engine.symbolic_theory.walkable_features
        )
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
                chosen_eff = spatial_effector_actions[0]
                return EpistemicProbeProposal(
                    action=chosen_eff,
                    action_data=engine.ground_effector_action(curr_grid, chosen_eff),
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
                a for a in available_actions if engine.inhibited_actions.get(a, 0) == 0
            ]
            cand_pool = uninhibited_acts if uninhibited_acts else list(available_actions)
            chosen_fallback = cand_pool[engine.step_counter % len(cand_pool)]
            return EpistemicProbeProposal(
                action=chosen_fallback,
                clear_active_probe=clear_active_probe,
                new_probe_target=new_probe_target,
                new_probe_id=new_probe_id,
                probed_entity_ids_to_add=tuple(probed_ids_to_add),
                entity_visit_counts_to_inc=tuple(entity_visits_to_inc),
                topological_doorway=doorway_reg,
                exploration_mode="step_counter_fallback",
            )

        # 4. Fallback probe
        if spatial_effector_actions:
            chosen_eff = spatial_effector_actions[0]
            return EpistemicProbeProposal(
                action=chosen_eff,
                action_data=engine.ground_effector_action(curr_grid, chosen_eff),
                exploration_mode="effector_global_fallback",
            )
        uninhibited_global = [
            a for a in available_actions if engine.inhibited_actions.get(a, 0) == 0
        ]
        fallback_global = uninhibited_global[0] if uninhibited_global else available_actions[0]
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


class MotorCortexEffector:
    """Allocentric spatial effector grounding and action affordance arbitration."""

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
            return ActionAffordance(
                action_id=aid,
                name=name,
                is_displacement=is_disp,
                requires_spatial_target=req_spatial,
                target_param_keys=param_keys if param_keys else ("x", "y"),
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
    def _score_working_memory_pairs(
        engine: AutonomousEpistemicEngine,
        H: int,
        W: int,
        bg: int,
    ) -> list[tuple[int, int, float]]:
        """dlPFC Working Memory Pattern Completion: Match revealed card/tile with partner."""
        candidates: list[tuple[int, int, float]] = []
        last_probed = engine.working_memory.visuospatial.last_probed_coord
        last_feat = engine.working_memory.visuospatial.last_probed_feature
        if last_probed is not None and last_feat is not None and last_feat != bg and last_feat != 0:
            pair_target = engine.working_memory.visuospatial.find_matching_pair(
                last_probed, last_feat
            )
            if (
                pair_target is not None
                and pair_target not in engine.quiescent_click_targets
                and 0 <= pair_target[0] < H
                and 0 <= pair_target[1] < W
            ):
                candidates.append((pair_target[0], pair_target[1], 450.0))
        return candidates

    @staticmethod
    def _score_symmetry_discrepancies(
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        engine: AutonomousEpistemicEngine,
    ) -> list[tuple[int, int, float]]:
        """LOC Ventral Stream Geometric Symmetry Discrepancies."""
        candidates: list[tuple[int, int, float]] = []
        sym_discrepancies = VisualSymmetryAnalyzer.extract_discrepancy_targets(
            curr_grid, background_color=bg, threshold=0.55
        )
        for dr, dc, _, sym_conf in sym_discrepancies:
            if (dr, dc) not in engine.quiescent_click_targets and 0 <= dr < H and 0 <= dc < W:
                v_count = engine.entity_visit_counts.get(f"click_{dr}_{dc}", 0)
                if v_count < 3:
                    is_void = int(curr_grid[dr, dc]) == bg
                    base = 120.0 if is_void else 320.0
                    cand_score = base + (sym_conf * 40.0) - float(v_count) * 15.0
                    candidates.append((dr, dc, cand_score))
        return candidates

    @staticmethod
    def _score_safe_cells(
        engine: AutonomousEpistemicEngine,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        """dlPFC Constraint Propagation: Target guaranteed safe cells."""
        candidates: list[tuple[int, int, float]] = []
        deduced_safe = engine.working_memory.get_unrevealed_safe_cells()
        for sr, sc in deduced_safe:
            if (sr, sc) not in engine.quiescent_click_targets and 0 <= sr < H and 0 <= sc < W:
                visit_count = engine.entity_visit_counts.get(f"click_{sr}_{sc}", 0)
                if visit_count == 0:
                    candidates.append((sr, sc, 300.0))
        return candidates

    @staticmethod
    def _score_structural_goals(
        curr_grid: np.ndarray,
        bg: int,
        av_feats: set[int],
        engine: AutonomousEpistemicEngine,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        """Optical alignment & symmetry completion structural targets."""
        candidates: list[tuple[int, int, float]] = []
        structural_goals = PerceptionEngine.detect_structural_goals(
            curr_grid,
            bg=bg,
            avatar_features=av_feats,
            avatar_feature=engine.avatar_feature,
            barrier_features=engine.symbolic_theory.barrier_features,
            known_lethal_features=engine.hazard_tracker.known_lethal_features,
        )
        for og in structural_goals:
            op = og.get("position")
            g_type = og.get("type")
            if (
                op is not None
                and op not in engine.quiescent_click_targets
                and 0 <= op[0] < H
                and 0 <= op[1] < W
            ):
                v_count = engine.entity_visit_counts.get(f"click_{op[0]}_{op[1]}", 0)
                if v_count < 3:
                    is_void = og.get("is_void", False) or (int(curr_grid[op[0], op[1]]) == bg)
                    if g_type in (
                        "relational_alignment",
                        "optical_mirror_target",
                        "reflection_target",
                    ):
                        if is_void:
                            cand_score = (
                                60.0 + (og.get("confidence", 0.3) * 20.0) - float(v_count) * 20.0
                            )
                        else:
                            cand_score = (
                                240.0 + (og.get("confidence", 0.95) * 40.0) - float(v_count) * 20.0
                            )
                        candidates.append((op[0], op[1], cand_score))
                    elif g_type == "symmetry_completion":
                        if is_void:
                            cand_score = (
                                50.0 + (og.get("confidence", 0.5) * 20.0) - float(v_count) * 20.0
                            )
                        else:
                            cand_score = (
                                220.0 + (og.get("confidence", 0.8) * 50.0) - float(v_count) * 20.0
                            )
                        candidates.append((op[0], op[1], cand_score))
        return candidates

    @staticmethod
    def _score_affordance_panels(
        entities: Sequence[SpatialEntity],
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        engine: AutonomousEpistemicEngine,
    ) -> list[tuple[int, int, float]]:
        """Gestalt Affordance Panels: Regular interactive arrays & pop-out minority items."""
        candidates: list[tuple[int, int, float]] = []
        panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
        for panel in panels:
            panel_coords = set(panel["item_coords"])
            has_effective_item = any(p in engine.effective_click_targets for p in panel_coords)

            for m_item in panel["minority_items"]:
                mr, mc = m_item.grid_pos
                if (mr, mc) not in engine.quiescent_click_targets and 0 <= mr < H and 0 <= mc < W:
                    visit_count = engine.entity_visit_counts.get(f"click_{mr}_{mc}", 0)
                    last_step = engine.last_effector_target_step.get((mr, mc), -999)
                    delta_t = max(1, engine.step_counter - last_step)
                    refractory = 35.0 if delta_t == 1 else (35.0 / float(delta_t))
                    visit_damping = min(30.0, float(visit_count) * 4.0)
                    score = 400.0 - refractory - visit_damping
                    candidates.append((mr, mc, score))

            for item in panel["items"]:
                ir, ic = item.grid_pos
                if (ir, ic) not in engine.quiescent_click_targets and 0 <= ir < H and 0 <= ic < W:
                    visit_count = engine.entity_visit_counts.get(f"click_{ir}_{ic}", 0)
                    bonus = 260.0 if has_effective_item else 200.0
                    feat_bonus = 40.0 if item.feature_id in engine.effective_features else 0.0
                    last_step = engine.last_effector_target_step.get((ir, ic), -999)
                    delta_t = max(1, engine.step_counter - last_step)
                    is_goal_converging = (
                        (ir, ic) == engine.active_goal_converging_coord
                        and engine.consecutive_goal_converging_clicks < 12
                    )
                    is_active_momentum = (
                        (ir, ic) == engine.last_effective_click_coord
                        and engine.consecutive_effective_clicks < 6
                    )
                    if is_goal_converging:
                        momentum_bonus = 120.0
                        ior_penalty = 0.0
                    elif is_active_momentum:
                        momentum_bonus = 60.0
                        ior_penalty = 0.0
                    else:
                        momentum_bonus = 0.0
                        refractory = 35.0 if delta_t == 1 else (35.0 / float(delta_t))
                        visit_damping = min(30.0, float(visit_count) * 4.0)
                        ior_penalty = refractory + visit_damping
                    score = bonus + feat_bonus + momentum_bonus - ior_penalty
                    candidates.append((ir, ic, score))
        return candidates

    @staticmethod
    def _score_kinetic_repulsion_targets(
        entities: Sequence[SpatialEntity],
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        engine: AutonomousEpistemicEngine,
    ) -> list[tuple[int, int, float]]:
        """Acoustic / wave repulsion targeting: click behind floating items to propel toward goal zones."""
        candidates: list[tuple[int, int, float]] = []
        target_zones = [
            e.grid_pos
            for e in entities
            if (
                e.role == EntityRole.GOAL
                or e.feature_id in engine.symbolic_theory.goal_features
                or e.grid_pos in engine.learned_goal_positions
            )
            and e.area >= 4
        ]
        if not target_zones:
            return candidates

        for e in entities:
            if (
                e.role in (EntityRole.MANIPULABLE, EntityRole.UNKNOWN)
                and 4 <= e.area <= 64
                and e.grid_pos not in target_zones
                and e.feature_id != bg
            ):
                er, ec = e.grid_pos
                tr, tc = min(target_zones, key=lambda z: abs(z[0] - er) + abs(z[1] - ec))
                dr = int(np.sign(tr - er))
                dc = int(np.sign(tc - ec))
                extent_r = (
                    max(1, (e.bounding_box[1] - e.bounding_box[0] + 1) // 2)
                    if e.bounding_box
                    else max(1, int(round(math.sqrt(max(1, e.area)) / 2.0)))
                )
                extent_c = (
                    max(1, (e.bounding_box[3] - e.bounding_box[2] + 1) // 2)
                    if e.bounding_box
                    else max(1, int(round(math.sqrt(max(1, e.area)) / 2.0)))
                )
                offset = max(2, max(extent_r, extent_c) + 1)
                click_r = er - dr * offset
                click_c = ec - dc * offset
                if (
                    0 <= click_r < H
                    and 0 <= click_c < W
                    and (click_r, click_c) not in engine.quiescent_click_targets
                ):
                    v_count = engine.entity_visit_counts.get(f"click_{click_r}_{click_c}", 0)
                    if v_count < 4:
                        candidates.append((click_r, click_c, 350.0 - float(v_count) * 25.0))
        return candidates

    @classmethod
    def ground_effector_action(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        action: Any = None,
    ) -> dict[str, Any]:
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

        # 1. dlPFC Working Memory Pattern Completion
        click_candidates.extend(cls._score_working_memory_pairs(engine, H, W, bg))

        # 2. LOC Ventral Stream Symmetry Discrepancies
        click_candidates.extend(cls._score_symmetry_discrepancies(curr_grid, bg, H, W, engine))

        # 3. dlPFC Constraint Propagation: Safe cells
        click_candidates.extend(cls._score_safe_cells(engine, H, W))

        # 4. Optical and symmetry completion goals
        click_candidates.extend(cls._score_structural_goals(curr_grid, bg, av_feats, engine, H, W))

        # 5. Gestalt Affordance Panels
        click_candidates.extend(cls._score_affordance_panels(entities, curr_grid, bg, H, W, engine))

        # 5b. Kinetic wave & repulsion targeting
        click_candidates.extend(
            cls._score_kinetic_repulsion_targets(entities, curr_grid, bg, H, W, engine)
        )

        # 6. Salient entities if no panel candidates exist
        if not click_candidates:
            for e in entities:
                if e.feature_id == bg or e.feature_id in av_feats or e.area > 120:
                    continue
                cr, cc = e.grid_pos
                if (cr, cc) in engine.quiescent_click_targets:
                    continue
                visit_count = engine.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                saliency = 100.0 / math.log2(2 + max(1, e.area))
                affordance_bonus = 60.0 if (cr, cc) in engine.effective_click_targets else 0.0
                feat_bias = 0.0
                if e.feature_id in engine.quiescent_features:
                    feat_bias -= 80.0
                elif e.feature_id in engine.effective_features:
                    feat_bias += 40.0
                ior_penalty = float(visit_count) * 30.0 + (float(visit_count) ** 2) * 15.0
                cand_score = saliency + affordance_bonus + feat_bias - ior_penalty
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
                if (cr, cc) in engine.quiescent_click_targets or feat == bg:
                    continue
                visit_count = engine.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                affordance_bonus = 60.0 if (cr, cc) in engine.effective_click_targets else 0.0
                feat_bias = 0.0
                if feat in engine.quiescent_features:
                    feat_bias -= 80.0
                elif feat in engine.effective_features:
                    feat_bias += 40.0
                cand_score = (
                    f.salience
                    + affordance_bonus
                    + feat_bias
                    - float(visit_count) * (0.1 if affordance_bonus > 0 else 0.35)
                )
                click_candidates.append((cr, cc, cand_score))

        # 9. Fallback: unvisited non-background cells
        if not click_candidates:
            non_bg = np.argwhere(curr_grid != bg)
            for r, c in non_bg:
                cr, cc = int(r), int(c)
                feat = int(curr_grid[cr, cc])
                if (cr, cc) in engine.quiescent_click_targets or (cr, cc) in av_feats:
                    continue
                visit_count = engine.entity_visit_counts.get(f"click_{cr}_{cc}", 0)
                feat_bias = 0.0
                if feat in engine.quiescent_features:
                    feat_bias -= 80.0
                elif feat in engine.effective_features:
                    feat_bias += 40.0
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
        else:
            # Ventral Stream Invariant: Never click featureless void background cells.
            # When candidates are exhausted, clear quiescent locks on entities and re-sample
            # the highest salience non-background object centroids.
            engine.quiescent_click_targets.clear()
            non_bg = np.argwhere(curr_grid != bg)
            if len(non_bg) > 0:
                cand_cells = [
                    (int(r), int(c)) for r, c in non_bg if (int(r), int(c)) not in av_feats
                ]
                if cand_cells:
                    cand_cells.sort(
                        key=lambda p: engine.entity_visit_counts.get(f"click_{p[0]}_{p[1]}", 0)
                    )
                    target_r, target_c = cand_cells[0]
                else:
                    target_r, target_c = int(non_bg[0][0]), int(non_bg[0][1])
            else:
                target_r, target_c = H // 2, W // 2

        coords: dict[str, Any] = {}
        for k in param_keys:
            if k in ("x", "col", "c", "column", "azimuth"):
                coords[k] = target_c
            elif k in ("y", "row", "r", "elevation", "distance"):
                coords[k] = target_r
            else:
                coords[k] = 0
        return coords

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
