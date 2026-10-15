from __future__ import annotations

"""Prefrontal Cortex Deliberation & Executive Arbitration Faculty.

Biologically modeled on mammalian dorsolateral / ventromedial prefrontal cortex (dlPFC / vmPFC)
and striatal gating:
1. Multi-Faculty Cognitive Arbitration:
   - Evaluates active mental plans against static lethal entities, spatiotemporal looming cones,
     dynamic patroller forward trajectories, and Habenular episodic action inhibition.
2. Cerebellar Phase-Locked Motor Gating:
   - Synchronizes kinetic movements to rhythmic oscillation windows.
3. Affordance Panel & Optical Symmetry Arbitration:
   - Directs spatial effectors toward interactive control panels and relational reflection midpoints.
4. Deliberate-to-Habitual Search Backoff (Daw, Niv, & Dayan; Dolan & Dayan):
   - Transitions between deliberate forward mental simulation and habitual macro-action exploration.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.cortex_planner import MentalSimulationStep

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Prefrontal Safety Gating Predicates (Stage 4 Active Inference)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SafetyGateResult:
    """Result of a cognitive safety gate evaluation."""

    is_safe: bool = True
    should_wait: bool = False
    wait_steps: int = 0
    reason: str = ""

    def to_causal_predicate(self) -> CausalPredicate:
        """Export safety gate result as a declarative causal predicate."""
        return CausalPredicate(
            variable="safety_gate_result",
            operator="==",
            value="SAFE" if self.is_safe else ("WAIT" if self.should_wait else "LETHAL"),
            relation=f"wait_steps:{self.wait_steps}",
            target_entity=self.reason or "nominal",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording safety gate result."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="safety_gate_result",
                node_data={
                    "is_safe": self.is_safe,
                    "should_wait": self.should_wait,
                    "wait_steps": self.wait_steps,
                    "reason": self.reason,
                },
            )
        ]


class PlanSafetyPredicate(ABC):
    """Abstract base class for prefrontal safety gating predicates."""

    @abstractmethod
    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        """Evaluate plan safety against specific hazard models."""
        ...


class StaticLethalHazardPredicate(PlanSafetyPredicate):
    """Rejects candidate trajectories intersecting known static lethal entities."""

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        H, W = curr_grid.shape
        for future_step in future_steps:
            if future_step.predicted_avatar_pos is None:
                continue
            fr, fc = future_step.predicted_avatar_pos
            if 0 <= fr < H and 0 <= fc < W:
                feat = int(curr_grid[fr, fc])
                if feat in engine.mobile_threat_features:
                    continue
                if feat in engine.hazard_tracker.known_lethal_features:
                    return SafetyGateResult(
                        is_safe=False, reason=f"Lethal feature {feat} at ({fr}, {fc})"
                    )
        return SafetyGateResult(is_safe=True)


class SpatiotemporalCollisionConePredicate(PlanSafetyPredicate):
    """Rejects steps intersecting spatiotemporal looming collision cones."""

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        if engine.avatar_pos is None:
            return SafetyGateResult(is_safe=True)

        ar, ac = engine.avatar_pos
        is_stationary = (
            next_step.predicted_avatar_pos is None or next_step.predicted_avatar_pos == (ar, ac)
        )
        if is_stationary:
            if any(
                engine.collision_cones.is_collision_hazard(
                    ar, ac, t_f, ignored_features=engine.mobile_threat_features
                )
                for t_f in (1, 2)
            ):
                return SafetyGateResult(
                    is_safe=False, reason="Stationary avatar in looming collision cone"
                )
        elif next_step.predicted_avatar_pos is not None:
            nr, nc = next_step.predicted_avatar_pos
            if engine.collision_cones.is_collision_hazard(
                nr, nc, 1, ignored_features=engine.mobile_threat_features
            ):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"Next step ({nr}, {nc}) in looming collision cone",
                )

        return SafetyGateResult(is_safe=True)


class DynamicPatrolTrajectoryPredicate(PlanSafetyPredicate):
    """Simulates dynamic threat patrollers forward in time to prevent kinetic collisions."""

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        try:
            H, W = curr_grid.shape
            bg = engine.estimate_background(curr_grid)
            step_size = engine.infer_motor_step_size(available_actions)
            entities = engine.extract_entities(curr_grid, bg)
            av_feats = engine.avatar_features or (
                {engine.avatar_feature} if engine.avatar_feature is not None else set()
            )
            live_threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                entities,
                bg=bg,
                step_size=step_size,
                avatar_pos=engine.avatar_pos,
                avatar_features=av_feats,
            )
            if not live_threats and not engine.mobile_threat_features:
                return SafetyGateResult(is_safe=True)

            # Oriented stationary hazard line-of-sight gate (e.g., directional sensors, optical cones)
            stationary_threats = [
                t for t in live_threats if t.feature_id not in engine.mobile_threat_features
            ]
            oriented_hazard_cells = {t.gaze_pos for t in stationary_threats if t.facing != (0, 0)}
            if (
                next_step.predicted_avatar_pos is not None
                and next_step.predicted_avatar_pos in oriented_hazard_cells
            ):
                return SafetyGateResult(
                    is_safe=False,
                    reason="Avatar destination steps directly into oriented hazard field-of-view",
                )
            for st in stationary_threats:
                if next_step.predicted_avatar_pos == st.pos and engine.avatar_pos == st.gaze_pos:
                    return SafetyGateResult(
                        is_safe=False,
                        reason="Frontal collision with oriented active hazard along line-of-sight",
                    )

            live_patrols: set[tuple[tuple[int, int], tuple[int, int]]] = {
                (t.pos, t.facing)
                for t in live_threats
                if t.feature_id in engine.mobile_threat_features
            }

            if not live_patrols:
                return SafetyGateResult(is_safe=True)

            half_step = max(1, step_size // 2)
            barrier_feats = engine.symbolic_theory.barrier_features
            walkable_and_safe = (
                engine.symbolic_theory.walkable_features | engine.verified_safe_features
            )
            bg_is_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
            patrol_walls: set[tuple[int, int]] = set(engine.learned_barriers)
            for rr in range(H):
                for cc in range(W):
                    v = int(curr_grid[rr, cc])
                    if (
                        engine.symbolic_theory.is_barrier(v)
                        or v in barrier_feats
                        or (bg_is_void and v == bg)
                    ):
                        patrol_walls.add((rr, cc))

            def _blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
                probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
                dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                for q in (probe, dest):
                    if not (0 <= q[0] < H and 0 <= q[1] < W) or q in patrol_walls:
                        return True
                return False

            sim_patrols = frozenset(live_patrols)
            for future_step in future_steps:
                avatar_dest = future_step.predicted_avatar_pos
                new_patrol_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                collision = False
                for pat_pos, pat_facing in sim_patrols:
                    if pat_pos == avatar_dest:
                        collision = True
                        break
                    dest = (
                        pat_pos[0] + pat_facing[0] * step_size,
                        pat_pos[1] + pat_facing[1] * step_size,
                    )
                    if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in patrol_walls:
                        new_f = (-pat_facing[0], -pat_facing[1])
                        new_patrol_set.add((pat_pos, new_f))
                        continue
                    if dest == avatar_dest:
                        collision = True
                        break
                    new_f = pat_facing
                    if _blocked(dest, pat_facing):
                        new_f = (-pat_facing[0], -pat_facing[1])
                    new_patrol_set.add((dest, new_f))
                if collision:
                    return SafetyGateResult(
                        is_safe=True,
                        should_wait=True,
                        wait_steps=2,
                        reason="Collision with simulated patrol threat: temporal pacing advised",
                    )
                sim_patrols = frozenset(new_patrol_set)

        except Exception as e:
            logger.debug("Dynamic patrol trajectory simulation error: %s", e)

        return SafetyGateResult(is_safe=True)


class HabenularInhibitionPredicate(PlanSafetyPredicate):
    """Inhibits repetitive looping motor actions via Lateral Habenula / Basal Ganglia IOR."""

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        if engine.inhibited_actions.get(next_step.action, 0) > 0:
            return SafetyGateResult(
                is_safe=False, reason=f"Action {next_step.action} in refractory inhibition"
            )
        if engine.working_memory.habenular_ior.is_action_inhibited(
            engine.avatar_pos, next_step.action
        ):
            return SafetyGateResult(is_safe=False, reason="Action inhibited by Habenular IOR")
        return SafetyGateResult(is_safe=True)


class CerebellarPhaseGatePredicate(PlanSafetyPredicate):
    """Evaluates rhythm-locked oscillation windows via Cerebellar phase clock."""

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        if next_step.predicted_avatar_pos is not None:
            phase_gate = engine.cerebellar_clock.evaluate_motion_hazard_gate(
                current_step=engine.step_counter,
                avatar_pos=engine.avatar_pos,
                target_pos=next_step.predicted_avatar_pos,
                hazard_tracker=engine.hazard_tracker,
                background_feature=engine.bg_feature,
                avatar_features=engine.avatar_features,
                walkable_features=engine.symbolic_theory.walkable_features
                | engine.verified_safe_features,
            )
            if phase_gate.should_wait:
                return SafetyGateResult(
                    is_safe=True,
                    should_wait=True,
                    wait_steps=phase_gate.wait_steps_recommended,
                    reason="Cerebellar phase clock advises wait window",
                )
        return SafetyGateResult(is_safe=True)


class PlanSafetyEvaluator:
    """Prefrontal safety arbitration engine composing typed PlanSafetyPredicates."""

    def __init__(self, predicates: Sequence[PlanSafetyPredicate] | None = None) -> None:
        self.predicates: list[PlanSafetyPredicate] = list(
            predicates
            if predicates is not None
            else [
                StaticLethalHazardPredicate(),
                SpatiotemporalCollisionConePredicate(),
                DynamicPatrolTrajectoryPredicate(),
                HabenularInhibitionPredicate(),
                CerebellarPhaseGatePredicate(),
            ]
        )

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        future_steps = [next_step] + list(engine.mental_plan)
        for pred in self.predicates:
            res = pred.evaluate(
                engine=engine,
                curr_grid=curr_grid,
                next_step=next_step,
                future_steps=future_steps,
                available_actions=available_actions,
            )
            if not res.is_safe or res.should_wait:
                return res
        return SafetyGateResult(is_safe=True)

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export plan safety evaluator capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="plan_safety_evaluator",
            operator="==",
            value="ACTIVE",
            relation="multivariate_prefrontal_safety_gating",
            target_entity="cortex_deliberation",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering plan safety evaluator."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="plan_safety_evaluator",
                node_data={
                    "faculty": "plan_safety_evaluator",
                    "status": "ACTIVE",
                },
            )
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Prefrontal Deliberation Engine
# ═══════════════════════════════════════════════════════════════════════════


class PrefrontalDeliberationEngine:
    """Executive prefrontal arbitration engine coordinating perception, mental simulation,
    affordance chunking, and motor output.
    """

    def __init__(self, safety_evaluator: PlanSafetyEvaluator | None = None) -> None:
        self.safety_evaluator = safety_evaluator or PlanSafetyEvaluator()

    def validate_plan_safety(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        next_step: MentalSimulationStep,
    ) -> tuple[bool, bool, int]:
        """Validate safety of next plan step with multi-faculty reactive checks:
        A) Static lethal entities along planned path
        A2) Spatiotemporal looming collision cones
        B) Dynamic patrol trajectory forward simulation
        C) Habenular episodic action inhibition
        D) Cerebellar phase gating

        Returns:
            (plan_safe, should_wait, wait_steps)
        """
        res = self.safety_evaluator.evaluate(
            engine=engine,
            curr_grid=curr_grid,
            next_step=next_step,
            available_actions=available_actions,
        )
        return res.is_safe, res.should_wait, res.wait_steps

    def arbitrate_affordances(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        known_av_feats: set[int],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Arbitrate control panel affordances and interactive targets via registered policies or generic fallbacks."""
        # 1. Query external registered deliberation policies first (e.g. ARC3DeliberationPolicy)
        for policy in getattr(engine, "deliberation_policies", []):
            if hasattr(policy, "arbitrate_affordances"):
                res = policy.arbitrate_affordances(
                    engine, curr_grid, available_actions, known_av_feats
                )
                if res is not None:
                    return res

        # 2. Domain-agnostic fallback for discrete transform / focus switch actions
        spatial_effector_actions = [a for a in available_actions if engine.is_spatial_effector(a)]
        discrete_transform_actions = [
            a
            for a in available_actions
            if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
        ]
        focus_switches = [a for a in discrete_transform_actions if engine.is_focus_switch_action(a)]
        if focus_switches:
            engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            return focus_switches[0], None

        if not spatial_effector_actions and discrete_transform_actions:
            has_displacement_actions = any(
                engine.is_displacement_action(a) for a in available_actions
            ) or any(
                not engine.is_spatial_effector(a) and a not in engine.action_dynamics
                for a in available_actions
            )
            is_stuck_or_looping = (
                engine.consecutive_stuck_steps >= 2
                or (
                    engine.avatar_pos is not None
                    and engine.recent_positions.count(engine.avatar_pos) >= 2
                )
                or (engine.avatar_pos is None and engine.step_counter > 4)
            )
            if not has_displacement_actions or is_stuck_or_looping:
                undo_acts = getattr(engine, "undo_actions", set())
                uninhibited_discrete = [
                    a
                    for a in discrete_transform_actions
                    if engine.inhibited_actions.get(a, 0) == 0 and a not in undo_acts
                ]
                chosen_action = (
                    uninhibited_discrete[0]
                    if uninhibited_discrete
                    else discrete_transform_actions[0]
                )
                engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen_action, None

        return None

    def decide(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action_schemas: Sequence[Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision function:
        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        """
        engine.step_counter += 1
        if action_schemas:
            engine.register_action_space(action_schemas)

        curr_grid = engine.normalize_sensory_input(curr_grid)

        # Metacognitive Urgency Modulation:
        # Under acute step countdown / strict budget constraints, suppress exploratory
        # refractory cooldowns to prioritize immediate exploitation toward known goals.
        if engine.is_under_strict_step_budget():
            engine.exploration_cooldown = 0
            engine.simulation_cooldown = 0

        # 1. Update background & assimilate sensory feedback from previous action
        engine.bg_feature = engine.estimate_background(curr_grid)
        if engine.prev_grid is not None and not engine._feedback_assimilated:
            engine.assimilate_feedback(
                curr_grid,
                available_actions,
                is_win=is_win,
                is_lost=is_lost,
                action=engine.last_action,
                action_data=engine.last_action_data,
            )
        engine._feedback_assimilated = False

        # Phasic Salience Reset: Environmental discovery or high surprise awakens deliberate forward search
        n_calibrated_actions = len(engine.action_dynamics)
        if (
            len(engine.learned_goal_positions) > engine._last_sim_goal_count
            or len(engine.learned_barriers) != engine._last_sim_barrier_count
            or n_calibrated_actions > engine._last_sim_action_count
            or engine.last_surprise >= 0.4
        ):
            engine.simulation_cooldown = 0
            engine.consecutive_simulation_failures = 0
            if n_calibrated_actions > engine._last_sim_action_count:
                engine.mental_plan.clear()
                engine.phase = (
                    EpistemicPhase.MENTAL_SIMULATION
                    if engine.is_motor_grounded()
                    else EpistemicPhase.EPISTEMIC_EXPLORATION
                )
                engine._last_sim_action_count = n_calibrated_actions
            engine._last_sim_goal_count = len(engine.learned_goal_positions)
            engine._last_sim_barrier_count = len(engine.learned_barriers)

        # Metacognitive Refractory Inhibition: decay refractory timers
        if engine.simulation_cooldown > 0:
            engine.simulation_cooldown -= 1
        to_uninhibited = [act for act, timer in engine.inhibited_actions.items() if timer <= 1]
        for act in engine.inhibited_actions:
            engine.inhibited_actions[act] -= 1
        for act in to_uninhibited:
            engine.inhibited_actions.pop(act, None)

        # Track recent avatar position on every decision step
        if engine.avatar_pos is not None:
            engine.recent_positions.append(engine.avatar_pos)

        # Anterior Mid-Cingulate & Lateral Habenula: Limit Cycle / Oscillation Detection
        osc_acts = engine.working_memory.habenular_ior.detect_cyclic_actions()
        if osc_acts:
            has_progress = (
                engine.phase == EpistemicPhase.EXPLOITATION
                and bool(engine.mental_plan)
                and any(
                    s.predicted_avatar_pos is not None
                    and s.predicted_avatar_pos != engine.avatar_pos
                    for s in engine.mental_plan
                )
            )
            if not has_progress:
                for osc_act in osc_acts:
                    engine.inhibited_actions[osc_act] = max(
                        engine.inhibited_actions.get(osc_act, 0), 6
                    )

        # Faculty: Anterior Cingulate Cortex (ACC) Limit Cycle & Conflict Suppression
        forced_acc_perturbation: Any | None = None
        engine.acc_conflict_monitor.record_step(engine.last_action, engine.avatar_pos)
        is_osc, cyclic_acts = engine.acc_conflict_monitor.detect_oscillation()
        if is_osc:
            # In cognitive architecture, an intentional temporal wait/dodge maneuver
            # planned by the prefrontal cortex must not be aborted as an unintentional deadlock
            # if the active mental plan progresses to novel coordinates beyond the cyclic positions.
            if engine.phase == EpistemicPhase.EXPLOITATION and engine.mental_plan:
                cyclic_positions = set(list(engine.acc_conflict_monitor.position_history)[-4:])
                future_plan_positions = {
                    s.predicted_avatar_pos
                    for s in engine.mental_plan
                    if s.predicted_avatar_pos is not None
                }
                if any(p not in cyclic_positions for p in future_plan_positions):
                    is_osc = False
        if is_osc:
            for cyc_a in cyclic_acts:
                engine.inhibited_actions[cyc_a] = max(engine.inhibited_actions.get(cyc_a, 0), 8)
            engine.mental_plan.clear()
            from hbllm.hcir.world.cortex_motor import ACCConflictMonitor

            ortho_acts = ACCConflictMonitor.get_orthogonal_actions(
                available_actions=available_actions,
                cyclic_actions=cyclic_acts,
                action_dynamics=engine.action_dynamics,
            )
            H, W = curr_grid.shape
            for cand_a in ortho_acts:
                if engine.inhibited_actions.get(cand_a, 0) > 0:
                    continue
                if (engine.avatar_pos, cand_a) in engine.failed_transitions:
                    continue
                dyn = engine.action_dynamics.get(cand_a)
                if dyn is not None and dyn.is_displacement_action():
                    dr, dc = dyn.get_displacement()
                    if engine.avatar_pos is not None:
                        nr, nc = engine.avatar_pos[0] + dr, engine.avatar_pos[1] + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if (nr, nc) in engine.hazard_tracker.static_lethal_positions:
                            continue
                        if (nr, nc) in engine.learned_barriers:
                            continue
                        feat_at_dest = int(curr_grid[nr, nc])
                        if feat_at_dest in engine.hazard_tracker.known_lethal_features:
                            continue
                        forced_acc_perturbation = cand_a
                        break
                else:
                    forced_acc_perturbation = cand_a
                    break

        # Faculty: ACC Recurring Catastrophe Conflict Resolution (Metacognitive Executive Shift)
        acc_death_conflict = getattr(engine.working_memory, "acc_conflict", None)
        if acc_death_conflict is not None and getattr(acc_death_conflict, "active_conflict", False):
            c_radius = float(getattr(acc_death_conflict, "cluster_radius", 3.0))
            # Invalidate any active plans routing into failure clusters
            if engine.mental_plan:
                for cl in acc_death_conflict.clusters:
                    if cl.count >= 2:
                        for step_s in engine.mental_plan:
                            if step_s.predicted_avatar_pos is not None:
                                pr, pc = step_s.predicted_avatar_pos
                                if math.hypot(pr - cl.center_r, pc - cl.center_c) <= c_radius:
                                    engine.mental_plan.clear()
                                    engine.current_simulated_goal = None
                                    break
                            if not engine.mental_plan:
                                break

            # Mark death cluster cells as static lethal hazards to force topological detours
            for cl in acc_death_conflict.clusters:
                if cl.count >= 2:
                    cr_int, cc_int = int(round(cl.center_r)), int(round(cl.center_c))
                    rad_int = int(math.ceil(c_radius))
                    for dr_c in range(-rad_int, rad_int + 1):
                        for dc_c in range(-rad_int, rad_int + 1):
                            if math.hypot(dr_c, dc_c) <= c_radius:
                                lr, lc = cr_int + dr_c, cc_int + dc_c
                                if 0 <= lr < curr_grid.shape[0] and 0 <= lc < curr_grid.shape[1]:
                                    engine.hazard_tracker.static_lethal_positions.add((lr, lc))
                                    cell_f = int(curr_grid[lr, lc])
                                    if cell_f in engine.symbolic_theory.goal_features:
                                        engine.symbolic_theory.goal_features.discard(cell_f)
                                        engine.hazard_tracker.known_lethal_features.add(cell_f)
                                        logger.warning(
                                            "ACC Metacognitive Shift: Inverted hypothesis for feature %d (GOAL -> LETHAL) due to death cluster at (%d, %d)",
                                            cell_f,
                                            lr,
                                            lc,
                                        )

        # Visual avatar localization
        known_av_feats = set(engine.avatar_features) if engine.avatar_features else set()
        if engine.avatar_feature is not None:
            known_av_feats.add(engine.avatar_feature)
        elif engine.prior_avatar_feature is not None:
            known_av_feats.add(engine.prior_avatar_feature)
        if known_av_feats:
            engine._update_avatar_position_from_grid(curr_grid, known_av_feats)
            if (
                engine.avatar_pos is not None
                and engine.avatar_feature is None
                and engine.prior_avatar_feature is not None
            ):
                engine.avatar_feature = engine.prior_avatar_feature
                engine.avatar_features.add(engine.prior_avatar_feature)

        # Observe creature motion
        engine._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        # Faculty E: Basal Ganglia Motor Gating & Phase Entrainment
        if engine.pending_phase_wait_steps > 0:
            engine.pending_phase_wait_steps -= 1
            undo_acts = getattr(engine, "undo_actions", set())
            non_disp = [
                a
                for a in available_actions
                if not engine.is_displacement_action(a)
                and not engine.is_spatial_effector(a)
                and a not in undo_acts
            ]
            if non_disp:
                return non_disp[0], None

        # Faculty: Brodmann Area 10 (aPFC) Hierarchical Subgoal Stack & Cognitive Branching
        if engine.subgoal_stack.has_pending_goals:
            cur_sg = engine.subgoal_stack.current_subgoal
            if cur_sg is not None and engine.subgoal_stack.is_subgoal_satisfied(
                cur_sg, set(), avatar_pos=engine.avatar_pos
            ):
                engine.subgoal_stack.pop_subgoal()

        chosen_action: Any
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None

        # 0. ACC Forced Perturbation: break limit-cycle oscillation immediately
        if forced_acc_perturbation is not None:
            logger.info(
                "Anterior Cingulate Cortex (ACC): Breaking limit cycle oscillation with orthogonal perturbation action %s",
                forced_acc_perturbation,
            )
            engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action = forced_acc_perturbation
            chosen_data = None

        # 1. Prefrontal Affordance Panel & Optical Symmetry Gating
        elif (
            affordance_choice := self.arbitrate_affordances(
                engine, curr_grid, available_actions, known_av_feats
            )
        ) is not None and affordance_choice[0] not in engine.inhibited_actions:
            chosen_action, chosen_data = affordance_choice

        # 2. Check active mental plan during EXPLOITATION phase
        elif engine.phase == EpistemicPhase.EXPLOITATION and engine.mental_plan:
            next_step = engine.mental_plan.popleft()
            if not engine.mental_plan and engine.object_planner.active_target_object_id:
                engine.object_planner.ledger.record_interaction(
                    engine.object_planner.active_target_object_id,
                    next_step.predicted_avatar_pos or engine.avatar_pos or (0, 0),
                )
                engine.object_planner.active_target_object_id = None

            plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                engine, curr_grid, available_actions, next_step
            )

            if should_wait:
                engine.consecutive_pacing_steps += 1
                if engine.consecutive_pacing_steps > 2:
                    engine.consecutive_pacing_steps = 0
                    engine.mental_plan.clear()
                    engine.transition_to_phase(
                        EpistemicPhase.REPLANNING,
                        reason="pacing_window_exhausted",
                    )
                    return engine.plan_epistemic_probe(curr_grid, available_actions)

                engine.mental_plan.appendleft(next_step)
                engine.pending_phase_wait_steps = wait_steps
                undo_acts = getattr(engine, "undo_actions", set())
                non_disp = [
                    a
                    for a in available_actions
                    if not engine.is_displacement_action(a)
                    and not engine.is_spatial_effector(a)
                    and a not in undo_acts
                ]
                if non_disp:
                    return non_disp[0], None

                # Cerebellar / Premotor Oscillatory Temporal Pacing:
                # In environments with strictly directional actions (no explicit wait),
                # synthesize an oscillatory burn step (e.g. step back / lateral dodge -> return)
                # to burn phase steps until dynamic hazards clear.
                curr_pos = engine.avatar_pos or getattr(engine, "current_actor_pos", None)
                H, W = curr_grid.shape
                # Dynamically derive opposite displacement action mapping from kinematic deltas
                reverse_map: dict[Any, Any] = {}
                for a1, d1 in engine.action_deltas.items():
                    for a2, d2 in engine.action_deltas.items():
                        if d1[0] == -d2[0] and d1[1] == -d2[1] and (d1[0] != 0 or d1[1] != 0):
                            reverse_map[a1] = a2
                if not reverse_map:
                    reverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
                pacing_candidates: list[tuple[Any, tuple[int, int]]] = []

                if curr_pos is not None:
                    action_deltas = engine.action_deltas
                    for act in available_actions:
                        if not engine.is_displacement_action(act) or act == next_step.action:
                            continue
                        delta = action_deltas.get(act)
                        if delta is None:
                            continue
                        dr, dc = delta
                        nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if (nr, nc) in engine.learned_barriers or (
                            nr,
                            nc,
                        ) in engine.current_level_lethal_positions:
                            continue
                        cand_step = MentalSimulationStep(
                            action=act,
                            predicted_avatar_pos=(nr, nc),
                            expected_mutation="oscillatory_pacing_step",
                        )
                        cand_res = self.safety_evaluator.evaluate(
                            engine=engine,
                            curr_grid=curr_grid,
                            next_step=cand_step,
                            available_actions=available_actions,
                        )
                        if cand_res.is_safe and not cand_res.should_wait:
                            pacing_candidates.append((act, (nr, nc)))

                if pacing_candidates:
                    # Prefer reverse of recent action to oscillate along existing safe path
                    chosen_pacing_act, _ = pacing_candidates[0]
                    if len(engine.recent_actions) > 0:
                        last_act = engine.recent_actions[-1]
                        rev_last = reverse_map.get(last_act)
                        for act, pos in pacing_candidates:
                            if act == rev_last:
                                chosen_pacing_act = act
                                break

                    rev_act = reverse_map.get(chosen_pacing_act)
                    # Enqueue return step to restore spatial trajectory once hazard passes
                    if (
                        rev_act is not None
                        and rev_act in available_actions
                        and curr_pos is not None
                    ):
                        engine.mental_plan.appendleft(
                            MentalSimulationStep(
                                action=rev_act,
                                predicted_avatar_pos=curr_pos,
                                expected_mutation="oscillatory_return_step",
                            )
                        )
                    return chosen_pacing_act, None

                # Invariant fallback: Invalidate plan and replan if no safe pacing candidate exists
                engine.mental_plan.clear()
                engine.phase = EpistemicPhase.REPLANNING
                return engine.plan_epistemic_probe(curr_grid, available_actions)

            engine.consecutive_pacing_steps = 0
            if not plan_safe:
                engine.mental_plan.clear()
                engine.phase = EpistemicPhase.REPLANNING
                # Primary replanning: Object-Centric Macro Option
                macro_plan = engine.object_planner.plan_macro_option(
                    engine, curr_grid, available_actions
                )
                if macro_plan:
                    engine.phase = EpistemicPhase.EXPLOITATION
                    engine.mental_plan = deque(macro_plan)
                    next_step = engine.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    danger_cells = (
                        {next_step.predicted_avatar_pos}
                        if next_step.predicted_avatar_pos is not None
                        else set()
                    )
                    simulated_plan = engine.simulate_in_mind(
                        curr_grid, available_actions, blocked_cells=danger_cells
                    )
                    if simulated_plan:
                        engine.consecutive_simulation_failures = 0
                        engine.simulation_cooldown = 0
                        engine.phase = EpistemicPhase.EXPLOITATION
                        engine.mental_plan = deque(simulated_plan)
                        next_step = engine.mental_plan.popleft()
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                    else:
                        chosen_action, chosen_data = engine.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
            elif (
                next_step.action in available_actions
                and engine.inhibited_actions.get(next_step.action, 0) == 0
                and not engine.working_memory.habenular_ior.is_action_inhibited(
                    engine.avatar_pos, next_step.action, next_step.action_data
                )
            ):
                chosen_action = next_step.action
                chosen_data = next_step.action_data
                predicted_pos = next_step.predicted_avatar_pos
            else:
                engine.mental_plan.clear()
                engine.phase = EpistemicPhase.REPLANNING
                macro_plan = engine.object_planner.plan_macro_option(
                    engine, curr_grid, available_actions
                )
                if macro_plan:
                    engine.phase = EpistemicPhase.EXPLOITATION
                    engine.mental_plan = deque(macro_plan)
                    next_step = engine.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    chosen_action, chosen_data = engine.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        # 3. Master Planning Branch (when no active plan is in flight)
        else:
            chosen_action = None
            chosen_data = None
            predicted_pos = None

            # 3A. Primary: Forward Mental Simulation over learned dynamics & verified exits
            # (Executive Directive 1 & 2: Check if exit is open, if open go through it)
            if engine.is_motor_grounded() and engine.simulation_cooldown == 0:
                simulated_plan = engine.simulate_in_mind(curr_grid, available_actions)
                if simulated_plan:
                    engine.consecutive_simulation_failures = 0
                    engine.simulation_cooldown = 0
                    engine.phase = EpistemicPhase.EXPLOITATION
                    engine.mental_plan = deque(simulated_plan)
                    next_step = engine.mental_plan.popleft()
                    plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                        engine, curr_grid, available_actions, next_step
                    )
                    if (
                        plan_safe
                        and not should_wait
                        and next_step.action in available_actions
                        and engine.inhibited_actions.get(next_step.action, 0) == 0
                        and not engine.working_memory.habenular_ior.is_action_inhibited(
                            engine.avatar_pos, next_step.action, next_step.action_data
                        )
                    ):
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                    else:
                        engine.mental_plan.clear()
                        engine.phase = EpistemicPhase.REPLANNING
                        chosen_action, chosen_data = engine.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
                else:
                    engine.consecutive_simulation_failures += 1
                    engine.simulation_cooldown = min(
                        12, 2 ** min(engine.consecutive_simulation_failures, 4)
                    )
                    engine._last_sim_goal_count = len(engine.learned_goal_positions)
                    engine._last_sim_barrier_count = len(engine.learned_barriers)

            # 3B. Secondary: Object-Centric Macro Action Planning (Directives 3, 4, 5)
            # If exit is not open/reachable or in search backoff, interact with intermediate objects
            if chosen_action is None:
                macro_plan = engine.object_planner.plan_macro_option(
                    engine, curr_grid, available_actions
                )
                if (
                    macro_plan
                    and macro_plan[0].action in available_actions
                    and engine.inhibited_actions.get(macro_plan[0].action, 0) == 0
                    and not engine.working_memory.habenular_ior.is_action_inhibited(
                        engine.avatar_pos, macro_plan[0].action, macro_plan[0].action_data
                    )
                ):
                    engine.phase = EpistemicPhase.EXPLOITATION
                    engine.mental_plan = deque(macro_plan)
                    next_step = engine.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    if not engine.mental_plan and engine.object_planner.active_target_object_id:
                        engine.object_planner.ledger.record_interaction(
                            engine.object_planner.active_target_object_id,
                            predicted_pos or engine.avatar_pos or (0, 0),
                        )
                        engine.object_planner.active_target_object_id = None
                else:
                    if engine.exploration_cooldown > 0:
                        engine.exploration_cooldown -= 1
                    engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    chosen_action, chosen_data = engine.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        # Effector Grounding Invariant
        if engine.is_spatial_effector(chosen_action):
            aff = engine.action_affordances.get(chosen_action)
            req_keys = aff.target_param_keys if aff else ("x", "y")
            if (
                chosen_data is None
                or not isinstance(chosen_data, dict)
                or not all(k in chosen_data for k in req_keys)
            ):
                chosen_data = engine.ground_effector_action(curr_grid, chosen_action)
                if chosen_data is None:
                    engine.inhibited_actions[chosen_action] = max(
                        engine.inhibited_actions.get(chosen_action, 0), 6
                    )
                    non_eff = [
                        a
                        for a in available_actions
                        if not engine.is_spatial_effector(a)
                        and engine.inhibited_actions.get(a, 0) == 0
                    ]
                    chosen_action = non_eff[0] if non_eff else available_actions[0]
                    chosen_data = None

        engine.prev_grid = curr_grid.copy()
        if hasattr(engine, "grid_history"):
            engine.grid_history.append(curr_grid.copy())
        engine.last_action = chosen_action
        engine.last_action_data = chosen_data
        engine.last_predicted_pos = predicted_pos
        return chosen_action, chosen_data

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export prefrontal deliberation engine capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="prefrontal_deliberation_engine",
            operator="==",
            value="ACTIVE",
            relation="dlpfc_vmpfc_arbitration",
            target_entity="cortex_deliberation",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering deliberation engine."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="deliberation_engine",
                node_data={
                    "faculty": "prefrontal_deliberation_engine",
                    "status": "ACTIVE",
                },
            )
        ]
