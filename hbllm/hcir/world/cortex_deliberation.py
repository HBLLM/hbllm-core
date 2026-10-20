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
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.skills.declarative import SkillEvaluationContext
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.cortex_planner import MentalSimulationStep
from hbllm.hcir.world.macro_actions import (
    MacroPlanSynthesizer,
)

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
            if any(engine.collision_cones.is_collision_hazard(ar, ac, t_f) for t_f in (1, 2)):
                return SafetyGateResult(
                    is_safe=False, reason="Stationary avatar in looming collision cone"
                )
        elif next_step.predicted_avatar_pos is not None:
            nr, nc = next_step.predicted_avatar_pos
            if engine.collision_cones.is_collision_hazard(nr, nc, 1):
                # Check if temporal wait clears the corridor
                acc_mon = getattr(engine.working_memory, "acc_conflict", None)
                if acc_mon is not None:
                    should_w, w_steps = acc_mon.should_advise_temporal_wait(
                        (ar, ac), (nr, nc), engine.collision_cones
                    )
                    if should_w:
                        return SafetyGateResult(
                            is_safe=False,
                            should_wait=True,
                            wait_steps=w_steps,
                            reason=f"looming collision cone at ({nr}, {nc}) clearing in {w_steps} steps",
                        )
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
            cargos_known = engine.symbolic_theory.cargo_features | engine.learned_cargo_features
            goals_known = set(engine.learned_goal_positions)
            live_threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                entities,
                bg=bg,
                step_size=step_size,
                avatar_pos=engine.avatar_pos,
                avatar_features=av_feats,
                goals=goals_known,
                cargo_features=cargos_known,
                barrier_features=engine.symbolic_theory.barrier_features,
                walkable_features=engine.symbolic_theory.walkable_features,
            )
            if not live_threats and not engine.mobile_threat_features:
                return SafetyGateResult(is_safe=True)

            # Oriented stationary hazard line-of-sight gate (e.g., directional sensors, optical cones)
            stationary_threats = [
                t
                for t in live_threats
                if t.feature_id not in engine.mobile_threat_features
                and t.feature_id not in cargos_known
                and t.feature_id not in engine.symbolic_theory.goal_features
                and t.feature_id not in engine.symbolic_theory.receptacle_features
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
            walkable_and_safe = engine.symbolic_theory.walkable_features
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
            gaze_range = min(3, max(1, step_size * 2))

            for step_idx, future_step in enumerate(future_steps):
                avatar_dest = future_step.predicted_avatar_pos
                if avatar_dest is None:
                    continue
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
                    turned = False
                    new_f = pat_facing

                    if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in patrol_walls:
                        # 90-degree rotational turn at corner before 180-degree reversal
                        for rot_f in [
                            (pat_facing[1], -pat_facing[0]),
                            (-pat_facing[1], pat_facing[0]),
                        ]:
                            rot_dest = (
                                pat_pos[0] + rot_f[0] * step_size,
                                pat_pos[1] + rot_f[1] * step_size,
                            )
                            if (
                                0 <= rot_dest[0] < H
                                and 0 <= rot_dest[1] < W
                                and rot_dest not in patrol_walls
                            ):
                                dest = rot_dest
                                new_f = rot_f
                                turned = True
                                break
                        if not turned:
                            new_f = (-pat_facing[0], -pat_facing[1])
                            dest = pat_pos
                    elif _blocked(dest, pat_facing):
                        for rot_f in [
                            (pat_facing[1], -pat_facing[0]),
                            (-pat_facing[1], pat_facing[0]),
                        ]:
                            rot_dest = (
                                dest[0] + rot_f[0] * step_size,
                                dest[1] + rot_f[1] * step_size,
                            )
                            if (
                                0 <= rot_dest[0] < H
                                and 0 <= rot_dest[1] < W
                                and rot_dest not in patrol_walls
                            ):
                                new_f = rot_f
                                turned = True
                                break
                        if not turned:
                            new_f = (-pat_facing[0], -pat_facing[1])

                    new_patrol_set.add((dest, new_f))

                    # 1. Direct body collision at projected patroller position
                    if dest == avatar_dest:
                        collision = True
                        break

                    # 2. Dynamic sentry gaze / sightline projection along projected heading new_f
                    if new_f != (0, 0):
                        for k in range(1, gaze_range + 1):
                            gr = dest[0] + new_f[0] * k
                            gc = dest[1] + new_f[1] * k
                            if not (0 <= gr < H and 0 <= gc < W) or (gr, gc) in patrol_walls:
                                break
                            if (gr, gc) == avatar_dest:
                                collision = True
                                break
                    if collision:
                        break

                if collision:
                    if (
                        next_step.expected_mutation == "oscillatory_return_step"
                        and future_step != next_step
                    ):
                        continue
                    # Check if next_step collision clears within 1 or 2 steps (temporal pacing window)
                    should_wait = False
                    wait_steps = 1
                    if step_idx == 0 and len(new_patrol_set) > 0:
                        # Forward probe patrollers 1 more step to check if avatar_dest becomes clear
                        next_next_patrols: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                        for p2, f2 in new_patrol_set:
                            d2 = (p2[0] + f2[0] * step_size, p2[1] + f2[1] * step_size)
                            if 0 <= d2[0] < H and 0 <= d2[1] < W and d2 not in patrol_walls:
                                next_next_patrols.add((d2, f2))
                            else:
                                next_next_patrols.add((p2, (-f2[0], -f2[1])))
                        # If destination and gaze are free on the next phase tick, advise pacing
                        is_still_hazard = any(p == avatar_dest for p, _ in next_next_patrols)
                        if not is_still_hazard:
                            should_wait = True
                            wait_steps = 1

                    return SafetyGateResult(
                        is_safe=False,
                        should_wait=should_wait,
                        wait_steps=wait_steps,
                        reason="Collision with projected sentry trajectory or gaze cone along plan",
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
        cargos_known_temp = engine.symbolic_theory.cargo_features | getattr(
            engine, "learned_cargo_features", set()
        )
        av_feats_temp = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        has_periodic = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
            and any(
                not (
                    p.hazardous_values
                    <= (cargos_known_temp | av_feats_temp | {0, getattr(engine, "bg_feature", 0)})
                )
                for p in engine.hazard_tracker.periodic_cells.values()
            )
        )
        has_temporal = (
            has_periodic
            or bool(engine.mobile_threat_features)
            or any(
                (
                    proj.feature_id in engine.mobile_threat_features
                    or proj.feature_id in engine.hazard_tracker.known_lethal_features
                )
                and proj.feature_id not in cargos_known_temp
                and proj.feature_id not in av_feats_temp
                for proj in engine.collision_cones.projected_trajectories
            )
        )
        # 0. Physical Feasibility Invariant:
        # A known failed physical transition can NEVER be safely executed from this avatar position,
        # regardless of whether it originates from an executive geodesic plan or exploratory option.
        if (
            engine.avatar_pos is not None
            and (engine.avatar_pos, next_step.action) in engine.failed_transitions
        ):
            return SafetyGateResult(
                is_safe=False,
                reason=f"Action {next_step.action} from {engine.avatar_pos} is a physically verified impassable transition",
            )

        # 1. Deliberate goal plans (deterministic geodesics, teleological subgoals) are executive-driven
        # and override stochastic exploratory refractory inhibition timers.
        is_executive_plan = (
            getattr(next_step, "is_deterministic_geodesic", False)
            or getattr(next_step, "expected_mutation", "")
            in (
                "topological_geodesic_step",
                "geodesic_path_step",
                "bilateral_convergence",
            )
            or engine.phase == EpistemicPhase.EXPLOITATION
        )
        if is_executive_plan:
            return SafetyGateResult(is_safe=True)

        # 2. Emergency escape / sole transit action:
        # If all alternative displacement actions are blocked by physical transition failures,
        # allow the sole unfailed escape route even if it has an exploratory habituation timer.
        if engine.avatar_pos is not None:
            viable_displacement_acts = [
                a
                for a in available_actions
                if (engine.avatar_pos, a) not in engine.failed_transitions
            ]
            if (
                len(viable_displacement_acts) == 1
                and viable_displacement_acts[0] == next_step.action
            ):
                return SafetyGateResult(is_safe=True)

        if not has_temporal and engine.inhibited_actions.get(next_step.action, 0) > 0:
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
                walkable_features=engine.symbolic_theory.walkable_features,
            )
            if phase_gate.should_wait:
                return SafetyGateResult(
                    is_safe=True,
                    should_wait=True,
                    wait_steps=phase_gate.wait_steps_recommended,
                    reason="Cerebellar phase clock advises wait window",
                )
        return SafetyGateResult(is_safe=True)


class AnteriorCingulateConflictPredicate(PlanSafetyPredicate):
    """Evaluates plan safety against Anterior Cingulate Cortex (ACC) failure clusters,
    frustration states, and dynamic collision wait recommendations.
    """

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        acc = getattr(engine.working_memory, "acc_conflict", None)
        if acc is None:
            return SafetyGateResult(is_safe=True)

        pred_pos = next_step.predicted_avatar_pos
        if pred_pos is not None:
            # 1. Reject paths routing directly into recurrent death clusters
            if hasattr(acc, "is_in_conflict_zone") and acc.is_in_conflict_zone(pred_pos):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"Candidate step {pred_pos} is inside an ACC recurrent failure cluster",
                )

            # 2. Check if ACC advises temporal wait due to collision cones crossing candidate pos
            curr_pos = engine.avatar_pos or (0, 0)
            collision_cones = getattr(engine, "collision_cones", None)
            if collision_cones is not None and hasattr(acc, "should_advise_temporal_wait"):
                should_wait, wait_steps = acc.should_advise_temporal_wait(
                    current_pos=curr_pos,
                    candidate_pos=pred_pos,
                    collision_cones=collision_cones,
                    max_steps=3,
                )
                if should_wait:
                    return SafetyGateResult(
                        is_safe=True,
                        should_wait=True,
                        wait_steps=wait_steps,
                        reason="ACC advises temporal wait until collision hazard clears",
                    )

        return SafetyGateResult(is_safe=True)


class CausalUndoSafetyPredicate(PlanSafetyPredicate):
    """Rejects candidate trajectories that trigger empirical undo / state reversion rollbacks."""

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        if next_step.expected_mutation in (
            "attach_cargo",
            "detach_cargo_at_goal",
            "orient_towards_cargo",
            "approach_cargo",
            "deliver_cargo",
            "interact",
        ) or next_step.action in getattr(engine.symbolic_theory, "interaction_actions", set()):
            return SafetyGateResult(is_safe=True)

        undo_acts = getattr(engine, "undo_actions", set())
        aff = engine.action_affordances.get(next_step.action)
        is_undo = next_step.action in undo_acts or (
            aff is not None and aff.name.upper() in ("UNDO", "RESET")
        )
        if is_undo and len(available_actions) > 1:
            return SafetyGateResult(
                is_safe=False,
                reason=f"Action {next_step.action} is an empirical state-reversion undo operator",
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
                AnteriorCingulateConflictPredicate(),
                CausalUndoSafetyPredicate(),
            ]
        )

    def evaluate(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        available_actions: Sequence[Any],
        future_steps: Sequence[MentalSimulationStep] | None = None,
    ) -> SafetyGateResult:
        if future_steps is None:
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
        if not res.is_safe or res.should_wait:
            logger.debug(
                "Plan safety check: is_safe=%s should_wait=%s reason=%s next_act=%s",
                res.is_safe,
                res.should_wait,
                res.reason,
                next_step.action,
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

        # 2. Declarative Neuro-Symbolic Skill Arbitration via PrefrontalWorkingMemory Procedural Library
        ctx = SkillEvaluationContext(
            grid=curr_grid,
            available_actions=list(available_actions),
            metadata={"engine": engine, "known_av_feats": known_av_feats},
        )
        for skill_name, skill in engine.working_memory.procedural_library.items():
            if skill.can_handle(
                curr_grid,
                list(available_actions),
                metadata={"engine": engine, "known_av_feats": known_av_feats},
            ):
                res = skill.solve(ctx)
                if res:
                    first = res[0]
                    if isinstance(first, tuple):
                        return first[0], first[1]
                    return first, None

        return None

    def _execute_temporal_pacing(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        next_step: MentalSimulationStep,
        wait_steps: int,
    ) -> tuple[Any, dict[str, Any] | None, tuple[int, int] | None]:
        """Cerebellar / Premotor oscillatory temporal pacing:
        In environments with strictly directional actions (no explicit wait),
        synthesize an oscillatory burn step (e.g. step back / lateral dodge -> return)
        to burn phase steps until dynamic hazards clear.
        """
        engine.consecutive_pacing_steps += 1
        if engine.consecutive_pacing_steps > 2:
            engine.consecutive_pacing_steps = 0
            engine.mental_plan.clear()
            engine.transition_to_phase(
                EpistemicPhase.REPLANNING,
                reason="pacing_window_exhausted",
            )
            act, data = engine.plan_epistemic_probe(curr_grid, available_actions)
            return act, data, None

        engine.mental_plan.appendleft(next_step)
        undo_acts = getattr(engine, "undo_actions", set())
        calibrated_waits = [
            a
            for a in available_actions
            if hasattr(engine, "is_calibrated_wait_action")
            and engine.is_calibrated_wait_action(a)
            and a not in undo_acts
            and engine.inhibited_actions.get(a, 0) == 0
        ]
        curr_pos = engine.avatar_pos or getattr(engine, "current_actor_pos", None)
        if calibrated_waits:
            engine.pending_phase_wait_steps = wait_steps
            return calibrated_waits[0], None, curr_pos

        engine.pending_phase_wait_steps = 0

        H, W = curr_grid.shape
        reverse_map: dict[Any, Any] = {}
        for a1, d1 in engine.action_deltas.items():
            for a2, d2 in engine.action_deltas.items():
                if d1[0] == -d2[0] and d1[1] == -d2[1] and (d1[0] != 0 or d1[1] != 0):
                    reverse_map[a1] = a2
        if not reverse_map:
            reverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
        pacing_candidates: list[tuple[Any, tuple[int, int]]] = []

        if curr_pos is not None:
            bg = engine.estimate_background(curr_grid)
            walkable_and_safe = engine.symbolic_theory.walkable_features
            other_walkables = {f for f in walkable_and_safe if f != bg and f != 0}
            bg_is_impassable_void = len(other_walkables) > 0 and bg not in walkable_and_safe
            if not bg_is_impassable_void:
                bg_count = int(np.count_nonzero(curr_grid == bg))
                bg_ratio = bg_count / max(1, H * W)
                ar, ac = curr_pos
                av_feats = engine.avatar_features or (
                    {engine.avatar_feature} if engine.avatar_feature is not None else set()
                )
                if av_feats:
                    avatar_cells = [
                        (int(r), int(c)) for r, c in np.argwhere(np.isin(curr_grid, list(av_feats)))
                    ]
                    avatar_on_bg = any(int(curr_grid[r, c]) == bg for r, c in avatar_cells)
                else:
                    avatar_on_bg = 0 <= ar < H and 0 <= ac < W and int(curr_grid[ar, ac]) == bg
                if bg_ratio >= 0.20 and not avatar_on_bg:
                    bg_is_impassable_void = True

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
                val = int(curr_grid[nr, nc])
                if (
                    engine.symbolic_theory.is_barrier(val)
                    or val in engine.symbolic_theory.barrier_features
                    or val in engine.hazard_tracker.known_lethal_features
                    or (bg_is_impassable_void and val == bg)
                ):
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
                    future_steps=[cand_step],
                )
                if cand_res.is_safe and not cand_res.should_wait:
                    pacing_candidates.append((act, (nr, nc)))

        if pacing_candidates:
            # Prefer reverse of recent action to oscillate along existing safe path
            chosen_pacing_act, chosen_pacing_pos = pacing_candidates[0]
            if len(engine.recent_actions) > 0:
                last_act = engine.recent_actions[-1]
                rev_last = reverse_map.get(last_act)
                for act, pos in pacing_candidates:
                    if act == rev_last:
                        chosen_pacing_act, chosen_pacing_pos = act, pos
                        break

            rev_act = reverse_map.get(chosen_pacing_act)
            # Enqueue return step to restore spatial trajectory once hazard passes
            if rev_act is not None and rev_act in available_actions and curr_pos is not None:
                engine.mental_plan.appendleft(
                    MentalSimulationStep(
                        action=rev_act,
                        predicted_avatar_pos=curr_pos,
                        expected_mutation="oscillatory_return_step",
                    )
                )
            return chosen_pacing_act, None, chosen_pacing_pos

        # Invariant fallback: Invalidate plan and replan if no safe pacing candidate exists
        engine.mental_plan.clear()
        engine.phase = EpistemicPhase.REPLANNING
        act, data = engine.plan_epistemic_probe(curr_grid, available_actions)
        return act, data, None

    def _detect_and_resolve_acc_conflict(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> Any | None:
        """Detect limit-cycle oscillations and select orthogonal perturbation action."""
        engine.acc_conflict_monitor.record_step(engine.last_action, engine.avatar_pos)
        is_osc, cyclic_acts = engine.acc_conflict_monitor.detect_oscillation()
        if not is_osc:
            return None

        # In cognitive architecture, an intentional temporal wait/dodge maneuver
        # planned by the prefrontal cortex must not be aborted as an unintentional deadlock
        # if the active mental plan progresses to novel coordinates beyond the cyclic positions.
        if engine.phase == EpistemicPhase.EXPLOITATION and engine.mental_plan:
            pos_hist = list(engine.acc_conflict_monitor.position_history)
            if len(pos_hist) >= 4:
                cyclic_positions = set(pos_hist[-4:])
                is_true_bounce = (
                    len(cyclic_positions) <= 2
                    and pos_hist[-1] == pos_hist[-3]
                    and pos_hist[-2] == pos_hist[-4]
                )
                if not is_true_bounce:
                    future_plan_positions = {
                        s.predicted_avatar_pos
                        for s in engine.mental_plan
                        if s.predicted_avatar_pos is not None
                    }
                    if any(p not in cyclic_positions for p in future_plan_positions):
                        return None

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
                    return cand_a
            else:
                return cand_a
        return None

    def _resolve_cluster_deadlock(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> Any | None:
        """Resolve spatial cluster or motor stagnation deadlocks when avatar is confined over 30 steps."""
        if engine.avatar_pos is None or len(engine.recent_positions) < 30:
            return None
        H, W = curr_grid.shape
        if H <= 5 and W <= 5:
            return None

        recent_cluster = list(engine.recent_positions)[-30:]
        min_r = min(p[0] for p in recent_cluster)
        max_r = max(p[0] for p in recent_cluster)
        min_c = min(p[1] for p in recent_cluster)
        max_c = max(p[1] for p in recent_cluster)
        span_r, span_c = max_r - min_r, max_c - min_c
        if span_r > 3 or span_c > 3:
            return None

        logger.warning(
            "Metacognitive Executive Control: Spatial cluster deadlock detected in %dx%d area over 30 steps. Triggering breakoff.",
            span_r + 1,
            span_c + 1,
        )
        if engine.mental_plan:
            engine.mental_plan.clear()
            engine.current_simulated_goal = None
        engine.recent_positions.clear()

        # Try frontier escape via calibrated displacement actions
        disp_acts = [
            (a, engine.action_dynamics[a].get_displacement())
            for a in available_actions
            if a in engine.action_dynamics and engine.action_dynamics[a].is_displacement_action()
        ]
        esc_act: Any | None = None
        if disp_acts and engine.avatar_pos is not None:
            cluster_center_r = (min_r + max_r) / 2.0
            cluster_center_c = (min_c + max_c) / 2.0
            best_cand: Any | None = None
            best_cand_score = -float("inf")
            for cand_a, (dr, dc) in disp_acts:
                nr = engine.avatar_pos[0] + dr
                nc = engine.avatar_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                dest_feat = int(curr_grid[nr, nc])
                if dest_feat in engine.hazard_tracker.known_lethal_features:
                    continue
                if engine.collision_cones.is_collision_hazard(nr, nc, 1):
                    continue
                dist_from_center = abs(nr - cluster_center_r) + abs(nc - cluster_center_c)
                pos_visits = engine.position_visit_counts.get((nr, nc), 0)
                is_failed = (engine.avatar_pos, cand_a) in engine.failed_transitions
                fail_penalty = 50.0 if is_failed else 0.0
                score = (dist_from_center * 10.0) - (pos_visits * 2.0) - fail_penalty
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand = cand_a
            esc_act = best_cand

        if esc_act is not None:
            engine.inhibited_actions.pop(esc_act, None)
            recent_acts = list(engine.recent_actions)[-10:]
            for act_used in set(recent_acts):
                if act_used != esc_act:
                    engine.inhibited_actions[act_used] = max(
                        engine.inhibited_actions.get(act_used, 0), 10
                    )
            return esc_act

        # Non-displacement / uncalibrated actions
        recent_acts = list(engine.recent_actions)[-10:]
        cluster_act_counts = Counter(recent_acts)
        stagnant_act = cluster_act_counts.most_common(1)[0][0] if cluster_act_counts else None
        if stagnant_act is not None:
            engine.inhibited_actions[stagnant_act] = max(
                engine.inhibited_actions.get(stagnant_act, 0), 8
            )
        undo_acts = set(getattr(engine, "undo_actions", set()))
        if hasattr(engine, "symbolic_theory") and hasattr(engine.symbolic_theory, "undo_actions"):
            undo_acts.update(engine.symbolic_theory.undo_actions)
        cand_orthogonal = [
            a
            for a in available_actions
            if a != stagnant_act and a not in undo_acts and engine.inhibited_actions.get(a, 0) == 0
        ]
        if cand_orthogonal:
            recent_list = list(engine.recent_actions)

            def _orth_key(a: Any) -> int:
                try:
                    return recent_list.index(a)
                except ValueError:
                    return -1

            cand_orthogonal.sort(key=_orth_key)
            chosen_orth = cand_orthogonal[0]
            engine.inhibited_actions.pop(chosen_orth, None)
            logger.info(
                "Anterior Cingulate Cortex (ACC): Resolving non-displacement cluster deadlock by inhibiting %s and selecting orthogonal perturbation action %s",
                stagnant_act,
                chosen_orth,
            )
            return chosen_orth
        return None

    def _deliberate_master_plan(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        blocked_cells: set[tuple[int, int]] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None, tuple[int, int] | None]:
        """Deliberate a new master plan via mental simulation -> macro option -> epistemic probe."""
        chosen_action: Any = None
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None

        # Multi-agent focus switch resolver
        focus_switch = next(
            (
                a
                for a in available_actions
                if (engine.is_focus_switch_action(a) or a in engine.focus_switch_actions)
                and engine.inhibited_actions.get(a, 0) == 0
                and not engine.working_memory.habenular_ior.is_action_inhibited(
                    engine.avatar_pos, a, None
                )
            ),
            None,
        )
        has_multi_agent = (
            getattr(engine, "bilateral_state", None) is not None
            or getattr(engine, "prior_avatar_feature", None) is not None
            or len(engine.working_memory.focus_switch_actions) > 0
            or len(engine.focus_switch_actions) > 0
        )
        if focus_switch is None and has_multi_agent:
            undo_acts = getattr(engine, "undo_actions", set())
            focus_switch = next(
                (
                    a
                    for a in available_actions
                    if not engine.is_displacement_action(a)
                    and not engine.is_spatial_effector(a)
                    and a not in undo_acts
                    and engine.inhibited_actions.get(a, 0) == 0
                    and not engine.working_memory.habenular_ior.is_action_inhibited(
                        engine.avatar_pos, a, None
                    )
                ),
                None,
            )

        is_at_terminal_goal = bool(
            engine.avatar_pos is not None
            and (
                engine.avatar_pos in engine.goals
                or engine.avatar_pos in getattr(engine, "learned_goal_positions", set())
                or (
                    hasattr(engine.working_memory, "target_constellations")
                    and any(
                        (
                            abs(engine.avatar_pos[0] - pos[0]) <= 1
                            and abs(engine.avatar_pos[1] - pos[1]) <= 1
                        )
                        for f, pos in engine.working_memory.target_constellations.items()
                        if (
                            f in engine.avatar_features
                            if engine.avatar_features
                            else f == engine.avatar_feature
                        )
                        and pos is not None
                    )
                )
            )
        )

        consecutive_focus_switches = getattr(engine.working_memory, "consecutive_focus_switches", 0)
        focus_switch_allowed = engine.last_action != focus_switch or consecutive_focus_switches < 4

        # 3A. Multi-Agent Terminal Goal Focus Switch:
        # If the active agent is already at its terminal goal, do not execute excursions away from the goal.
        # Immediately hand over control via focus switch.
        if (
            chosen_action is None
            and focus_switch is not None
            and has_multi_agent
            and focus_switch_allowed
            and is_at_terminal_goal
        ):
            logger.info(
                "PrefrontalDeliberationEngine: Intentional Multi-Agent Focus Switch at Terminal Goal via action %s (pos=%s)",
                focus_switch,
                engine.avatar_pos,
            )
            chosen_action = focus_switch
            chosen_data = None
        # 3B1. Discrete Macro-Action Synthesis (Options Framework)
        if chosen_action is None and engine.is_motor_grounded() and not blocked_cells:
            synth_plan = MacroPlanSynthesizer.synthesize_plan(
                engine, curr_grid, available_actions, objects=objects, micro_actions=micro_actions
            )
            if synth_plan is not None and not synth_plan.is_empty():
                engine.working_memory.macro_plan = synth_plan
                step_res = engine.working_memory.macro_plan.step(
                    engine, curr_grid, available_actions
                )
                if step_res is not None:
                    chosen_action, chosen_data = step_res
                    engine.phase = EpistemicPhase.EXPLOITATION
                    return chosen_action, chosen_data, None

        # 3B. Primary: Forward Mental Simulation over learned dynamics & verified exits
        # (Executive Directive 1 & 2: Check if exit is open, if open go through it)
        simulated_plan = None
        if (
            chosen_action is None
            and engine.is_motor_grounded()
            and (engine.simulation_cooldown == 0 or blocked_cells)
        ):
            simulated_plan = engine.simulate_in_mind(
                curr_grid, available_actions, blocked_cells=blocked_cells
            )
            if simulated_plan is not None and len(simulated_plan) == 0:
                engine.consecutive_simulation_failures = 0
                engine.simulation_cooldown = 0
                # Active agent has reached its goal: if multi-agent focus switch is available, trigger handover
                if focus_switch is not None and has_multi_agent and focus_switch_allowed:
                    logger.info(
                        "PrefrontalDeliberationEngine: Multi-agent active avatar reached goal (0 steps). Switching control via %s",
                        focus_switch,
                    )
                    chosen_action = focus_switch
                    chosen_data = None
            elif simulated_plan:
                engine.consecutive_simulation_failures = 0
                engine.simulation_cooldown = 0
                engine.phase = EpistemicPhase.EXPLOITATION
                engine.mental_plan = deque(simulated_plan)
                next_step = engine.mental_plan.popleft()
                plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                    engine, curr_grid, available_actions, next_step
                )
                if should_wait:
                    chosen_action, chosen_data, predicted_pos = self._execute_temporal_pacing(
                        engine, curr_grid, available_actions, next_step, wait_steps
                    )
                elif (
                    plan_safe
                    and next_step.action in available_actions
                    and (
                        getattr(next_step, "is_deterministic_geodesic", False)
                        or getattr(next_step, "expected_mutation", "")
                        in (
                            "topological_geodesic_step",
                            "geodesic_path_step",
                            "bilateral_convergence",
                        )
                        or not engine.working_memory.habenular_ior.is_action_inhibited(
                            engine.avatar_pos, next_step.action, next_step.action_data
                        )
                    )
                ):
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    engine.mental_plan.clear()
                    engine.phase = EpistemicPhase.REPLANNING
                    engine.consecutive_simulation_failures += 1
                    engine.simulation_cooldown = min(
                        12, 2 ** min(engine.consecutive_simulation_failures, 4)
                    )
                    engine.working_memory.update_salience_baseline(engine)
            else:
                engine.consecutive_simulation_failures += 1
                engine.simulation_cooldown = min(
                    12, 2 ** min(engine.consecutive_simulation_failures, 4)
                )
                engine.working_memory.update_salience_baseline(engine)

        # 3C. Secondary: Object-Centric Macro Action Planning (Directives 3, 4, 5)
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
            else:
                if (
                    focus_switch is not None
                    and has_multi_agent
                    and focus_switch_allowed
                    and (
                        engine.consecutive_simulation_failures >= 1
                        or getattr(engine, "is_avatar_immobile", False)
                    )
                ):
                    logger.info(
                        "PrefrontalDeliberationEngine: Intentional Multi-Agent Focus Switch via action %s (active agent at %s failed=%d)",
                        focus_switch,
                        engine.avatar_pos,
                        engine.consecutive_simulation_failures,
                    )
                    chosen_action = focus_switch
                    chosen_data = None
                    engine.consecutive_simulation_failures = 0
                else:
                    if engine.exploration_cooldown > 0:
                        engine.exploration_cooldown -= 1
                    engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    chosen_action, chosen_data = engine.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        if focus_switch is not None and chosen_action == focus_switch:
            engine.working_memory.consecutive_focus_switches = consecutive_focus_switches + 1
        else:
            engine.working_memory.consecutive_focus_switches = 0

        return chosen_action, chosen_data, predicted_pos

    def decide(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action_schemas: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
        macro_actions: Sequence[Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision function:
        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        """
        engine.step_counter += 1
        if action_schemas:
            engine.register_action_space(action_schemas)
        if objects or micro_actions:
            from hbllm.hcir.world.macro_actions import MacroPlanSynthesizer

            MacroPlanSynthesizer.ingest_knowledge(
                engine, objects=objects, micro_actions=micro_actions
            )
        if macro_actions:
            engine.working_memory.macro_plan.extend(macro_actions)

        curr_grid = engine.normalize_sensory_input(curr_grid)

        # Metacognitive Urgency Modulation:
        # Under acute step countdown / strict budget constraints, suppress exploratory
        # refractory cooldowns to prioritize immediate exploitation toward known goals.
        if engine.is_under_strict_step_budget():
            engine.exploration_cooldown = 0
            engine.simulation_cooldown = 0

        # 1. Update background & assimilate sensory feedback from previous action
        old_bg = engine.bg_feature
        engine.bg_feature = engine.estimate_background(curr_grid)
        if engine.bg_feature != old_bg:
            if (
                old_bg == 0
                and 0 in engine.symbolic_theory.walkable_features
                and not np.any(curr_grid == 0)
            ):
                engine.symbolic_theory.walkable_features.discard(0)
            if engine.bg_feature in engine.verified_safe_features:
                engine.symbolic_theory.induce_walkable(engine.bg_feature)
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

        # Phasic Salience Reset: Environmental discovery, dynamic threat emergence, or high surprise
        # Query prefrontal working memory for salience shift to awaken deliberate forward search
        if engine.working_memory.has_salience_shift(engine):
            engine.simulation_cooldown = 0
            engine.consecutive_simulation_failures = 0
            if not (engine.phase == EpistemicPhase.EXPLOITATION and engine.mental_plan):
                engine.mental_plan.clear()
                engine.phase = (
                    EpistemicPhase.MENTAL_SIMULATION
                    if engine.is_motor_grounded()
                    else EpistemicPhase.EPISTEMIC_EXPLORATION
                )

        cargos_known_temp2 = engine.symbolic_theory.cargo_features | getattr(
            engine, "learned_cargo_features", set()
        )
        av_feats_temp2 = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        has_periodic2 = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
            and any(
                not (
                    p.hazardous_values
                    <= (cargos_known_temp2 | av_feats_temp2 | {0, getattr(engine, "bg_feature", 0)})
                )
                for p in engine.hazard_tracker.periodic_cells.values()
            )
        )
        has_temporal = (
            has_periodic2
            or bool(engine.mobile_threat_features)
            or any(
                (
                    proj.feature_id in engine.mobile_threat_features
                    or proj.feature_id in engine.hazard_tracker.known_lethal_features
                )
                and proj.feature_id not in cargos_known_temp2
                and proj.feature_id not in av_feats_temp2
                for proj in engine.collision_cones.projected_trajectories
            )
        )

        # Metacognitive Refractory Inhibition: decay refractory timers

        if engine.simulation_cooldown > 0:
            engine.simulation_cooldown -= 1
        to_uninhibited = [act for act, timer in engine.inhibited_actions.items() if timer <= 1]
        for act in engine.inhibited_actions:
            engine.inhibited_actions[act] -= 1
        for act in to_uninhibited:
            engine.inhibited_actions.pop(act, None)

        # Track recent avatar position and visit counts on every decision step
        H, W = curr_grid.shape
        if engine.avatar_pos is not None:
            engine.recent_positions.append(engine.avatar_pos)
            engine.position_visit_counts[engine.avatar_pos] = (
                engine.position_visit_counts.get(engine.avatar_pos, 0) + 1
            )

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
                    k = getattr(
                        getattr(engine, "symbolic_theory", None),
                        "get_cycle_period",
                        lambda a, default=4: default,
                    )(osc_act, default=4)
                    if (
                        osc_act in engine.value_mutator_actions
                        and getattr(engine.working_memory, "dial_mutation_streak", 0) < k
                    ):
                        continue
                    engine.inhibited_actions[osc_act] = max(
                        engine.inhibited_actions.get(osc_act, 0), 6
                    )

        # Faculty: Anterior Cingulate Cortex (ACC) Limit Cycle & Conflict Suppression
        forced_acc_perturbation = self._detect_and_resolve_acc_conflict(
            engine, curr_grid, available_actions
        )
        if forced_acc_perturbation is None:
            forced_acc_perturbation = self._resolve_cluster_deadlock(
                engine, curr_grid, available_actions
            )

        # Visual avatar localization
        known_av_feats = set(engine.avatar_features) if engine.avatar_features else set()
        if engine.avatar_feature is not None:
            known_av_feats.add(engine.avatar_feature)
        elif engine.prior_avatar_feature is not None and not engine.is_focus_switch_action(
            engine.last_action
        ):
            known_av_feats.add(engine.prior_avatar_feature)
        if known_av_feats:
            engine._update_avatar_position_from_grid(curr_grid, known_av_feats)
            if (
                engine.avatar_pos is not None
                and engine.avatar_feature is None
                and engine.prior_avatar_feature is not None
                and not engine.is_focus_switch_action(engine.last_action)
            ):
                engine.avatar_feature = engine.prior_avatar_feature
                engine.avatar_features.add(engine.prior_avatar_feature)

        # Faculty: Active Manipulandum Localization
        if engine.avatar_pos is None and engine.working_memory.body_schema.has_manipulandum():
            active_manip = engine.working_memory.body_schema.active_manipulandum
            if active_manip is not None:
                engine.avatar_pos = active_manip.pos

        # Observe creature motion
        engine._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        # Faculty E: Basal Ganglia Motor Gating & Phase Entrainment
        if engine.pending_phase_wait_steps > 0:
            engine.pending_phase_wait_steps -= 1
            undo_acts = getattr(engine, "undo_actions", set())
            calibrated_waits = [
                a
                for a in available_actions
                if hasattr(engine, "is_calibrated_wait_action")
                and engine.is_calibrated_wait_action(a)
                and a not in undo_acts
                and engine.inhibited_actions.get(a, 0) == 0
            ]
            if calibrated_waits:
                return calibrated_waits[0], None
            engine.pending_phase_wait_steps = 0

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
        affordance_choice: tuple[Any, dict[str, Any] | None] | None = None

        # 0. ACC Forced Perturbation: break limit-cycle oscillation immediately
        if forced_acc_perturbation is not None:
            logger.info(
                "Anterior Cingulate Cortex (ACC): Breaking limit cycle oscillation with orthogonal perturbation action %s",
                forced_acc_perturbation,
            )
            engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action = forced_acc_perturbation
            chosen_data = None

        # 1. Macro-Action Plan Execution (Discrete Operators: NavigateTo, InteractWith, PickUpCargo, etc.)
        elif (
            hasattr(engine.working_memory, "macro_plan")
            and not engine.working_memory.macro_plan.is_empty()
            and (
                macro_step := engine.working_memory.macro_plan.step(
                    engine, curr_grid, available_actions
                )
            )
            is not None
        ):
            chosen_action, chosen_data = macro_step
            engine.phase = EpistemicPhase.EXPLOITATION

        # 2. Check active mental plan during EXPLOITATION phase
        elif engine.phase == EpistemicPhase.EXPLOITATION and engine.mental_plan:
            # Check for physical plan divergence: avatar failed to reach predicted location
            diverged = False
            if (
                engine.is_focus_switch_action(engine.last_action)
                or engine.last_action in getattr(engine, "focus_switch_actions", set())
                or engine.last_action
                in getattr(getattr(engine, "working_memory", None), "focus_switch_actions", set())
            ):
                engine.last_predicted_pos = None

            if engine.last_predicted_pos is not None and engine.avatar_pos is not None:
                dr_err = abs(engine.avatar_pos[0] - engine.last_predicted_pos[0])
                dc_err = abs(engine.avatar_pos[1] - engine.last_predicted_pos[1])
                stride = max(1, getattr(engine, "lattice_stride", 1) or 1)
                av_sz = getattr(engine, "avatar_size", 1) or 1
                max_tol = max(
                    1 if stride > 2 else 0,
                    int(round(np.sqrt(av_sz) / 2)) if av_sz > 1 else 0,
                )
                if dr_err > max_tol or dc_err > max_tol:
                    diverged = True

            if diverged:
                engine.consecutive_pacing_steps = 0
                engine.mental_plan.clear()
                engine.transition_to_phase(
                    EpistemicPhase.REPLANNING,
                    reason="plan_divergence_from_predicted_avatar_pos",
                )
                chosen_action, chosen_data, predicted_pos = self._deliberate_master_plan(
                    engine, curr_grid, available_actions
                )
            else:
                next_step = engine.mental_plan.popleft()

                plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                    engine, curr_grid, available_actions, next_step
                )

                if should_wait:
                    chosen_action, chosen_data, predicted_pos = self._execute_temporal_pacing(
                        engine, curr_grid, available_actions, next_step, wait_steps
                    )
                elif not plan_safe:
                    engine.consecutive_pacing_steps = 0
                    engine.mental_plan.clear()

                    engine.phase = EpistemicPhase.REPLANNING
                    danger_cells = (
                        {next_step.predicted_avatar_pos}
                        if next_step.predicted_avatar_pos is not None and not has_temporal
                        else set()
                    )
                    chosen_action, chosen_data, predicted_pos = self._deliberate_master_plan(
                        engine, curr_grid, available_actions, blocked_cells=danger_cells
                    )
                elif next_step.action in available_actions and (
                    getattr(next_step, "is_deterministic_geodesic", False)
                    or getattr(next_step, "expected_mutation", "")
                    in (
                        "topological_geodesic_step",
                        "geodesic_path_step",
                        "bilateral_convergence",
                    )
                    or not engine.working_memory.habenular_ior.is_action_inhibited(
                        engine.avatar_pos, next_step.action, next_step.action_data
                    )
                ):
                    engine.consecutive_pacing_steps = 0
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos

                else:
                    engine.mental_plan.clear()
                    engine.phase = EpistemicPhase.REPLANNING
                    chosen_action, chosen_data, predicted_pos = self._deliberate_master_plan(
                        engine, curr_grid, available_actions
                    )

        # 2. Prefrontal Affordance Panel & Optical Symmetry Gating
        elif (
            affordance_choice := self.arbitrate_affordances(
                engine, curr_grid, available_actions, known_av_feats
            )
        ) is not None:
            chosen_action, chosen_data = affordance_choice
            if not engine.is_spatial_effector(chosen_action):
                engine.inhibited_actions.pop(chosen_action, None)

        # 3. Master Planning Branch (when no active plan is in flight)
        else:
            chosen_action, chosen_data, predicted_pos = self._deliberate_master_plan(
                engine,
                curr_grid,
                available_actions,
                objects=objects,
                micro_actions=micro_actions,
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

        # Compute principled epistemic confidence based on decision source & motor dynamics
        base_dyn_conf = 0.95
        if chosen_action in engine.action_dynamics:
            base_dyn_conf = float(engine.action_dynamics[chosen_action].confidence)

        if engine.phase == EpistemicPhase.EXPLOITATION:
            # Deterministic plan rollout towards verified goal/exit
            last_conf = min(0.98, max(0.70, base_dyn_conf))
        elif affordance_choice is not None:
            # Procedural declarative neuro-symbolic skill execution (dials, effectors, focus switches)
            last_conf = min(0.90, max(0.75, base_dyn_conf * 0.90))
        elif getattr(engine.object_planner, "active_target_object_id", None) is not None:
            # Macro option (Directive 4 known tool vs Directive 5 curious probe)
            if any(
                chosen_action in affs
                for affs in getattr(engine.symbolic_theory, "tool_barrier_affinities", {}).values()
            ):
                last_conf = 0.85
            else:
                last_conf = 0.60
        elif forced_acc_perturbation is not None:
            # ACC deadlock breakoff
            last_conf = 0.25
        else:
            # Epistemic probe / frontier curiosity exploration
            last_conf = max(0.20, min(0.50, base_dyn_conf * 0.5))

        engine.last_action_confidence = float(last_conf)
        engine.prev_grid = curr_grid.copy()
        if hasattr(engine, "grid_history"):
            engine.grid_history.append(curr_grid.copy())
        engine.last_action = chosen_action
        engine.last_action_data = chosen_data
        engine.last_predicted_pos = predicted_pos
        if engine.is_spatial_effector(chosen_action) and isinstance(chosen_data, dict):
            raw_r = chosen_data.get("y", chosen_data.get("r", chosen_data.get("row")))
            raw_c = chosen_data.get("x", chosen_data.get("c", chosen_data.get("col")))
            if raw_r is not None and raw_c is not None:
                if hasattr(engine, "working_memory") and hasattr(
                    engine.working_memory, "record_effector_coord"
                ):
                    engine.working_memory.record_effector_coord((int(raw_r), int(raw_c)))
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
