"""Prefrontal Cortex Safety Gating Predicates (Stage 4 Active Inference)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.cortex_planner import MentalSimulationStep
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.engine_state import get_engine_state_attr

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


from hbllm.hcir.world.cortex_deliberation import get_sub_attr


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
        view: Any,
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
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        H, W = curr_grid.shape
        for future_step in future_steps:
            pos = future_step.predicted_avatar_pos
            if (
                not isinstance(pos, tuple)
                or len(pos) < 2
                or not isinstance(pos[0], (int, float))
                or not isinstance(pos[1], (int, float))
            ):
                continue
            fr, fc = int(pos[0]), int(pos[1])
            if 0 <= fr < H and 0 <= fc < W:
                feat = int(curr_grid[fr, fc])
                if feat in view.mobile_threat_features:
                    continue
                ev = view.symbolic_theory.evaluate_transition(feat, locus=(fr, fc))
                if feat in view.hazard_tracker.known_lethal_features or ev.risk_cost >= 40.0:
                    return SafetyGateResult(
                        is_safe=False, reason=f"Lethal feature {feat} at ({fr}, {fc})"
                    )
        return SafetyGateResult(is_safe=True)


class DynamicPatrolTrajectoryPredicate(PlanSafetyPredicate):
    """Simulates dynamic threat patrollers forward in time to prevent kinetic collisions."""

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        try:
            H, W = curr_grid.shape
            bg = view.estimate_background(curr_grid)
            step_size = view.infer_motor_step_size(available_actions)
            entities = view.extract_entities(curr_grid, bg)
            av_feats = view.avatar_features or (
                {view.avatar_feature} if view.avatar_feature is not None else set()
            )
            live_threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                entities,
                bg=bg,
                step_size=step_size,
                avatar_pos=view.avatar_pos,
                avatar_features=av_feats,
            )
            cargo_feats = set(view.symbolic_theory.cargo_features) | set(
                view.learned_cargo_features
            )
            mob_avoid = view.hazard_tracker.get_mobile_avoidance_cells(
                curr_grid,
                set(view.symbolic_theory.barrier_features),
                bg,
                cargo_features=cargo_feats,
                avatar_features=av_feats,
            )
            if (
                next_step.predicted_avatar_pos is not None
                and next_step.predicted_avatar_pos in mob_avoid
            ):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"Mobile entity collision zone at {next_step.predicted_avatar_pos}",
                )
            if not live_threats and not view.mobile_threat_features:
                return SafetyGateResult(is_safe=True)

            # Oriented stationary hazard line-of-sight gate (e.g., directional sensors, optical cones)
            stationary_threats = [
                t for t in live_threats if t.feature_id not in view.mobile_threat_features
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
                if next_step.predicted_avatar_pos == st.pos and view.avatar_pos == st.gaze_pos:
                    return SafetyGateResult(
                        is_safe=False,
                        reason="Frontal collision with oriented active hazard along line-of-sight",
                    )

            live_patrols: set[tuple[tuple[int, int], tuple[int, int]]] = {
                (t.pos, t.facing)
                for t in live_threats
                if (
                    t.feature_id in view.mobile_threat_features
                    or any(
                        e.properties.get("dynamic_hazard")
                        for e in entities
                        if e.feature_id == t.feature_id
                    )
                )
            }

            if not live_patrols:
                return SafetyGateResult(is_safe=True)

            half_step = max(1, step_size // 2)
            walkable_and_safe = view.symbolic_theory.walkable_features
            bg_is_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
            patrol_walls: set[tuple[int, int]] = set(view.learned_barriers)
            kb = view.symbolic_theory.transition_kb
            for rr in range(H):
                for cc in range(W):
                    v = int(curr_grid[rr, cc])
                    ev = view.symbolic_theory.evaluate_transition(v, locus=(rr, cc))
                    if not ev.traversable or ev.risk_cost >= 40.0 or (bg_is_void and v == bg):
                        patrol_walls.add((rr, cc))
                    elif (
                        kb is not None
                        and kb.get_belief((rr, cc), scope="locus") is not None
                        and ev.traversable
                        and ev.confidence > 0.0
                    ):
                        patrol_walls.discard((rr, cc))

            def _blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
                probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
                dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                for q in (probe, dest):
                    if not (0 <= q[0] < H and 0 <= q[1] < W) or q in patrol_walls:
                        return True
                return False

            sim_patrols = frozenset(live_patrols)
            cur_av_pos = view.avatar_pos
            for future_step in future_steps:
                avatar_dest = future_step.predicted_avatar_pos
                new_patrol_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                collision = False
                for pat_pos, pat_facing in sim_patrols:
                    dest = (
                        pat_pos[0] + pat_facing[0] * step_size,
                        pat_pos[1] + pat_facing[1] * step_size,
                    )
                    if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in patrol_walls:
                        new_f = (-pat_facing[0], -pat_facing[1])
                        new_patrol_set.add((pat_pos, new_f))
                        if pat_pos == avatar_dest:
                            collision = True
                            break
                        continue
                    if dest == avatar_dest:
                        collision = True
                        break
                    # Physical head-on swap collision: entities cross each other on the corridor edge
                    if cur_av_pos is not None and pat_pos == avatar_dest and dest == cur_av_pos:
                        collision = True
                        break
                    new_f = pat_facing
                    if _blocked(dest, pat_facing):
                        new_f = (-pat_facing[0], -pat_facing[1])
                    new_patrol_set.add((dest, new_f))
                if collision:
                    if (
                        next_step.expected_mutation == "oscillatory_return_step"
                        and future_step != next_step
                    ):
                        continue
                    if future_step == next_step:
                        return SafetyGateResult(
                            is_safe=False,
                            reason="Immediate collision with simulated patrol threat: space-time replanning required",
                        )
                    # Distal collision along trajectory: the immediate next_step does not collide with the threat.
                    # Allow next_step to proceed toward the corridor while re-evaluating each step online.
                    continue
                sim_patrols = frozenset(new_patrol_set)
                cur_av_pos = avatar_dest

        except Exception as e:
            logger.debug("Dynamic patrol trajectory simulation error: %s", e)

        return SafetyGateResult(is_safe=True)


class HabenularInhibitionPredicate(PlanSafetyPredicate):
    """Inhibits repetitive looping motor actions via Lateral Habenula / Basal Ganglia IOR."""

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        has_temporal = bool(view.hazard_tracker.periodic_cells) or bool(view.mobile_threat_features)
        is_action_trigger = getattr(next_step, "expected_mutation", "") in (
            "ACTION_TRIGGER",
            "INTERACT_TRIGGER",
        )
        is_failed_transition = (view.avatar_pos, next_step.action) in getattr(
            view, "failed_transitions", set()
        )
        if (
            not has_temporal
            and not is_action_trigger
            and is_failed_transition
            and view.inhibited_actions.get(next_step.action, 0) > 0
        ):
            return SafetyGateResult(
                is_safe=False, reason=f"Action {next_step.action} in refractory inhibition"
            )
        if is_failed_transition and view.working_memory.habenular_ior.is_action_inhibited(
            view.avatar_pos, next_step.action
        ):
            return SafetyGateResult(is_safe=False, reason="Action inhibited by Habenular IOR")
        return SafetyGateResult(is_safe=True)


class AnteriorCingulateConflictPredicate(PlanSafetyPredicate):
    """Evaluates plan safety against Anterior Cingulate Cortex (ACC) failure clusters,
    frustration states, and dynamic collision wait recommendations.
    """

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        acc = get_sub_attr(view.working_memory, "acc_conflict", None)
        if acc is None:
            return SafetyGateResult(is_safe=True)

        pred_pos = next_step.predicted_avatar_pos
        if pred_pos is not None:
            # Reject paths routing directly into recurrent death clusters
            if hasattr(acc, "is_in_conflict_zone") and acc.is_in_conflict_zone(pred_pos):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"Candidate step {pred_pos} is inside an ACC recurrent failure cluster",
                )

        return SafetyGateResult(is_safe=True)


class CausalUndoSafetyPredicate(PlanSafetyPredicate):
    """Rejects candidate trajectories that trigger empirical undo / state reversion rollbacks."""

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        undo_acts = get_engine_state_attr(view, "undo_actions", set())
        if next_step.action in undo_acts and not view.is_displacement_action(next_step.action):
            return SafetyGateResult(
                is_safe=False,
                reason=f"Action {next_step.action} is an empirical state-reversion undo operator",
            )
        return SafetyGateResult(is_safe=True)


class OFCCounterfactualMentalSimulationPredicate(PlanSafetyPredicate):
    """Orbitofrontal Cortex (OFC) Counterfactual Mental Simulation Gating:
    Simulates candidate trajectory forward in imagination up to depth K (default 3 steps).
    Evaluates:
    1. Irreversible corner deadlock on pushable cargo or avatar entrapment.
    2. Boundary wall deadlock without accessible goals.
    3. Impasse into spatiotemporal hazard cones or lethal entities along the simulated path.
    """

    def __init__(self, simulation_depth: int = 3) -> None:
        self.simulation_depth = simulation_depth

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        if next_step.predicted_avatar_pos is None:
            return SafetyGateResult(is_safe=True)

        H, W = curr_grid.shape
        goals = set(get_engine_state_attr(view, "learned_goal_positions", set()))
        static_barriers = set(get_engine_state_attr(view, "learned_barriers", set()))
        if hasattr(view, "current_context_hazards"):
            static_barriers.update(view.current_context_hazards)

        sim_steps: list[MentalSimulationStep] = []
        if next_step is not None:
            sim_steps.append(next_step)
        for f_step in future_steps:
            if f_step is not next_step and len(sim_steps) < self.simulation_depth:
                sim_steps.append(f_step)

        for step_idx, step in enumerate(sim_steps):
            pos = step.predicted_avatar_pos
            if (
                not isinstance(pos, tuple)
                or len(pos) < 2
                or not isinstance(pos[0], (int, float))
                or not isinstance(pos[1], (int, float))
            ):
                continue
            r, c = int(pos[0]), int(pos[1])
            if not (0 <= r < H and 0 <= c < W):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"OFC mental rollout: Step {step_idx} out of grid bounds at ({r}, {c})",
                )

            # Check corner deadlock if position is trapped and not a goal
            has_cargo_ctx = bool(
                view.symbolic_theory.cargo_features or getattr(view, "learned_cargo_features", None)
            )
            is_trapped_avatar = all(
                not (0 <= r + dr < H and 0 <= c + dc < W) or (r + dr, c + dc) in static_barriers
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if (r, c) not in goals and (
                is_trapped_avatar
                or (
                    has_cargo_ctx
                    and CounterfactualDeadlockDetector.is_corner_deadlock(
                        (r, c), goals, static_barriers, (H, W)
                    )
                )
            ):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"OFC mental rollout detected corner deadlock at ({r}, {c}) on step {step_idx}",
                )

        return SafetyGateResult(is_safe=True)


class PlanSafetyEvaluator:
    """Prefrontal safety arbitration view composing typed PlanSafetyPredicates."""

    def __init__(self, predicates: Sequence[PlanSafetyPredicate] | None = None) -> None:
        self.predicates: list[PlanSafetyPredicate] = list(
            predicates
            if predicates is not None
            else [
                StaticLethalHazardPredicate(),
                DynamicPatrolTrajectoryPredicate(),
                HabenularInhibitionPredicate(),
                AnteriorCingulateConflictPredicate(),
                CausalUndoSafetyPredicate(),
                OFCCounterfactualMentalSimulationPredicate(),
            ]
        )

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        available_actions: Sequence[Any],
        future_steps: Sequence[MentalSimulationStep] | None = None,
    ) -> SafetyGateResult:
        if future_steps is None:
            future_steps = [next_step] + list(view.mental_plan)
        for pred in self.predicates:
            res = pred.evaluate(
                view=view,
                curr_grid=curr_grid,
                next_step=next_step,
                future_steps=future_steps,
                available_actions=available_actions,
            )
            if not res.is_safe or res.should_wait:
                return res
        return SafetyGateResult(is_safe=True)

    @classmethod
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export plan safety evaluator capability and active hazard count as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "multivariate_prefrontal_safety_gating"
        if view is not None:
            lethals = len(
                getattr(getattr(view, "hazard_tracker", None), "known_lethal_features", set())
                or set()
            )
            rel = f"multivariate_prefrontal_safety_gating:lethal_features:{lethals}"
        return CausalPredicate(
            variable="plan_safety_evaluator",
            operator="==",
            value=val,
            relation=rel,
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
