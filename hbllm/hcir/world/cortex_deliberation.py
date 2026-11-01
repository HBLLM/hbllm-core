from __future__ import annotations


def get_sub_attr(target: Any, attr: str, default: Any = None) -> Any:
    return getattr(target, attr, default) if target is not None else default


from hbllm.hcir.world.engine_state import get_engine_state_attr

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
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.cortex_planner import EpistemicGoalScheduler, MentalSimulationStep
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.engine_state import CognitiveContext

if TYPE_CHECKING:
    pass

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
            if future_step.predicted_avatar_pos is None:
                continue
            fr, fc = future_step.predicted_avatar_pos
            if 0 <= fr < H and 0 <= fc < W:
                feat = int(curr_grid[fr, fc])
                if feat in view.mobile_threat_features:
                    continue
                if feat in view.hazard_tracker.known_lethal_features:
                    return SafetyGateResult(
                        is_safe=False, reason=f"Lethal feature {feat} at ({fr}, {fc})"
                    )
        return SafetyGateResult(is_safe=True)


class SpatiotemporalCollisionConePredicate(PlanSafetyPredicate):
    """Rejects steps intersecting spatiotemporal looming collision cones."""

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        if view.avatar_pos is None:
            return SafetyGateResult(is_safe=True)

        ar, ac = view.avatar_pos
        is_stationary = (
            next_step.predicted_avatar_pos is None or next_step.predicted_avatar_pos == (ar, ac)
        )
        if is_stationary:
            if any(view.collision_cones.is_collision_hazard(ar, ac, t_f) for t_f in (1, 2)):
                return SafetyGateResult(
                    is_safe=False, reason="Stationary avatar in looming collision cone"
                )
        elif next_step.predicted_avatar_pos is not None:
            nr, nc = next_step.predicted_avatar_pos
            if view.collision_cones.is_collision_hazard(nr, nc, 1):
                # Check if temporal wait clears the corridor
                acc_mon = get_sub_attr(view.working_memory, "acc_conflict", None)
                if acc_mon is not None:
                    should_w, w_steps = acc_mon.should_advise_temporal_wait(
                        (ar, ac), (nr, nc), view.collision_cones
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
            barrier_feats = view.symbolic_theory.barrier_features
            walkable_and_safe = view.symbolic_theory.walkable_features
            bg_is_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
            patrol_walls: set[tuple[int, int]] = set(view.learned_barriers)
            for rr in range(H):
                for cc in range(W):
                    v = int(curr_grid[rr, cc])
                    if (
                        view.symbolic_theory.is_barrier(v)
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
        has_temporal = (
            bool(view.hazard_tracker.periodic_cells)
            or bool(view.mobile_threat_features)
            or bool(view.collision_cones.projected_trajectories)
        )
        if not has_temporal and view.inhibited_actions.get(next_step.action, 0) > 0:
            return SafetyGateResult(
                is_safe=False, reason=f"Action {next_step.action} in refractory inhibition"
            )
        if view.working_memory.habenular_ior.is_action_inhibited(view.avatar_pos, next_step.action):
            return SafetyGateResult(is_safe=False, reason="Action inhibited by Habenular IOR")
        return SafetyGateResult(is_safe=True)


class CerebellarPhaseGatePredicate(PlanSafetyPredicate):
    """Evaluates rhythm-locked oscillation windows via Cerebellar phase clock."""

    def evaluate(
        self,
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        if next_step.predicted_avatar_pos is not None:
            phase_gate = view.cerebellar_clock.evaluate_motion_hazard_gate(
                current_step=view.step_counter,
                avatar_pos=view.avatar_pos,
                target_pos=next_step.predicted_avatar_pos,
                hazard_tracker=view.hazard_tracker,
                background_feature=view.bg_feature,
                avatar_features=view.avatar_features,
                walkable_features=view.symbolic_theory.walkable_features,
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
            # 1. Reject paths routing directly into recurrent death clusters
            if hasattr(acc, "is_in_conflict_zone") and acc.is_in_conflict_zone(pred_pos):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"Candidate step {pred_pos} is inside an ACC recurrent failure cluster",
                )

            # 2. Check if ACC advises temporal wait due to collision cones crossing candidate pos
            curr_pos = view.avatar_pos or (0, 0)
            collision_cones = get_engine_state_attr(view, "collision_cones", None)
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
        view: Any,
        curr_grid: np.ndarray,
        next_step: MentalSimulationStep,
        future_steps: Sequence[MentalSimulationStep],
        available_actions: Sequence[Any],
    ) -> SafetyGateResult:
        undo_acts = get_engine_state_attr(view, "undo_actions", set())
        if next_step.action in undo_acts:
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
        if hasattr(view, "current_level_lethal_positions"):
            static_barriers.update(view.current_level_lethal_positions)

        sim_steps: list[MentalSimulationStep] = []
        if next_step is not None:
            sim_steps.append(next_step)
        for f_step in future_steps:
            if f_step is not next_step and len(sim_steps) < self.simulation_depth:
                sim_steps.append(f_step)

        for step_idx, step in enumerate(sim_steps):
            pos = step.predicted_avatar_pos
            if pos is None:
                continue
            r, c = pos
            if not (0 <= r < H and 0 <= c < W):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"OFC mental rollout: Step {step_idx} out of grid bounds at ({r}, {c})",
                )

            # Check corner deadlock if position is trapped and not a goal
            if (r, c) not in goals and CounterfactualDeadlockDetector.is_corner_deadlock(
                (r, c), goals, static_barriers, (H, W)
            ):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"OFC mental rollout detected corner deadlock at ({r}, {c}) on step {step_idx}",
                )

            # Check collision hazard at future timestep
            cones = get_engine_state_attr(view, "collision_cones", None)
            if cones is not None and cones.is_collision_hazard(r, c, time_step=step_idx + 1):
                return SafetyGateResult(
                    is_safe=False,
                    reason=f"OFC mental rollout detected collision cone intersection at ({r}, {c}) on step {step_idx + 1}",
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
                SpatiotemporalCollisionConePredicate(),
                DynamicPatrolTrajectoryPredicate(),
                HabenularInhibitionPredicate(),
                CerebellarPhaseGatePredicate(),
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
# Basal Ganglia Cortico-Striatal Motor Arbitration (Stage 4 Deliberation)
# ═══════════════════════════════════════════════════════════════════════════


class BasalGangliaPathway(StrEnum):
    """Cortico-striatal motor pathway classification."""

    DIRECT_GO = "DIRECT_GO"  # Striatal D1 MSN facilitation (Go)
    INDIRECT_NOGO = "INDIRECT_NOGO"  # Striatal D2 MSN suppression (NoGo)
    HYPERDIRECT_BRAKE = "HYPERDIRECT_BRAKE"  # Subthalamic Nucleus (STN) global pause/brake


@dataclass
class BasalGangliaDecision:
    """Action arbitration outcome from the Basal Ganglia tri-pathway circuit."""

    action: Any
    pathway: BasalGangliaPathway
    d1_go_drive: float
    d2_nogo_drive: float
    stn_brake_signal: float
    net_drive: float
    is_suppressed: bool
    reason: str = ""

    def to_causal_predicate(self) -> CausalPredicate:
        """Export basal ganglia decision as a declarative causal predicate."""
        return CausalPredicate(
            variable="basal_ganglia:pathway",
            operator="==",
            value=self.pathway.value,
            relation=f"action:{self.action}:drive:{self.net_drive:.3f}:suppressed:{self.is_suppressed}",
            target_entity=str(self.action),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording basal ganglia decision."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"bg_decision_{self.action}",
                node_data={
                    "action": str(self.action),
                    "pathway": self.pathway.value,
                    "d1_go_drive": float(self.d1_go_drive),
                    "d2_nogo_drive": float(self.d2_nogo_drive),
                    "stn_brake_signal": float(self.stn_brake_signal),
                    "net_drive": float(self.net_drive),
                    "is_suppressed": self.is_suppressed,
                    "reason": self.reason,
                },
            )
        ]


class BasalGangliaGatingCircuit:
    """Biologically-modeled Basal Ganglia Tri-Pathway Action Arbitration Circuit.

    Models the competition between:
    1. Direct Pathway (Striatal D1 MSNs): Facilitates action execution by disinhibiting the thalamus.
       Modulated positively by dopamine: higher tonic DA / positive RPE increases D1 Go gain.
    2. Indirect Pathway (Striatal D2 MSNs): Suppresses action execution by increasing GPi inhibition.
       Modulated inversely by dopamine: lower tonic DA / negative RPE increases D2 NoGo drive.
    3. Hyperdirect Pathway (Cortical ACC/preSMA -> Subthalamic Nucleus / STN -> GPi):
       Fires upon high conflict (ACC), looming threats, or high surprise, exerting a global
       emergency brake that suppresses candidate execution before catastrophic motor commitment.
    """

    def __init__(
        self,
        w_d1: float = 1.0,
        w_d2: float = 1.2,
        brake_threshold: float = 0.65,
        da_sensitivity: float = 0.5,
    ) -> None:
        self.w_d1 = w_d1
        self.w_d2 = w_d2
        self.brake_threshold = brake_threshold
        self.da_sensitivity = da_sensitivity
        self.arbitration_history: deque[BasalGangliaDecision] = deque(maxlen=64)

    def compute_hyperdirect_brake(
        self,
        acc_conflict: float = 0.0,
        threat_looming: float = 0.0,
        surprise_entropy: float = 0.0,
    ) -> float:
        """Compute instantaneous Subthalamic Nucleus (STN) hyperdirect brake signal."""
        combined = 0.5 * acc_conflict + 0.35 * threat_looming + 0.15 * surprise_entropy
        return float(1.0 / (1.0 + math.exp(-6.0 * (combined - 0.45))))

    def arbitrate_candidates(
        self,
        candidate_utilities: dict[Any, float],
        dopamine_level: float = 1.0,
        acc_conflict: float = 0.0,
        threat_looming: float = 0.0,
        surprise_entropy: float = 0.0,
        habenular_inhibitions: dict[Any, float] | None = None,
        risk_penalties: dict[Any, float] | None = None,
    ) -> tuple[BasalGangliaDecision | None, list[BasalGangliaDecision]]:
        """Arbitrate candidate actions through the 3 cortico-striatal pathways.

        Returns:
            (winning_decision_or_none, all_evaluated_decisions)
        """
        if not candidate_utilities:
            return None, []

        stn_brake = self.compute_hyperdirect_brake(
            acc_conflict=acc_conflict,
            threat_looming=threat_looming,
            surprise_entropy=surprise_entropy,
        )
        global_braked = stn_brake >= self.brake_threshold

        hab_dict = habenular_inhibitions or {}
        risk_dict = risk_penalties or {}

        # Dopamine gain modulation (D1 boosted by DA, D2 attenuated by DA)
        da_delta = float(np.clip(dopamine_level - 1.0, -1.0, 2.0))
        d1_mod = max(0.1, 1.0 + self.da_sensitivity * da_delta)
        d2_mod = max(0.1, 1.0 - self.da_sensitivity * da_delta)

        decisions: list[BasalGangliaDecision] = []

        for action, utility in candidate_utilities.items():
            # D1 Go pathway: utility scaled by D1 gain
            d1_go = float(1.0 / (1.0 + math.exp(-3.0 * (self.w_d1 * utility * d1_mod - 0.2))))

            # D2 NoGo pathway: Habenular failure history + risk penalty scaled by D2 gain
            hab_cost = hab_dict.get(action, 0.0)
            risk_cost = risk_dict.get(action, 0.0)
            nogo_input = (hab_cost + risk_cost) * self.w_d2 * d2_mod
            d2_nogo = float(1.0 / (1.0 + math.exp(-3.0 * (nogo_input - 0.3))))

            net_drive = d1_go - d2_nogo - (stn_brake if global_braked else 0.0)

            if global_braked:
                pathway = BasalGangliaPathway.HYPERDIRECT_BRAKE
                suppressed = True
                reason = f"Hyperdirect STN brake triggered (brake={stn_brake:.2f} >= {self.brake_threshold})"
            elif d2_nogo > d1_go:
                pathway = BasalGangliaPathway.INDIRECT_NOGO
                suppressed = True
                reason = f"Indirect D2 suppression (nogo={d2_nogo:.2f} > go={d1_go:.2f})"
            else:
                pathway = BasalGangliaPathway.DIRECT_GO
                suppressed = False
                reason = f"Direct D1 disinhibition (go={d1_go:.2f} > nogo={d2_nogo:.2f})"

            dec = BasalGangliaDecision(
                action=action,
                pathway=pathway,
                d1_go_drive=d1_go,
                d2_nogo_drive=d2_nogo,
                stn_brake_signal=stn_brake,
                net_drive=net_drive,
                is_suppressed=suppressed,
                reason=reason,
            )
            decisions.append(dec)

        # Sort candidate decisions by net drive descending
        decisions.sort(key=lambda d: d.net_drive, reverse=True)

        winning: BasalGangliaDecision | None = None
        if not global_braked:
            for d in decisions:
                if not d.is_suppressed and d.pathway == BasalGangliaPathway.DIRECT_GO:
                    winning = d
                    break
        else:
            winning = decisions[0] if decisions else None

        if winning is not None:
            self.arbitration_history.append(winning)

        return winning, decisions


# ═══════════════════════════════════════════════════════════════════════════
# Dual Neuromodulation: VTA Dopamine & LC Noradrenaline (Adaptive Gain)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NeuromodulatoryState:
    """Composite biological neuromodulatory state governing exploration vs. exploitation."""

    dopamine_tonic: float = 1.0
    dopamine_phasic_rpe: float = 0.0
    noradrenaline_tonic: float = 0.2
    noradrenaline_phasic: float = 0.1
    exploration_gain: float = 1.0
    exploitation_gain: float = 1.0
    is_disengaged: bool = False
    regime: str = "EXPLOITATION"  # "EXPLOITATION", "FOCAL_ENGAGEMENT", "GLOBAL_EXPLORATION"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export neuromodulatory state as a declarative causal predicate."""
        return CausalPredicate(
            variable="neuromodulation:state",
            operator="==",
            value=self.regime,
            relation=f"da_tonic:{self.dopamine_tonic:.2f}:na_tonic:{self.noradrenaline_tonic:.2f}:disengaged:{self.is_disengaged}",
            target_entity="adaptive_governor",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording neuromodulatory state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="neuromodulatory_state",
                node_data={
                    "dopamine_tonic": float(self.dopamine_tonic),
                    "dopamine_phasic_rpe": float(self.dopamine_phasic_rpe),
                    "noradrenaline_tonic": float(self.noradrenaline_tonic),
                    "noradrenaline_phasic": float(self.noradrenaline_phasic),
                    "exploration_gain": float(self.exploration_gain),
                    "exploitation_gain": float(self.exploitation_gain),
                    "is_disengaged": self.is_disengaged,
                    "regime": self.regime,
                },
            )
        ]


class AdaptiveNeuromodulationGovernor:
    """Dual Neuromodulatory Governor: VTA Dopamine (RPE) & LC Noradrenaline (Adaptive Gain).

    Modeled on:
    - Schultz (1998): Ventral Tegmental Area (VTA) dopaminergic Reward Prediction Error (RPE).
    - Aston-Jones & Cohen (2005) Adaptive Gain Theory: Locus Coeruleus (LC) noradrenergic gating.
      * Phasic LC bursts optimize task exploitation and signal-to-noise ratio.
      * High tonic LC promotes behavioral disengagement from unproductive local loops,
        triggering broad global exploration.
    """

    def __init__(
        self,
        gamma: float = 0.9,
        da_decay: float = 0.1,
        na_decay: float = 0.15,
        disengagement_threshold: float = 0.65,
    ) -> None:
        self.gamma = gamma
        self.da_decay = da_decay
        self.na_decay = na_decay
        self.disengagement_threshold = disengagement_threshold

        self.current_state = NeuromodulatoryState()
        self.history: deque[NeuromodulatoryState] = deque(maxlen=64)

    def step_dynamics(
        self,
        reward: float = 0.0,
        predicted_value: float = 0.0,
        actual_value: float = 0.0,
        conflict_level: float = 0.0,
        stuck_steps: int = 0,
    ) -> NeuromodulatoryState:
        """Step VTA-LC coupled dynamics based on reward progress and cognitive conflict."""
        # 1. VTA Dopamine: Reward Prediction Error (RPE)
        rpe = reward + self.gamma * actual_value - predicted_value
        tonic_da = (1.0 - self.da_decay) * self.current_state.dopamine_tonic + self.da_decay * (
            1.0 + rpe
        )
        tonic_da = float(np.clip(tonic_da, 0.1, 2.5))

        # 2. LC Noradrenaline: Phasic burst on acute conflict / salient prediction error
        phasic_na = float(1.0 / (1.0 + math.exp(-5.0 * (abs(rpe) + conflict_level - 0.5))))

        # Tonic NA: Slowly integrates persistent lack of progress / stuck counter
        stuck_pressure = min(2.0, float(stuck_steps) / 3.0)
        tonic_na = (
            1.0 - self.na_decay
        ) * self.current_state.noradrenaline_tonic + self.na_decay * stuck_pressure
        tonic_na = float(np.clip(tonic_na, 0.05, 2.0))

        # 3. Behavioral Regime Determination
        is_disengaged = tonic_na >= self.disengagement_threshold or stuck_steps >= 4

        if is_disengaged:
            regime = "GLOBAL_EXPLORATION"
            exploration_gain = 1.0 + 1.5 * tonic_na
            exploitation_gain = max(0.1, 1.0 - 0.7 * tonic_na)
        elif phasic_na > 0.6 and tonic_da > 0.8:
            regime = "FOCAL_ENGAGEMENT"
            exploration_gain = 0.4
            exploitation_gain = 1.0 + 0.8 * tonic_da
        else:
            regime = "EXPLOITATION"
            exploration_gain = 1.0
            exploitation_gain = 1.0

        new_state = NeuromodulatoryState(
            dopamine_tonic=tonic_da,
            dopamine_phasic_rpe=float(rpe),
            noradrenaline_tonic=tonic_na,
            noradrenaline_phasic=phasic_na,
            exploration_gain=exploration_gain,
            exploitation_gain=exploitation_gain,
            is_disengaged=is_disengaged,
            regime=regime,
        )
        self.current_state = new_state
        self.history.append(new_state)
        return new_state


# ═══════════════════════════════════════════════════════════════════════════
# Prefrontal Deliberation Engine
# ═══════════════════════════════════════════════════════════════════════════


class PrefrontalDeliberationEngine:
    """Executive prefrontal arbitration view coordinating perception, mental simulation,
    affordance chunking, and motor output.
    """

    def __init__(
        self,
        safety_evaluator: PlanSafetyEvaluator | None = None,
        basal_ganglia: BasalGangliaGatingCircuit | None = None,
        neuromodulation: AdaptiveNeuromodulationGovernor | None = None,
    ) -> None:
        self.safety_evaluator = safety_evaluator or PlanSafetyEvaluator()
        self.basal_ganglia = basal_ganglia or BasalGangliaGatingCircuit()
        self.neuromodulation = neuromodulation or AdaptiveNeuromodulationGovernor()

    def arbitrate_basal_ganglia_action(
        self,
        view: Any,
        candidate_utilities: dict[Any, float],
        acc_conflict: float = 0.0,
        threat_looming: float = 0.0,
        surprise_entropy: float = 0.0,
        risk_penalties: dict[Any, float] | None = None,
    ) -> tuple[BasalGangliaDecision | None, list[BasalGangliaDecision]]:
        """Arbitrate candidate actions using the Basal Ganglia tri-pathway circuit."""
        dopamine = 1.0
        if hasattr(view, "ventral_tegmental") and hasattr(view.ventral_tegmental, "tonic_level"):
            dopamine = float(view.ventral_tegmental.tonic_level)

        habenular_dict: dict[Any, float] = {}
        if hasattr(view, "inhibited_actions"):
            for act, cnt in view.inhibited_actions.items():
                if cnt > 0:
                    habenular_dict[act] = min(1.0, 0.2 * cnt)

        return self.basal_ganglia.arbitrate_candidates(
            candidate_utilities=candidate_utilities,
            dopamine_level=dopamine,
            acc_conflict=acc_conflict,
            threat_looming=threat_looming,
            surprise_entropy=surprise_entropy,
            habenular_inhibitions=habenular_dict,
            risk_penalties=risk_penalties,
        )

    def step_neuromodulatory_dynamics(
        self,
        view: Any,
        reward: float = 0.0,
        predicted_value: float = 0.0,
        actual_value: float = 0.0,
        conflict_level: float = 0.0,
    ) -> NeuromodulatoryState:
        """Update VTA-LC dual neuromodulatory state using view tracking context."""
        stuck = get_engine_state_attr(view, "consecutive_stuck_steps", 0)
        return self.neuromodulation.step_dynamics(
            reward=reward,
            predicted_value=predicted_value,
            actual_value=actual_value,
            conflict_level=conflict_level,
            stuck_steps=stuck,
        )

    def validate_plan_safety(
        self,
        view: Any,
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
            view=view,
            curr_grid=curr_grid,
            next_step=next_step,
            available_actions=available_actions,
        )
        return res.is_safe, res.should_wait, res.wait_steps

    def arbitrate_affordances(
        self,
        view: Any,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        known_av_feats: set[int],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Arbitrate control panel affordances and interactive targets via registered policies or generic fallbacks."""
        # 1. Query external registered deliberation policies first (e.g. ARC3DeliberationPolicy)
        for policy in get_engine_state_attr(view, "deliberation_policies", []):
            if hasattr(policy, "arbitrate_affordances"):
                res = policy.arbitrate_affordances(
                    view, curr_grid, available_actions, known_av_feats
                )
                if res is not None:
                    return res

        # 2. Domain-agnostic fallback for discrete transform / focus switch actions
        spatial_effector_actions = [a for a in available_actions if view.is_spatial_effector(a)]
        discrete_transform_actions = [
            a
            for a in available_actions
            if not view.is_displacement_action(a) and not view.is_spatial_effector(a)
        ]
        focus_switches = [
            a
            for a in discrete_transform_actions
            if view.is_focus_switch_action(a) and view.inhibited_actions.get(a, 0) == 0
        ]
        if focus_switches:
            view.inhibited_actions[focus_switches[0]] = max(
                view.inhibited_actions.get(focus_switches[0], 0), 8
            )
            view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            return focus_switches[0], None

        # Faculty: Dial / State-Cycler Interleaved Deliberation (Norman Parameter Space Navigation)
        index_selectors = [
            a
            for a in available_actions
            if a in view.index_selector_actions and view.inhibited_actions.get(a, 0) == 0
        ]
        value_mutators = [
            a
            for a in available_actions
            if a in view.value_mutator_actions and view.inhibited_actions.get(a, 0) == 0
        ]
        if index_selectors and value_mutators:
            mut_streak = get_sub_attr(view.working_memory, "dial_mutation_streak", 0)
            if mut_streak < 2:
                chosen = value_mutators[view.step_counter % len(value_mutators)]
                view.working_memory.dial_mutation_streak = mut_streak + 1
                view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen, None
            else:
                chosen = index_selectors[view.step_counter % len(index_selectors)]
                view.working_memory.dial_mutation_streak = 0
                view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen, None

        # Modal Select-Then-Direct Gating: In pure effector domains without displacement, reach for interactive entities
        if (
            not view.has_displacement_capabilities(available_actions)
            and view.avatar_pos is None
            and not view.working_memory.body_schema.has_manipulandum()
            and spatial_effector_actions
        ):
            bg = view.estimate_background(curr_grid)
            entities = view.extract_entities(curr_grid, bg)
            has_interactive = bool(
                entities
                or get_engine_state_attr(view, "learned_goal_positions", None)
                or get_engine_state_attr(view, "target_scorers", None)
            )
            if has_interactive:
                chosen_eff = spatial_effector_actions[0]
                eff_data = view.ground_effector_action(curr_grid, chosen_eff)
                if eff_data is not None:
                    view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    return chosen_eff, eff_data

        is_stuck_or_looping = (
            view.consecutive_stuck_steps >= 2
            or (view.avatar_pos is not None and view.recent_positions.count(view.avatar_pos) >= 2)
            or (view.avatar_pos is None and view.step_counter > 4)
        )
        has_displacement_actions = any(
            view.is_displacement_action(a) for a in available_actions
        ) or any(
            not view.is_spatial_effector(a) and a not in view.action_dynamics
            for a in available_actions
        )

        if not has_displacement_actions or is_stuck_or_looping:
            if spatial_effector_actions:
                # When spatial effectors are present and no panel exists,
                # arbitrate_affordances must return None instead of blind discrete execution
                return None

            undo_acts = get_engine_state_attr(view, "undo_actions", set())
            uninhibited_discrete = [
                a
                for a in discrete_transform_actions
                if view.inhibited_actions.get(a, 0) == 0 and a not in undo_acts
            ]
            if uninhibited_discrete:
                # WP0.3: prefer the action least dominated by no_op/blocked
                # samples and never repeat a null-effect action >2x in a row
                # (unless every action is null-dominated).
                def _null_score(a: Any) -> tuple[int, float, int]:
                    post = view.motor_posterior(a) if hasattr(view, "motor_posterior") else {}
                    null_p = post.get("no_op", 0.0) + post.get("blocked", 0.0)
                    consec = (
                        view.consecutive_null_count(a)
                        if hasattr(view, "consecutive_null_count")
                        else 0
                    )
                    return (consec >= 2, null_p, a)

                viable = [a for a in uninhibited_discrete if not _null_score(a)[0]]
                pool = viable or uninhibited_discrete
                chosen_action = min(pool, key=_null_score)
                view.inhibited_actions[chosen_action] = max(
                    view.inhibited_actions.get(chosen_action, 0), 4
                )
                view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen_action, None

        return None

    def _execute_temporal_pacing(
        self,
        view: Any,
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
        view.consecutive_pacing_steps += 1
        if view.consecutive_pacing_steps > 2:
            view.state.consecutive_pacing_steps = 0
            view.mental_plan.clear()
            view.transition_to_phase(
                EpistemicPhase.REPLANNING,
                reason="pacing_window_exhausted",
            )
            act, data = view.plan_epistemic_probe(curr_grid, available_actions)
            return act, data, None

        view.mental_plan.appendleft(next_step)
        view.state.pending_phase_wait_steps = wait_steps
        undo_acts = get_engine_state_attr(view, "undo_actions", set())
        non_disp = [
            a
            for a in available_actions
            if not view.is_displacement_action(a)
            and not view.is_spatial_effector(a)
            and a not in undo_acts
        ]
        curr_pos = view.avatar_pos or get_engine_state_attr(view, "current_actor_pos", None)
        if non_disp:
            return non_disp[0], None, curr_pos

        H, W = curr_grid.shape
        reverse_map: dict[Any, Any] = {}
        for a1, d1 in view.action_deltas.items():
            for a2, d2 in view.action_deltas.items():
                if d1[0] == -d2[0] and d1[1] == -d2[1] and (d1[0] != 0 or d1[1] != 0):
                    reverse_map[a1] = a2
        if not reverse_map:
            reverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
        pacing_candidates: list[tuple[Any, tuple[int, int]]] = []

        if curr_pos is not None:
            bg = view.estimate_background(curr_grid)
            walkable_and_safe = view.symbolic_theory.walkable_features
            bg_is_impassable_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
            if not bg_is_impassable_void:
                bg_count = int(np.count_nonzero(curr_grid == bg))
                bg_ratio = bg_count / max(1, H * W)
                ar, ac = curr_pos
                avatar_on_bg = (
                    0 <= ar < H
                    and 0 <= ac < W
                    and (
                        int(curr_grid[ar, ac]) == bg
                        or any(
                            0 <= ar + dr_a < H
                            and 0 <= ac + dc_a < W
                            and int(curr_grid[ar + dr_a, ac + dc_a]) == bg
                            for dr_a, dc_a in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        )
                    )
                )
                if bg_ratio >= 0.20 and not avatar_on_bg:
                    bg_is_impassable_void = True

            action_deltas = view.action_deltas
            for act in available_actions:
                if not view.is_displacement_action(act) or act == next_step.action:
                    continue
                delta = action_deltas.get(act)
                if delta is None:
                    continue
                dr, dc = delta
                nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in view.learned_barriers or (
                    nr,
                    nc,
                ) in view.current_level_lethal_positions:
                    continue
                val = int(curr_grid[nr, nc])
                if (
                    view.symbolic_theory.is_barrier(val)
                    or val in view.symbolic_theory.barrier_features
                    or val in view.hazard_tracker.known_lethal_features
                    or (bg_is_impassable_void and val == bg)
                ):
                    continue
                cand_step = MentalSimulationStep(
                    action=act,
                    predicted_avatar_pos=(nr, nc),
                    expected_mutation="oscillatory_pacing_step",
                )
                cand_res = self.safety_evaluator.evaluate(
                    view=view,
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
            if len(view.recent_actions) > 0:
                last_act = view.recent_actions[-1]
                rev_last = reverse_map.get(last_act)
                for act, pos in pacing_candidates:
                    if act == rev_last:
                        chosen_pacing_act, chosen_pacing_pos = act, pos
                        break

            rev_act = reverse_map.get(chosen_pacing_act)
            # Enqueue return step to restore spatial trajectory once hazard passes
            if rev_act is not None and rev_act in available_actions and curr_pos is not None:
                view.mental_plan.appendleft(
                    MentalSimulationStep(
                        action=rev_act,
                        predicted_avatar_pos=curr_pos,
                        expected_mutation="oscillatory_return_step",
                    )
                )
            return chosen_pacing_act, None, chosen_pacing_pos

        # Invariant fallback: Invalidate plan and replan if no safe pacing candidate exists
        view.mental_plan.clear()
        view.state.phase = EpistemicPhase.REPLANNING
        act, data = view.plan_epistemic_probe(curr_grid, available_actions)
        return act, data, None

    def prepare(
        self,
        view: Any,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: Any | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action_schemas: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Pre-deliberation preamble: terminal resolution, step counter,
        normalization, bg estimate, and sensory assimilation (WP1)."""
        from hbllm.hcir.world.percept import TerminalSignal

        if terminal is None:
            if is_success:
                terminal = TerminalSignal.SUCCESS
            elif is_failure:
                terminal = TerminalSignal.FAILURE

        view.step_counter += 1
        if action_schemas:
            view.register_action_space(action_schemas)

        curr_grid = view.normalize_sensory_input(curr_grid)

        # Metacognitive Urgency Modulation:
        if view.is_under_strict_step_budget():
            view.exploration_cooldown = 0
            view.simulation_cooldown = 0

        # 1. Update background & assimilate sensory feedback from previous action
        view.state.bg_feature = view.estimate_background(curr_grid)
        if view.prev_grid is not None and not view._feedback_assimilated:
            view.assimilate_feedback(
                curr_grid,
                available_actions,
                terminal=terminal,
                is_success=is_success,
                is_failure=is_failure,
                action=view.last_action,
                action_data=view.last_action_data,
                **kwargs,
            )
        view.state._feedback_assimilated = False
        return curr_grid, terminal

    def decide(
        self,
        view: Any,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: Any | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action_schemas: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision (prepare + deliberate; kept for direct callers)."""
        g, term = self.prepare(
            view,
            curr_grid,
            available_actions,
            terminal=terminal,
            is_success=is_success,
            is_failure=is_failure,
            action_schemas=action_schemas,
            **kwargs,
        )
        proposal = self.deliberate(
            view,
            g,
            available_actions,
            terminal=term,
            is_success=is_success,
            is_failure=is_failure,
            action_schemas=action_schemas,
            objects=objects,
            micro_actions=micro_actions,
            **kwargs,
        )
        if isinstance(proposal, ActionProposal):
            if proposal.finalized:
                return proposal.action, proposal.action_data
            return self._finalize_step(
                view,
                proposal.grid,
                proposal.action,
                proposal.action_data,
                predicted_pos=proposal.predicted_pos,
            )
        return proposal

    def deliberate(
        self,
        view: Any,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: Any | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action_schemas: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> ActionProposal | tuple[Any, dict[str, Any] | None]:
        """Post-assimilation deliberation body (WP1 split from decide)."""
        # Track recent avatar position on every decision step (before macro or micro action planning)
        if view.avatar_pos is not None:
            view.recent_positions.append(view.avatar_pos)

        # Metacognitive Fail-Fast & Entropy Monitoring (Phase I)
        if hasattr(view, "entropy_monitor") and view.entropy_monitor is not None:
            cur_h = float("inf")
            if view.avatar_pos is not None and view.learned_goal_positions:
                cur_h = min(
                    abs(view.avatar_pos[0] - g[0]) + abs(view.avatar_pos[1] - g[1])
                    for g in view.learned_goal_positions
                )
            if view.entropy_monitor.update(view.avatar_pos, cur_h):
                view.entropy_monitor.trigger_fail_fast(view)

        # Faculty: Executive Macro-Action Plan Execution & Synthesis
        def _is_macro_step_safe(plan: Any, act: Any, data: dict[str, Any] | None) -> bool:
            # Executive prefrontal macro plans supersede subcortical motor refractory inhibition (inhibited_actions)
            # and exploratory sensory foraging IOR (habenular_ior).
            # If the macro plan is marked is_strict, trust executive dispatch completely (e.g. verified evasion/solution sequence).
            if getattr(plan, "is_strict", False):
                return True
            cur_op = getattr(plan, "current", lambda: None)()
            if cur_op is not None and getattr(cur_op, "is_strict", False):
                return True

            # Safety gating aborts if displacement moves into collision cone or ping-pongs in limit cycle.
            if view.avatar_pos is not None and act in view.action_dynamics:
                dyn = view.action_dynamics[act]
                if dyn.is_displacement_action():
                    dr, dc = dyn.delta_r, dyn.delta_c
                    nr, nc = view.avatar_pos[0] + dr, view.avatar_pos[1] + dc
                    if hasattr(
                        view, "collision_cones"
                    ) and view.collision_cones.is_collision_hazard(nr, nc, 1):
                        return False
                    # Abort if displacement moves into limit cycle or oscillation
                    rec = list(get_engine_state_attr(view, "recent_positions", []))
                    if rec:
                        # Immediate 2-step reversal
                        if len(rec) >= 2 and (nr, nc) == rec[-2]:
                            return False
                        # 4-step limit cycle
                        if len(rec) >= 4 and (nr, nc) == rec[-4]:
                            return False
                        # Repeated local trapping in small window
                        if len(rec) >= 6 and rec[-6:].count((nr, nc)) >= 2:
                            return False
            return True

        macro_plan = get_sub_attr(view.working_memory, "macro_plan", None)
        if macro_plan is not None and not macro_plan.is_empty():
            macro_step = macro_plan.step(
                view, curr_grid, available_actions, prev_grid=view.prev_grid
            )
            if macro_step is not None:
                m_act, m_data = macro_step
                if m_act in available_actions:
                    if _is_macro_step_safe(macro_plan, m_act, m_data):
                        return ActionProposal(m_act, m_data, branch="macro_option", grid=curr_grid)
                    else:
                        macro_plan.clear()

        from hbllm.hcir.world.macro_actions import (
            MacroPlanSynthesizer,
            ProceduralReplayAction,
        )

        # ── Procedural Memory Recall ───────────────────────────────────────
        # Before synthesizing a new plan from scratch, check if episodic memory
        # has a consolidated procedural macro applicable to the current observation.
        # This is the retrieval arm of the procedural consolidation loop (Phase A):
        #   Episode win → consolidate macro → query on future levels → replay.
        if (
            (macro_plan is None or macro_plan.is_empty())
            and not view.mental_plan
            and hasattr(view, "episodic_cortex")
            and view.episodic_cortex.consolidated_macros
        ):
            curr_entities = view.extract_entities(curr_grid, view.bg_feature)
            applicable_macros = view.episodic_cortex.query_applicable_macros(
                grid=curr_grid,
                available_actions=set(available_actions),
                top_k=1,
                min_confidence=0.5,
                entities=curr_entities,
            )
            if applicable_macros:
                best_macro = applicable_macros[0]
                logger.info(
                    "Procedural Recall: Replaying consolidated macro '%s' "
                    "(%d actions, confidence=%.2f)",
                    best_macro.macro_id,
                    len(best_macro.action_sequence),
                    best_macro.confidence,
                )
                replay_action = ProceduralReplayAction(macro=best_macro)
                from hbllm.hcir.world.macro_actions import MacroActionPlan

                replay_plan = MacroActionPlan(
                    actions=deque([replay_action]),
                    plan_id=f"procedural_replay_{best_macro.macro_id}",
                    is_strict=True,
                )
                view.working_memory.macro_plan = replay_plan
                first_step = replay_plan.step(
                    view, curr_grid, available_actions, prev_grid=view.prev_grid
                )
                if first_step is not None:
                    m_act, m_data = first_step
                    if m_act in available_actions:
                        return ActionProposal(m_act, m_data, branch="macro_recall", grid=curr_grid)

        # ── Relational Subgoal Macro Recall (Phase E) ──────────────────────
        # When exact concrete macro does not match, query coordinate-invariant
        # relational macros and plan dynamically through intermediate waypoints.
        if (
            (macro_plan is None or macro_plan.is_empty())
            and not view.mental_plan
            and hasattr(view, "episodic_cortex")
            and get_sub_attr(view.episodic_cortex, "relational_macros", None)
            and view.avatar_pos is not None
            and view.is_motor_grounded()
        ):
            applicable_rel = view.episodic_cortex.query_applicable_relational_macros(
                grid=curr_grid,
                available_actions=set(available_actions),
                top_k=1,
            )
            if applicable_rel:
                rel_macro = applicable_rel[0]
                from hbllm.hcir.world.autonomous_learning import CausalTheoryPlanner

                ctx = CausalTheoryPlanner.extract_planning_context(view, curr_grid)
                rel_path = CausalTheoryPlanner.plan_relational_macro(
                    view.avatar_pos, curr_grid, ctx, rel_macro
                )
                if rel_path:
                    from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                    rel_plan: list[MentalSimulationStep] = []
                    sim_pos = view.avatar_pos
                    for act_id in rel_path:
                        dr, dc = ctx.action_displacements.get(act_id, (0, 0))
                        next_pos = (sim_pos[0] + dr, sim_pos[1] + dc)
                        rel_plan.append(
                            MentalSimulationStep(
                                action=act_id,
                                predicted_avatar_pos=next_pos,
                            )
                        )
                        sim_pos = next_pos
                    view.state.phase = EpistemicPhase.EXPLOITATION
                    view.state.mental_plan = deque(rel_plan)
                    next_step = view.mental_plan.popleft()
                    logger.info(
                        "Relational Macro Recall: Dynamically planned %d-step path for '%s'",
                        len(rel_path),
                        rel_macro.macro_id,
                    )
                    return ActionProposal(
                        next_step.action,
                        next_step.action_data,
                        branch="relational_macro",
                        grid=curr_grid,
                    )

        can_synthesize = view.is_motor_grounded() or not view.has_displacement_capabilities(
            available_actions
        )

        if (
            (objects or micro_actions)
            and (macro_plan is None or macro_plan.is_empty())
            and not view.mental_plan
            and can_synthesize
        ):
            synthesized = MacroPlanSynthesizer.synthesize_plan(
                view,
                curr_grid,
                available_actions,
                objects=objects,
                micro_actions=micro_actions,
            )
            if synthesized is not None and not synthesized.is_empty():
                view.working_memory.macro_plan = synthesized
                macro_step = synthesized.step(
                    view, curr_grid, available_actions, prev_grid=view.prev_grid
                )
                if macro_step is not None:
                    m_act, m_data = macro_step
                    if m_act in available_actions:
                        if _is_macro_step_safe(synthesized, m_act, m_data):
                            if (
                                view.phase == EpistemicPhase.MOTOR_GROUNDING
                                and view.is_motor_grounded()
                            ):
                                view.state.phase = EpistemicPhase.EXPLOITATION
                            pred_pos = getattr(
                                synthesized.current_action, "next_predicted_pos", None
                            )
                            if pred_pos is None and view.mental_plan:
                                pred_pos = view.mental_plan[0].predicted_avatar_pos
                            return ActionProposal(
                                m_act,
                                m_data,
                                branch="synthesized_schema",
                                predicted_pos=pred_pos,
                                grid=curr_grid,
                            )
                        else:
                            view.working_memory.macro_plan.clear()

        # Phasic Salience Reset: Environmental discovery, dynamic threat emergence, or high surprise
        # Query prefrontal working memory for salience shift to awaken deliberate forward search
        if view.working_memory.has_salience_shift(view):
            view.simulation_cooldown = 0
            view.consecutive_simulation_failures = 0
            view.mental_plan.clear()
            view.state.phase = (
                EpistemicPhase.MENTAL_SIMULATION
                if view.is_motor_grounded()
                else EpistemicPhase.EPISTEMIC_EXPLORATION
            )

        has_temporal = (
            bool(view.hazard_tracker.periodic_cells)
            or bool(view.mobile_threat_features)
            or bool(view.collision_cones.projected_trajectories)
        )

        # Metacognitive Refractory Inhibition: decay refractory timers

        if view.simulation_cooldown > 0:
            view.simulation_cooldown -= 1
        to_uninhibited = [act for act, timer in view.inhibited_actions.items() if timer <= 1]
        for act in view.inhibited_actions:
            view.inhibited_actions[act] -= 1
        for act in to_uninhibited:
            view.inhibited_actions.pop(act, None)

        # Anterior Mid-Cingulate & Lateral Habenula: Limit Cycle / Oscillation Detection
        osc_acts = view.working_memory.habenular_ior.detect_cyclic_actions()
        if osc_acts:
            cyclic_positions = set(list(view.recent_positions)[-4:])
            if (
                len(cyclic_positions) <= 2 and len(view.recent_positions) >= 4
            ) or view.consecutive_stuck_steps >= 2:
                has_progress = False
            else:
                has_progress = (
                    view.phase == EpistemicPhase.EXPLOITATION
                    and bool(view.mental_plan)
                    and any(
                        s.predicted_avatar_pos is not None
                        and s.predicted_avatar_pos != view.avatar_pos
                        for s in view.mental_plan
                    )
                )
            if not has_progress:
                for osc_act in osc_acts:
                    view.inhibited_actions[osc_act] = max(view.inhibited_actions.get(osc_act, 0), 6)

        # Faculty: Anterior Cingulate Cortex (ACC) Limit Cycle & Conflict Suppression
        forced_acc_perturbation: Any | None = None
        view.acc_conflict_monitor.record_step(view.last_action, view.avatar_pos)
        is_osc, cyclic_acts = view.acc_conflict_monitor.detect_oscillation()
        if is_osc:
            # Purposeful Active Inference & Hesitation Invariant:
            # An action oscillation is NOT a pathological stereotypy if the organism is engaged
            # in intentional temporal pacing/evasion around dynamic threats, or if an active
            # mental plan exhibits future forward progress once the looming hazard clears.
            has_progress = (
                view.phase == EpistemicPhase.EXPLOITATION
                and bool(view.mental_plan)
                and any(
                    s.predicted_avatar_pos is not None and s.predicted_avatar_pos != view.avatar_pos
                    for s in view.mental_plan
                )
            )
            is_deliberate_pacing = has_temporal and (
                view.consecutive_pacing_steps > 0
                or bool(
                    get_engine_state_attr(view, "collision_cones", None)
                    and view.collision_cones.projected_trajectories
                )
            )
            if not has_progress and not is_deliberate_pacing:
                for cyc_a in cyclic_acts:
                    view.inhibited_actions[cyc_a] = max(view.inhibited_actions.get(cyc_a, 0), 8)
                view.mental_plan.clear()
                from hbllm.hcir.world.cortex_motor import ACCConflictMonitor

                ortho_acts = ACCConflictMonitor.get_orthogonal_actions(
                    available_actions=available_actions,
                    cyclic_actions=cyclic_acts,
                    action_dynamics=view.action_dynamics,
                )
                H, W = curr_grid.shape
                for cand_a in ortho_acts:
                    if view.inhibited_actions.get(cand_a, 0) > 0:
                        continue
                    if (view.avatar_pos, cand_a) in view.failed_transitions:
                        continue
                    dyn = view.action_dynamics.get(cand_a)
                    if dyn is not None and dyn.is_displacement_action():
                        dr, dc = dyn.get_displacement()
                        if view.avatar_pos is not None:
                            nr, nc = view.avatar_pos[0] + dr, view.avatar_pos[1] + dc
                            if not (0 <= nr < H and 0 <= nc < W):
                                continue
                            if (nr, nc) in view.hazard_tracker.static_lethal_positions:
                                continue
                            if (nr, nc) in view.learned_barriers:
                                continue
                            feat_at_dest = int(curr_grid[nr, nc])
                            if feat_at_dest in view.hazard_tracker.known_lethal_features:
                                continue
                            forced_acc_perturbation = cand_a
                            break
                    else:
                        forced_acc_perturbation = cand_a
                        break

        # Faculty: ACC Recurring Catastrophe Conflict Resolution (Metacognitive Executive Shift)
        acc_death_conflict = get_sub_attr(view.working_memory, "acc_conflict", None)
        if acc_death_conflict is not None and getattr(acc_death_conflict, "active_conflict", False):
            c_radius = float(getattr(acc_death_conflict, "cluster_radius", 3.0))
            # Invalidate any active plans routing into failure clusters
            if view.mental_plan:
                for cl in acc_death_conflict.clusters:
                    if cl.count >= 2:
                        for step_s in view.mental_plan:
                            if step_s.predicted_avatar_pos is not None:
                                pr, pc = step_s.predicted_avatar_pos
                                if math.hypot(pr - cl.center_r, pc - cl.center_c) <= c_radius:
                                    view.mental_plan.clear()
                                    view.state.current_simulated_goal = None
                                    break
                            if not view.mental_plan:
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
                                    view.hazard_tracker.static_lethal_positions.add((lr, lc))
                                    cell_f = int(curr_grid[lr, lc])
                                    if cell_f in view.symbolic_theory.goal_features:
                                        view.symbolic_theory.goal_features.discard(cell_f)
                                        view.hazard_tracker.known_lethal_features.add(cell_f)
                                        logger.warning(
                                            "ACC Metacognitive Shift: Inverted hypothesis for feature %d (GOAL -> LETHAL) due to death cluster at (%d, %d)",
                                            cell_f,
                                            lr,
                                            lc,
                                        )

            # Invalidate deadlocked subgoals in FrontopolarSubgoalStack if they target conflict zones
            if view.subgoal_stack.has_pending_goals:
                cur_sg = view.subgoal_stack.current_subgoal
                if cur_sg is not None:
                    target = cur_sg.target_entity_pos or cur_sg.target_destination
                    if target and acc_death_conflict.is_in_conflict_zone(target):
                        logger.warning(
                            "ACC Metacognitive Gating: Aborting deadlocked subgoal %s targeting conflict zone %s",
                            cur_sg.subgoal_id,
                            target,
                        )
                        view.subgoal_stack.clear_subgoals()
                        view.transition_to_phase(
                            EpistemicPhase.REPLANNING,
                            reason="acc_subgoal_conflict_abort",
                        )

        # Visual avatar localization
        known_av_feats = set(view.avatar_features) if view.avatar_features else set()
        if view.avatar_feature is not None:
            known_av_feats.add(view.avatar_feature)
        elif view.prior_avatar_feature is not None:
            known_av_feats.add(view.prior_avatar_feature)
        if known_av_feats:
            view._update_avatar_position_from_grid(curr_grid, known_av_feats)
            if (
                view.avatar_pos is not None
                and view.avatar_feature is None
                and view.prior_avatar_feature is not None
            ):
                view.state.avatar_feature = view.prior_avatar_feature
                view.avatar_features.add(view.prior_avatar_feature)

        # Faculty: Active Manipulandum Localization
        if view.avatar_pos is None and view.working_memory.body_schema.has_manipulandum():
            manip = get_sub_attr(view.working_memory.body_schema, "active_manipulandum", None)
            if manip is not None and getattr(manip, "pos", None) is not None:
                view.state.avatar_pos = manip.pos

        # Observe creature motion
        view._observe_oriented_threat_motion(curr_grid, available_actions, known_av_feats)

        # Faculty: Epistemic & Perceptual Goal Scheduling (Stage 1 Active Inference)
        # Identify structural and candidate goals prior to forward mental simulation once avatar is recognized.
        if not view.learned_goal_positions and (
            view.avatar_pos is not None or bool(view.avatar_features)
        ):
            bg_val = (
                view.bg_feature
                if view.bg_feature is not None
                else view.estimate_background(curr_grid)
            )
            entities = view.extract_entities(curr_grid, bg_val)
            av_feats_cand = view.avatar_features or (
                {view.avatar_feature} if view.avatar_feature is not None else set()
            )
            proposal = EpistemicGoalScheduler.schedule_goals(
                engine=view,
                curr_grid=curr_grid,
                entities=entities,
                bg=bg_val,
                av_feats=av_feats_cand,
                target_role=EntityRole.GOAL,
                available_actions=available_actions,
            )
            for gp in proposal.new_learned_goal_positions:
                view.record_learned_goal(gp)
            for feat in proposal.new_candidate_features:
                view.symbolic_theory.candidate_goal_features.add(feat)

        # Faculty E: Basal Ganglia Motor Gating & Phase Entrainment
        if view.pending_phase_wait_steps > 0:
            view.pending_phase_wait_steps -= 1
            undo_acts = get_engine_state_attr(view, "undo_actions", set())
            non_disp = [
                a
                for a in available_actions
                if not view.is_displacement_action(a)
                and not view.is_spatial_effector(a)
                and a not in undo_acts
            ]
            if non_disp:
                return ActionProposal(
                    non_disp[0], None, branch="non_displacement_fallback", grid=curr_grid
                )

        # Faculty: Brodmann Area 10 (aPFC) Hierarchical Subgoal Stack & Cognitive Branching
        if view.subgoal_stack.has_pending_goals:
            cur_sg = view.subgoal_stack.current_subgoal
            if cur_sg is not None and view.subgoal_stack.is_subgoal_satisfied(
                cur_sg, set(), avatar_pos=view.avatar_pos
            ):
                view.subgoal_stack.pop_subgoal()

        chosen_action: Any
        chosen_data: dict[str, Any] | None = None
        predicted_pos: tuple[int, int] | None = None

        # 0. ACC Forced Perturbation: break limit-cycle oscillation immediately
        if forced_acc_perturbation is not None:
            logger.info(
                "Anterior Cingulate Cortex (ACC): Breaking limit cycle oscillation with orthogonal perturbation action %s",
                forced_acc_perturbation,
            )
            view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
            chosen_action = forced_acc_perturbation
            chosen_data = None

        # 1. Prefrontal Affordance Panel & Optical Symmetry Gating
        elif (
            affordance_choice := self.arbitrate_affordances(
                view, curr_grid, available_actions, known_av_feats
            )
        ) is not None and view.inhibited_actions.get(affordance_choice[0], 0) == 0:
            chosen_action, chosen_data = affordance_choice

        # 2. Check active mental plan during EXPLOITATION phase
        elif view.phase == EpistemicPhase.EXPLOITATION and view.mental_plan:
            # Active Inference & Efference Copy Divergence:
            # If the organism is physically stuck or the previous step produced zero movement,
            # the counterfactual mental rollout is falsified. Clear stale plan immediately.
            if view.consecutive_stuck_steps > 0:
                view.mental_plan.clear()
                view.state.phase = EpistemicPhase.REPLANNING
                macro_plan = view.object_planner.plan_macro_option(
                    view, curr_grid, available_actions
                )
                if macro_plan:
                    view.phase = EpistemicPhase.EXPLOITATION
                    view.mental_plan = deque(macro_plan)
                    next_step = view.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    return ActionProposal(
                        chosen_action,
                        chosen_data,
                        branch="macro_plan",
                        predicted_pos=predicted_pos,
                        grid=curr_grid,
                    )
                else:
                    chosen_action, chosen_data = view.plan_epistemic_probe(
                        curr_grid, available_actions
                    )
                    return ActionProposal(
                        chosen_action, chosen_data, branch="epistemic_probe", grid=curr_grid
                    )
            else:
                next_step = view.mental_plan.popleft()
            if not view.mental_plan:
                view.state.phase = EpistemicPhase.REPLANNING
                if view.object_planner.active_target_object_id:
                    view.object_planner.ledger.record_interaction(
                        view.object_planner.active_target_object_id,
                        next_step.predicted_avatar_pos or view.avatar_pos or (0, 0),
                    )
                    view.object_planner.active_target_object_id = None

            plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                view, curr_grid, available_actions, next_step
            )

            if should_wait:
                chosen_action, chosen_data, predicted_pos = self._execute_temporal_pacing(
                    view, curr_grid, available_actions, next_step, wait_steps
                )
            elif not plan_safe:
                view.state.consecutive_pacing_steps = 0
                view.mental_plan.clear()

                view.state.phase = EpistemicPhase.REPLANNING
                # Primary replanning: Object-Centric Macro Option
                macro_plan = view.object_planner.plan_macro_option(
                    view, curr_grid, available_actions
                )
                if macro_plan:
                    view.phase = EpistemicPhase.EXPLOITATION
                    view.mental_plan = deque(macro_plan)
                    next_step = view.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    danger_cells = (
                        {next_step.predicted_avatar_pos}
                        if next_step.predicted_avatar_pos is not None and not has_temporal
                        else set()
                    )
                    simulated_plan = view.simulate_in_mind(
                        curr_grid, available_actions, blocked_cells=danger_cells
                    )
                    if simulated_plan:
                        view.consecutive_simulation_failures = 0
                        view.simulation_cooldown = 0
                        view.state.phase = EpistemicPhase.EXPLOITATION
                        view.state.mental_plan = deque(simulated_plan)
                    if view.mental_plan:
                        next_step = view.mental_plan.popleft()
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                    else:
                        chosen_action, chosen_data = view.plan_epistemic_probe(
                            curr_grid, available_actions
                        )
            elif (
                next_step.action in available_actions
                and (has_temporal or view.inhibited_actions.get(next_step.action, 0) == 0)
                and not view.working_memory.habenular_ior.is_action_inhibited(
                    view.avatar_pos, next_step.action, next_step.action_data
                )
            ):
                view.state.consecutive_pacing_steps = 0
                chosen_action = next_step.action
                chosen_data = next_step.action_data
                predicted_pos = next_step.predicted_avatar_pos

            else:
                view.mental_plan.clear()
                view.state.phase = EpistemicPhase.REPLANNING
                macro_plan = view.object_planner.plan_macro_option(
                    view, curr_grid, available_actions
                )
                if macro_plan:
                    view.phase = EpistemicPhase.EXPLOITATION
                    view.mental_plan = deque(macro_plan)
                    next_step = view.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                else:
                    chosen_action, chosen_data = view.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        # 3. Master Planning Branch (when no active plan is in flight)
        else:
            chosen_action = None
            chosen_data = None
            predicted_pos = None

            # 3A. Primary: Forward Mental Simulation over learned dynamics & verified exits
            # (Executive Directive 1 & 2: Check if exit is open, if open go through it)
            if view.is_motor_grounded() and view.simulation_cooldown == 0:
                simulated_plan = view.simulate_in_mind(curr_grid, available_actions)
                if simulated_plan:
                    view.consecutive_simulation_failures = 0
                    view.simulation_cooldown = 0
                    view.phase = EpistemicPhase.EXPLOITATION
                    view.mental_plan = deque(simulated_plan)
                    next_step = view.mental_plan.popleft()
                    plan_safe, should_wait, wait_steps = self.validate_plan_safety(
                        view, curr_grid, available_actions, next_step
                    )
                    if should_wait:
                        chosen_action, chosen_data, predicted_pos = self._execute_temporal_pacing(
                            view, curr_grid, available_actions, next_step, wait_steps
                        )
                    elif (
                        plan_safe
                        and next_step.action in available_actions
                        and (has_temporal or view.inhibited_actions.get(next_step.action, 0) == 0)
                        and not view.working_memory.habenular_ior.is_action_inhibited(
                            view.avatar_pos, next_step.action, next_step.action_data
                        )
                    ):
                        chosen_action = next_step.action
                        chosen_data = next_step.action_data
                        predicted_pos = next_step.predicted_avatar_pos
                    else:
                        view.mental_plan.clear()
                        view.state.phase = EpistemicPhase.REPLANNING
                        chosen_action, chosen_data = view.plan_epistemic_probe(
                            curr_grid, available_actions
                        )

                else:
                    view.consecutive_simulation_failures += 1
                    view.simulation_cooldown = min(
                        12, 2 ** min(view.consecutive_simulation_failures, 4)
                    )
                    view.working_memory.update_salience_baseline(view)

            # 3A-bis. Causal Theory Planning (Phase B):
            # If mental simulation failed/cooled down, try constructing a plan
            # from discovered causal predicates and action dynamics.
            if chosen_action is None and view.is_motor_grounded():
                try:
                    from hbllm.hcir.world.autonomous_learning import CausalTheoryPlanner

                    if view.avatar_pos is not None:
                        ctx = CausalTheoryPlanner.extract_planning_context(view, curr_grid)
                        if ctx.goal_positions and ctx.action_displacements:
                            causal_path = CausalTheoryPlanner.plan_path(
                                view.avatar_pos,
                                curr_grid,
                                ctx,
                                max_expansions=300,
                            )
                            # Phase F: Multi-Stage Chained Prerequisite Backchaining
                            if not causal_path:
                                tool_affinities = getattr(
                                    view.working_memory, "tool_barrier_affinities", {}
                                )
                                if not tool_affinities and hasattr(view, "symbolic_theory"):
                                    tool_affinities = getattr(
                                        view.symbolic_theory, "tool_barrier_affinities", {}
                                    )
                                causal_path = CausalTheoryPlanner.plan_chained_path(
                                    view.avatar_pos,
                                    curr_grid,
                                    ctx,
                                    tool_barrier_affinities=tool_affinities,
                                    max_expansions=400,
                                )
                            if causal_path:
                                from hbllm.hcir.world.cortex_planner import MentalSimulationStep

                                causal_plan: list[MentalSimulationStep] = []
                                sim_pos = view.avatar_pos
                                for act_id in causal_path:
                                    dr, dc = ctx.action_displacements.get(act_id, (0, 0))
                                    next_pos = (sim_pos[0] + dr, sim_pos[1] + dc)
                                    causal_plan.append(
                                        MentalSimulationStep(
                                            action=act_id,
                                            predicted_avatar_pos=next_pos,
                                        )
                                    )
                                    sim_pos = next_pos
                                view.state.phase = EpistemicPhase.EXPLOITATION
                                view.state.mental_plan = deque(causal_plan)
                                next_step = view.mental_plan.popleft()
                                chosen_action = next_step.action
                                chosen_data = next_step.action_data
                                predicted_pos = next_step.predicted_avatar_pos
                                logger.info(
                                    "Phase B: CausalTheoryPlanner produced %d-step path",
                                    len(causal_path),
                                )
                except Exception:
                    pass  # Graceful degradation if causal planner fails

            # 3B. Secondary: Object-Centric Macro Action Planning (Directives 3, 4, 5)
            # If exit is not open/reachable or in search backoff, interact with intermediate objects
            if chosen_action is None:
                macro_plan = view.object_planner.plan_macro_option(
                    view, curr_grid, available_actions
                )
                if (
                    macro_plan
                    and macro_plan[0].action in available_actions
                    and view.inhibited_actions.get(macro_plan[0].action, 0) == 0
                    and not view.working_memory.habenular_ior.is_action_inhibited(
                        view.avatar_pos, macro_plan[0].action, macro_plan[0].action_data
                    )
                ):
                    view.phase = EpistemicPhase.EXPLOITATION
                    view.mental_plan = deque(macro_plan)
                    next_step = view.mental_plan.popleft()
                    chosen_action = next_step.action
                    chosen_data = next_step.action_data
                    predicted_pos = next_step.predicted_avatar_pos
                    if not view.mental_plan:
                        view.state.phase = EpistemicPhase.REPLANNING
                        if view.object_planner.active_target_object_id:
                            view.object_planner.ledger.record_interaction(
                                view.object_planner.active_target_object_id,
                                predicted_pos or view.avatar_pos or (0, 0),
                            )
                            view.object_planner.active_target_object_id = None
                else:
                    if view.exploration_cooldown > 0:
                        view.exploration_cooldown -= 1
                    view.state.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    chosen_action, chosen_data = view.plan_epistemic_probe(
                        curr_grid, available_actions
                    )

        # Effector Grounding Invariant
        if view.is_spatial_effector(chosen_action):
            aff = view.action_affordances.get(chosen_action)
            req_keys = aff.target_param_keys if aff else ("x", "y")
            if (
                chosen_data is None
                or not isinstance(chosen_data, dict)
                or not all(k in chosen_data for k in req_keys)
            ):
                chosen_data = view.ground_effector_action(curr_grid, chosen_action)
                if chosen_data is None:
                    view.inhibited_actions[chosen_action] = max(
                        view.inhibited_actions.get(chosen_action, 0), 6
                    )
                    non_eff = [
                        a
                        for a in available_actions
                        if not view.is_spatial_effector(a) and view.inhibited_actions.get(a, 0) == 0
                    ]
                    chosen_action = non_eff[0] if non_eff else available_actions[0]
                    chosen_data = None

        # Fallback to exploratory spatial effector affordance if any spatial effector action is available
        if (
            (chosen_action is None or chosen_action not in available_actions)
            and hasattr(view, "point_actuation_learner")
            and view.point_actuation_learner is not None
        ):
            spatial_eff_acts = [a for a in available_actions if view.is_spatial_effector(a)]
            if spatial_eff_acts:
                eff_act = spatial_eff_acts[0]
                actuation_target = view.point_actuation_learner.pick_exploratory_actuation(
                    curr_grid, bg_feature=get_engine_state_attr(view, "bg_feature", 0)
                )
                if actuation_target is not None:
                    chosen_action = eff_act
                    aff = view.action_affordances.get(eff_act)
                    req_keys = aff.target_param_keys if aff else ("row", "col")
                    if "row" in req_keys and "col" in req_keys:
                        chosen_data = {"row": actuation_target[0], "col": actuation_target[1]}
                    elif "y" in req_keys and "x" in req_keys:
                        chosen_data = {"y": actuation_target[0], "x": actuation_target[1]}
                    else:
                        chosen_data = view.ground_effector_action(curr_grid, eff_act)

        return ActionProposal(
            chosen_action,
            chosen_data,
            branch="default_arbitration",
            predicted_pos=predicted_pos,
            grid=curr_grid,
        )

    @classmethod
    def _finalize_step(
        cls,
        view: Any,
        curr_grid: np.ndarray,
        chosen_action: Any,
        chosen_data: dict[str, Any] | None,
        predicted_pos: tuple[int, int] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        view.state.prev_grid = curr_grid.copy()
        if hasattr(view, "grid_history"):
            view.grid_history.append(curr_grid.copy())
        view.state.last_action = chosen_action
        view.state.last_action_data = chosen_data
        view.state.last_predicted_pos = predicted_pos
        return chosen_action, chosen_data

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export prefrontal deliberation view capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="prefrontal_deliberation_engine",
            operator="==",
            value="ACTIVE",
            relation="dlpfc_vmpfc_arbitration",
            target_entity="cortex_deliberation",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering deliberation view."""
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


@dataclass
@dataclass(frozen=True)
class ActionProposal:
    """Deliberation's decision intent — finalised by MotorStage (WP1)."""

    action: Any
    action_data: Any = None
    branch: str = "unknown"
    predicted_pos: Any = None
    grid: Any = None
    finalized: bool = False


class DeliberationStage(CorticalStage):
    """Cortical Stage 4: Prefrontal Deliberation & Arbitration.

    Receives CognitiveContext containing candidate proposals from upstream stages
    (such as simulation, macro actions, or episodic habit) and outputs a pure StageResult
    with arbitrated action and proposed transaction operations.
    """

    name: str = "deliberation_stage"

    def observe(self, ctx: CognitiveContext) -> StageResult:
        engine = ctx.signals.get("_engine")
        if engine is not None:
            # WP1: the real faculty — runs the full PrefrontalDeliberationEngine.decide
            # via a journaled view. NOTE: Perception/Assimilation/Simulation/Motor are
            # folded into this call (assimilate_feedback is invoked internally at
            # cortex_deliberation:~1302 and finalize in _finalize_step) — decomposing
            # them would duplicate work; documented per WP1 stage-fold allowance.
            from hbllm.hcir.world.engine_state import (
                FacultyWriteJournal,
                JournaledEngineView,
            )

            journal = FacultyWriteJournal(author="deliberation")
            proposal = engine.deliberation_engine.deliberate(
                JournaledEngineView(engine, journal),
                curr_grid=ctx.signals.get("_curr_grid"),
                available_actions=ctx.signals.get("_available_actions", []),
                terminal=ctx.signals.get("_terminal"),
                is_success=ctx.signals.get("_is_success", False),
                is_failure=ctx.signals.get("_is_failure", False),
                action_schemas=ctx.signals.get("_action_schemas"),
                objects=ctx.signals.get("_objects"),
                micro_actions=ctx.signals.get("_micro_actions"),
                **ctx.signals.get("_kwargs", {}),
            )
            return StageResult(
                author=self.name,
                journal=journal,
                signals={"_proposal": proposal},
            )
        act = ctx.signals.get("mental_plan_action")
        if act is None:
            act = ctx.signals.get("macro_action")
        if act is None:
            act = ctx.signals.get("curiosity_action")
        if act is None:
            actions = list(ctx.action_dynamics.keys())
            act = actions[0] if actions else 1

        ops: list[TransactionOperation] = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="deliberation_decision",
                node_data={
                    "stage": self.name,
                    "arbitrated_action": str(act),
                    "step": ctx.step_counter,
                },
            )
        ]
        return StageResult(
            author=self.name,
            operations=ops,
            signals={"arbitrated_action": act},
            proposed_action=act,
        )
