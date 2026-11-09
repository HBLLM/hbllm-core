"""Homeostatic Intrinsic Motivation & Goal Arbitration System (Gap 5, WP10).

Biologically grounded in:
1. Karl Friston's Expected Free Energy (EFE) formulation of Active Inference:
   G(goal) = w_extrinsic * Extrinsic_Value + w_epistemic * Epistemic_Value - w_cost * Estimated_Cost
2. Kenji Doya's Neuromodulatory Meta-Parameter Regulation:
   - Dopamine (DA): Modulates w_extrinsic (surges exploitation of confirmed rewards & instructables).
   - Noradrenaline (NE): Modulates w_epistemic (surges curiosity & exploration on unexpected uncertainty).
   - Serotonin (5-HT): Modulates w_cost (regulates temporal patience; low 5-HT penalizes costly/distant paths).
3. Allostatic & Homeostatic Regulation in Primate Prefrontal Cortex:
   - Social instructions provide overriding extrinsic attractors, preempting open-ended curiosity until satisfied.
   - Low-energy states (energy_budget <= 25.0) heavily penalize costs and risks, prioritizing low-cost,
     high-safety subgoals over open exploration.
4. Event-Sourced Tiered Workspace Binding:
   - Selecting a winning goal activates / creates a corresponding TaskFrame in the Working Tier.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.workspace_tiers import TaskFrame, TieredWorkspace
    from hbllm.hcir.world.prefrontal_working_memory import HomeostaticState

logger = logging.getLogger(__name__)


@dataclass
class CandidateGoal:
    """A candidate attractor state considered for active inference goal arbitration."""

    goal_id: str
    target: tuple[int, int] | str
    goal_type: str  # "social_instruction", "confirmed_extrinsic", "epistemic_exploration", "safety_preservation"
    extrinsic_utility: float = 0.0
    epistemic_utility: float = 0.0
    estimated_cost: float = 1.0
    hazard_risk: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export candidate goal state as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"candidate_goal:{self.goal_id}",
            operator="==",
            value=str(self.target),
            relation=f"type:{self.goal_type}:cost:{self.estimated_cost:.1f}",
            target_entity="goal_arbiter",
        )


@dataclass
class ArbitrationResult:
    """The result of an Expected Free Energy goal arbitration cycle."""

    selected_goal: CandidateGoal
    expected_free_energy: float
    all_scores: dict[str, float] = field(default_factory=dict)
    active_task_frame: Any | None = None  # TaskFrame
    applied_weights: dict[str, float] = field(default_factory=dict)

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction logging arbitration result."""
        node_id = f"arbitration_{self.selected_goal.goal_id}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "selected_goal_id": self.selected_goal.goal_id,
                    "target": list(self.selected_goal.target)
                    if isinstance(self.selected_goal.target, tuple)
                    else str(self.selected_goal.target),
                    "goal_type": self.selected_goal.goal_type,
                    "expected_free_energy": float(self.expected_free_energy),
                    "applied_weights": dict(self.applied_weights),
                    "all_scores": dict(self.all_scores),
                },
            )
        ]


class GoalArbiter:
    """Homeostatic & Neuromodulatory Goal Arbiter (Gap 5, WP10).

    Computes Expected Free Energy (EFE) across candidate goals, balancing extrinsic utility,
    epistemic information gain, and action costs, modulated dynamically by neuromodulators
    and allostatic energy constraints.
    """

    def __init__(
        self,
        base_extrinsic_weight: float = 1.0,
        base_epistemic_weight: float = 1.0,
        base_cost_weight: float = 1.0,
    ) -> None:
        self.base_extrinsic_weight = base_extrinsic_weight
        self.base_epistemic_weight = base_epistemic_weight
        self.base_cost_weight = base_cost_weight
        self.last_arbitration: ArbitrationResult | None = None

    def compute_arbitration_weights(
        self,
        homeostatic_state: HomeostaticState | None = None,
        neuromodulator_system: Any | None = None,
    ) -> dict[str, float]:
        """Compute dynamic weights (w_extrinsic, w_epistemic, w_cost) from biological states."""
        da = 1.0
        ne = 1.0
        serotonin = 1.0

        if neuromodulator_system is not None:
            cur = getattr(neuromodulator_system, "current", neuromodulator_system)
            da = float(getattr(cur, "dopamine", 1.0))
            ne = float(getattr(cur, "noradrenaline", 1.0))
            serotonin = float(getattr(cur, "serotonin", 1.0))

        # Neuromodulatory dynamic modulation:
        # 1. Dopamine scales extrinsic value exploitation
        w_ext = self.base_extrinsic_weight * max(0.1, min(4.0, da))
        # 2. Noradrenaline scales epistemic information seeking
        w_epi = self.base_epistemic_weight * max(0.1, min(4.0, ne))
        # 3. Serotonin scales patience (inversely scaling cost penalty)
        w_cost = self.base_cost_weight * (1.0 / max(0.25, min(4.0, serotonin)))

        # Homeostatic state modulation:
        if homeostatic_state is not None:
            energy = max(0.0, min(100.0, float(homeostatic_state.energy_budget)))
            energy_ratio = energy / 100.0

            if homeostatic_state.is_critical_energy:
                # Critical energy depletion (<= 25.0):
                # Cost penalty quadruples and epistemic curiosity is suppressed by 90%
                w_cost *= 1.0 + 3.0 * (1.0 - energy_ratio)
                w_epi *= 0.10
            else:
                # Mild energy deficit linearly increases cost caution and adjusts curiosity
                w_cost *= 1.0 + 0.5 * (1.0 - energy_ratio)
                w_epi *= max(0.25, energy_ratio)

        return {
            "w_extrinsic": w_ext,
            "w_epistemic": w_epi,
            "w_cost": w_cost,
        }

    def compute_expected_free_energy(
        self,
        candidate: CandidateGoal,
        weights: dict[str, float],
        homeostatic_state: HomeostaticState | None = None,
    ) -> float:
        """Compute Friston's Expected Free Energy (EFE) score for a candidate attractor.

        Higher score indicates higher subjective desirability.
        G = w_ext * Extrinsic + w_epi * Epistemic - w_cost * Cost - Risk_Penalty
        """
        w_ext = weights.get("w_extrinsic", 1.0)
        w_epi = weights.get("w_epistemic", 1.0)
        w_cost = weights.get("w_cost", 1.0)

        # Baseline EFE evaluation
        score = (
            w_ext * candidate.extrinsic_utility
            + w_epi * candidate.epistemic_utility
            - w_cost * candidate.estimated_cost
            - (w_cost * 2.5 * candidate.hazard_risk)
        )

        # 1. Social Instruction Gating: Instructed goals receive decisive priority (+100.0)
        if candidate.goal_type == "social_instruction":
            score += 100.0

        # 2. Critical Energy Self-Preservation Gating: Safety retreat goals receive decisive priority (+60.0)
        if homeostatic_state is not None and homeostatic_state.is_critical_energy:
            if candidate.goal_type == "safety_preservation":
                score += 60.0

        return score

    def extract_candidate_goals(
        self,
        engine: Any,
        ctx: Any | None = None,
    ) -> list[CandidateGoal]:
        """Extract candidate goals from working memory, theory, and environmental state."""
        candidates: list[CandidateGoal] = []
        avatar_pos = getattr(engine, "avatar_pos", None)
        homeostatic = getattr(engine, "homeostatic_state", None)

        # 1. Social Instructions (highest explicit priority)
        if homeostatic is not None and homeostatic.social_instructions:
            for idx, ins in enumerate(homeostatic.social_instructions):
                gid = ins.get("goal_id", f"instructed_{idx}")
                target = ins.get("target", (0, 0))
                cost = float(ins.get("cost", 1.0))
                if avatar_pos is not None and isinstance(target, tuple) and len(target) == 2:
                    cost = max(
                        1.0, float(abs(avatar_pos[0] - target[0]) + abs(avatar_pos[1] - target[1]))
                    )
                candidates.append(
                    CandidateGoal(
                        goal_id=gid,
                        target=target,
                        goal_type="social_instruction",
                        extrinsic_utility=float(ins.get("utility", 80.0)),
                        epistemic_utility=0.0,
                        estimated_cost=cost,
                        hazard_risk=0.0,
                        metadata=dict(ins),
                    )
                )

        # 2. Confirmed Extrinsic Goals (known reward attractors)
        goals = getattr(engine, "goals", []) or []
        for gp in goals:
            if avatar_pos is not None and gp == avatar_pos:
                continue
            cost = 1.0
            if avatar_pos is not None and isinstance(gp, tuple) and len(gp) == 2:
                cost = max(1.0, float(abs(avatar_pos[0] - gp[0]) + abs(avatar_pos[1] - gp[1])))
            candidates.append(
                CandidateGoal(
                    goal_id=f"goal_{gp[0]}_{gp[1]}",
                    target=gp,
                    goal_type="confirmed_extrinsic",
                    extrinsic_utility=40.0,
                    epistemic_utility=2.0,
                    estimated_cost=cost,
                    hazard_risk=0.0,
                )
            )

        # 3. Epistemic Exploratory Attractors (unverified candidates, frontiers)
        theory = getattr(engine, "symbolic_theory", None)
        if theory is not None and hasattr(theory, "candidate_goal_features"):
            grid = getattr(engine, "prev_grid", None)
            if grid is not None and hasattr(grid, "shape"):
                H, W = grid.shape
                for feat in theory.candidate_goal_features:
                    coords = list(zip(*((grid == feat).nonzero())))
                    for r, c in coords[:3]:  # sample top candidate positions
                        cost = 1.0
                        if avatar_pos is not None:
                            cost = max(1.0, float(abs(avatar_pos[0] - r) + abs(avatar_pos[1] - c)))
                        candidates.append(
                            CandidateGoal(
                                goal_id=f"explore_feat_{feat}_{r}_{c}",
                                target=(int(r), int(c)),
                                goal_type="epistemic_exploration",
                                extrinsic_utility=5.0,
                                epistemic_utility=25.0,
                                estimated_cost=cost,
                                hazard_risk=0.0,
                                metadata={"feature_id": feat},
                            )
                        )

        # 4. Safety Preservation Attractors (when energy is critical)
        if homeostatic is not None and homeostatic.is_critical_energy and avatar_pos is not None:
            # Nearest safe low-cost waypoint (immediate resting / sheltering)
            candidates.append(
                CandidateGoal(
                    goal_id="safety_haven_local",
                    target=avatar_pos,
                    goal_type="safety_preservation",
                    extrinsic_utility=30.0,
                    epistemic_utility=0.0,
                    estimated_cost=0.5,
                    hazard_risk=0.0,
                )
            )

        return candidates

    def arbitrate(
        self,
        candidates: Sequence[CandidateGoal],
        engine: Any = None,
        homeostatic_state: HomeostaticState | None = None,
        neuromodulator_system: Any | None = None,
        tiered_workspace: TieredWorkspace | Any | None = None,
    ) -> ArbitrationResult | None:
        """Execute Expected Free Energy arbitration to select the active goal and bind TaskFrame."""
        if not candidates:
            return None

        # Resolve homeostatic state and neuromodulatory system from engine if not provided
        if homeostatic_state is None and engine is not None:
            homeostatic_state = getattr(engine, "homeostatic_state", None)
        if neuromodulator_system is None and engine is not None:
            neuromodulator_system = getattr(engine, "neuromodulatory_system", None)
        if tiered_workspace is None and engine is not None:
            tiered_workspace = getattr(engine, "tiered_workspace", None)

        # 1. Compute dynamic weights
        weights = self.compute_arbitration_weights(
            homeostatic_state=homeostatic_state,
            neuromodulator_system=neuromodulator_system,
        )

        # 2. Score all candidate goals under EFE
        scores: dict[str, float] = {}
        for cand in candidates:
            score = self.compute_expected_free_energy(
                candidate=cand,
                weights=weights,
                homeostatic_state=homeostatic_state,
            )
            scores[cand.goal_id] = score

        # 3. Select winning candidate maximizing EFE
        winning_candidate = max(candidates, key=lambda c: scores[c.goal_id])
        winning_score = scores[winning_candidate.goal_id]

        # 4. Activate or create corresponding TaskFrame in TieredWorkspace
        active_frame: TaskFrame | None = None
        if tiered_workspace is not None:
            try:
                existing_frame = None
                if hasattr(tiered_workspace, "get_task_frame_by_goal"):
                    existing_frame = tiered_workspace.get_task_frame_by_goal(
                        winning_candidate.goal_id
                    )
                elif hasattr(tiered_workspace, "get_frame_by_goal"):
                    existing_frame = tiered_workspace.get_frame_by_goal(winning_candidate.goal_id)
                elif hasattr(tiered_workspace, "working") and hasattr(
                    tiered_workspace.working, "get_frame_by_goal"
                ):
                    existing_frame = tiered_workspace.working.get_frame_by_goal(
                        winning_candidate.goal_id
                    )

                if existing_frame is not None and getattr(existing_frame, "is_active", False):
                    active_frame = existing_frame
                elif hasattr(tiered_workspace, "create_task_frame"):
                    active_frame = tiered_workspace.create_task_frame(
                        goal_id=winning_candidate.goal_id
                    )

                if engine is not None:
                    engine.active_task_frame = active_frame
            except Exception as e:
                logger.debug("GoalArbiter: Could not bind TaskFrame: %s", e)

        result = ArbitrationResult(
            selected_goal=winning_candidate,
            expected_free_energy=winning_score,
            all_scores=scores,
            active_task_frame=active_frame,
            applied_weights=weights,
        )
        self.last_arbitration = result

        if engine is not None:
            setattr(engine, "active_arbitrated_goal", winning_candidate)
            setattr(engine, "_last_arbitrated_goal", winning_candidate)

        return result
