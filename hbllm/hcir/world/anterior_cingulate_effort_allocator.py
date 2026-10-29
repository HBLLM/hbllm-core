"""Anterior Cingulate Cortex (ACC) Conflict Monitoring & Cognitive Effort Allocation.

Biologically grounded in primate anterior cingulate cortex (ACC) conflict monitoring
and Expected Value of Control (EVC) theory (Botvinick et al., 2001; Frank, 2006;
Shenhav, Botvinick, & Cohen, 2013):
1. Response Conflict Computation:
   - Evaluates competition between alternative action channels:
     Conflict_response = 1.0 - (|Q_best - Q_second| / (sum |Q| + epsilon)).
   - Integrates environmental hazard proximity and state novelty.
2. Expected Value of Control (EVC) Optimization:
   - Weighs the anticipated value gain of cognitive deliberation against its
     metabolic/temporal cost:
     EVC = Expected_Reward(Deliberation) - Cost(Deliberative_Effort).
3. Dynamic Control Gating:
   - Low Conflict / High Familiarity: Allocates minimal effort (depth k=1, 0 rollouts),
     reverting to fast, habitual, low-energy reflexive execution.
   - High Conflict / Critical Hazard: Scales deliberative search depth (k=4..8) and
     counterfactual simulation budgets to avoid fatal traps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class ConflictProfile:
    """Multi-attribute conflict and urgency profile computed by ACC."""

    response_conflict: float  # [0, 1] action value competition
    hazard_urgency: float  # [0, 1] proximity to lethal threat
    novelty_uncertainty: float  # [0, 1] lack of state familiarity
    composite_conflict: float  # [0, 1] weighted overall conflict level


@dataclass
class EffortAllocationDecision:
    """The executive allocation of deliberative effort and simulation budget."""

    effort_level: float  # [0, 1] continuous intensity of cognitive control
    search_depth: int  # Recommended tree search / rollout depth [1, 8]
    simulation_budget: int  # Number of forward MCTS simulation rollouts
    is_habitual: bool  # True if fast reflexive execution is sufficient
    cognitive_cost: float  # Energy penalty deducted from interoceptive reserves
    rationale: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Export effort allocation decision as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="cognitive_effort_allocation",
            operator="==",
            value=f"depth:{self.search_depth}:budget:{self.simulation_budget}:hab:{self.is_habitual}",
            relation="acc_evc_gating",
            target_entity="prefrontal_cortex",
        )


class AnteriorCingulateEffortAllocator:
    """ACC executive controller optimizing cognitive effort and search budgets."""

    def __init__(
        self,
        base_depth: int = 2,
        max_depth: int = 6,
        base_simulations: int = 5,
        max_simulations: int = 25,
        conflict_threshold_habit: float = 0.25,
    ) -> None:
        """Initialize anterior cingulate effort allocator.

        Args:
            base_depth: Default shallow search depth for routine tasks.
            max_depth: Upper bound for deep deliberative tree searches.
            base_simulations: Routine simulation rollout budget.
            max_simulations: Ceiling on MCTS rollouts during high conflict.
            conflict_threshold_habit: Threshold below which habitual execution is favored.
        """
        self.base_depth = max(1, int(base_depth))
        self.max_depth = max(self.base_depth, int(max_depth))
        self.base_simulations = max(0, int(base_simulations))
        self.max_simulations = max(self.base_simulations, int(max_simulations))
        self.habit_threshold = max(0.05, float(conflict_threshold_habit))

        self.last_decision: EffortAllocationDecision | None = None
        self.total_allocations: int = 0

    def compute_conflict_profile(
        self,
        action_values: dict[Any, float],
        hazard_distance: float | None = None,
        state_familiarity: float = 1.0,
    ) -> ConflictProfile:
        """Evaluate response competition, hazard urgency, and epistemic uncertainty."""
        # 1. Action response conflict
        if len(action_values) >= 2:
            sorted_vals = sorted(action_values.values(), reverse=True)
            v1, v2 = sorted_vals[0], sorted_vals[1]
            diff = abs(v1 - v2)
            total_mag = sum(abs(v) for v in sorted_vals) + 1e-5
            # Close competing values yield high conflict near 1.0
            response_conflict = float(max(0.0, min(1.0, 1.0 - (diff / total_mag))))
        else:
            response_conflict = 0.0

        # 2. Hazard urgency
        if hazard_distance is not None:
            if hazard_distance <= 1.5:
                hazard_urgency = 0.95
            elif hazard_distance <= 3.0:
                hazard_urgency = 0.60
            else:
                hazard_urgency = max(0.0, 1.0 / (1.0 + 0.3 * hazard_distance))
        else:
            hazard_urgency = 0.0

        # 3. Novelty uncertainty
        novelty = float(max(0.0, min(1.0, 1.0 - state_familiarity)))

        # Composite conflict
        composite = 0.45 * response_conflict + 0.35 * hazard_urgency + 0.20 * novelty
        composite = float(max(0.0, min(1.0, composite)))

        return ConflictProfile(
            response_conflict=response_conflict,
            hazard_urgency=hazard_urgency,
            novelty_uncertainty=novelty,
            composite_conflict=composite,
        )

    def allocate_effort(
        self,
        profile: ConflictProfile,
        energy_reserve: float = 100.0,
    ) -> EffortAllocationDecision:
        """Determine optimal search depth and simulation budget based on EVC balance.

        Args:
            profile: Current ACC conflict profile.
            energy_reserve: Visceral energetic budget remaining in [0, 100].

        Returns:
            EffortAllocationDecision gating forward deliberation.
        """
        self.total_allocations += 1

        # Check if energetic reserves are depleted (fatigue inhibits deep deliberation)
        energy_factor = max(0.1, min(1.0, energy_reserve / 100.0))

        # Check for habitual regime
        if profile.composite_conflict <= self.habit_threshold and profile.hazard_urgency < 0.3:
            decision = EffortAllocationDecision(
                effort_level=0.1,
                search_depth=1,
                simulation_budget=0,
                is_habitual=True,
                cognitive_cost=0.05,
                rationale="habitual_reflexive_execution_low_conflict",
            )
            self.last_decision = decision
            return decision

        # Scale effort continuously between [0.2, 1.0] based on conflict and energy
        raw_effort = profile.composite_conflict * energy_factor
        effort_level = float(max(0.2, min(1.0, raw_effort)))

        # Compute search depth and simulation count
        depth_range = self.max_depth - self.base_depth
        search_depth = self.base_depth + int(round(effort_level * depth_range))

        sim_range = self.max_simulations - self.base_simulations
        simulation_budget = self.base_simulations + int(round(effort_level * sim_range))

        # Effort cost scales quadratically with depth and simulations
        cognitive_cost = float(0.5 * (search_depth**1.5) + 0.1 * simulation_budget)

        decision = EffortAllocationDecision(
            effort_level=effort_level,
            search_depth=search_depth,
            simulation_budget=simulation_budget,
            is_habitual=False,
            cognitive_cost=cognitive_cost,
            rationale=f"deliberative_evc_gated_conflict_{profile.composite_conflict:.2f}",
        )
        self.last_decision = decision
        return decision

    def reset(self) -> None:
        """Reset internal history and state."""
        self.last_decision = None
        self.total_allocations = 0
