"""Frontal Polar Cortex (FPC / BA10) Counterfactual Branch Switching.

Biologically modeled on mammalian frontopolar cortex (BA10) tracking of
unchosen counterfactual alternatives and exploration horizons
(Koechlin & Summerfield, 2007; Boorman et al., 2009; Rushworth et al., 2011):
1. Counterfactual Option Tracking:
   - While the agent commits executive focus to a foreground goal plan, BA10
     tracks alternative, unchosen counterfactual branches in background working memory.
2. Value Dynamics & Decay:
   - Foreground value decays when progress stalls or unexpected friction arises:
     V_foreground(t+1) = V_foreground(t) * (1 - lambda_decay) + reward.
   - Background branches accumulate evidence and an exploration uncertainty bonus:
     V_eff(b) = V_projected(b) + beta_explore * uncertainty(b).
3. Executive Branch Swap Trigger:
   - When the net advantage of a background alternative exceeds the decaying
     foreground plan by a switching threshold:
     V_eff(b*) - V_foreground > Delta_switch,
     BA10 issues an executive branch-swap recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualBranch:
    """A latent, unchosen behavioral branch maintained in BA10 working memory."""

    branch_id: str
    intent: str
    target_position: tuple[int, int] | None = None
    projected_value: float = 0.5  # Expected return [0, 1]
    evidence_count: int = 1
    uncertainty: float = 0.5  # Epistemic uncertainty [0, 1]
    dormant_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_value(self) -> float:
        """Effective decision value combining projected return and uncertainty bonus."""
        return self.projected_value + 0.25 * self.uncertainty


@dataclass
class BranchSwitchDecision:
    """Outcome of BA10 counterfactual branch evaluation."""

    should_switch: bool
    active_branch_id: str
    recommended_branch_id: str | None
    value_difference: float
    confidence: float
    reason: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Export branch switch decision as a declarative HCIR CausalPredicate."""
        rec = self.recommended_branch_id or "none"
        return CausalPredicate(
            variable="branch_switch_decision",
            operator="==",
            value=f"switch:{self.should_switch}:rec:{rec}",
            relation="ba10_counterfactual_choice",
            target_entity="frontopolar_cortex",
        )


class FrontopolarExplorationHorizonEngine:
    """BA10 executive controller maintaining latent counterfactual branches and horizon switches."""

    def __init__(
        self,
        switch_threshold: float = 0.20,
        foreground_decay_rate: float = 0.05,
        exploration_bonus_weight: float = 0.30,
        max_branches: int = 5,
    ) -> None:
        """Initialize frontopolar exploration horizon engine.

        Args:
            switch_threshold: Minimum margin by which a background branch must exceed foreground.
            foreground_decay_rate: Rate at which foreground plan value decays without progress.
            exploration_bonus_weight: Multiplier on epistemic uncertainty bonus.
            max_branches: Maximum number of dormant counterfactual branches to track.
        """
        self.switch_threshold = max(0.05, float(switch_threshold))
        self.foreground_decay_rate = max(0.01, float(foreground_decay_rate))
        self.exploration_bonus_weight = max(0.0, float(exploration_bonus_weight))
        self.max_branches = max(2, int(max_branches))

        self.active_branch_id: str = "foreground_primary"
        self.foreground_value: float = 0.70
        self.branches: dict[str, CounterfactualBranch] = {}
        self.total_switches: int = 0
        self._step_counter: int = 0

    def register_counterfactual_branch(
        self,
        branch_id: str,
        intent: str,
        target_position: tuple[int, int] | None = None,
        initial_value: float = 0.50,
        uncertainty: float = 0.50,
        metadata: dict[str, Any] | None = None,
    ) -> CounterfactualBranch:
        """Register or update a candidate alternative plan in background memory."""
        if branch_id in self.branches:
            branch = self.branches[branch_id]
            branch.intent = intent
            branch.target_position = target_position
            branch.projected_value = max(0.0, min(1.0, float(initial_value)))
            branch.uncertainty = max(0.0, min(1.0, float(uncertainty)))
            return branch

        # If capacity reached, evict branch with lowest effective value
        if len(self.branches) >= self.max_branches:
            worst_id = min(self.branches.keys(), key=lambda k: self.branches[k].effective_value)
            self.branches.pop(worst_id, None)

        branch = CounterfactualBranch(
            branch_id=branch_id,
            intent=intent,
            target_position=target_position,
            projected_value=max(0.0, min(1.0, float(initial_value))),
            uncertainty=max(0.0, min(1.0, float(uncertainty))),
            dormant_steps=0,
            metadata=metadata or {},
        )
        self.branches[branch_id] = branch
        return branch

    def step_foreground_progress(
        self,
        reward: float = 0.0,
        progress_made: bool = True,
    ) -> float:
        """Update foreground value based on execution feedback and time decay."""
        self._step_counter += 1

        # Age dormant branches
        for b in self.branches.values():
            b.dormant_steps += 1

        if progress_made:
            self.foreground_value = min(1.0, self.foreground_value * 0.95 + 0.15 + reward)
        else:
            # Plan is stalling; decay foreground value
            self.foreground_value = max(
                0.05, self.foreground_value * (1.0 - self.foreground_decay_rate)
            )

        return self.foreground_value

    def update_background_evidence(
        self,
        branch_id: str,
        evidence_delta: float,
        uncertainty_reduction: float = 0.05,
    ) -> None:
        """Incorporate sensory evidence into a background branch without executing it."""
        if branch_id not in self.branches:
            return

        branch = self.branches[branch_id]
        branch.evidence_count += 1
        branch.projected_value = max(0.0, min(1.0, branch.projected_value + evidence_delta))
        branch.uncertainty = max(0.05, branch.uncertainty - max(0.0, uncertainty_reduction))

    def evaluate_branch_switch(self) -> BranchSwitchDecision:
        """Evaluate whether an unchosen counterfactual branch outvalues the active plan."""
        if not self.branches:
            return BranchSwitchDecision(
                should_switch=False,
                active_branch_id=self.active_branch_id,
                recommended_branch_id=None,
                value_difference=0.0,
                confidence=self.foreground_value,
                reason="no_counterfactual_branches_registered",
            )

        # Find best background branch
        best_id: str | None = None
        best_eff_val: float = -1.0

        for bid, branch in self.branches.items():
            eff = branch.projected_value + self.exploration_bonus_weight * branch.uncertainty
            if eff > best_eff_val:
                best_eff_val = eff
                best_id = bid

        if best_id is None:
            return BranchSwitchDecision(
                should_switch=False,
                active_branch_id=self.active_branch_id,
                recommended_branch_id=None,
                value_difference=0.0,
                confidence=self.foreground_value,
                reason="empty_candidate_evaluation",
            )

        diff = best_eff_val - self.foreground_value
        should_switch = diff > self.switch_threshold

        reason = (
            f"counterfactual_advantage_{diff:.2f}_exceeds_threshold_{self.switch_threshold:.2f}"
            if should_switch
            else f"foreground_value_{self.foreground_value:.2f}_remains_dominant"
        )

        return BranchSwitchDecision(
            should_switch=should_switch,
            active_branch_id=self.active_branch_id,
            recommended_branch_id=best_id if should_switch else None,
            value_difference=float(diff),
            confidence=max(0.1, min(1.0, best_eff_val)),
            reason=reason,
        )

    def execute_branch_switch(self, decision: BranchSwitchDecision) -> str:
        """Commit an executive branch switch, swapping the active plan with the chosen alternative."""
        if not decision.should_switch or not decision.recommended_branch_id:
            return self.active_branch_id

        old_active = self.active_branch_id
        new_active = decision.recommended_branch_id

        # Swap: old active becomes a counterfactual branch
        promoted = self.branches.pop(new_active)
        self.register_counterfactual_branch(
            branch_id=old_active,
            intent="demoted_foreground_plan",
            initial_value=self.foreground_value,
            uncertainty=0.4,
        )

        self.active_branch_id = new_active
        self.foreground_value = promoted.projected_value
        self.total_switches += 1

        logger.info(
            "BA10 executive branch swap committed: %s -> %s (val: %.2f)",
            old_active,
            new_active,
            self.foreground_value,
        )
        return self.active_branch_id

    def reset(self) -> None:
        """Reset internal branch tracking and state."""
        self.active_branch_id = "foreground_primary"
        self.foreground_value = 0.70
        self.branches.clear()
        self.total_switches = 0
        self._step_counter = 0
