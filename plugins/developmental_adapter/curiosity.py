"""Autonomous Epistemic Curiosity Engine (Stage D8).

Drives self-generated active exploration to maximize information gain
and reduce epistemic uncertainty across candidate causal models and affordances
without any task-specific external reward.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .types import (
    BabyActionType,
    BeliefTransitionEvent,
    BeliefTransitionType,
    CausalHypothesis,
    EpistemicUncertaintyReport,
)

logger = logging.getLogger(__name__)


class EpistemicCuriosityEngine:
    """Intrinsically motivated active learner maximizing expected information gain."""

    def __init__(self, substrate: BlankBrainSubstrate, env: BabyWorldEnvironment) -> None:
        self.substrate = substrate
        self.env = env
        self.candidate_hypotheses: list[CausalHypothesis] = []
        self.exploration_history: list[dict[str, Any]] = []

    def compute_entropy(self, hypotheses: list[CausalHypothesis] | None = None) -> float:
        """Calculate Shannon entropy over the epistemic confidence distribution."""
        hyps = hypotheses if hypotheses is not None else self.candidate_hypotheses
        if not hyps:
            return 0.0

        total_entropy = 0.0
        for h in hyps:
            p = max(1e-6, min(1.0 - 1e-6, h.confidence))
            # Binary entropy for each hypothesis
            h_entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
            total_entropy += h_entropy

        return total_entropy

    def generate_candidate_hypotheses_from_environment(self) -> list[CausalHypothesis]:
        """Generate candidate questions/hypotheses about unprobed objects and properties."""
        self.candidate_hypotheses.clear()

        # Probe actions across all visible objects
        actions_to_probe = [BabyActionType.PUSH, BabyActionType.ROLL, BabyActionType.GRASP]
        for oid, obj in self.env.objects.items():
            for act in actions_to_probe:
                # E.g. hypothesis: Action on this object causes MOVEMENT
                hyp = CausalHypothesis(
                    hypothesis_id=f"curiosity_{act.value}::{oid}",
                    action=act,
                    variable="mass",
                    operator="<=",
                    value=obj.mass,
                    consequence="MOVES",
                    confidence=0.5,  # Maximum uncertainty prior
                )
                self.candidate_hypotheses.append(hyp)

        return self.candidate_hypotheses

    def select_highest_information_gain_action(
        self,
    ) -> tuple[CausalHypothesis | None, BabyActionType, str | None]:
        """Select the action-target pair that provides maximum expected entropy reduction."""
        if not self.candidate_hypotheses:
            self.generate_candidate_hypotheses_from_environment()

        # Hypotheses closest to p=0.5 have maximum entropy, so testing them yields highest reduction
        best_hyp = None
        max_uncertainty = -1.0

        for h in self.candidate_hypotheses:
            if h.falsified or h.confirmed:
                continue
            # Distance from 0.5 (smaller is higher uncertainty)
            uncertainty = 1.0 - abs(h.confidence - 0.5) * 2.0
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                best_hyp = h

        if not best_hyp:
            return None, BabyActionType.LOOK, None

        # Extract target object id from hypothesis_id
        target_id = best_hyp.hypothesis_id.split("::")[-1]
        return best_hyp, best_hyp.action, target_id

    def run_curiosity_cycle(self, max_steps: int = 10) -> EpistemicUncertaintyReport:
        """Autonomously conduct information-seeking exploration cycles."""
        self.generate_candidate_hypotheses_from_environment()
        initial_entropy = self.compute_entropy()

        interventions_count = 0
        discovered_rules: list[str] = []

        for step_i in range(max_steps):
            hyp, action, target_id = self.select_highest_information_gain_action()
            if not hyp or not target_id or target_id not in self.env.objects:
                break

            target_obj = self.env.objects[target_id]
            initial_pos = (target_obj.position.x, target_obj.position.y)

            # Move arm near object if not close
            dist = self.env.agent_hand_position.distance_to(target_obj.position)
            if dist > self.env.REACH_DISTANCE:
                self.env.step(
                    BabyActionType.MOVE, target_id=target_id, parameter=target_obj.position
                )

            # Execute exploratory intervention
            self.env.step(action, target_id=target_id)
            interventions_count += 1

            hyp.interventions_tested += 1

            new_pos = (target_obj.position.x, target_obj.position.y)
            moved = (
                abs(new_pos[0] - initial_pos[0]) > 0.05
                or abs(new_pos[1] - initial_pos[1]) > 0.05
                or target_obj.held_by_agent
            )

            # Bayesian belief update on hypothesis
            if moved:
                hyp.confidence = min(0.99, hyp.confidence + 0.35)
                if hyp.confidence >= 0.80:
                    hyp.confirmed = True

                    rule_str = f"{action.value}({target_obj.object_type.value}) -> MOVES"
                    if rule_str not in discovered_rules:
                        discovered_rules.append(rule_str)
                        self.substrate.causal_rules.append(
                            {
                                "action": action.value,
                                "object_type": target_obj.object_type.value,
                                "consequence": "MOVES",
                                "confidence": hyp.confidence,
                            }
                        )
            else:
                hyp.confidence = max(0.01, hyp.confidence - 0.3)
                if hyp.confidence <= 0.15:
                    hyp.falsified = True

            self.exploration_history.append(
                {
                    "step": step_i,
                    "target": target_id,
                    "action": action.value,
                    "outcome_moved": moved,
                    "updated_confidence": hyp.confidence,
                }
            )

            # Record event in substrate transitions
            if hasattr(self.substrate, "profile") and hasattr(
                self.substrate.profile, "belief_transitions"
            ):
                self.substrate.profile.belief_transitions.append(
                    BeliefTransitionEvent(
                        event_type=BeliefTransitionType.CURIOSITY_EXPLORATION,
                        step_index=step_i,
                        variable="intrinsic_uncertainty",
                        condition=f"{action.value}({target_id})",
                        posterior_confidence=hyp.confidence,
                        evidence={"moved": moved},
                    )
                )

        final_entropy = self.compute_entropy()
        reduction = max(0.0, initial_entropy - final_entropy)

        return EpistemicUncertaintyReport(
            initial_entropy=initial_entropy,
            final_entropy=final_entropy,
            entropy_reduction=reduction,
            interventions_executed=interventions_count,
            hypotheses_evaluated=len(self.candidate_hypotheses),
            discovered_rules=discovered_rules,
        )
