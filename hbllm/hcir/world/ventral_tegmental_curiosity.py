"""Ventral Tegmental Area (VTA) Dopaminergic Precision & Epistemic Curiosity Drive.

Biologically grounded in mesolimbic VTA dopaminergic signaling and active inference
(Schultz, 1998; Berlyne, 1960; Friston et al., 2015; Oudeyer & Kaplan, 2007):
1. Epistemic Value & Bayesian Information Gain:
   - Computes intrinsic epistemic curiosity reward based on state novelty and reduction
     in environmental parameter uncertainty:
     R_curiosity(s) = eta / sqrt(N(s) + 1).
2. Autonomous Exploration Without Extrinsic Reward:
   - Prioritizes uncharted topological regions and ungrounded mechanical interactions,
     driving active inference during early epistemic phases.
3. Adaptive Habituation & Curiosity Quenching:
   - As an environmental sub-space becomes thoroughly mapped and predictable (N(s) >> 1),
     epistemic drive naturally quenches, yielding executive control to goal-directed exploitation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class EpistemicInformationGain:
    """Quantitative measurement of epistemic information gain for a state or action."""

    state_key: str
    visit_count: int
    novelty_score: float  # [0.0, 1.0] where 1.0 = completely uncharted
    information_gain: float  # Expected reduction in entropy / surprise
    curiosity_bonus: float  # Additive intrinsic reward signal

    def to_causal_predicate(self) -> CausalPredicate:
        """Export epistemic information gain as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="epistemic_curiosity_gain",
            operator="==",
            value=f"state:{self.state_key}:novelty:{self.novelty_score:.2f}:bonus:{self.curiosity_bonus:.2f}",
            relation="vta_epistemic_curiosity_drive",
            target_entity="vta_mesolimbic",
        )


@dataclass
class CuriosityDriveSignal:
    """Composite curiosity motivation landscape across candidate action choices."""

    action_curiosity_bonuses: dict[Any, float]
    highest_novelty_action: Any
    mean_novelty: float
    exploration_recommended: bool


class VentralTegmentalCuriosityEngine:
    """VTA mesolimbic engine generating intrinsic epistemic curiosity bonuses."""

    def __init__(
        self,
        base_curiosity_coefficient: float = 0.60,
        novelty_decay_rate: float = 0.50,
        max_tracked_states: int = 5000,
    ) -> None:
        """Initialize VTA curiosity engine.

        Args:
            base_curiosity_coefficient: Baseline intrinsic reward scale eta.
            novelty_decay_rate: Exponential decay power governing visit count habituation.
            max_tracked_states: Maximum number of distinct state keys to retain.
        """
        self.eta = float(base_curiosity_coefficient)
        self.decay_power = float(novelty_decay_rate)
        self.max_tracked = max(100, int(max_tracked_states))

        self.visit_counts: dict[str, int] = {}
        self.total_observations: int = 0

    def record_visit(self, state_key: str) -> EpistemicInformationGain:
        """Record an observed state visit and calculate its epistemic information gain.

        Args:
            state_key: String representation of coordinate (r, c) or state hash.

        Returns:
            EpistemicInformationGain detailing novelty and curiosity bonus.
        """
        self.total_observations += 1
        current_count = self.visit_counts.get(state_key, 0)
        new_count = current_count + 1

        if len(self.visit_counts) >= self.max_tracked and state_key not in self.visit_counts:
            # Evict state with highest visit count (most familiar / quenched)
            most_familiar = max(self.visit_counts.keys(), key=lambda k: self.visit_counts[k])
            self.visit_counts.pop(most_familiar, None)

        self.visit_counts[state_key] = new_count

        # Compute count-based novelty: 1.0 for first visit, decaying with counts
        novelty = 1.0 / ((new_count) ** self.decay_power)
        novelty = min(1.0, max(0.01, float(novelty)))

        # Information gain proportional to surprise reduction
        info_gain = float(math.log(1.0 + novelty))
        curiosity_bonus = float(self.eta * novelty)

        return EpistemicInformationGain(
            state_key=state_key,
            visit_count=new_count,
            novelty_score=novelty,
            information_gain=info_gain,
            curiosity_bonus=curiosity_bonus,
        )

    def evaluate_candidate_curiosity(
        self,
        candidate_actions: list[Any],
        projected_state_keys: dict[Any, str],
    ) -> CuriosityDriveSignal:
        """Evaluate epistemic curiosity bonuses across projected candidate actions.

        Args:
            candidate_actions: List of possible motor actions.
            projected_state_keys: Mapping of action -> predicted next state key.

        Returns:
            CuriosityDriveSignal detailing intrinsic bonuses and top novelty choices.
        """
        action_bonuses: dict[Any, float] = {}
        novelties: list[float] = []

        for act in candidate_actions:
            s_key = projected_state_keys.get(act)
            if s_key is not None:
                count = self.visit_counts.get(s_key, 0)
                # Unvisited state gets full novelty bonus
                nov = 1.0 / ((count + 1) ** self.decay_power)
            else:
                nov = 0.50  # Neutral unprojected novelty

            bonus = self.eta * nov
            action_bonuses[act] = float(bonus)
            novelties.append(nov)

        mean_nov = sum(novelties) / len(novelties) if novelties else 0.0
        best_act = (
            max(action_bonuses.keys(), key=lambda a: action_bonuses[a]) if action_bonuses else None
        )
        explore_rec = mean_nov > 0.40

        return CuriosityDriveSignal(
            action_curiosity_bonuses=action_bonuses,
            highest_novelty_action=best_act,
            mean_novelty=float(mean_nov),
            exploration_recommended=explore_rec,
        )

    def get_novelty(self, state_key: str) -> float:
        """Query novelty score for a given state key."""
        count = self.visit_counts.get(state_key, 0)
        return float(1.0 / ((count + 1) ** self.decay_power))

    def reset(self) -> None:
        """Clear visit counts and history."""
        self.visit_counts.clear()
        self.total_observations = 0
