"""Opportunity Market.

Scores, ranks, and resolves conflicts and dependencies for candidate opportunities.
Gated by available cognitive budget.
"""

from __future__ import annotations

import logging
import time

from hbllm.brain.autonomy.cognitive_budget import CognitiveBudget
from hbllm.brain.autonomy.opportunity import Opportunity

logger = logging.getLogger(__name__)


class OpportunityMarket:
    """Coordinates the cognitive economy by ranking opportunities via utility scoring.

    Resolves dependencies, blocks, and resource allocations with CognitiveBudget.
    """

    def __init__(self, budget: CognitiveBudget | None = None) -> None:
        self.budget = budget or CognitiveBudget()
        self.running_opportunities: set[str] = set()
        self.completed_opportunities: set[str] = set()

    def calculate_utility(self, opp: Opportunity) -> float:
        """Compute the utility score of an opportunity.

        Formula:
            Utility = (Priority * Urgency * ExpectedValue * Confidence) /
                      (ResourceCost + InterruptionCost + 1e-5)
        """
        denominator = opp.resource_cost + opp.interruption_cost
        if denominator <= 0.0:
            denominator = 1e-5

        numerator = opp.priority * opp.urgency * opp.expected_value * opp.confidence
        return numerator / denominator

    def select_opportunities(
        self,
        candidates: list[Opportunity],
        budget: CognitiveBudget | None = None,
    ) -> list[Opportunity]:
        """Rank, filter, and resolve dependencies/conflicts among candidates.

        Args:
            candidates: List of candidate Opportunity objects.
            budget: Optional override for the current CognitiveBudget.

        Returns:
            A sorted list of opportunities that can be run right now.
        """
        active_budget = budget or self.budget
        now = time.time()

        # 1. Recalculate priorities using aging and filter expired
        active_candidates: list[Opportunity] = []
        for opp in candidates:
            opp.update_priority(now)
            if opp.expires_at is None or now < opp.expires_at:
                active_candidates.append(opp)

        # 2. Score and sort by utility
        scored = [(opp, self.calculate_utility(opp)) for opp in active_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        valid_opps: list[Opportunity] = []

        for opp, score in scored:
            # Check resource budget
            if not active_budget.can_afford(opp):
                logger.debug("Market: Opportunity %s rejected by budget limit", opp.id)
                continue

            # Check dependencies: requires
            missing_dep = False
            for req in opp.requires:
                if req not in self.completed_opportunities:
                    missing_dep = True
                    break
            if missing_dep:
                logger.debug("Market: Opportunity %s rejected due to missing dependencies", opp.id)
                continue

            # Check conflicts: blocks, conflicts
            has_conflict = False
            # Check if this opportunity conflicts with any currently running opportunity
            for running in self.running_opportunities:
                if running in opp.conflicts or opp.id in opp.conflicts:
                    has_conflict = True
                    break

            # Also check if it conflicts with already selected opportunities in this cycle
            for selected in valid_opps:
                if selected.id in opp.conflicts or opp.id in selected.conflicts:
                    has_conflict = True
                    break
                # Or if the selected opportunity blocks this category
                if selected.category in opp.blocks or opp.category in selected.blocks:
                    has_conflict = True
                    break

            if has_conflict:
                logger.debug("Market: Opportunity %s rejected due to conflicts", opp.id)
                continue

            valid_opps.append(opp)

        return valid_opps
