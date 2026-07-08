"""Cognitive Budget Representation.

Tracks available system tokens, GPU/CPU resources, attention capacity, and
interruption capacity to gate opportunity execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hbllm.brain.autonomy.opportunity import Opportunity

logger = logging.getLogger(__name__)


@dataclass
class CognitiveBudget:
    """Represents the available resources for execution in the cognitive economy."""

    available_tokens: int = 100000
    available_gpu_time: float = 1.0
    available_cpu: float = 1.0
    attention_capacity: float = 1.0
    interruption_capacity: float = 1.0

    def can_afford(self, opportunity: Opportunity) -> bool:
        """Evaluate if the opportunity can be run given current budgets.

        Args:
            opportunity: The candidate Opportunity to check.

        Returns:
            True if all resource and attention requirements are met, False otherwise.
        """
        # If the interruption cost exceeds interruption capacity, we cannot afford it now.
        if opportunity.interruption_cost > self.interruption_capacity:
            logger.debug(
                "[CognitiveBudget] Opportunity %s rejected: interruption cost (%.2f) > capacity (%.2f)",
                opportunity.id,
                opportunity.interruption_cost,
                self.interruption_capacity,
            )
            return False

        # If resource cost exceeds available CPU or GPU time, reject.
        limit = min(self.available_cpu, self.available_gpu_time)
        if opportunity.resource_cost > limit:
            logger.debug(
                "[CognitiveBudget] Opportunity %s rejected: resource cost (%.2f) > available CPU/GPU (%.2f/%.2f)",
                opportunity.id,
                opportunity.resource_cost,
                self.available_cpu,
                self.available_gpu_time,
            )
            return False

        return True
