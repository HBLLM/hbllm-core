"""
Cognitive Scheduler Policy — multi-criteria priority scoring.

Implements the multi-criteria execution score formula:

                      ( expected_value × urgency × confidence )
    execution_score = ───────────────────────────────────────────
                       ( resource_cost × interruption_cost )

Where:
    expected_value     (0.01 – 1.0): Expected utility/impact of the task
    urgency            (0.01 – 1.0): Time sensitivity / attention salience
    confidence         (0.01 – 1.0): Uncertainty vector confidence
    resource_cost      (0.01 – 10.0): Estimated token/compute/energy cost
    interruption_cost  (0.01 – 10.0): Context switching or focus break penalty
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TaskScoringFactors:
    """Inputs to the multi-criteria cognitive priority score formula."""

    expected_value: float = 0.5
    urgency: float = 0.5
    confidence: float = 0.8
    resource_cost: float = 1.0
    interruption_cost: float = 1.0

    def compute_score(self) -> float:
        """Compute execution score according to the cognitive OS formula.

        Prevents division by zero by clamping inputs to minimum 0.001.
        Higher score = higher priority.
        """
        val = max(0.001, min(1.0, self.expected_value))
        urg = max(0.001, min(1.0, self.urgency))
        conf = max(0.001, min(1.0, self.confidence))
        rcost = max(0.001, self.resource_cost)
        icost = max(0.001, self.interruption_cost)

        numerator = val * urg * conf
        denominator = rcost * icost
        return numerator / denominator


class CognitiveScoreCalculator:
    """Calculator for cognitive process scheduling priority."""

    @staticmethod
    def score_task(
        expected_value: float = 0.5,
        urgency: float = 0.5,
        confidence: float = 0.8,
        resource_cost: float = 1.0,
        interruption_cost: float = 1.0,
    ) -> float:
        """Compute execution score for a task."""
        factors = TaskScoringFactors(
            expected_value=expected_value,
            urgency=urgency,
            confidence=confidence,
            resource_cost=resource_cost,
            interruption_cost=interruption_cost,
        )
        return factors.compute_score()
