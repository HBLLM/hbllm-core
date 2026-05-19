"""Cognitive Governance: Hard Budgets and Soft Cognitive Pressure."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CognitiveBudget:
    """Hard, non-bypassable limits for cognitive processes."""

    simulation_budget: int = 50  # Max tasks simulated per planning session
    recursion_budget: int = 5  # Max depth of self-triggered thoughts
    branch_budget: int = 10  # Max concurrent counterfactual branches
    llm_calls_per_hour: int = 100  # Abstract budget for reasoning throttling
    correction_budget: int = 3  # Max retries for a failing task
    max_pre_execution_deliberation_ms: int = 2000  # Hard timeout for planning


class CognitiveGovernanceEngine:
    """Manages cognitive budgets and computes dynamic cognitive pressure."""

    def __init__(self, budget: CognitiveBudget | None = None) -> None:
        self.budget = budget or CognitiveBudget()
        self._llm_calls_this_hour = 0
        self._hour_start_time = time.time()

    def consume_llm_call(self) -> bool:
        """Attempt to consume an LLM call from the abstract budget.

        Returns:
            True if allowed, False if budget exceeded.
        """
        now = time.time()
        if now - self._hour_start_time > 3600:
            # Reset bucket
            self._llm_calls_this_hour = 0
            self._hour_start_time = now

        if self._llm_calls_this_hour >= self.budget.llm_calls_per_hour:
            logger.warning("LLM Budget exceeded (%d/hour)", self.budget.llm_calls_per_hour)
            return False

        self._llm_calls_this_hour += 1
        return True

    def get_cognitive_pressure(
        self, memory_pressure: float, active_goals: int, queue_depth: int
    ) -> float:
        """Calculate soft cognitive pressure (0.0 to 1.0).

        High pressure creates graceful degradation in the planner.
        """
        # Normalize inputs (heuristics)
        norm_memory = min(1.0, memory_pressure)
        norm_goals = min(1.0, active_goals / 10.0)
        norm_queue = min(1.0, queue_depth / 50.0)

        # LLM usage rate impact
        now = time.time()
        elapsed_hour_fraction = max(0.01, (now - self._hour_start_time) / 3600.0)
        llm_usage_rate = self._llm_calls_this_hour / (
            self.budget.llm_calls_per_hour * elapsed_hour_fraction
        )
        norm_llm_rate = min(1.0, llm_usage_rate)

        # Aggregate
        pressure = (
            (norm_memory * 0.3) + (norm_goals * 0.2) + (norm_queue * 0.3) + (norm_llm_rate * 0.2)
        )
        return min(1.0, pressure)

    def get_degradation_profile(self, pressure: float) -> dict[str, Any]:
        """Convert cognitive pressure into a graceful degradation profile."""
        profile = {"max_simulation_depth": 3, "allow_speculation": True, "force_heuristic": False}

        if pressure > 0.75:
            # Severe overload
            profile["max_simulation_depth"] = 0
            profile["allow_speculation"] = False
            profile["force_heuristic"] = True
        elif pressure > 0.5:
            # Moderate overload
            profile["max_simulation_depth"] = 1
            profile["allow_speculation"] = False

        return profile
