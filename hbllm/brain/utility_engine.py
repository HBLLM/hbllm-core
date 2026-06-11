"""
Unified Cognitive Utility Engine and Thought Budget Abstraction.

Enables bounded rationality by tracking token, time, and branch consumption,
and evaluates cognitive decisions using a multi-factor utility function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtBudget:
    """
    Universally consumable thought budget tracking token, time, and branch constraints.
    """

    max_tokens: int
    max_time_ms: float
    max_branches: int

    tokens_spent: int = 0
    branches_spent: int = 0
    start_time: float = field(default_factory=time.time)

    def spend_tokens(self, count: int) -> None:
        """Record token consumption."""
        self.tokens_spent += count

    def spend_branch(self, count: int = 1) -> None:
        """Record branch/node creation."""
        self.branches_spent += count

    @property
    def elapsed_time_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000.0

    @property
    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self.tokens_spent)

    @property
    def remaining_time_ms(self) -> float:
        """Get remaining time budget in milliseconds."""
        return max(0.0, self.max_time_ms - self.elapsed_time_ms)

    @property
    def remaining_branches(self) -> int:
        """Get remaining branch budget."""
        return max(0, self.max_branches - self.branches_spent)

    def is_exhausted(self) -> bool:
        """
        Check if any budget dimension has been exhausted.
        """
        if self.tokens_spent >= self.max_tokens:
            return True
        if self.elapsed_time_ms >= self.max_time_ms:
            return True
        if self.branches_spent >= self.max_branches:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert budget state to a dictionary for telemetry/debugging."""
        return {
            "max_tokens": self.max_tokens,
            "max_time_ms": self.max_time_ms,
            "max_branches": self.max_branches,
            "tokens_spent": self.tokens_spent,
            "branches_spent": self.branches_spent,
            "elapsed_time_ms": self.elapsed_time_ms,
            "remaining_tokens": self.remaining_tokens,
            "remaining_time_ms": self.remaining_time_ms,
            "remaining_branches": self.remaining_branches,
            "is_exhausted": self.is_exhausted(),
        }


@dataclass
class UtilityBreakdown:
    """Breakdown of factors contributing to utility score."""

    utility: float
    progress_score: float
    token_cost: float
    latency_cost: float
    risk_cost: float

    def to_dict(self) -> dict[str, float]:
        return {
            "utility": self.utility,
            "progress_score": self.progress_score,
            "token_cost": self.token_cost,
            "latency_cost": self.latency_cost,
            "risk_cost": self.risk_cost,
        }


class CognitiveUtilityEngine:
    """
    Evaluates utility of cognitive transitions or actions.

    Formula:
    Utility = (weight_progress * progress_score)
              - (weight_token * tokens_used)
              - (weight_latency * latency_ms)
              - (weight_risk * risk_score)
    """

    def __init__(
        self,
        weight_progress: float = 1.0,
        weight_token: float = 0.0001,
        weight_latency: float = 0.0001,
        weight_risk: float = 1.0,
    ) -> None:
        self.weight_progress = weight_progress
        self.weight_token = weight_token
        self.weight_latency = weight_latency
        self.weight_risk = weight_risk

    def calculate_utility(
        self,
        progress_score: float,
        tokens_used: int,
        latency_ms: float,
        risk_score: float,
    ) -> UtilityBreakdown:
        """
        Compute utility and return the detailed breakdown.
        """
        token_cost = self.weight_token * tokens_used
        latency_cost = self.weight_latency * latency_ms
        risk_cost = self.weight_risk * risk_score
        utility = (self.weight_progress * progress_score) - token_cost - latency_cost - risk_cost

        return UtilityBreakdown(
            utility=utility,
            progress_score=progress_score,
            token_cost=token_cost,
            latency_cost=latency_cost,
            risk_cost=risk_cost,
        )
