"""Cognitive Resource Budgeting and Load Management for A21.

Models computation constraints as explicit budgets.
Enforces the safety invariant: Throttling simulation depth or branches
explicitly incurs an uncertainty penalty rather than silently compromising correctness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CognitiveBudget:
    """Resource budget parameters for cognitive simulation and search."""

    max_simulation_depth: int = 5
    max_branches: int = 8
    max_search_nodes: int = 50
    max_probe_count: int = 3
    current_load: float = 0.0  # 0.0 (idle) to 1.0 (maximum load)


@dataclass
class BudgetDecision:
    """The outcome of a cognitive resource allocation query."""

    allocated_depth: int
    allocated_branches: int
    allocated_search_nodes: int
    truncated: bool
    uncertainty_penalty: float  # Explicit epistemic penalty if resources were truncated
    reason: str


class CognitiveBudgetManager:
    """Dynamically manages cognitive resource allocations for A18 simulation and A19 search."""

    def __init__(self, base_budget: CognitiveBudget | None = None) -> None:
        self.budget = base_budget or CognitiveBudget()

    def allocate_simulation_budget(
        self,
        requested_depth: int = 5,
        requested_branches: int = 8,
        task_stake: float = 0.50,  # 0.0 (trivial) to 1.0 (mission critical)
    ) -> BudgetDecision:
        """Allocate simulation depth and branches, applying an uncertainty penalty if constrained."""
        load = self.budget.current_load

        # If system is under heavy load (> 0.70) and task stake is not critical (< 0.80)
        if load >= 0.70 and task_stake < 0.80:
            alloc_depth = max(2, min(requested_depth, self.budget.max_simulation_depth - 2))
            alloc_branches = max(2, min(requested_branches, self.budget.max_branches // 2))
            truncated = (alloc_depth < requested_depth) or (alloc_branches < requested_branches)
            uncertainty_pen = 0.25 if truncated else 0.05
            reason = f"Throttled simulation due to elevated cognitive load ({load:.2f})"
        elif load >= 0.40:
            alloc_depth = min(requested_depth, self.budget.max_simulation_depth)
            alloc_branches = max(3, min(requested_branches, self.budget.max_branches - 2))
            truncated = alloc_branches < requested_branches
            uncertainty_pen = 0.10 if truncated else 0.0
            reason = "Moderate load: branches constrained"
        else:
            alloc_depth = min(requested_depth, self.budget.max_simulation_depth)
            alloc_branches = min(requested_branches, self.budget.max_branches)
            truncated = False
            uncertainty_pen = 0.0
            reason = "Optimal budget allocated"

        return BudgetDecision(
            allocated_depth=alloc_depth,
            allocated_branches=alloc_branches,
            allocated_search_nodes=self.budget.max_search_nodes,
            truncated=truncated,
            uncertainty_penalty=round(uncertainty_pen, 4),
            reason=reason,
        )

    def set_load(self, load: float) -> None:
        """Update current system cognitive load in [0.0, 1.0]."""
        self.budget.current_load = max(0.0, min(1.0, load))
