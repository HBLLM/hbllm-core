"""Domain-General Epistemic Budget Governor (W187).

Biological basis: Lateral Habenular (LHb) signaling of repeated frustration/refractory
inhibition modulating dopaminergic exploration drive in midbrain VTA.
"""

from __future__ import annotations


class ActionBudgetGovernor:
    """Manages episode budget consumption and controls exploration vs. exploitation throttling (W187).

    Biological basis: Lateral Habenular (LHb) signaling of repeated frustration/refractory inhibition,
    modulating dopaminergic exploration drive in midbrain VTA.
    """

    def __init__(self, max_budget: int = 1000) -> None:
        self.max_budget = max_budget
        self.steps_consumed = 0
        self.ior_cache: dict[str, int] = {}  # Habenular Inhibition of Return (key -> step tested)

    def reset(self, max_budget: int | None = None) -> None:
        """Reset budget tracker for a new episode."""
        if max_budget is not None:
            self.max_budget = max_budget
        self.steps_consumed = 0
        self.ior_cache.clear()

    def record_step(self) -> int:
        """Increment consumed steps and return remaining budget."""
        self.steps_consumed += 1
        return self.get_remaining_budget()

    def get_remaining_budget(self) -> int:
        """Return remaining step budget."""
        return max(0, self.max_budget - self.steps_consumed)

    def get_exploration_weight(self) -> float:
        """Dynamically scales epistemic exploration drive based on remaining budget (W187)."""
        rem_ratio = self.get_remaining_budget() / float(max(1, self.max_budget))
        if rem_ratio > 0.60:
            return 1.0  # Full active exploration
        elif rem_ratio > 0.25:
            return rem_ratio  # Linear throttle
        else:
            return 0.10  # Emergency exploitation mode

    def register_experiment_trial(self, experiment_signature: str) -> None:
        """Record that an experiment was executed at the current step (W188 IOR)."""
        self.ior_cache[experiment_signature] = self.steps_consumed

    def is_inhibited_by_ior(self, experiment_signature: str, refractory_period: int = 15) -> bool:
        """True if experiment was recently tested and is currently inhibited by Habenular IOR (W188)."""
        last_step = self.ior_cache.get(experiment_signature)
        if last_step is None:
            return False
        return (self.steps_consumed - last_step) < refractory_period

    def clear_ior_cache(self) -> None:
        """Clear refractory inhibition cache."""
        self.ior_cache.clear()
