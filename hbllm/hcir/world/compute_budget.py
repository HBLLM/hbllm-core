"""Anytime cognition budget (WP0.2): deadline-aware facultative compute.

The budget is denominated in deterministic WORK UNITS (node expansions,
rollout steps, entity comparisons) so seeded replays produce identical
decisions regardless of machine speed. A wall-clock deadline is only used
when ``deterministic=False`` (an explicit production opt-in); every
wall-clock-triggered cutoff is counted in ``wallclock_cutoffs`` so traces
expose non-determinism.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Cost classes -> work-unit charge per item of that class.
COST_CLASS_UNITS: dict[str, int] = {
    "cheap": 1,
    "moderate": 25,
    "expensive": 100,
    "deferred": 250,
}


@dataclass
class ComputeBudget:
    """Per-decision compute budget with deterministic work-unit semantics."""

    work_units: int
    deadline_s: float | None = None
    deterministic: bool = True
    remaining_actions: int | None = None
    charged: int = 0
    wallclock_cutoffs: int = 0
    _start_s: float = field(default_factory=time.monotonic)

    @classmethod
    def for_step(
        cls,
        work_units: int,
        remaining_actions: int | None = None,
        deterministic: bool = True,
        step_budget_ms: float | None = None,
    ) -> ComputeBudget:
        deadline = (
            (time.monotonic() + step_budget_ms / 1000.0) if step_budget_ms is not None else None
        )
        return cls(
            work_units=work_units,
            deadline_s=deadline,
            deterministic=deterministic,
            remaining_actions=remaining_actions,
        )

    @property
    def remaining_units(self) -> int:
        return max(0, self.work_units - self.charged)

    def charge(self, units: int = 1) -> None:
        """Record work done, in deterministic units."""
        self.charged += units

    def remaining_ms(self) -> float:
        if self.deadline_s is None:
            return float("inf")
        return max(0.0, (self.deadline_s - time.monotonic()) * 1000.0)

    def _expired(self) -> bool:
        if self.deterministic:
            return self.remaining_units <= 0
        out = self.remaining_units <= 0
        if not out and self.deadline_s is not None and time.monotonic() >= self.deadline_s:
            self.wallclock_cutoffs += 1
            return True
        return out

    def allows(self, cost_class: str) -> bool:
        """Whether a work item of the given cost class may start.

        Under ``deterministic=True`` this consults ONLY remaining work
        units — identical decisions under identical seeds/replays.
        """
        if self._expired():
            return False
        units = COST_CLASS_UNITS.get(cost_class, 1)
        if self.deterministic:
            return self.remaining_units >= units
        # non-deterministic mode additionally honours the wall-clock deadline
        return self.remaining_ms() > 0 and self.remaining_units >= units

    def fraction_remaining(self) -> float:
        return self.remaining_units / max(1, self.work_units)
