"""
Cognitive Budget — Per-request resource allocation and enforcement.

Every cognitive request (user message, background task, plugin call) gets
a budget that constrains how much work the system will do. This prevents
runaway chains, infinite tool loops, and excessive LLM calls.

Budget dimensions:
    - max_llm_calls: Maximum LLM inference calls for this request
    - max_tool_calls: Maximum tool invocations
    - max_memory_reads: Maximum memory queries
    - max_latency_ms: Hard timeout for the entire request
    - max_tokens_out: Maximum tokens in the response
    - max_reasoning_depth: Maximum chain-of-thought/planning depth

Profiles set default budgets, but they can be overridden per-request.

Architecture::

    Request arrives
        ↓
    CognitiveBudgetManager.allocate(request, profile)
        ↓
    RequestBudget (limits)
        ↓
    Executive checks budget.can_afford("llm_call") before each action
        ↓
    budget.spend("llm_call")
        ↓
    If exhausted → graceful degradation or early return

Usage::

    from hbllm.brain.attention.cognitive_budget import CognitiveBudgetManager

    manager = CognitiveBudgetManager()
    budget = manager.allocate("interactive", profile)

    if budget.can_afford("llm_call"):
        budget.spend("llm_call")
        result = await llm.call(...)

    if budget.is_exhausted:
        return "Budget exhausted — returning best-effort response."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Budget Limits
# ═══════════════════════════════════════════════════════════════════════════

# Default budget tiers
_DEFAULT_BUDGETS: dict[str, dict[str, int | float]] = {
    "interactive": {
        "max_llm_calls": 10,
        "max_tool_calls": 20,
        "max_memory_reads": 15,
        "max_latency_ms": 30_000,
        "max_tokens_out": 4096,
        "max_reasoning_depth": 5,
    },
    "background": {
        "max_llm_calls": 5,
        "max_tool_calls": 10,
        "max_memory_reads": 10,
        "max_latency_ms": 60_000,
        "max_tokens_out": 2048,
        "max_reasoning_depth": 3,
    },
    "maintenance": {
        "max_llm_calls": 3,
        "max_tool_calls": 5,
        "max_memory_reads": 20,
        "max_latency_ms": 120_000,
        "max_tokens_out": 1024,
        "max_reasoning_depth": 2,
    },
    "lite": {
        "max_llm_calls": 3,
        "max_tool_calls": 5,
        "max_memory_reads": 5,
        "max_latency_ms": 15_000,
        "max_tokens_out": 2048,
        "max_reasoning_depth": 2,
    },
    "unlimited": {
        "max_llm_calls": 100,
        "max_tool_calls": 200,
        "max_memory_reads": 100,
        "max_latency_ms": 300_000,
        "max_tokens_out": 16384,
        "max_reasoning_depth": 10,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Request Budget
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RequestBudget:
    """Resource budget for a single cognitive request.

    Tracks limits and current spend. Every action checks
    ``can_afford()`` before proceeding and calls ``spend()`` after.
    """

    # Limits
    max_llm_calls: int = 10
    max_tool_calls: int = 20
    max_memory_reads: int = 15
    max_latency_ms: float = 30_000
    max_tokens_out: int = 4096
    max_reasoning_depth: int = 5

    # Current spend
    llm_calls_used: int = 0
    tool_calls_used: int = 0
    memory_reads_used: int = 0
    tokens_out_used: int = 0
    reasoning_depth_used: int = 0

    # Timing
    started_at: float = field(default_factory=time.time)

    # ── Spend Tracking ───────────────────────────────────────────────

    def can_afford(self, resource: str, amount: int = 1) -> bool:
        """Check if the budget allows spending on a resource.

        Args:
            resource: One of "llm_call", "tool_call", "memory_read",
                      "token_out", "reasoning_depth".
            amount: How much to check.

        Returns:
            True if within budget.
        """
        if self._is_timed_out:
            return False

        checks: dict[str, bool] = {
            "llm_call": self.llm_calls_used + amount <= self.max_llm_calls,
            "tool_call": self.tool_calls_used + amount <= self.max_tool_calls,
            "memory_read": self.memory_reads_used + amount <= self.max_memory_reads,
            "token_out": self.tokens_out_used + amount <= self.max_tokens_out,
            "reasoning_depth": self.reasoning_depth_used + amount <= self.max_reasoning_depth,
        }

        return checks.get(resource, True)

    def spend(self, resource: str, amount: int = 1) -> None:
        """Record spending on a resource.

        Args:
            resource: Resource type to spend.
            amount: How much to spend.
        """
        if resource == "llm_call":
            self.llm_calls_used += amount
        elif resource == "tool_call":
            self.tool_calls_used += amount
        elif resource == "memory_read":
            self.memory_reads_used += amount
        elif resource == "token_out":
            self.tokens_out_used += amount
        elif resource == "reasoning_depth":
            self.reasoning_depth_used += amount

    @property
    def _is_timed_out(self) -> bool:
        elapsed_ms = (time.time() - self.started_at) * 1000
        return elapsed_ms > self.max_latency_ms

    @property
    def is_exhausted(self) -> bool:
        """Check if any budget dimension is exhausted."""
        return (
            self._is_timed_out
            or self.llm_calls_used >= self.max_llm_calls
            or self.tool_calls_used >= self.max_tool_calls
        )

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.started_at) * 1000

    @property
    def remaining_llm_calls(self) -> int:
        return max(0, self.max_llm_calls - self.llm_calls_used)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls_used)

    @property
    def utilization(self) -> dict[str, float]:
        """Resource utilization as percentages."""
        def _pct(used: int, limit: int) -> float:
            return round(used / limit * 100, 1) if limit > 0 else 0.0

        return {
            "llm_calls": _pct(self.llm_calls_used, self.max_llm_calls),
            "tool_calls": _pct(self.tool_calls_used, self.max_tool_calls),
            "memory_reads": _pct(self.memory_reads_used, self.max_memory_reads),
            "tokens_out": _pct(self.tokens_out_used, self.max_tokens_out),
            "time_elapsed_ms": round(self.elapsed_ms, 1),
        }

    def summary(self) -> str:
        """Human-readable budget summary."""
        return (
            f"LLM: {self.llm_calls_used}/{self.max_llm_calls} | "
            f"Tools: {self.tool_calls_used}/{self.max_tool_calls} | "
            f"Memory: {self.memory_reads_used}/{self.max_memory_reads} | "
            f"Time: {self.elapsed_ms:.0f}/{self.max_latency_ms:.0f}ms"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Budget Manager
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveBudgetManager:
    """Allocates and tracks cognitive budgets per request.

    Uses budget tiers (interactive, background, maintenance, lite)
    that can be customized or overridden per-request.
    """

    def __init__(self) -> None:
        self._tiers = dict(_DEFAULT_BUDGETS)
        self._active_budgets: dict[str, RequestBudget] = {}
        self._total_allocated = 0

    def register_tier(self, name: str, limits: dict[str, int | float]) -> None:
        """Register a custom budget tier."""
        self._tiers[name] = limits

    def allocate(
        self,
        tier: str = "interactive",
        *,
        request_id: str = "",
        overrides: dict[str, int | float] | None = None,
        profile: Any = None,
    ) -> RequestBudget:
        """Allocate a budget for a request.

        Args:
            tier: Budget tier name.
            request_id: Optional ID for tracking active budgets.
            overrides: Per-request limit overrides.
            profile: Optional BrainProfile to adjust limits.

        Returns:
            A new RequestBudget instance.
        """
        # Start with tier defaults
        limits = dict(self._tiers.get(tier, self._tiers["interactive"]))

        # Apply profile constraints
        if profile:
            max_ram = getattr(profile, "max_ram_mb", None)
            if max_ram and max_ram < 512:
                # Constrained device — reduce budgets
                limits["max_llm_calls"] = min(limits.get("max_llm_calls", 10), 5)
                limits["max_tool_calls"] = min(limits.get("max_tool_calls", 20), 10)

        # Apply per-request overrides
        if overrides:
            limits.update(overrides)

        budget = RequestBudget(
            max_llm_calls=int(limits.get("max_llm_calls", 10)),
            max_tool_calls=int(limits.get("max_tool_calls", 20)),
            max_memory_reads=int(limits.get("max_memory_reads", 15)),
            max_latency_ms=float(limits.get("max_latency_ms", 30_000)),
            max_tokens_out=int(limits.get("max_tokens_out", 4096)),
            max_reasoning_depth=int(limits.get("max_reasoning_depth", 5)),
        )

        if request_id:
            self._active_budgets[request_id] = budget

        self._total_allocated += 1
        return budget

    def release(self, request_id: str) -> RequestBudget | None:
        """Release a tracked budget (request completed)."""
        return self._active_budgets.pop(request_id, None)

    def get_active(self, request_id: str) -> RequestBudget | None:
        """Get an active budget by request ID."""
        return self._active_budgets.get(request_id)

    def stats(self) -> dict[str, Any]:
        return {
            "total_allocated": self._total_allocated,
            "active_budgets": len(self._active_budgets),
            "tiers": list(self._tiers.keys()),
        }
