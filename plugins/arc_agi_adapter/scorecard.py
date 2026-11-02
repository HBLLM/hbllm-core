"""Shared scorecard metrics for the ARC harness + local suite (WP0.9)."""

from __future__ import annotations

import statistics
from typing import Any


def _p95(latencies_ms: list[float]) -> float:
    if len(latencies_ms) > 1:
        return round(statistics.quantiles(latencies_ms, n=20)[18], 2)
    return latencies_ms[0] if latencies_ms else 0.0


def compute_game_metrics(
    *,
    steps: int,
    latencies_ms: list[float],
    stagnant_steps: int,
    noop_actions: int,
    contract_violations: int,
    levels_completed: int,
    win_levels: int,
    state: str,
    stop_reason: str,
    baseline_actions: int | None = None,
    decision_exceptions: int = 0,
    decide_latencies_ms: list[float] | None = None,
    max_stagnation_streak: int = 0,
    agent_yield_step: int | None = None,
    agent_yield_reason: str | None = None,
    stop_detail: str | None = None,
) -> dict[str, Any]:
    """One game's scorecard row — identical shape for ARC + local env."""
    decide = decide_latencies_ms or []
    return {
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "state": state,
        "steps": steps,
        "stop_reason": stop_reason,
        "stop_detail": stop_detail,
        "actions_per_level": round(steps / max(1, levels_completed + 1), 2),
        "baseline_actions": baseline_actions,
        "mean_latency_ms": round(statistics.fmean(latencies_ms), 2) if latencies_ms else 0.0,
        "p95_latency_ms": _p95(latencies_ms),
        "mean_decide_latency_ms": round(statistics.fmean(decide), 2) if decide else 0.0,
        "p95_decide_latency_ms": _p95(decide),
        "stagnant_steps": stagnant_steps,
        "max_stagnation_streak": max_stagnation_streak,
        "noop_actions": noop_actions,
        "contract_violations": contract_violations,
        "decision_exceptions": decision_exceptions,
        "agent_yield_step": agent_yield_step,
        "agent_yield_reason": agent_yield_reason,
    }
