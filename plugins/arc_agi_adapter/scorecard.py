"""Shared scorecard metrics for the ARC harness + local suite (WP0.9)."""

from __future__ import annotations

import statistics
from typing import Any


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
) -> dict[str, Any]:
    """One game's scorecard row — identical shape for ARC + local env."""
    return {
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "state": state,
        "steps": steps,
        "stop_reason": stop_reason,
        "actions_per_level": round(steps / max(1, levels_completed + 1), 2),
        "baseline_actions": baseline_actions,
        "mean_latency_ms": round(statistics.fmean(latencies_ms), 2) if latencies_ms else 0.0,
        "p95_latency_ms": (
            round(statistics.quantiles(latencies_ms, n=20)[18], 2)
            if len(latencies_ms) > 1
            else (latencies_ms[0] if latencies_ms else 0.0)
        ),
        "stagnant_steps": stagnant_steps,
        "noop_actions": noop_actions,
        "contract_violations": contract_violations,
    }
