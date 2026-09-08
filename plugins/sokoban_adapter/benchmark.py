"""
Sokoban Multi-Tier Benchmark Suite.

Evaluates Pure HCIR against canonical combinatorial push tiers reporting
exact 95% Wilson confidence intervals, dead-end avoidance rates, and mean steps.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .action import SokobanActionAdapter
from .environment import make_sokoban_env
from .perception import SokobanPerceptionAdapter
from .types import SokobanAction, SokobanObservation, SokobanTier

logger = logging.getLogger(__name__)


def compute_wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute exact two-sided 95% Wilson score confidence interval."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2.0 * total)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z**2) / (4.0 * (total**2))))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


class PureHCIRSokobanAgent:
    """Pure HCIR zero-token deterministic solver for Sokoban."""

    def __init__(self) -> None:
        self.perception = SokobanPerceptionAdapter()
        self.action = SokobanActionAdapter()

    def reset_episode(self) -> None:
        """Reset internal solver buffers."""
        self.perception.reset()
        self.action.reset()

    def select_action(self, obs: SokobanObservation) -> SokobanAction:
        """Process observation and yield optimal action."""
        percept = self.perception.process_observation(obs)
        return self.action.select_action(obs, percept)


def run_sokoban_tier(
    tier: SokobanTier | str,
    episodes: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run evaluation on a single Sokoban tier."""
    agent = PureHCIRSokobanAgent()
    tier_enum = SokobanTier(tier)

    successes = 0
    total_steps = 0
    deadlocks = 0

    for ep in range(episodes):
        agent.reset_episode()
        env = make_sokoban_env(tier=tier_enum, seed=seed + ep)
        obs = env.reset()

        done = False
        while not done:
            action = agent.select_action(obs)
            obs, _reward, done, info = env.step(action)

        if obs.won:
            successes += 1
        if obs.deadlock_detected:
            deadlocks += 1
        total_steps += obs.step_count

    rate = successes / episodes if episodes > 0 else 0.0
    ci_low, ci_high = compute_wilson_ci(successes, episodes)
    mean_steps = total_steps / episodes if episodes > 0 else 0.0

    return {
        "tier": tier_enum.value,
        "episodes": episodes,
        "successes": successes,
        "success_rate": rate,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(mean_steps, 1),
        "deadlock_count": deadlocks,
    }


def run_sokoban_benchmark(
    episodes_per_tier: int = 5,
    seed: int = 42,
    episodes: int | None = None,
) -> dict[str, Any]:
    """Run full-spectrum 5-tier Sokoban benchmark."""
    if episodes is not None:
        episodes_per_tier = max(1, episodes // len(SokobanTier))

    tiers_results = {}
    total_successes = 0
    total_episodes = 0
    total_steps = 0

    for tier in SokobanTier:
        res = run_sokoban_tier(tier, episodes=episodes_per_tier, seed=seed)
        tiers_results[tier.value] = res
        total_successes += res["successes"]
        total_episodes += res["episodes"]
        total_steps += res["mean_steps"] * res["episodes"]

    overall_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    ci_low, ci_high = compute_wilson_ci(total_successes, total_episodes)
    mean_steps = total_steps / total_episodes if total_episodes > 0 else 0.0

    return {
        "benchmark": "sokoban",
        "tiers": tiers_results,
        "total_episodes": total_episodes,
        "overall_success_rate": overall_rate,
        "success_rate": overall_rate,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "overall_mean_steps": round(mean_steps, 1),
    }
