"""
Overcooked-AI Multi-Tier Benchmark Suite.

Evaluates Pure HCIR multi-agent coordination, recipe pipelining, and counter
contention across 5 canonical cooperative kitchen tiers.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .action import OvercookedActionAdapter
from .environment import make_overcooked_env
from .perception import OvercookedPerceptionAdapter
from .types import OvercookedAction, OvercookedObservation, OvercookedTier

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


class PureHCIROvercookedAgent:
    """Zero-token causal coordinator for Overcooked kitchen tasks."""

    def __init__(self) -> None:
        self.perception = OvercookedPerceptionAdapter()
        self.action = OvercookedActionAdapter()

    def reset_episode(self) -> None:
        """Reset solver buffers."""
        self.perception.reset()
        self.action.reset()

    def select_action(self, obs: OvercookedObservation) -> OvercookedAction:
        """Process observation and select optimal recipe action."""
        percept = self.perception.process_observation(obs)
        return self.action.select_action(obs, percept)


def run_overcooked_tier(
    tier: OvercookedTier | str,
    episodes: int = 5,
    seed: int = 42,
    prefer_native: bool = False,
) -> dict[str, Any]:
    """Run evaluation on a single Overcooked cooperative tier."""
    agent = PureHCIROvercookedAgent()
    tier_enum = OvercookedTier(tier)

    successes = 0
    total_steps = 0
    total_soups = 0

    for ep in range(episodes):
        agent.reset_episode()
        env = make_overcooked_env(tier=tier_enum, seed=seed + ep, prefer_native=prefer_native)
        obs = env.reset()

        done = False
        while not done:
            action = agent.select_action(obs)
            obs, _reward, done, info = env.step(action)

        if obs.won:
            successes += 1
        total_soups += obs.soups_delivered
        total_steps += obs.step_count

    rate = successes / episodes if episodes > 0 else 0.0
    ci_low, ci_high = compute_wilson_ci(successes, episodes)
    mean_steps = total_steps / episodes if episodes > 0 else 0.0
    mean_soups = total_soups / episodes if episodes > 0 else 0.0

    return {
        "tier": tier_enum.value,
        "episodes": episodes,
        "successes": successes,
        "success_rate": rate,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(mean_steps, 1),
        "mean_soups_delivered": round(mean_soups, 2),
    }


def run_overcooked_benchmark(
    episodes_per_tier: int = 5,
    seed: int = 42,
    episodes: int | None = None,
    prefer_native: bool = False,
) -> dict[str, Any]:
    """Run full 5-tier Overcooked cooperative benchmark."""
    if episodes is not None:
        episodes_per_tier = max(1, episodes // len(OvercookedTier))

    tiers_results = {}
    total_successes = 0
    total_episodes = 0
    total_steps = 0

    for tier in OvercookedTier:
        res = run_overcooked_tier(
            tier,
            episodes=episodes_per_tier,
            seed=seed,
            prefer_native=prefer_native,
        )
        tiers_results[tier.value] = res
        total_successes += res["successes"]
        total_episodes += res["episodes"]
        total_steps += res["mean_steps"] * res["episodes"]

    overall_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    ci_low, ci_high = compute_wilson_ci(total_successes, total_episodes)
    mean_steps = total_steps / total_episodes if total_episodes > 0 else 0.0

    return {
        "benchmark": "overcooked",
        "tiers": tiers_results,
        "total_episodes": total_episodes,
        "overall_success_rate": overall_rate,
        "success_rate": overall_rate,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "overall_mean_steps": round(mean_steps, 1),
    }
