"""
Digital Agent Multi-Tier Benchmark Suite.

Evaluates Pure HCIR digital tool use, autonomous debugging loops, DOM workflows,
and zero-violation safety policy compliance ($C=0$).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .action import DigitalActionAdapter
from .environment import make_digital_env
from .perception import DigitalPerceptionAdapter
from .types import DigitalAction, DigitalObservation, DigitalTier

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


class PureHCIRDigitalAgent:
    """Safe, zero-token deterministic digital agent."""

    def __init__(self) -> None:
        self.perception = DigitalPerceptionAdapter()
        self.action = DigitalActionAdapter()

    def reset_episode(self) -> None:
        """Reset internal buffers."""
        self.perception.reset()
        self.action.reset()

    def select_action(self, obs: DigitalObservation) -> DigitalAction:
        """Process observation and yield safe digital operation."""
        percept = self.perception.process_observation(obs)
        return self.action.select_action(obs, percept)


def run_digital_tier(
    tier: DigitalTier | str,
    episodes: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run evaluation on a single digital embodiment tier."""
    agent = PureHCIRDigitalAgent()
    tier_enum = DigitalTier(tier)

    successes = 0
    total_steps = 0
    total_violations = 0

    for ep in range(episodes):
        agent.reset_episode()
        env = make_digital_env(tier=tier_enum, seed=seed + ep)
        obs = env.reset()

        done = False
        while not done:
            action = agent.select_action(obs)
            obs, _reward, done, info = env.step(action)

        if obs.won:
            successes += 1
        total_violations += obs.safety_violations
        total_steps += obs.step_count

    rate = successes / episodes if episodes > 0 else 0.0
    zero_violation_rate = 1.0 if total_violations == 0 else 0.0
    ci_low, ci_high = compute_wilson_ci(successes, episodes)
    mean_steps = total_steps / episodes if episodes > 0 else 0.0

    return {
        "tier": tier_enum.value,
        "episodes": episodes,
        "successes": successes,
        "success_rate": rate,
        "zero_violation_rate": zero_violation_rate,
        "total_safety_violations": total_violations,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(mean_steps, 1),
    }


def run_digital_benchmark(
    episodes_per_tier: int = 5,
    seed: int = 42,
    episodes: int | None = None,
) -> dict[str, Any]:
    """Run full 5-tier digital embodiment benchmark."""
    if episodes is not None:
        episodes_per_tier = max(1, episodes // len(DigitalTier))

    tiers_results = {}
    total_successes = 0
    total_episodes = 0
    total_steps = 0
    total_violations = 0

    for tier in DigitalTier:
        res = run_digital_tier(tier, episodes=episodes_per_tier, seed=seed)
        tiers_results[tier.value] = res
        total_successes += res["successes"]
        total_episodes += res["episodes"]
        total_steps += res["mean_steps"] * res["episodes"]
        total_violations += res["total_safety_violations"]

    overall_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    ci_low, ci_high = compute_wilson_ci(total_successes, total_episodes)
    mean_steps = total_steps / total_episodes if total_episodes > 0 else 0.0

    return {
        "benchmark": "digital_agent",
        "tiers": tiers_results,
        "total_episodes": total_episodes,
        "overall_success_rate": overall_rate,
        "success_rate": overall_rate,
        "zero_violation_rate": 1.0 if total_violations == 0 else 0.0,
        "total_safety_violations": total_violations,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "overall_mean_steps": round(mean_steps, 1),
    }
