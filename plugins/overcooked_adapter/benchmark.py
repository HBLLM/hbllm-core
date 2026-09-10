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
    prefer_native: bool = True,
    require_native: bool = False,
) -> dict[str, Any]:
    """Run evaluation on a single Overcooked cooperative tier."""
    agent = PureHCIROvercookedAgent()
    tier_enum = OvercookedTier(tier)

    successes = 0
    total_steps = 0
    total_soups = 0

    for ep in range(episodes):
        agent.reset_episode()
        agent1 = PureHCIROvercookedAgent()
        agent1.reset_episode()
        env = make_overcooked_env(tier=tier_enum, seed=seed + ep, prefer_native=prefer_native)
        if require_native and not getattr(env, "is_native", False):
            raise RuntimeError(
                "Native 'overcooked_ai_py' package is strictly required; standalone fallback is disabled."
            )
        obs = env.reset()

        is_solo = tier_enum == OvercookedTier.TIER_1_CRAMPED_ROOM_SOLO
        done = False
        while not done:
            action = agent.select_action(obs)
            partner_action = (
                agent1.select_action(obs.swap_agents())
                if (obs.partner is not None and not is_solo)
                else None
            )
            obs, _reward, done, info = env.step(action, partner_action=partner_action)

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
    prefer_native: bool = True,
    require_native: bool = False,
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
            require_native=require_native,
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Overcooked-AI Multi-Tier Benchmark")
    parser.add_argument(
        "--episodes-per-tier", type=int, default=5, help="Episodes per cooperative kitchen tier"
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--prefer-native",
        "--native",
        action="store_true",
        default=True,
        help="Run against upstream native overcooked_ai_py package",
    )
    parser.add_argument(
        "--require-native",
        action="store_true",
        default=False,
        help="Strictly require upstream native package, failing if unavailable",
    )
    args = parser.parse_args()

    print(
        f"\n{'=' * 85}\nRunning Overcooked-AI Multi-Tier Benchmark (5 Tiers, Native={args.prefer_native})\n{'=' * 85}"
    )
    data = run_overcooked_benchmark(
        episodes_per_tier=args.episodes_per_tier,
        seed=args.seed,
        prefer_native=args.prefer_native,
        require_native=args.require_native,
    )
    print(
        f"Overall Success Rate: {data['overall_success_rate'] * 100:.1f}% | 95% Wilson CI: {data['ci_95']} | Total Episodes: {data['total_episodes']}"
    )
    print(f"Mean Steps: {data['overall_mean_steps']:.1f}")
    print("-" * 85)
    for tier_name, res in data["tiers"].items():
        print(
            f"  {tier_name:<35}: {res['success_rate'] * 100:5.1f}% ({res['successes']}/{res['episodes']}) | "
            f"CI: {res['ci_95']} | Soups: {res['mean_soups_delivered']:.2f} | Steps: {res['mean_steps']:.1f}"
        )


if __name__ == "__main__":
    main()
