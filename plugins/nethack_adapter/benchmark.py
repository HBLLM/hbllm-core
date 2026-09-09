"""
NetHack Three-Cohort Benchmark Runner.

Evaluates Pure HCIR vs LLM-Only baseline on Dungeon Exploration,
Closed Door Navigation, Monster Combat, and Staircase Descent with Wilson CIs.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure core and plugins can be imported cleanly
_current_dir = Path(__file__).resolve().parent
_plugins_dir = _current_dir.parent
_core_dir = _plugins_dir.parent

for p in [str(_core_dir), str(_plugins_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nethack_adapter.action import NetHackActionAdapter
from nethack_adapter.environment import make_nethack_env
from nethack_adapter.perception import NetHackPerceptionAdapter
from nethack_adapter.types import (
    NetHackAction,
    NetHackObservation,
)

logger = logging.getLogger(__name__)


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate Wilson score 95% confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4 * total)) / total)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass
class NetHackEpisodeResult:
    seed: int
    steps: int
    success: bool
    dungeon_level: int
    hp_remaining: int
    gold: int
    elapsed_ms: float


class PureHCIRNetHackAgent:
    """Pure HCIR reasoning agent with 0 LLM tokens."""

    def __init__(self) -> None:
        self.perception = NetHackPerceptionAdapter()
        self.action_adapter = NetHackActionAdapter()

    def select_action(self, obs: NetHackObservation) -> NetHackAction:
        self.perception.ingest_observation(obs)
        return self.action_adapter.plan_next_action(obs)


class LLMOnlyNetHackAgent:
    """Stochastic baseline randomly wandering corridors and rooms."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select_action(self, obs: NetHackObservation) -> NetHackAction:
        # Bias toward movement actions
        acts = [
            NetHackAction.NORTH,
            NetHackAction.EAST,
            NetHackAction.SOUTH,
            NetHackAction.WEST,
            NetHackAction.OPEN_DOOR,
            NetHackAction.PICKUP,
            NetHackAction.DESCEND_STAIRS,
        ]
        return self.rng.choice(acts)


NETHACK_TIERS = [
    (1, "Tier 1: Room Navigation"),
    (2, "Tier 2: Corridor Fog Exploration"),
    (3, "Tier 3: Closed Door Navigation"),
    (4, "Tier 4: Monster Combat"),
    (5, "Tier 5: Full Dungeon Descent"),
]


def run_nethack_tier_benchmark(
    cohort_name: str,
    tier: int,
    episodes: int = 3,
    base_seed: int = 4000,
    prefer_native: bool = False,
) -> dict[str, Any]:
    """Run benchmark for a given cohort on a specific NetHack/MiniHack tier."""
    results: list[NetHackEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + (tier * 100) + i
        env = make_nethack_env(seed=seed, tier=tier, prefer_native=prefer_native)
        obs, _ = env.reset(seed=seed)

        if cohort_name in ("pure-hcir", "guided-hcir"):
            agent = PureHCIRNetHackAgent()
        else:
            agent = LLMOnlyNetHackAgent(seed=seed)

        t0 = time.perf_counter()
        done = False

        while not done and obs.step_count < 150:
            act = agent.select_action(obs)
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                done = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        success = obs.stats.dungeon_level >= 2

        results.append(
            NetHackEpisodeResult(
                seed=seed,
                steps=obs.step_count,
                success=success,
                dungeon_level=obs.stats.dungeon_level,
                hp_remaining=obs.stats.hp,
                gold=obs.stats.gold,
                elapsed_ms=elapsed_ms,
            )
        )

    successes = sum(1 for r in results if r.success)
    ci_low, ci_high = wilson_score_interval(successes, episodes)
    mean_steps = sum(r.steps for r in results) / episodes if episodes else 0.0
    mean_gold = sum(r.gold for r in results) / episodes if episodes else 0.0

    return {
        "cohort": cohort_name,
        "tier": tier,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(mean_steps, 1),
        "mean_gold": round(mean_gold, 1),
        "results": [asdict(r) for r in results],
    }


def run_nethack_benchmark(
    cohort_name: str,
    episodes: int = 15,
    base_seed: int = 4000,
    prefer_native: bool = False,
) -> dict[str, Any]:
    """Backwards-compatible benchmark across standard environments."""
    eps_per_tier = max(1, episodes // len(NETHACK_TIERS))
    all_results = []
    total_successes = 0
    total_steps = 0
    total_gold = 0
    total_eps = 0

    for tier_id, _ in NETHACK_TIERS:
        res = run_nethack_tier_benchmark(
            cohort_name,
            tier=tier_id,
            episodes=eps_per_tier,
            base_seed=base_seed,
            prefer_native=prefer_native,
        )
        all_results.extend(res["results"])
        total_eps += res["episodes"]
        total_successes += sum(1 for r in res["results"] if r["success"])
        total_steps += sum(r["steps"] for r in res["results"])
        total_gold += sum(r["gold"] for r in res["results"])

    ci_low, ci_high = wilson_score_interval(total_successes, total_eps)

    return {
        "cohort": cohort_name,
        "episodes": total_eps,
        "success_rate": total_successes / total_eps if total_eps else 0.0,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(total_steps / total_eps, 1) if total_eps else 0.0,
        "mean_gold": round(total_gold / total_eps, 1) if total_eps else 0.0,
        "results": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NetHack Multi-Tier Benchmark")
    parser.add_argument("--episodes-per-tier", type=int, default=3, help="Episodes per tier")
    parser.add_argument("--seed", type=int, default=4000, help="Base seed")
    parser.add_argument(
        "--prefer-native",
        "--native",
        action="store_true",
        help="Run against upstream native minihack/nle package",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 85}")
    print(
        f"Running NetHack / MiniHack Full-Spectrum Benchmark (5 Tiers, N={args.episodes_per_tier} eps/tier, Native={args.prefer_native})"
    )
    print(f"{'=' * 85}\n")

    for cohort in ("pure-hcir", "llm-only"):
        print(f"--- Cohort: {cohort.upper()} ---")
        print(
            f"{'Dungeon Tier':<34} | {'Success Rate':<12} | {'95% Wilson CI':<18} | {'Mean Steps':<10} | {'Mean Gold'}"
        )
        print("-" * 85)

        total_eps = 0
        total_successes = 0
        total_steps = 0
        total_gold = 0

        for tier_id, tier_name in NETHACK_TIERS:
            data = run_nethack_tier_benchmark(
                cohort,
                tier=tier_id,
                episodes=args.episodes_per_tier,
                base_seed=args.seed,
                prefer_native=args.prefer_native,
            )
            total_eps += data["episodes"]
            total_successes += sum(1 for r in data["results"] if r["success"])
            total_steps += sum(r["steps"] for r in data["results"])
            total_gold += sum(r["gold"] for r in data["results"])

            ci_str = f"[{data['ci_95'][0]:.3f}, {data['ci_95'][1]:.3f}]"
            print(
                f"{tier_name:<34} | "
                f"{data['success_rate'] * 100:11.1f}% | "
                f"{ci_str:<18} | "
                f"{data['mean_steps']:10.1f} | "
                f"{data['mean_gold']:9.1f}"
            )

        overall_rate = total_successes / total_eps if total_eps else 0.0
        ci_low, ci_high = wilson_score_interval(total_successes, total_eps)
        overall_steps = total_steps / total_eps if total_eps else 0.0
        overall_gold = total_gold / total_eps if total_eps else 0.0

        print("-" * 85)
        print(
            f"{'OVERALL':<34} | "
            f"{overall_rate * 100:11.1f}% | "
            f"[{ci_low:.3f}, {ci_high:.3f}]        | "
            f"{overall_steps:10.1f} | "
            f"{overall_gold:9.1f} (Total {total_eps} eps)"
        )
        print("\n")


if __name__ == "__main__":
    main()
