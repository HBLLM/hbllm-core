"""
Crafter Official 22-Achievement Unconstrained Hafner Benchmark Runner.

Evaluates Pure HCIR reasoning (0 LLM tokens) on the official, unconstrained
Crafter protocol (Danijar Hafner, ICLR 2022):
- Single continuous survival episode (up to 300 steps).
- No oracle milestone resets or subgoals provided.
- Evaluates unlock rates across all 22 canonical achievements.
- Computes official logarithmic Crafter Score:
    Score = exp( (1/22) * sum(ln(1 + rate_i)) ) - 1
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure core and plugins can be imported cleanly
_current_dir = Path(__file__).resolve().parent
_plugins_dir = _current_dir.parent
_core_dir = _plugins_dir.parent

for p in [str(_core_dir), str(_plugins_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from crafter_adapter.action import CrafterActionAdapter
from crafter_adapter.environment import make_crafter_env
from crafter_adapter.perception import CrafterPerceptionAdapter
from crafter_adapter.types import (
    CrafterAchievement,
    CrafterAction,
    CrafterObservation,
)

ALL_22_ACHIEVEMENTS = [
    CrafterAchievement.COLLECT_WOOD,
    CrafterAchievement.PLACE_TABLE,
    CrafterAchievement.EAT_COW,
    CrafterAchievement.COLLECT_SAPLING,
    CrafterAchievement.COLLECT_DRINK,
    CrafterAchievement.MAKE_WOOD_PICKAXE,
    CrafterAchievement.MAKE_WOOD_SWORD,
    CrafterAchievement.PLACE_PLANT,
    CrafterAchievement.DEFEAT_ZOMBIE,
    CrafterAchievement.COLLECT_STONE,
    CrafterAchievement.PLACE_STONE,
    CrafterAchievement.EAT_PLANT,
    CrafterAchievement.DEFEAT_SKELETON,
    CrafterAchievement.MAKE_STONE_PICKAXE,
    CrafterAchievement.MAKE_STONE_SWORD,
    CrafterAchievement.PLACE_FURNACE,
    CrafterAchievement.COLLECT_COAL,
    CrafterAchievement.COLLECT_IRON,
    CrafterAchievement.MAKE_IRON_PICKAXE,
    CrafterAchievement.MAKE_IRON_SWORD,
    CrafterAchievement.COLLECT_DIAMOND,
    CrafterAchievement.WAKE_UP,
]


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
class HafnerEpisodeResult:
    seed: int
    steps: int
    reward: float
    achievements: set[str]
    elapsed_ms: float


class PureHCIRCrafterHafnerAgent:
    """Pure HCIR reasoning agent for unconstrained survival (0 LLM tokens)."""

    def __init__(self) -> None:
        self.perception = CrafterPerceptionAdapter()
        self.action_adapter = CrafterActionAdapter()

    def select_action(self, obs: CrafterObservation) -> CrafterAction:
        return self.action_adapter.plan_next_action(obs, goal=None)


def run_hafner_benchmark(
    episodes: int = 5,
    base_seed: int = 1000,
    max_steps: int = 300,
) -> dict[str, Any]:
    """Execute official unconstrained Hafner Crafter benchmark."""
    achievement_counts: dict[str, int] = {a.value: 0 for a in ALL_22_ACHIEVEMENTS}
    episode_results: list[HafnerEpisodeResult] = []

    for ep in range(episodes):
        seed = base_seed + ep
        env = make_crafter_env(seed=seed)
        obs, _ = env.reset(seed=seed)
        agent = PureHCIRCrafterHafnerAgent()

        t0 = time.perf_counter()
        total_reward = 0.0
        done = False

        while not done and obs.step_count < max_steps:
            act = agent.select_action(obs)
            obs, r, term, trunc, _ = env.step(act)
            total_reward += r
            if term or trunc:
                done = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        unlocked = {a.value for a in obs.achievements}

        for ach in ALL_22_ACHIEVEMENTS:
            if ach.value in unlocked:
                achievement_counts[ach.value] += 1

        res = HafnerEpisodeResult(
            seed=seed,
            steps=obs.step_count,
            reward=total_reward,
            achievements=unlocked,
            elapsed_ms=elapsed_ms,
        )
        episode_results.append(res)
        if hasattr(env, "close"):
            env.close()

    # Compute Hafner Score: exp((1/22) * sum(ln(1 + rate_i))) - 1
    log_sum = 0.0
    rates: dict[str, float] = {}
    for ach in ALL_22_ACHIEVEMENTS:
        rate = achievement_counts[ach.value] / episodes if episodes else 0.0
        rates[ach.value] = rate
        log_sum += math.log(1.0 + rate)

    crafter_score = math.exp(log_sum / len(ALL_22_ACHIEVEMENTS)) - 1.0
    mean_steps = sum(r.steps for r in episode_results) / episodes if episodes else 0.0
    mean_reward = sum(r.reward for r in episode_results) / episodes if episodes else 0.0

    return {
        "episodes": episodes,
        "crafter_score": round(crafter_score * 100.0, 2),
        "mean_steps": round(mean_steps, 1),
        "mean_reward": round(mean_reward, 2),
        "achievement_rates": {k: round(v * 100.0, 1) for k, v in rates.items()},
        "results": episode_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crafter 22-Achievement Hafner Benchmark")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=1000, help="Base random seed")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"Running Official Crafter 22-Achievement Hafner Benchmark ({args.episodes} episodes)")
    print(f"{'=' * 80}\n")

    res = run_hafner_benchmark(
        episodes=args.episodes, base_seed=args.seed, max_steps=args.max_steps
    )

    print(f"Official Hafner Crafter Score: {res['crafter_score']}%\n")
    print(f"Mean Steps: {res['mean_steps']} | Mean Reward: {res['mean_reward']}\n")
    print(f"{'Achievement':<25} | {'Unlock Rate':<12}")
    print("-" * 40)
    for ach, rate in res["achievement_rates"].items():
        print(f"{ach:<25} | {rate:5.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
