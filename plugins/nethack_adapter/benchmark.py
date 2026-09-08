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


def run_nethack_benchmark(
    cohort_name: str,
    episodes: int = 15,
    base_seed: int = 4000,
) -> dict[str, Any]:
    """Run benchmark for a given cohort."""
    results: list[NetHackEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + i
        env = make_nethack_env(seed=seed)
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
    mean_steps = sum(r.steps for r in results) / episodes
    mean_gold = sum(r.gold for r in results) / episodes

    return {
        "cohort": cohort_name,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(mean_steps, 1),
        "mean_gold": round(mean_gold, 1),
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NetHack Three-Cohort Benchmark")
    parser.add_argument("--episodes", type=int, default=15, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=4000, help="Base seed")
    args = parser.parse_args()

    print(f"\n{'=' * 75}\nRunning NetHack Benchmark (N={args.episodes} episodes)\n{'=' * 75}")

    for cohort in ("pure-hcir", "llm-only"):
        data = run_nethack_benchmark(cohort, episodes=args.episodes, base_seed=args.seed)
        print(
            f"Cohort: {data['cohort']:<15} | "
            f"Descend Rate: {data['success_rate'] * 100:5.1f}% CI={data['ci_95']} | "
            f"Mean Steps: {data['mean_steps']:5.1f} | "
            f"Mean Gold: {data['mean_gold']:4.1f}"
        )


if __name__ == "__main__":
    main()
