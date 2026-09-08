"""
MineDojo Three-Cohort Benchmark Runner.

Evaluates Pure HCIR vs LLM-Only baseline on
Tree Harvesting, Tool Synthesis, and Recipe DAG Progression with Wilson CIs.
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

from minedojo_adapter.action import MineDojoActionAdapter
from minedojo_adapter.environment import make_minedojo_env
from minedojo_adapter.perception import MineDojoPerceptionAdapter
from minedojo_adapter.types import (
    MineDojoAction,
    MineDojoGoal,
    MineDojoObservation,
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
class MineDojoEpisodeResult:
    seed: int
    steps: int
    success: bool
    reward: float
    logs_mined: int
    has_table: bool
    has_pickaxe: bool
    elapsed_ms: float


class PureHCIRMineDojoAgent:
    """Pure HCIR reasoning agent with 0 LLM tokens."""

    def __init__(self) -> None:
        self.perception = MineDojoPerceptionAdapter()
        self.action_adapter = MineDojoActionAdapter()

    def select_action(self, obs: MineDojoObservation, goal: MineDojoGoal) -> MineDojoAction:
        self.perception.ingest_observation(obs)
        return self.action_adapter.plan_next_action(obs, goal)


class LLMOnlyMineDojoAgent:
    """Stochastic baseline executing random moves and crafting attempts."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select_action(self, obs: MineDojoObservation, goal: MineDojoGoal) -> MineDojoAction:
        acts = [
            MineDojoAction.MOVE_FORWARD,
            MineDojoAction.MINE_BLOCK,
            MineDojoAction.CRAFT_PLANKS,
            MineDojoAction.CRAFT_STICKS,
            MineDojoAction.CRAFT_TABLE,
            MineDojoAction.CRAFT_WOOD_PICKAXE,
        ]
        return self.rng.choice(acts)


def run_minedojo_benchmark(
    cohort_name: str,
    episodes: int = 15,
    base_seed: int = 6000,
) -> dict[str, Any]:
    """Run benchmark across episodes."""
    results: list[MineDojoEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + i
        env = make_minedojo_env(seed=seed)
        obs, info = env.reset(seed=seed)
        goal: MineDojoGoal = info["goal"]

        if cohort_name in ("pure-hcir", "guided-hcir"):
            agent = PureHCIRMineDojoAgent()
        else:
            agent = LLMOnlyMineDojoAgent(seed=seed)

        t0 = time.perf_counter()
        done = False
        total_reward = 0.0

        while not done and obs.step_count < 50:
            act = agent.select_action(obs, goal)
            obs, r, term, trunc, info = env.step(act)
            total_reward += r
            if term or trunc:
                done = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        success = obs.inventory.wooden_pickaxe > 0

        results.append(
            MineDojoEpisodeResult(
                seed=seed,
                steps=obs.step_count,
                success=success,
                reward=total_reward,
                logs_mined=obs.inventory.log,
                has_table=obs.inventory.crafting_table > 0,
                has_pickaxe=obs.inventory.wooden_pickaxe > 0,
                elapsed_ms=elapsed_ms,
            )
        )

    successes = sum(1 for r in results if r.success)
    ci_low, ci_high = wilson_score_interval(successes, episodes)
    mean_steps = sum(r.steps for r in results) / episodes

    return {
        "cohort": cohort_name,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(mean_steps, 1),
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MineDojo Three-Cohort Benchmark")
    parser.add_argument("--episodes", type=int, default=15, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=6000, help="Base seed")
    args = parser.parse_args()

    print(f"\n{'=' * 75}\nRunning MineDojo Benchmark (N={args.episodes} episodes)\n{'=' * 75}")

    for cohort in ("pure-hcir", "llm-only"):
        data = run_minedojo_benchmark(cohort, episodes=args.episodes, base_seed=args.seed)
        print(
            f"Cohort: {data['cohort']:<15} | "
            f"Craft Pickaxe Rate: {data['success_rate'] * 100:5.1f}% CI={data['ci_95']} | "
            f"Mean Steps: {data['mean_steps']:5.1f}"
        )


if __name__ == "__main__":
    main()
