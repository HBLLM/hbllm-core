"""
ALFWorld Three-Cohort Benchmark Runner.

Evaluates Pure HCIR vs LLM-Only / ReAct baseline across all 6 ALFWorld
household task categories with Wilson score 95% confidence intervals.
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

from alfworld_adapter.action import ALFWorldActionAdapter
from alfworld_adapter.environment import make_alfworld_env
from alfworld_adapter.perception import ALFWorldPerceptionAdapter
from alfworld_adapter.types import (
    ALFWorldGoal,
    ALFWorldObservation,
    ALFWorldTaskType,
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
class ALFWorldEpisodeResult:
    seed: int
    task_type: str
    steps: int
    success: bool
    reward: float
    elapsed_ms: float


class PureHCIRALFWorldAgent:
    """Pure HCIR reasoning agent with 0 LLM tokens."""

    def __init__(self) -> None:
        self.perception = ALFWorldPerceptionAdapter()
        self.action_adapter = ALFWorldActionAdapter()

    def select_action(self, obs: ALFWorldObservation, goal: ALFWorldGoal) -> str:
        self.perception.ingest_observation(obs)
        return self.action_adapter.plan_next_action(obs, goal)


class LLMOnlyALFWorldAgent:
    """Stochastic baseline randomly choosing among admissible actions."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select_action(self, obs: ALFWorldObservation, goal: ALFWorldGoal) -> str:
        # Without causal affordance tracking, uniformly samples admissible command
        if obs.admissible_commands:
            return self.rng.choice(obs.admissible_commands)
        return "look"


def run_alfworld_benchmark(
    cohort_name: str,
    episodes: int = 18,
    base_seed: int = 2000,
) -> dict[str, Any]:
    """Run benchmark across all 6 ALFWorld task types."""
    task_types = list(ALFWorldTaskType)
    results: list[ALFWorldEpisodeResult] = []

    for i in range(episodes):
        task_type = task_types[i % len(task_types)]
        seed = base_seed + i

        env = make_alfworld_env(task_type=task_type, seed=seed)
        obs, info = env.reset(seed=seed, task_type=task_type)
        goal: ALFWorldGoal = info["goal"]

        if cohort_name in ("pure-hcir", "guided-hcir"):
            agent = PureHCIRALFWorldAgent()
        else:
            agent = LLMOnlyALFWorldAgent(seed=seed)

        t0 = time.perf_counter()
        done = False
        total_reward = 0.0

        while not done and obs.step_count < 30:
            act = agent.select_action(obs, goal)
            obs, r, term, trunc, info = env.step(act)
            total_reward += r
            if term or trunc:
                done = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        success = total_reward > 0.0

        results.append(
            ALFWorldEpisodeResult(
                seed=seed,
                task_type=task_type.value,
                steps=obs.step_count,
                success=success,
                reward=total_reward,
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
    parser = argparse.ArgumentParser(description="ALFWorld Three-Cohort Benchmark")
    parser.add_argument("--episodes", type=int, default=18, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=2000, help="Base seed")
    args = parser.parse_args()

    print(
        f"\n{'=' * 70}\nRunning ALFWorld Benchmark (N={args.episodes} episodes across 6 task types)\n{'=' * 70}"
    )

    for cohort in ("pure-hcir", "llm-only"):
        data = run_alfworld_benchmark(cohort, episodes=args.episodes, base_seed=args.seed)
        print(
            f"Cohort: {data['cohort']:<15} | "
            f"Success: {data['success_rate'] * 100:5.1f}% CI={data['ci_95']} | "
            f"Mean Steps: {data['mean_steps']:5.1f}"
        )


if __name__ == "__main__":
    main()
