"""
AI2-THOR Three-Cohort Benchmark Runner.

Evaluates Pure HCIR vs LLM-Only / Stochastic baseline on
3D Embodied Object Manipulation and Receptacle Placement with Wilson CIs.
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

from ai2thor_adapter.action import AI2ThorActionAdapter
from ai2thor_adapter.environment import make_ai2thor_env
from ai2thor_adapter.perception import AI2ThorPerceptionAdapter
from ai2thor_adapter.types import (
    AI2ThorActionType,
    AI2ThorGoal,
    AI2ThorObservation,
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
class AI2ThorEpisodeResult:
    seed: int
    steps: int
    success: bool
    reward: float
    elapsed_ms: float


class PureHCIRAI2ThorAgent:
    """Pure HCIR reasoning agent with 0 LLM tokens."""

    def __init__(self) -> None:
        self.perception = AI2ThorPerceptionAdapter()
        self.action_adapter = AI2ThorActionAdapter()

    def select_action(self, obs: AI2ThorObservation, goal: AI2ThorGoal) -> dict[str, Any] | str:
        self.perception.ingest_observation(obs)
        return self.action_adapter.plan_next_action(obs, goal)


class LLMOnlyAI2ThorAgent:
    """Stochastic baseline wandering 3D room."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select_action(self, obs: AI2ThorObservation, goal: AI2ThorGoal) -> dict[str, Any] | str:
        acts = [
            AI2ThorActionType.MOVE_AHEAD,
            AI2ThorActionType.ROTATE_RIGHT,
            AI2ThorActionType.ROTATE_LEFT,
            AI2ThorActionType.MOVE_LEFT,
            AI2ThorActionType.MOVE_RIGHT,
        ]
        return self.rng.choice(acts)


def run_ai2thor_benchmark(
    cohort_name: str,
    episodes: int = 15,
    base_seed: int = 5000,
) -> dict[str, Any]:
    """Run benchmark across episodes."""
    results: list[AI2ThorEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + i
        env = make_ai2thor_env(seed=seed)
        obs, info = env.reset(seed=seed)
        goal: AI2ThorGoal = info["goal"]

        if cohort_name in ("pure-hcir", "guided-hcir"):
            agent = PureHCIRAI2ThorAgent()
        else:
            agent = LLMOnlyAI2ThorAgent(seed=seed)

        t0 = time.perf_counter()
        done = False
        total_reward = 0.0

        while not done and obs.step_count < 60:
            act = agent.select_action(obs, goal)
            obs, r, term, trunc, info = env.step(act)
            total_reward += r
            if term or trunc:
                done = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        success = total_reward > 0.0

        results.append(
            AI2ThorEpisodeResult(
                seed=seed,
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
    parser = argparse.ArgumentParser(description="AI2-THOR Three-Cohort Benchmark")
    parser.add_argument("--episodes", type=int, default=15, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=5000, help="Base seed")
    args = parser.parse_args()

    print(f"\n{'=' * 75}\nRunning AI2-THOR Benchmark (N={args.episodes} episodes)\n{'=' * 75}")

    for cohort in ("pure-hcir", "llm-only"):
        data = run_ai2thor_benchmark(cohort, episodes=args.episodes, base_seed=args.seed)
        print(
            f"Cohort: {data['cohort']:<15} | "
            f"Success Rate: {data['success_rate'] * 100:5.1f}% CI={data['ci_95']} | "
            f"Mean Steps: {data['mean_steps']:5.1f}"
        )


if __name__ == "__main__":
    main()
