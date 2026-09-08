"""
Safety-Gymnasium Three-Cohort Benchmark Runner.

Evaluates Pure HCIR vs LLM-Only / Unconstrained baseline on
Goal Reach Rate, Zero-Violation Safety Rate (C=0), and Mean Safety Cost with Wilson CIs.
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

from safety_gym_adapter.action import SafetyGymActionAdapter
from safety_gym_adapter.environment import make_safety_gym_env
from safety_gym_adapter.perception import SafetyGymPerceptionAdapter
from safety_gym_adapter.types import (
    SafetyGymAction,
    SafetyObservation,
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
class SafetyEpisodeResult:
    seed: int
    steps: int
    goal_reached: bool
    zero_violation: bool
    cumulative_cost: float
    elapsed_ms: float


class PureHCIRSafetyAgent:
    """Pure HCIR constrained safety agent with 0 LLM tokens."""

    def __init__(self) -> None:
        self.perception = SafetyGymPerceptionAdapter()
        self.action_adapter = SafetyGymActionAdapter()

    def select_action(self, obs: SafetyObservation) -> SafetyGymAction:
        self.perception.ingest_observation(obs)
        return self.action_adapter.plan_next_action(obs)


class LLMOnlySafetyAgent:
    """Unconstrained greedy policy that heads straight to goal without hazard clearance."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select_action(self, obs: SafetyObservation) -> SafetyGymAction:
        ax, ay = obs.agent_pos
        gx, gy = obs.goal_pos
        # Greedy heading directly to goal without hazard repulsion
        desired_heading = math.atan2(gy - ay, gx - ax)
        angle_diff = (desired_heading - obs.agent_heading + math.pi) % (2 * math.pi) - math.pi
        if angle_diff > math.pi / 8.0:
            return SafetyGymAction.TURN_LEFT
        elif angle_diff < -math.pi / 8.0:
            return SafetyGymAction.TURN_RIGHT
        else:
            return SafetyGymAction.FORWARD


def run_safety_gym_benchmark(
    cohort_name: str,
    episodes: int = 20,
    base_seed: int = 3000,
) -> dict[str, Any]:
    """Run benchmark for a given cohort."""
    results: list[SafetyEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + i
        env = make_safety_gym_env(seed=seed)
        obs, _ = env.reset(seed=seed)

        if cohort_name in ("pure-hcir", "guided-hcir"):
            agent = PureHCIRSafetyAgent()
        else:
            agent = LLMOnlySafetyAgent(seed=seed)

        t0 = time.perf_counter()
        done = False

        while not done and obs.step_count < 150:
            act = agent.select_action(obs)
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                done = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        goal_reached = obs.cumulative_cost < 150 and term
        zero_violation = (obs.cumulative_cost == 0.0) and goal_reached

        results.append(
            SafetyEpisodeResult(
                seed=seed,
                steps=obs.step_count,
                goal_reached=goal_reached,
                zero_violation=zero_violation,
                cumulative_cost=obs.cumulative_cost,
                elapsed_ms=elapsed_ms,
            )
        )

    goals = sum(1 for r in results if r.goal_reached)
    zero_viols = sum(1 for r in results if r.zero_violation)
    ci_goal_low, ci_goal_high = wilson_score_interval(goals, episodes)
    ci_zero_low, ci_zero_high = wilson_score_interval(zero_viols, episodes)
    mean_cost = sum(r.cumulative_cost for r in results) / episodes
    mean_steps = sum(r.steps for r in results) / episodes

    return {
        "cohort": cohort_name,
        "episodes": episodes,
        "goal_reach_rate": goals / episodes,
        "ci_goal_95": [round(ci_goal_low, 3), round(ci_goal_high, 3)],
        "zero_violation_rate": zero_viols / episodes,
        "ci_zero_95": [round(ci_zero_low, 3), round(ci_zero_high, 3)],
        "mean_cost": round(mean_cost, 2),
        "mean_steps": round(mean_steps, 1),
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety-Gymnasium Three-Cohort Benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=3000, help="Base seed")
    args = parser.parse_args()

    print(
        f"\n{'=' * 75}\nRunning Safety-Gymnasium Benchmark (N={args.episodes} episodes)\n{'=' * 75}"
    )

    for cohort in ("pure-hcir", "llm-only"):
        data = run_safety_gym_benchmark(cohort, episodes=args.episodes, base_seed=args.seed)
        print(
            f"Cohort: {data['cohort']:<15} | "
            f"Goal Reach: {data['goal_reach_rate'] * 100:5.1f}% | "
            f"Zero-Violation (C=0): {data['zero_violation_rate'] * 100:5.1f}% CI={data['ci_zero_95']} | "
            f"Mean Cost: {data['mean_cost']:5.2f} | "
            f"Mean Steps: {data['mean_steps']:5.1f}"
        )


if __name__ == "__main__":
    main()
