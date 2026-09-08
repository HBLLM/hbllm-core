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


SAFETY_TIERS = [
    (1, "Tier 1: Open Navigation"),
    (2, "Tier 2: Static Hazards"),
    (3, "Tier 3: Dynamic Gremlins"),
    (4, "Tier 4: Constrained Corridor"),
]


def run_safety_gym_tier_benchmark(
    cohort_name: str,
    tier: int,
    episodes: int = 5,
    base_seed: int = 3000,
) -> dict[str, Any]:
    """Run benchmark for a given cohort on a specific safety tier."""
    results: list[SafetyEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + (tier * 100) + i
        env = make_safety_gym_env(seed=seed, tier=tier)
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
    mean_cost = sum(r.cumulative_cost for r in results) / episodes if episodes else 0.0
    mean_steps = sum(r.steps for r in results) / episodes if episodes else 0.0

    return {
        "cohort": cohort_name,
        "tier": tier,
        "episodes": episodes,
        "goal_reach_rate": goals / episodes if episodes else 0.0,
        "ci_goal_95": [round(ci_goal_low, 3), round(ci_goal_high, 3)],
        "zero_violation_rate": zero_viols / episodes if episodes else 0.0,
        "ci_zero_95": [round(ci_zero_low, 3), round(ci_zero_high, 3)],
        "mean_cost": round(mean_cost, 2),
        "mean_steps": round(mean_steps, 1),
        "results": [asdict(r) for r in results],
    }


def run_safety_gym_benchmark(
    cohort_name: str,
    episodes: int = 20,
    base_seed: int = 3000,
) -> dict[str, Any]:
    """Backwards-compatible benchmark across standard environments."""
    eps_per_tier = max(1, episodes // len(SAFETY_TIERS))
    all_results = []
    total_goals = 0
    total_zeros = 0
    total_cost = 0.0
    total_steps = 0
    total_eps = 0

    for tier_id, _ in SAFETY_TIERS:
        res = run_safety_gym_tier_benchmark(
            cohort_name, tier=tier_id, episodes=eps_per_tier, base_seed=base_seed
        )
        all_results.extend(res["results"])
        total_eps += res["episodes"]
        total_goals += sum(1 for r in res["results"] if r["goal_reached"])
        total_zeros += sum(1 for r in res["results"] if r["zero_violation"])
        total_cost += sum(r["cumulative_cost"] for r in res["results"])
        total_steps += sum(r["steps"] for r in res["results"])

    ci_goal_low, ci_goal_high = wilson_score_interval(total_goals, total_eps)
    ci_zero_low, ci_zero_high = wilson_score_interval(total_zeros, total_eps)

    return {
        "cohort": cohort_name,
        "episodes": total_eps,
        "goal_reach_rate": total_goals / total_eps if total_eps else 0.0,
        "ci_goal_95": [round(ci_goal_low, 3), round(ci_goal_high, 3)],
        "zero_violation_rate": total_zeros / total_eps if total_eps else 0.0,
        "ci_zero_95": [round(ci_zero_low, 3), round(ci_zero_high, 3)],
        "mean_cost": round(total_cost / total_eps, 2) if total_eps else 0.0,
        "mean_steps": round(total_steps / total_eps, 1) if total_eps else 0.0,
        "results": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety-Gymnasium Multi-Tier Benchmark")
    parser.add_argument("--episodes-per-tier", type=int, default=5, help="Episodes per tier")
    parser.add_argument("--seed", type=int, default=3000, help="Base seed")
    args = parser.parse_args()

    print(f"\n{'=' * 95}")
    print(
        f"Running Safety-Gymnasium Full-Spectrum Benchmark (4 Tiers, N={args.episodes_per_tier} eps/tier)"
    )
    print(f"{'=' * 95}\n")

    for cohort in ("pure-hcir", "llm-only"):
        print(f"--- Cohort: {cohort.upper()} ---")
        print(
            f"{'Safety Tier':<28} | {'Goal Rate':<9} | {'C=0 Rate':<9} | {'95% Wilson CI (C=0)':<19} | {'Mean Cost':<9} | {'Mean Steps'}"
        )
        print("-" * 95)

        total_eps = 0
        total_goals = 0
        total_zeros = 0
        total_cost = 0.0
        total_steps = 0

        for tier_id, tier_name in SAFETY_TIERS:
            data = run_safety_gym_tier_benchmark(
                cohort,
                tier=tier_id,
                episodes=args.episodes_per_tier,
                base_seed=args.seed,
            )
            total_eps += data["episodes"]
            total_goals += sum(1 for r in data["results"] if r["goal_reached"])
            total_zeros += sum(1 for r in data["results"] if r["zero_violation"])
            total_cost += sum(r["cumulative_cost"] for r in data["results"])
            total_steps += sum(r["steps"] for r in data["results"])

            ci_str = f"[{data['ci_zero_95'][0]:.3f}, {data['ci_zero_95'][1]:.3f}]"
            print(
                f"{tier_name:<28} | "
                f"{data['goal_reach_rate'] * 100:8.1f}% | "
                f"{data['zero_violation_rate'] * 100:8.1f}% | "
                f"{ci_str:<19} | "
                f"{data['mean_cost']:9.2f} | "
                f"{data['mean_steps']:10.1f}"
            )

        overall_goal_rate = total_goals / total_eps if total_eps else 0.0
        overall_zero_rate = total_zeros / total_eps if total_eps else 0.0
        ci_zero_low, ci_zero_high = wilson_score_interval(total_zeros, total_eps)
        overall_cost = total_cost / total_eps if total_eps else 0.0
        overall_steps = total_steps / total_eps if total_eps else 0.0

        print("-" * 95)
        print(
            f"{'OVERALL':<28} | "
            f"{overall_goal_rate * 100:8.1f}% | "
            f"{overall_zero_rate * 100:8.1f}% | "
            f"[{ci_zero_low:.3f}, {ci_zero_high:.3f}]       | "
            f"{overall_cost:9.2f} | "
            f"{overall_steps:10.1f} (Total {total_eps} eps)"
        )
        print("\n")


if __name__ == "__main__":
    main()
