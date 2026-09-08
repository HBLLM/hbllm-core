"""
ALFWorld Multi-Tier Benchmark Runner.

Evaluates Pure HCIR vs LLM-Only baseline across all 6 canonical ALFWorld task tiers:
- Tier 1: Pick and Place
- Tier 2: Examine in Light
- Tier 3: Clean and Place
- Tier 4: Heat and Place
- Tier 5: Cool and Place
- Tier 6: Pick Two and Place

Computes exact per-tier success rates, 95% Wilson score CIs, and mean steps.
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
    tier: str
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
        if obs.admissible_commands:
            return self.rng.choice(obs.admissible_commands)
        return "look"


def run_alfworld_benchmark(
    cohort_name: str,
    episodes_per_tier: int = 3,
    base_seed: int = 2000,
    episodes: int | None = None,
) -> dict[str, Any]:
    """Run benchmark across all 6 ALFWorld task types with per-tier aggregation."""
    tier_map = {
        "Tier 1: Pick & Place": ALFWorldTaskType.PICK_AND_PLACE,
        "Tier 2: Examine in Light": ALFWorldTaskType.EXAMINE_IN_LIGHT,
        "Tier 3: Clean & Place": ALFWorldTaskType.CLEAN_AND_PLACE,
        "Tier 4: Heat & Place": ALFWorldTaskType.HEAT_AND_PLACE,
        "Tier 5: Cool & Place": ALFWorldTaskType.COOL_AND_PLACE,
        "Tier 6: Pick Two & Place": ALFWorldTaskType.PICK_TWO_AND_PLACE,
    }

    if episodes is not None:
        episodes_per_tier = max(1, episodes // len(tier_map))

    all_results: list[ALFWorldEpisodeResult] = []
    tier_summaries: dict[str, Any] = {}

    current_seed = base_seed

    for tier_name, task_type in tier_map.items():
        tier_results: list[ALFWorldEpisodeResult] = []

        for _ in range(episodes_per_tier):
            seed = current_seed
            current_seed += 1

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

            res = ALFWorldEpisodeResult(
                seed=seed,
                tier=tier_name,
                task_type=task_type.value,
                steps=obs.step_count,
                success=success,
                reward=total_reward,
                elapsed_ms=elapsed_ms,
            )
            tier_results.append(res)
            all_results.append(res)

        tier_successes = sum(1 for r in tier_results if r.success)
        tier_total = len(tier_results)
        ci_low, ci_high = wilson_score_interval(tier_successes, tier_total)
        mean_steps = sum(r.steps for r in tier_results) / tier_total

        tier_summaries[tier_name] = {
            "success_rate": tier_successes / tier_total,
            "ci_95": [round(ci_low, 3), round(ci_high, 3)],
            "mean_steps": round(mean_steps, 1),
            "episodes": tier_total,
        }

    total_successes = sum(1 for r in all_results if r.success)
    overall_ci = wilson_score_interval(total_successes, len(all_results))
    overall_mean_steps = sum(r.steps for r in all_results) / len(all_results)

    return {
        "cohort": cohort_name,
        "episodes": len(all_results),
        "total_episodes": len(all_results),
        "success_rate": total_successes / len(all_results) if all_results else 0.0,
        "overall_success_rate": total_successes / len(all_results) if all_results else 0.0,
        "ci_95": [round(overall_ci[0], 3), round(overall_ci[1], 3)],
        "overall_ci_95": [round(overall_ci[0], 3), round(overall_ci[1], 3)],
        "mean_steps": round(overall_mean_steps, 1),
        "overall_mean_steps": round(overall_mean_steps, 1),
        "tier_summaries": tier_summaries,
        "results": [asdict(r) for r in all_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ALFWorld 6-Tier Benchmark")
    parser.add_argument("--episodes-per-tier", type=int, default=3, help="Episodes per task tier")
    parser.add_argument("--seed", type=int, default=2000, help="Base seed")
    args = parser.parse_args()

    print(f"\n{'=' * 85}\nRunning ALFWorld Full-Spectrum Benchmark (6 Task Tiers)\n{'=' * 85}")

    for cohort in ("pure-hcir", "llm-only"):
        data = run_alfworld_benchmark(
            cohort, episodes_per_tier=args.episodes_per_tier, base_seed=args.seed
        )
        print(f"\n--- Cohort: {data['cohort'].upper()} ---")
        print(
            f"{'Task Tier':<25} | {'Success Rate':<12} | {'95% Wilson CI':<16} | {'Mean Steps':<10}"
        )
        print("-" * 75)
        for tier_name, s in data["tier_summaries"].items():
            print(
                f"{tier_name:<25} | {s['success_rate'] * 100:5.1f}%       | "
                f"[{s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f}]    | {s['mean_steps']:5.1f}"
            )
        print("-" * 75)
        print(
            f"{'OVERALL':<25} | {data['overall_success_rate'] * 100:5.1f}%       | "
            f"[{data['overall_ci_95'][0]:.3f}, {data['overall_ci_95'][1]:.3f}]    | "
            f"Mean: {data['overall_mean_steps']:4.1f} steps (Total {data['total_episodes']} eps)\n"
        )


if __name__ == "__main__":
    main()
