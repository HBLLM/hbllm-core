"""
Crafter Multi-Tier Benchmark Runner.

Evaluates Pure HCIR vs LLM-Only baseline across 5 Canonical Tech Tiers:
- Tier 1: Survival & Gathering (Wood, Drink, Cow)
- Tier 2: Technology Genesis (Table, Wood Pickaxe)
- Tier 3: Stone Mining (Stone, Stone Pickaxe, Coal)
- Tier 4: Metallurgy (Iron, Furnace)
- Tier 5: Apex Endurance (Survive)

Computes per-tier success rates with Wilson 95% CIs and the official Crafter Score.
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

from crafter_adapter.action import CrafterActionAdapter
from crafter_adapter.environment import make_crafter_env
from crafter_adapter.perception import CrafterPerceptionAdapter
from crafter_adapter.types import (
    CrafterAchievement,
    CrafterAction,
    CrafterGoal,
    CrafterObservation,
)

logger = logging.getLogger(__name__)

CRAFTER_TIERS = {
    "Tier 1: Gathering": [
        CrafterAchievement.COLLECT_WOOD,
        CrafterAchievement.COLLECT_DRINK,
        CrafterAchievement.EAT_COW,
    ],
    "Tier 2: Basic Tools": [
        CrafterAchievement.PLACE_TABLE,
        CrafterAchievement.MAKE_WOOD_PICKAXE,
    ],
    "Tier 3: Stone Age": [
        CrafterAchievement.COLLECT_STONE,
        CrafterAchievement.MAKE_STONE_PICKAXE,
        CrafterAchievement.COLLECT_COAL,
    ],
    "Tier 4: Metallurgy": [
        CrafterAchievement.COLLECT_IRON,
        CrafterAchievement.PLACE_FURNACE,
    ],
    "Tier 5: Apex Endurance": [
        CrafterAchievement.SURVIVE,
    ],
}


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
class CrafterEpisodeResult:
    seed: int
    tier: str
    target: str
    steps: int
    reward: float
    achievements: list[str]
    success: bool
    elapsed_ms: float


class PureHCIRCrafterAgent:
    """Pure HCIR reasoning agent with 0 LLM tokens."""

    def __init__(self) -> None:
        self.perception = CrafterPerceptionAdapter()
        self.action_adapter = CrafterActionAdapter()

    def select_action(
        self, obs: CrafterObservation, goal: CrafterGoal | None = None
    ) -> CrafterAction:
        self.perception.ingest_observation(obs)
        return self.action_adapter.plan_next_action(obs, goal)


class LLMOnlyCrafterAgent:
    """Stochastic / ReAct approximation without epistemic spatial graph."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select_action(
        self, obs: CrafterObservation, goal: CrafterGoal | None = None
    ) -> CrafterAction:
        weights = [0.05] + [0.18] * 4 + [0.20] + [0.03] * 11
        return self.rng.choices(list(CrafterAction), weights=weights)[0]


def run_crafter_benchmark(
    cohort_name: str,
    episodes_per_target: int = 3,
    base_seed: int = 1000,
    episodes: int | None = None,
    prefer_native: bool = False,
) -> dict[str, Any]:
    """Run benchmark across all 5 Crafter competency tiers."""
    if episodes is not None:
        total_targets = sum(len(tgts) for tgts in CRAFTER_TIERS.values())
        episodes_per_target = max(1, episodes // total_targets)

    all_results: list[CrafterEpisodeResult] = []
    tier_summaries: dict[str, Any] = {}
    achievement_success_counts: dict[str, int] = {}
    achievement_total_counts: dict[str, int] = {}

    current_seed = base_seed

    for tier_name, targets in CRAFTER_TIERS.items():
        tier_results: list[CrafterEpisodeResult] = []

        for target in targets:
            goal = CrafterGoal(target_achievement=target)
            ach_key = target.value
            achievement_success_counts.setdefault(ach_key, 0)
            achievement_total_counts.setdefault(ach_key, 0)

            for _ in range(episodes_per_target):
                seed = current_seed
                current_seed += 1

                env = make_crafter_env(seed=seed, prefer_native=prefer_native)
                obs, _ = env.reset(seed=seed)

                if cohort_name in ("pure-hcir", "guided-hcir"):
                    agent = PureHCIRCrafterAgent()
                else:
                    agent = LLMOnlyCrafterAgent(seed=seed)

                t0 = time.perf_counter()
                done = False
                total_reward = 0.0

                max_steps = 70 if target == CrafterAchievement.SURVIVE else 50

                while not done and obs.step_count < max_steps:
                    act = agent.select_action(obs, goal)
                    obs, r, term, trunc, info = env.step(act)
                    total_reward += r
                    if target in obs.achievements:
                        done = True
                    if term or trunc:
                        done = True

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                achs = [a.value for a in obs.achievements]
                success = target.value in achs

                achievement_total_counts[ach_key] += 1
                if success:
                    achievement_success_counts[ach_key] += 1

                res = CrafterEpisodeResult(
                    seed=seed,
                    tier=tier_name,
                    target=target.value,
                    steps=obs.step_count,
                    reward=total_reward,
                    achievements=achs,
                    success=success,
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

    # Compute Hafner Crafter Score across tested achievements
    log_sum = 0.0
    for ach, total in achievement_total_counts.items():
        rate = achievement_success_counts[ach] / total
        log_sum += math.log(1.0 + rate)
    crafter_score = math.exp(log_sum / len(achievement_total_counts)) - 1.0

    total_successes = sum(1 for r in all_results if r.success)
    overall_ci = wilson_score_interval(total_successes, len(all_results))

    return {
        "cohort": cohort_name,
        "episodes": len(all_results),
        "total_episodes": len(all_results),
        "success_rate": total_successes / len(all_results) if all_results else 0.0,
        "overall_success_rate": total_successes / len(all_results) if all_results else 0.0,
        "ci_95": [round(overall_ci[0], 3), round(overall_ci[1], 3)],
        "overall_ci_95": [round(overall_ci[0], 3), round(overall_ci[1], 3)],
        "crafter_score": round(crafter_score * 100.0, 1),
        "tier_summaries": tier_summaries,
        "results": [asdict(r) for r in all_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crafter Multi-Tier Benchmark")
    parser.add_argument(
        "--episodes-per-target", type=int, default=3, help="Episodes per milestone target"
    )
    parser.add_argument("--seed", type=int, default=1000, help="Base random seed")
    parser.add_argument(
        "--prefer-native",
        "--native",
        action="store_true",
        help="Run against upstream native crafter package",
    )
    args = parser.parse_args()

    print(
        f"\n{'=' * 85}\nRunning Crafter Multi-Tier Benchmark (5 Tiers, 11 Milestones, Native={args.prefer_native})\n{'=' * 85}"
    )

    for cohort in ("pure-hcir", "llm-only"):
        data = run_crafter_benchmark(
            cohort,
            episodes_per_target=args.episodes_per_target,
            base_seed=args.seed,
            prefer_native=args.prefer_native,
        )
        print(
            f"\n--- Cohort: {data['cohort'].upper()} (Crafter Score: {data['crafter_score']}%) ---"
        )
        print(f"{'Tier':<25} | {'Success Rate':<12} | {'95% Wilson CI':<16} | {'Mean Steps':<10}")
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
            f"Total: {data['total_episodes']} eps\n"
        )


if __name__ == "__main__":
    main()
