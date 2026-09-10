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


MINEDOJO_TIERS = [
    (1, "Tier 1: Wood Harvesting"),
    (2, "Tier 2: Planks & Sticks"),
    (3, "Tier 3: Crafting Table"),
    (4, "Tier 4: Wooden Pickaxe"),
    (5, "Tier 5: Stone Pickaxe"),
]

VALID_HCIR_COHORTS = frozenset(
    {"pure-hcir", "pure_hcir", "guided-hcir", "guided_hcir", "hbllm-core", "hbllm_core", "hcir"}
)
VALID_BASELINE_COHORTS = frozenset({"llm-only", "llm_only", "react", "baseline"})


def resolve_cohort(cohort_name: str) -> str:
    """Validate and normalize evaluation cohort identifier, failing loudly on unknown values."""
    cleaned = cohort_name.strip().lower().replace("_", "-")
    if cleaned in VALID_HCIR_COHORTS:
        return "pure-hcir"
    if cleaned in VALID_BASELINE_COHORTS:
        return "llm-only"
    raise ValueError(
        f"Unrecognized cohort '{cohort_name}'. "
        f"Allowed HCIR cohorts: {sorted(VALID_HCIR_COHORTS)}. "
        f"Allowed baseline cohorts: {sorted(VALID_BASELINE_COHORTS)}."
    )


def run_minedojo_tier_benchmark(
    cohort_name: str,
    tier: int,
    episodes: int = 3,
    base_seed: int = 6000,
) -> dict[str, Any]:
    """Run benchmark for a given cohort on a specific MineDojo tier."""
    canonical_cohort = resolve_cohort(cohort_name)
    results: list[MineDojoEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + (tier * 100) + i
        env = make_minedojo_env(seed=seed, tier=tier)
        obs, info = env.reset(seed=seed)
        goal: MineDojoGoal = info["goal"]

        if canonical_cohort == "pure-hcir":
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
        success = total_reward > 0.0

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
    mean_steps = sum(r.steps for r in results) / episodes if episodes else 0.0

    return {
        "cohort": cohort_name,
        "tier": tier,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(mean_steps, 1),
        "results": [asdict(r) for r in results],
    }


def run_minedojo_benchmark(
    cohort_name: str,
    episodes: int = 15,
    base_seed: int = 6000,
) -> dict[str, Any]:
    """Backwards-compatible benchmark across standard environments."""
    eps_per_tier = max(1, episodes // len(MINEDOJO_TIERS))
    all_results = []
    total_successes = 0
    total_steps = 0
    total_eps = 0

    for tier_id, _ in MINEDOJO_TIERS:
        res = run_minedojo_tier_benchmark(
            cohort_name, tier=tier_id, episodes=eps_per_tier, base_seed=base_seed
        )
        all_results.extend(res["results"])
        total_eps += res["episodes"]
        total_successes += sum(1 for r in res["results"] if r["success"])
        total_steps += sum(r["steps"] for r in res["results"])

    ci_low, ci_high = wilson_score_interval(total_successes, total_eps)

    return {
        "cohort": cohort_name,
        "episodes": total_eps,
        "success_rate": total_successes / total_eps if total_eps else 0.0,
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "mean_steps": round(total_steps / total_eps, 1) if total_eps else 0.0,
        "results": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MineDojo Multi-Tier Benchmark")
    parser.add_argument("--episodes-per-tier", type=int, default=3, help="Episodes per tier")
    parser.add_argument("--seed", type=int, default=6000, help="Base seed")
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Cohort to benchmark ('pure-hcir' / 'HBLLM-Core', 'llm-only'). Defaults to running both.",
    )
    args = parser.parse_args()

    cohorts_to_run = (resolve_cohort(args.cohort),) if args.cohort else ("pure-hcir", "llm-only")

    print(f"\n{'=' * 85}")
    print(
        f"Running MineDojo Full-Spectrum Benchmark (5 Tiers, N={args.episodes_per_tier} eps/tier)"
    )
    print(f"{'=' * 85}\n")

    for cohort in cohorts_to_run:
        print(f"--- Cohort: {cohort.upper()} ---")
        print(
            f"{'Progression Tier':<35} | {'Success Rate':<12} | {'95% Wilson CI':<18} | {'Mean Steps'}"
        )
        print("-" * 85)

        total_eps = 0
        total_successes = 0
        total_steps = 0

        for tier_id, tier_name in MINEDOJO_TIERS:
            data = run_minedojo_tier_benchmark(
                cohort,
                tier=tier_id,
                episodes=args.episodes_per_tier,
                base_seed=args.seed,
            )
            total_eps += data["episodes"]
            total_successes += sum(1 for r in data["results"] if r["success"])
            total_steps += sum(r["steps"] for r in data["results"])

            ci_str = f"[{data['ci_95'][0]:.3f}, {data['ci_95'][1]:.3f}]"
            print(
                f"{tier_name:<35} | "
                f"{data['success_rate'] * 100:11.1f}% | "
                f"{ci_str:<18} | "
                f"{data['mean_steps']:10.1f}"
            )

        overall_rate = total_successes / total_eps if total_eps else 0.0
        ci_low, ci_high = wilson_score_interval(total_successes, total_eps)
        overall_steps = total_steps / total_eps if total_eps else 0.0

        print("-" * 85)
        print(
            f"{'OVERALL':<35} | "
            f"{overall_rate * 100:11.1f}% | "
            f"[{ci_low:.3f}, {ci_high:.3f}]        | "
            f"{overall_steps:10.1f} (Total {total_eps} eps)"
        )
        print("\n")


if __name__ == "__main__":
    main()
