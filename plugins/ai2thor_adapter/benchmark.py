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


AI2THOR_TIERS = [
    (1, "Tier 1: Object Retrieval"),
    (2, "Tier 2: State Toggling / Opening"),
    (3, "Tier 3: Surface Relocation"),
    (4, "Tier 4: Container Transfer"),
]


def run_ai2thor_tier_benchmark(
    cohort_name: str,
    tier: int,
    episodes: int = 3,
    base_seed: int = 5000,
) -> dict[str, Any]:
    """Run benchmark for a given cohort on a specific AI2-THOR manipulation tier."""
    results: list[AI2ThorEpisodeResult] = []

    for i in range(episodes):
        seed = base_seed + (tier * 100) + i
        env = make_ai2thor_env(seed=seed, tier=tier)
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


def run_ai2thor_benchmark(
    cohort_name: str,
    episodes: int = 15,
    base_seed: int = 5000,
) -> dict[str, Any]:
    """Backwards-compatible benchmark across standard environments."""
    eps_per_tier = max(1, episodes // len(AI2THOR_TIERS))
    all_results = []
    total_successes = 0
    total_steps = 0
    total_eps = 0

    for tier_id, _ in AI2THOR_TIERS:
        res = run_ai2thor_tier_benchmark(
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
    parser = argparse.ArgumentParser(description="AI2-THOR Multi-Tier Benchmark")
    parser.add_argument("--episodes-per-tier", type=int, default=3, help="Episodes per tier")
    parser.add_argument("--seed", type=int, default=5000, help="Base seed")
    args = parser.parse_args()

    print(f"\n{'=' * 85}")
    print(
        f"Running AI2-THOR Full-Spectrum Benchmark (4 Tiers, N={args.episodes_per_tier} eps/tier)"
    )
    print(f"{'=' * 85}\n")

    for cohort in ("pure-hcir", "llm-only"):
        print(f"--- Cohort: {cohort.upper()} ---")
        print(
            f"{'Manipulation Tier':<35} | {'Success Rate':<12} | {'95% Wilson CI':<18} | {'Mean Steps'}"
        )
        print("-" * 85)

        total_eps = 0
        total_successes = 0
        total_steps = 0

        for tier_id, tier_name in AI2THOR_TIERS:
            data = run_ai2thor_tier_benchmark(
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
