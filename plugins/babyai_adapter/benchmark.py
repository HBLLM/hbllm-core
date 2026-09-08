"""Rigorous Multi-Seed Empirical Benchmark on Farama Gymnasium BabyAI Suite.

Evaluates HBLLM-Core across all BabyAI competency tiers across N>=100 random seeds,
aggregating empirical success rates, Wilson score 95% confidence intervals, mean steps,
and latency via ExperimentStatistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure core and plugins can be imported cleanly
_current_dir = Path(__file__).resolve().parent
_plugins_dir = _current_dir.parent
_core_dir = _plugins_dir.parent

for p in [str(_core_dir), str(_plugins_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from babyai_adapter import (
    BabyAIActionAdapter,
    BabyAIMissionParser,
    BabyAIPerceptionAdapter,
    MiniGridAction,
    make_gym_babyai_level,
)
from hbllm.experiment.statistics import ExperimentStatistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("babyai_benchmark")


@dataclass
class TierConfig:
    """Configuration for a BabyAI benchmark tier."""

    tier_id: str
    env_name: str
    description: str
    room_size: int = 10
    max_steps_override: int | None = None


TIER_CONFIGS: list[TierConfig] = [
    TierConfig(
        tier_id="Tier 1a: GoTo",
        env_name="BabyAI-GoToObj-v0",
        description="Single-room navigation to specific target with distractors",
        room_size=8,
    ),
    TierConfig(
        tier_id="Tier 1b: Pickup",
        env_name="BabyAI-PickupDist-v0",
        description="Single-room object pickup with distractor items",
        room_size=8,
    ),
    TierConfig(
        tier_id="Tier 2: Doors",
        env_name="BabyAI-OpenRedDoor-v0",
        description="Spatial navigation across partitioned room via closed door",
        room_size=10,
    ),
    TierConfig(
        tier_id="Tier 3: Unlock",
        env_name="BabyAI-UnlockLocal-v0",
        description="Prerequisite key retrieval and locked door opening",
        room_size=16,
    ),
    TierConfig(
        tier_id="Tier 4: PutNext",
        env_name="BabyAI-PutNextLocal-v0",
        description="Relational spatial goal placement (item next to landmark)",
        room_size=10,
    ),
    TierConfig(
        tier_id="Tier 5: Unblock",
        env_name="BabyAI-BlockedUnlockPickup-v0",
        description="Causal obstacle unblocking detour + key fetch + locked door",
        room_size=16,
    ),
    TierConfig(
        tier_id="Tier 6: Sequence",
        env_name="BabyAI-GoToSeqS5R2-v0",
        description="Sequential multi-subgoal instructions across partitioned spaces",
        room_size=12,
    ),
    TierConfig(
        tier_id="Tier 7: Synthesis",
        env_name="BabyAI-SynthS5R2-v0",
        description="Full compositional synthesis combining doors, keys, and objects",
        room_size=12,
    ),
    TierConfig(
        tier_id="Apex: BossLevel",
        env_name="BabyAI-BossLevel-v0",
        description="Full-horizon compound missions across multi-room mazes",
        room_size=20,
        max_steps_override=384,
    ),
]


@dataclass
class EpisodeResult:
    """Telemetry captured from a single benchmark episode."""

    tier_id: str
    env_name: str
    seed: int
    mission: str
    success: bool
    steps: int
    reward: float
    duration_ms: float
    llm_tokens: int = 0
    failure_reason: str = ""


@dataclass
class TierBenchmarkSummary:
    """Aggregated statistical report for a benchmark tier."""

    tier_id: str
    env_name: str
    description: str
    n_episodes: int
    n_successes: int
    success_rate: float
    ci_95_low: float
    ci_95_high: float
    mean_steps: float
    std_steps: float
    median_steps: float
    mean_reward: float
    std_reward: float
    mean_duration_ms: float
    total_llm_tokens: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)


def run_single_episode(
    env_name: str,
    tier_id: str,
    seed: int,
    room_size: int = 10,
    max_steps_override: int | None = None,
) -> EpisodeResult:
    """Execute a single closed-loop BabyAI episode using pure HCIR."""
    t_start = time.perf_counter()
    env = make_gym_babyai_level(env_name)
    obs, info = env.reset(seed=seed)

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=room_size)
    parser = BabyAIMissionParser()

    mission = obs.get("mission", "")
    try:
        goal = parser.parse(mission)
    except Exception as e:
        env.close()
        t_end = time.perf_counter()
        return EpisodeResult(
            tier_id=tier_id,
            env_name=env_name,
            seed=seed,
            mission=mission,
            success=False,
            steps=0,
            reward=0.0,
            duration_ms=(t_end - t_start) * 1000.0,
            failure_reason=f"Mission parse error: {e}",
        )

    max_steps = (
        max_steps_override
        if max_steps_override is not None
        else getattr(env.unwrapped, "max_steps", 256)
    )

    steps = 0
    success = False
    reward = 0.0
    failure_reason = ""

    # Phase 0: 360-degree initial visual orientation scan
    for _ in range(4):
        adapter.ingest_observation(
            obs,
            known_agent_pos=getattr(env.unwrapped, "agent_pos", None),
            known_carrying=getattr(env.unwrapped, "carrying", None),
        )
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))
        steps += 1
        if term and r > 0.0:
            success = True
            reward = float(r)
            break

    # Phase 1: Closed-loop HCIR execution
    if not success:
        while steps < max_steps:
            adapter.ingest_observation(
                obs,
                known_agent_pos=getattr(env.unwrapped, "agent_pos", None),
                known_carrying=getattr(env.unwrapped, "carrying", None),
            )
            act = planner.plan_next_action(adapter.graph, goal)
            obs, r, term, trunc, info = env.step(int(act))
            steps += 1
            if term and r > 0.0:
                success = True
                reward = float(r)
                break
            if term:
                failure_reason = "Premature termination (term=True, r=0)"
                break
            if trunc:
                failure_reason = "Environment truncated"
                break

    if not success and not failure_reason:
        failure_reason = f"Step budget exhausted ({steps}/{max_steps})"

    env.close()
    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000.0

    return EpisodeResult(
        tier_id=tier_id,
        env_name=env_name,
        seed=seed,
        mission=mission,
        success=success,
        steps=steps,
        reward=reward,
        duration_ms=duration_ms,
        llm_tokens=0,
        failure_reason=failure_reason if not success else "",
    )


def evaluate_tier(
    cfg: TierConfig,
    n_episodes: int,
    seed_start: int = 1,
) -> TierBenchmarkSummary:
    """Run N independent episodes for a given tier and aggregate statistics."""
    logger.info("Evaluating %s (%s) over %d episodes...", cfg.tier_id, cfg.env_name, n_episodes)
    results: list[EpisodeResult] = []
    for i in range(n_episodes):
        seed = seed_start + i
        res = run_single_episode(
            env_name=cfg.env_name,
            tier_id=cfg.tier_id,
            seed=seed,
            room_size=cfg.room_size,
            max_steps_override=cfg.max_steps_override,
        )
        results.append(res)

    successes = sum(1 for r in results if r.success)
    prop_stat = ExperimentStatistics.summarize_proportion(
        f"{cfg.tier_id}_success_rate", successes, n_episodes
    )

    # Compute step stats on successful episodes
    success_steps = [float(r.steps) for r in results if r.success]
    step_stat = ExperimentStatistics.summarize(f"{cfg.tier_id}_steps", success_steps)

    rewards = [float(r.reward) for r in results]
    reward_stat = ExperimentStatistics.summarize(f"{cfg.tier_id}_reward", rewards)

    durations = [float(r.duration_ms) for r in results]
    duration_stat = ExperimentStatistics.summarize(f"{cfg.tier_id}_duration", durations)

    failures: dict[str, int] = {}
    for r in results:
        if not r.success:
            failures[r.failure_reason] = failures.get(r.failure_reason, 0) + 1

    summary = TierBenchmarkSummary(
        tier_id=cfg.tier_id,
        env_name=cfg.env_name,
        description=cfg.description,
        n_episodes=n_episodes,
        n_successes=successes,
        success_rate=prop_stat.rate,
        ci_95_low=prop_stat.ci_95_low,
        ci_95_high=prop_stat.ci_95_high,
        mean_steps=step_stat.mean,
        std_steps=step_stat.std,
        median_steps=step_stat.median,
        mean_reward=reward_stat.mean,
        std_reward=reward_stat.std,
        mean_duration_ms=duration_stat.mean,
        total_llm_tokens=0,
        failure_reasons=failures,
    )

    logger.info(
        "Finished %s: Success Rate = %.1f%% [95%% CI: %.1f%% - %.1f%%], Mean Steps = %.1f ± %.1f, Avg Latency = %.1f ms",
        cfg.tier_id,
        summary.success_rate * 100.0,
        summary.ci_95_low * 100.0,
        summary.ci_95_high * 100.0,
        summary.mean_steps,
        summary.std_steps,
        summary.mean_duration_ms,
    )
    return summary


def format_markdown_table(summaries: list[TierBenchmarkSummary]) -> str:
    """Format benchmark summaries into a GitHub Flavored Markdown table."""
    lines = [
        "| Benchmark Tier | Environment | Episodes | Success Rate (95% Wilson CI) | Mean Steps (Solved) | Mean Reward | Latency / Episode | LLM Tokens |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        ci_str = f"**{s.success_rate * 100.0:.1f}%** `[{s.ci_95_low * 100.0:.1f}%, {s.ci_95_high * 100.0:.1f}%]`"
        steps_str = (
            f"{s.mean_steps:.1f} ± {s.std_steps:.1f} (med: {s.median_steps:.0f})"
            if s.n_successes > 0
            else "N/A"
        )
        reward_str = f"{s.mean_reward:.3f} ± {s.std_reward:.3f}"
        latency_str = f"{s.mean_duration_ms:.1f} ms"
        lines.append(
            f"| **{s.tier_id}** | `{s.env_name}` | {s.n_episodes} | {ci_str} | {steps_str} | {reward_str} | {latency_str} | **0** |"
        )
    return "\n".join(lines)


def run_benchmark_suite(
    n_episodes_per_tier: int = 100,
    seed_start: int = 1,
    tier_filters: list[str] | None = None,
    output_json_path: str | None = None,
) -> list[TierBenchmarkSummary]:
    """Execute the full BabyAI benchmark battery across all requested tiers."""
    configs = TIER_CONFIGS
    if tier_filters:
        configs = [
            c
            for c in configs
            if any(
                f.lower() in c.tier_id.lower() or f.lower() in c.env_name.lower()
                for f in tier_filters
            )
        ]

    logger.info(
        "Starting BabyAI Benchmark Battery: %d tiers, %d episodes each (Total: %d episodes)",
        len(configs),
        n_episodes_per_tier,
        len(configs) * n_episodes_per_tier,
    )

    t_suite_start = time.perf_counter()
    summaries: list[TierBenchmarkSummary] = []
    for cfg in configs:
        summary = evaluate_tier(cfg, n_episodes=n_episodes_per_tier, seed_start=seed_start)
        summaries.append(summary)

    t_suite_end = time.perf_counter()
    suite_duration = t_suite_end - t_suite_start

    total_episodes = sum(s.n_episodes for s in summaries)
    total_successes = sum(s.n_successes for s in summaries)
    overall_prop = ExperimentStatistics.summarize_proportion(
        "overall_success", total_successes, total_episodes
    )

    logger.info(
        "All tiers complete in %.2f seconds! Overall: %d/%d (%.1f%% [95%% CI: %.1f%% - %.1f%%])",
        suite_duration,
        total_successes,
        total_episodes,
        overall_prop.rate * 100.0,
        overall_prop.ci_95_low * 100.0,
        overall_prop.ci_95_high * 100.0,
    )

    if output_json_path:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([asdict(s) for s in summaries], f, indent=2)
        logger.info("Saved raw benchmark results to %s", out_path)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rigorous BabyAI benchmark evaluation.")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes per tier (default: 100)"
    )
    parser.add_argument(
        "--seed-start", type=int, default=1, help="Starting seed integer (default: 1)"
    )
    parser.add_argument(
        "--tiers", type=str, default="", help="Comma-separated tier filters (e.g. 'Tier 1,Boss')"
    )
    parser.add_argument("--out", type=str, default=None, help="Output JSON results path")
    parser.add_argument("--quick", action="store_true", help="Quick mode: runs 5 episodes per tier")
    args = parser.parse_args()

    n_eps = 5 if args.quick else args.episodes
    tier_filters = [t.strip() for t in args.tiers.split(",") if t.strip()] if args.tiers else None

    summaries = run_benchmark_suite(
        n_episodes_per_tier=n_eps,
        seed_start=args.seed_start,
        tier_filters=tier_filters,
        output_json_path=args.out,
    )

    print("\n" + "=" * 80)
    print("BABYAI BENCHMARK RESULTS (FORMAL STATISTICAL SUMMARY)")
    print("=" * 80 + "\n")
    print(format_markdown_table(summaries))
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
