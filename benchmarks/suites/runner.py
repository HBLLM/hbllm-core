"""Test suite runner for dev and held-out benchmark splits.

Executes true closed-loop evaluation where agent actions are fed back to environments
and success is determined strictly by the environment simulation.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from hbllm.benchmarks.protocol import EvaluationProtocol, TaskResult
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def _load_env_class(class_path: str) -> type:
    """Dynamically import environment class from module string."""
    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def run_closed_loop_episode(
    env: Any,
    seed: int,
    max_budget: int,
    baseline: str = "system",
    perception_mode: str = "handcrafted",
) -> tuple[bool, int, list[float], list[str]]:
    """Execute a single episode in closed loop.

    Returns:
        (success, steps, latencies_ms, errors)
    """
    latencies: list[float] = []
    errors: list[str] = []
    obs = env.reset(seed=seed)

    system = None
    if baseline in ("system", "no_grounding"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = CognitiveSystemConfig(
                world_id=f"bench_{seed}",
                memory_dir=tmp_dir,
                enable_grounding=(baseline == "system"),
                perception_mode=perception_mode,
            )
            system = build_cognitive_system(config)
            return _run_loop(env, obs, system.engine, baseline, max_budget, latencies, errors)

    return _run_loop(env, obs, None, baseline, max_budget, latencies, errors)


def _run_loop(
    env: Any,
    obs: Any,
    engine: Any | None,
    baseline: str,
    max_budget: int,
    latencies: list[float],
    errors: list[str],
) -> tuple[bool, int, list[float], list[str]]:
    success = False
    step_count = 0

    for step in range(max_budget):
        step_count = step + 1
        t0 = time.perf_counter()

        action = None
        try:
            if baseline == "stub_agent":
                # Always emits constant action (e.g. ACTION_UP = 1)
                action = 1
            elif baseline == "random":
                # Select random action from available actions
                avail = getattr(obs, "available_actions", [1, 2, 3, 4])
                action = random.choice(list(avail))
            elif baseline == "oracle":
                # Compute optimal action from ground-truth state
                action = env.get_oracle_action()
            else:
                # AutonomousEpistemicEngine step
                assert engine is not None
                action = engine.step(obs)
        except Exception as e:
            errors.append(f"Action selection error at step {step_count}: {e}")
            break

        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Environment closed-loop transition
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            errors.append(f"Environment step error at step {step_count}: {e}")
            break

        if done:
            success = bool(info.get("success", False))
            break

    return success, step_count, latencies, errors


def run_suite(
    split: str = "dev",
    reports_dir: str = "reports",
    perception_mode: str = "handcrafted",
    baseline: str = "system",
    manifest_path: str | None = None,
) -> int:
    """Executes the closed-loop benchmark evaluation for the given split."""
    manifest_file = (
        Path(manifest_path) if manifest_path else Path(__file__).parent / f"{split}.yaml"
    )

    if not manifest_file.exists():
        print(f"Error: Manifest file {manifest_file} not found.")
        return 1

    with open(manifest_file, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    protocol = EvaluationProtocol(
        reports_dir=reports_dir,
        perception_mode=perception_mode,
        config={
            "model": "deterministic_hcir",
            "version": "1.0.0",
            "perception_mode": perception_mode,
            "baseline": baseline,
            "split": split,
        },
    )

    if split == "heldout" and baseline == "system":
        protocol.verify_heldout_gate()

    print(
        f"=== Running HBLLM Generality Suite [{split.upper()}] (baseline={baseline}, perception={perception_mode}) ==="
    )
    print(f"Git SHA: {protocol.git_sha} | Config Hash: {protocol.config_hash}")

    tasks: list[TaskResult] = []

    for task_def in manifest.get("tasks", []):
        task_id = task_def["id"]
        family = task_def["family"]
        env_cls = _load_env_class(task_def["env_class"])
        seeds = task_def.get("seeds", [0])
        max_budget = task_def.get("max_budget", 100)

        print(
            f"-> Evaluating task [{task_id}] ({family}, {len(seeds)} seeds, budget={max_budget})..."
        )

        def make_task_runner(t_seeds: list[int], t_budget: int):
            def run_task() -> tuple[bool, int, list[float], list[str]]:
                all_successes: list[bool] = []
                all_steps: list[int] = []
                all_latencies: list[float] = []
                all_errors: list[str] = []

                for s in t_seeds:
                    env_instance = env_cls(max_steps=t_budget)
                    succ, st, lats, errs = run_closed_loop_episode(
                        env=env_instance,
                        seed=s,
                        max_budget=t_budget,
                        baseline=baseline,
                        perception_mode=perception_mode,
                    )
                    all_successes.append(succ)
                    all_steps.append(st)
                    all_latencies.extend(lats)
                    all_errors.extend(errs)

                solve_rate = sum(1 for s in all_successes if s) / len(all_successes)
                overall_success = solve_rate >= 0.5
                mean_steps = int(sum(all_steps) / len(all_steps))
                return overall_success, mean_steps, all_latencies, all_errors

            return run_task

        t_res = protocol.evaluate_task(
            task_id=f"{task_id}_{baseline}_{perception_mode}",
            family=family,
            split=split,
            run_fn=make_task_runner(seeds, max_budget),
            max_budget=max_budget,
        )
        tasks.append(t_res)

    report = protocol.compute_report(split=split, tasks=tasks)

    print("\n=== Benchmark Results Summary ===")
    print(f"Split: {report.split.upper()} | Baseline: {baseline}")
    for fam, score in report.family_scores.items():
        print(f"  Family [{fam}]: {score:.4f}")
    print(f"Generality Score: {report.generality_score:.4f}")
    print(
        f"Mean Latency: p50={report.mean_p50_latency_ms:.2f}ms | p95={report.mean_p95_latency_ms:.2f}ms"
    )
    print("Report written successfully.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HBLLM Closed-Loop Benchmark Suite Runner")
    parser.add_argument("split", choices=["dev", "heldout"], help="Split to execute")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for reports")
    parser.add_argument(
        "--perception-mode",
        default="handcrafted",
        choices=["handcrafted", "snn", "learned"],
        help="Perception lifting mode",
    )
    parser.add_argument(
        "--baseline",
        default="system",
        choices=["system", "stub_agent", "random", "oracle", "no_grounding"],
        help="Evaluation policy baseline",
    )
    parser.add_argument("--manifest", default=None, help="Custom manifest YAML path")
    args = parser.parse_args()

    return run_suite(
        split=args.split,
        reports_dir=args.reports_dir,
        perception_mode=args.perception_mode,
        baseline=args.baseline,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    sys.exit(main())
