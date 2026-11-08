"""WP0: Test suite runner for dev and held-out benchmark splits."""

from __future__ import annotations

import argparse
import sys

from benchmarks.suites.continuous.dev import run_continuous_dev_task
from benchmarks.suites.continuous.heldout import run_continuous_heldout_task
from benchmarks.suites.grid.dev import run_grid_dev_task
from benchmarks.suites.grid.heldout import run_grid_heldout_task
from benchmarks.suites.symbolic.dev import run_symbolic_dev_task
from benchmarks.suites.symbolic.heldout import run_symbolic_heldout_task
from hbllm.benchmarks.protocol import EvaluationProtocol, TaskResult


def run_suite(
    split: str = "dev",
    reports_dir: str = "reports",
    perception_mode: str = "handcrafted",
) -> int:
    """Executes the benchmark evaluation for the given split (dev or heldout)."""
    protocol = EvaluationProtocol(reports_dir=reports_dir, perception_mode=perception_mode)

    if split == "heldout":
        # Enforce anti-tamper check before running any tasks
        protocol.verify_heldout_gate()

    tasks: list[TaskResult] = []

    print(
        f"=== Running HBLLM Generality Suite [{split.upper()}] (perception={perception_mode}) ==="
    )
    print(f"Git SHA: {protocol.git_sha} | Config Hash: {protocol.config_hash}")

    if split == "dev":
        # Grid dev
        print(f"-> Running grid dev task (perception={perception_mode})...")
        t_grid = protocol.evaluate_task(
            task_id=f"grid_dev_nav_{perception_mode}",
            family="grid",
            split=split,
            run_fn=lambda: run_grid_dev_task(
                task_name=f"grid_dev_nav_{perception_mode}", perception_mode=perception_mode
            ),
            max_budget=50,
        )
        tasks.append(t_grid)

        # Continuous dev
        print("-> Running continuous dev task...")
        t_cont = protocol.evaluate_task(
            task_id="robotics_pick_place_dev",
            family="continuous",
            split=split,
            run_fn=run_continuous_dev_task,
            max_budget=50,
        )
        tasks.append(t_cont)

        # Symbolic dev
        print("-> Running symbolic dev task...")
        t_sym = protocol.evaluate_task(
            task_id="language_grounding_dev",
            family="symbolic",
            split=split,
            run_fn=run_symbolic_dev_task,
            max_budget=50,
        )
        tasks.append(t_sym)

    elif split == "heldout":
        # Grid heldout
        print(f"-> Running grid heldout task (perception={perception_mode})...")
        t_grid = protocol.evaluate_task(
            task_id=f"heldout_dual_barrier_nav_{perception_mode}",
            family="grid",
            split=split,
            run_fn=lambda: run_grid_heldout_task(
                task_name=f"heldout_dual_barrier_nav_{perception_mode}",
                perception_mode=perception_mode,
            ),
            max_budget=100,
        )
        tasks.append(t_grid)

        # Continuous heldout
        print("-> Running continuous heldout task...")
        t_cont = protocol.evaluate_task(
            task_id="continuous_obstacle_avoidance_heldout",
            family="continuous",
            split=split,
            run_fn=run_continuous_heldout_task,
            max_budget=100,
        )
        tasks.append(t_cont)

        # Symbolic heldout
        print("-> Running symbolic heldout task...")
        t_sym = protocol.evaluate_task(
            task_id="symbolic_rule_induction_heldout",
            family="symbolic",
            split=split,
            run_fn=run_symbolic_heldout_task,
            max_budget=100,
        )
        tasks.append(t_sym)

    else:
        print(f"Error: Unknown split {split}. Must be 'dev' or 'heldout'.")
        return 1

    report = protocol.compute_report(split=split, tasks=tasks)

    print("\n=== Benchmark Results Summary ===")
    print(f"Split: {report.split.upper()}")
    for fam, score in report.family_scores.items():
        print(f"  Family [{fam}]: {score:.4f}")
    print(f"Generality Score: {report.generality_score:.4f}")
    print(
        f"Mean Latency: p50={report.mean_p50_latency_ms:.2f}ms | p95={report.mean_p95_latency_ms:.2f}ms"
    )
    print("Report written successfully.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HBLLM Benchmark Suite Runner")
    parser.add_argument("split", choices=["dev", "heldout"], help="Split to execute")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for reports")
    args = parser.parse_args()

    return run_suite(split=args.split, reports_dir=args.reports_dir)


if __name__ == "__main__":
    sys.exit(main())
