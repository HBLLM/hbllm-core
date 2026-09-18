"""CLI Runner for the Inductive HCIR Learner on ARC-AGI-3 environments."""

import argparse
import json
import logging
import time
from pathlib import Path

from plugins.arc_agi_adapter.inductive_learner import (
    InductiveARC3BenchmarkRunner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("inductive_runner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Inductive HCIR Learner on ARC-AGI-3.")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["wa30", "ls20"],
        help="List of ARC-AGI-3 game IDs to evaluate.",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=2,
        help="Maximum levels per game to evaluate (default: 2).",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="arc_agi_3_inductive_report.md",
        help="Path to save Markdown report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="arc_agi_3_inductive_scorecard.json",
        help="Path to save JSON scorecard.",
    )
    args = parser.parse_args()

    try:
        from arc_agi import Arcade

        arcade_client = Arcade()
    except Exception as e:
        logger.error(f"Failed to initialize ARC Arcade client: {e}")
        return

    runner = InductiveARC3BenchmarkRunner(max_steps_per_level=120)
    results = []
    start = time.time()
    for gid in args.games:
        try:
            max_lvl = 1 if (gid == "su15" and args.max_levels == 2) else args.max_levels
            res = runner.run_environment(arcade_client, gid, max_levels=max_lvl)
            results.append(res)
        except Exception as e:
            logger.error(f"Error evaluating game {gid}: {e}")

    duration = time.time() - start
    logger.info(f"Inductive evaluation completed in {duration:.2f}s across {len(results)} games.")
    for r in results:
        logger.info(
            f"Game: {r.game_id} | Completed: {r.levels_completed}/{r.total_levels} | Actions: {r.total_actions}"
        )
        for lvl in r.level_results:
            logger.info(
                f"  Level {lvl.level_index + 1}: Passed={lvl.completed}, Actions={lvl.actions_taken}, "
                f"Baseline={lvl.baseline_actions}, Eff={lvl.efficiency_ratio * 100:.1f}%, Probes={lvl.epistemic_probes}"
            )

    # Format Markdown report
    report_lines = [
        "# Inductive HCIR Learner Benchmark Report",
        f"**Duration**: {duration:.2f}s across {len(results)} game(s)",
        "",
        "| Environment | Levels Completed | Total Actions | Total Baseline | Mean Efficiency |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        report_lines.append(
            f"| `{r.game_id}` | {r.levels_completed}/{r.total_levels} | {r.total_actions} | {r.total_baseline} | **{r.mean_efficiency * 100:.1f}%** |"
        )
    report_lines.append("")
    report_lines.append("## Level-by-Level Breakdown")
    for r in results:
        report_lines.append(f"### Environment: `{r.game_id}`")
        report_lines.append(
            "| Level | Completed | Actions | Baseline | Efficiency | Epistemic Probes |"
        )
        report_lines.append("|---|---|---|---|---|---|")
        for lvl in r.level_results:
            status = "**PASSED**" if lvl.completed else "ACTIVE"
            report_lines.append(
                f"| Level {lvl.level_index + 1} | {status} | {lvl.actions_taken} | {lvl.baseline_actions} | {lvl.efficiency_ratio * 100:.1f}% | {lvl.epistemic_probes} |"
            )
        report_lines.append("")

    report_md = "\n".join(report_lines)
    Path(args.output_report).write_text(report_md, encoding="utf-8")

    scorecard_data = {
        "games": [
            {
                "game_id": r.game_id,
                "levels_completed": r.levels_completed,
                "total_levels": r.total_levels,
                "total_actions": r.total_actions,
                "total_baseline": r.total_baseline,
                "mean_efficiency": r.mean_efficiency,
                "level_results": [
                    {
                        "level_index": lvl.level_index,
                        "completed": lvl.completed,
                        "actions_taken": lvl.actions_taken,
                        "baseline_actions": lvl.baseline_actions,
                        "efficiency_ratio": lvl.efficiency_ratio,
                        "epistemic_probes": lvl.epistemic_probes,
                    }
                    for lvl in r.level_results
                ],
            }
            for r in results
        ]
    }
    Path(args.output_json).write_text(json.dumps(scorecard_data, indent=2), encoding="utf-8")
    print("\n" + report_md + "\n")


if __name__ == "__main__":
    main()
