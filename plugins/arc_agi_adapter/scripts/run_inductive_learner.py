"""CLI Runner for the Inductive HCIR Learner on ARC-AGI-3 environments."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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
        "--max-retries",
        "--retries",
        dest="max_retries",
        type=int,
        default=2,
        help="Number of retries per level if attempt fails (default: 2, total attempts = 1 + retries).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Maximum action steps per level attempt (default: 120).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ARC_API_KEY", ""),
        help="ARC-AGI API key (defaults to ARC_API_KEY env var).",
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
    parser.add_argument(
        "--knowledge-dir",
        type=str,
        default="data/cognitive_memory/arc_agi_3",
        help="Directory to persist and load core KnowledgeGraph files (default: data/cognitive_memory/arc_agi_3).",
    )
    parser.add_argument(
        "--disable-archetypes",
        action="store_true",
        help="Bypass all 8 specialized archetype solvers and force all games through core CognitiveBlackbox / HCIR reasoning.",
    )
    args = parser.parse_args()

    try:
        from arc_agi import Arcade

        arcade_client = Arcade(arc_api_key=args.api_key) if args.api_key else Arcade()
    except Exception as e:
        logger.error(f"Failed to initialize ARC Arcade client: {e}")
        return

    runner = InductiveARC3BenchmarkRunner(
        max_steps_per_level=args.max_steps,
        max_retries_per_level=args.max_retries,
        knowledge_dir=args.knowledge_dir,
        disable_archetypes=args.disable_archetypes,
    )
    results = []
    start = time.time()
    game_list = args.games
    if game_list == ["all"] or "all" in game_list:
        envs = (
            arcade_client.get_environments() if hasattr(arcade_client, "get_environments") else []
        )
        game_list = sorted([e.game_id.split("-")[0] for e in envs])

    for idx, gid in enumerate(game_list, 1):
        try:
            max_lvl = (
                1 if (gid in ["su15", "lf52", "ka59"] and args.max_levels == 2) else args.max_levels
            )
            res = runner.run_environment(
                arcade_client,
                gid,
                max_levels=max_lvl,
                max_retries_per_level=args.max_retries,
            )
            results.append(res)
            logger.info(
                f"[{idx}/{len(game_list)}] Completed {gid}: {res.levels_completed}/{res.total_levels} levels passed ({res.total_actions} actions)"
            )
        except Exception as e:
            logger.error(f"Error evaluating game {gid}: {e}")

    duration = time.time() - start
    logger.info(f"Inductive evaluation completed in {duration:.2f}s across {len(results)} games.")
    total_completed = sum(r.levels_completed for r in results)
    total_levels = sum(r.total_levels for r in results)
    logger.info(
        f"OVERALL SCORE: {total_completed}/{total_levels} ({total_completed / total_levels * 100:.1f}%)"
    )
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
        f"**Total Score**: **{total_completed}/{total_levels} ({total_completed / total_levels * 100:.1f}%)**",
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
        "total_levels_completed": total_completed,
        "total_levels": total_levels,
        "duration_seconds": duration,
        "results": [
            {
                "game_id": r.game_id,
                "levels_completed": r.levels_completed,
                "total_levels": r.total_levels,
                "level_results": [lvl.completed for lvl in r.level_results],
                "level_attempts": [getattr(lvl, "attempts", 1) for lvl in r.level_results],
                "duration": round(sum(lvl.time_seconds for lvl in r.level_results), 2),
            }
            for r in results
        ],
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
        ],
    }
    Path(args.output_json).write_text(json.dumps(scorecard_data, indent=2), encoding="utf-8")
    print("\n" + report_md + "\n")


if __name__ == "__main__":
    main()
