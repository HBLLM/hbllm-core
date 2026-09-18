"""CLI Runner for the Inductive HCIR Learner on ARC-AGI-3 environments."""

import argparse
import logging
import time

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
            res = runner.run_environment(arcade_client, gid, max_levels=args.max_levels)
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


if __name__ == "__main__":
    main()
