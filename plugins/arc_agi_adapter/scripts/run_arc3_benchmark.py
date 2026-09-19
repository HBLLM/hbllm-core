"""Run real official ARC-AGI-3 interactive benchmark evaluation against HBLLM."""

import argparse
import json
import logging
import time
from pathlib import Path

from plugins.arc_agi_adapter.arc_agi_3_runner import ARC3BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("arc3_runner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute ARC-AGI-3 benchmark against HBLLM.")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["ls20", "wa30", "cd82"],
        help="List of ARC-AGI-3 game IDs to evaluate (default: ls20 wa30 cd82).",
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
        default="arc_agi_3_report.md",
        help="Path to save the Markdown report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="arc_agi_3_scorecard.json",
        help="Path to save the raw JSON scorecard.",
    )
    args = parser.parse_args()

    logger.info(f"Initializing HBLLM ARC-AGI-3 Benchmark Runner on games: {args.games}...")
    runner = ARC3BenchmarkRunner(max_steps_per_level=120)

    start = time.time()
    report = runner.run_benchmark(game_ids=args.games, max_levels_per_game=args.max_levels)
    duration = time.time() - start

    md = report.format_markdown()
    p_report = Path(args.output_report)
    p_report.parent.mkdir(parents=True, exist_ok=True)
    p_report.write_text(md, encoding="utf-8")

    p_json = Path(args.output_json)
    p_json.parent.mkdir(parents=True, exist_ok=True)
    p_json.write_text(json.dumps(report.raw_scorecard, indent=2), encoding="utf-8")

    logger.info(f"Benchmark completed in {duration:.2f}s!")
    logger.info(f"Completion Rate: {report.overall_completion_rate * 100:.1f}%")
    logger.info(f"Efficiency: {report.mean_action_efficiency * 100:.1f}%")
    logger.info(f"Saved Markdown report to: {args.output_report}")
    logger.info(f"Saved JSON scorecard to: {args.output_json}")

    print("\n" + md + "\n")


if __name__ == "__main__":
    main()
