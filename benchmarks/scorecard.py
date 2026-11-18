from __future__ import annotations

"""Scorecard Generator — generates standardised JSON scorecard from benchmark results.

Usage::

    from benchmarks.scorecard import generate_scorecard
    from benchmarks.sge_integration_benchmark import SGEBenchmarkRunner

    runner = SGEBenchmarkRunner()
    scorecard = runner.run_all()
    generate_scorecard(scorecard, output_path="reports/sge_scorecard.json")
"""

import json
from pathlib import Path
from typing import Any

from benchmarks.sge_integration_benchmark import BenchmarkScorecard


def generate_scorecard(
    scorecard: BenchmarkScorecard,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generates a standardised JSON scorecard.

    If ``output_path`` is provided, writes the JSON to disk.
    Always returns the scorecard dict.
    """
    data = scorecard.to_dict()

    # Add summary statistics
    data["summary"] = {
        "pass_rate": (round(scorecard.passed_scenarios / max(1, scorecard.total_scenarios), 3)),
        "all_passed": scorecard.passed_scenarios == scorecard.total_scenarios,
        "mean_step_time_ms": (
            round(
                sum(s.mean_step_time_ms for s in scorecard.scenarios)
                / max(1, len(scorecard.scenarios)),
                2,
            )
        ),
        "total_entities_tracked": sum(s.entities_tracked for s in scorecard.scenarios),
        "total_concepts_formed": sum(s.concepts_formed for s in scorecard.scenarios),
        "total_surprise_alerts": sum(s.surprise_alerts for s in scorecard.scenarios),
        "total_actions_authorised": sum(s.actions_authorised for s in scorecard.scenarios),
        "total_actions_rejected": sum(s.actions_rejected for s in scorecard.scenarios),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return data
