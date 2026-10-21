"""Integration test for the Master Scientific Reproducibility Suite.

Verifies:
1. Exact Wilson Score 95% Confidence Interval calculations.
2. Artifact export routines: LaTeX table, JSON matrix, and Markdown report.
3. Execution of targeted benchmark suites via MasterReproducibilityRunner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reproduce_all_benchmarks import (
    MasterReproducibilityRunner,
    wilson_score_interval,
)


def test_wilson_score_confidence_intervals() -> None:
    """Verify Wilson score confidence interval boundary calculations."""
    # 0 successes out of 10
    low, high = wilson_score_interval(0, 10)
    assert low == 0.0
    assert pytest.approx(high, abs=0.01) == 0.278

    # 10 successes out of 10
    low, high = wilson_score_interval(10, 10)
    assert pytest.approx(low, 0.01) == 0.722
    assert high == 1.0

    # 5 successes out of 10 (50%)
    low, high = wilson_score_interval(5, 10)
    assert pytest.approx(low, 0.01) == 0.237
    assert pytest.approx(high, 0.01) == 0.763

    # Total 0
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_reproducibility_export_pipeline(tmp_path: Path) -> None:
    """Verify LaTeX, JSON, and Markdown generation from mock benchmark results."""
    runner = MasterReproducibilityRunner(quick=True, output_dir=tmp_path)

    # Populate mock results
    runner.results = {
        "ai2thor": {
            "domain": "AI2-THOR",
            "simulator": "Native Unity 3D",
            "literature_baseline": "35-45% (RL)",
            "llm_baseline": "0.0%",
            "episodes": 12,
            "success_rate": 1.0,
            "ci_95": [0.758, 1.0],
            "metric_name": "Success Rate",
        },
        "crafter": {
            "domain": "Crafter (Hafner)",
            "simulator": "Native crafter",
            "literature_baseline": "10.0% (DreamerV2)",
            "llm_baseline": "6.1%",
            "episodes": 5,
            "score": 53.9,
            "ci_95": [0.741, 1.0],
            "metric_name": "Logarithmic Crafter Score",
        },
    }

    # 1. Export LaTeX
    tex_path = runner.export_latex()
    assert tex_path.exists()
    tex_content = tex_path.read_text()
    assert r"\begin{table*}" in tex_content
    assert "AI2-THOR" in tex_content
    assert "Crafter (Hafner)" in tex_content
    assert r"\textbf{53.9\%} (Score)" in tex_content

    # 2. Export JSON
    json_path = runner.export_json()
    assert json_path.exists()
    json_data = json.loads(json_path.read_text())
    assert json_data["token_cost_total"] == 0
    assert json_data["dollar_cost_total"] == 0.0
    assert "ai2thor" in json_data["domains"]
    assert json_data["domains"]["crafter"]["score"] == 53.9

    # 3. Export Markdown
    md_path = runner.export_markdown()
    assert md_path.exists()
    md_content = md_path.read_text()
    assert "# Master Scientific Benchmark & Reproducibility Report" in md_content
    assert "| **AI2-THOR** |" in md_content
    assert "**0 tokens** ($0.00)" in md_content


def test_quick_suite_execution_language(tmp_path: Path) -> None:
    """Execute quick language suite via MasterReproducibilityRunner and verify structure."""
    runner = MasterReproducibilityRunner(quick=True, output_dir=tmp_path)
    results = runner.execute_all(suite="language")

    assert "alfworld" in results
    assert "babyai" in results
    assert results["alfworld"]["token_cost"] == 0
    assert results["babyai"]["token_cost"] == 0
    assert results["alfworld"]["success_rate"] == 1.0
    assert results["babyai"]["success_rate"] == 1.0
