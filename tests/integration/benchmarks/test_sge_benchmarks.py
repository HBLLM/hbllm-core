"""Benchmark Integration Tests — runs each benchmark scenario as a pytest.

Validates that all 4 domain scenarios execute without errors,
meet minimum quality thresholds, and produce a valid scorecard.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


class TestSGEBenchmarkScenarios:
    """Runs each benchmark scenario individually and validates results."""

    def test_grid_spatial_scenario(self) -> None:
        from benchmarks.scenarios.grid_spatial import GridSpatialScenario

        scenario = GridSpatialScenario()
        metrics = scenario.run()

        assert metrics.scenario_name == "grid_spatial"
        assert metrics.steps == 10
        assert metrics.passed, f"Grid spatial failed: {metrics.errors}"
        assert metrics.entities_tracked >= 2
        assert metrics.total_time_ms > 0
        assert metrics.mean_step_time_ms > 0

    def test_robotics_manipulation_scenario(self) -> None:
        from benchmarks.scenarios.robotics_manipulation import RoboticsManipulationScenario

        scenario = RoboticsManipulationScenario()
        metrics = scenario.run()

        assert metrics.scenario_name == "robotics_manipulation"
        assert metrics.steps >= 3
        assert metrics.passed, f"Robotics manipulation failed: {metrics.errors}"
        assert metrics.total_time_ms > 0

    def test_language_grounding_scenario(self) -> None:
        from benchmarks.scenarios.language_grounding import LanguageGroundingScenario

        scenario = LanguageGroundingScenario()
        metrics = scenario.run()

        assert metrics.scenario_name == "language_grounding"
        assert metrics.steps >= 1
        assert metrics.passed, f"Language grounding failed: {metrics.errors}"
        assert metrics.total_time_ms > 0

    def test_adversarial_surprise_scenario(self) -> None:
        from benchmarks.scenarios.adversarial_surprise import AdversarialSurpriseScenario

        scenario = AdversarialSurpriseScenario()
        metrics = scenario.run()

        assert metrics.scenario_name == "adversarial_surprise"
        assert metrics.steps >= 1
        assert metrics.passed, f"Adversarial surprise failed: {metrics.errors}"
        assert metrics.surprise_alerts >= 1
        assert metrics.total_time_ms > 0


class TestSGEBenchmarkRunner:
    """Tests the full benchmark runner and scorecard generation."""

    def test_run_all_scenarios(self) -> None:
        from benchmarks.sge_integration_benchmark import SGEBenchmarkRunner

        runner = SGEBenchmarkRunner()
        scorecard = runner.run_all()

        assert scorecard.total_scenarios == 4
        assert scorecard.passed_scenarios == 4, (
            f"Expected 4 passed, got {scorecard.passed_scenarios}. "
            f"Failures: {[s.errors for s in scorecard.scenarios if not s.passed]}"
        )
        assert scorecard.failed_scenarios == 0
        assert scorecard.total_time_ms > 0

    def test_scorecard_to_dict(self) -> None:
        from benchmarks.sge_integration_benchmark import SGEBenchmarkRunner

        runner = SGEBenchmarkRunner()
        scorecard = runner.run_all()
        data = scorecard.to_dict()

        assert "timestamp" in data
        assert "scenarios" in data
        assert len(data["scenarios"]) == 4
        for scenario_data in data["scenarios"]:
            assert "name" in scenario_data
            assert "passed" in scenario_data
            assert "steps" in scenario_data

    def test_scorecard_json_output(self) -> None:
        from benchmarks.scorecard import generate_scorecard
        from benchmarks.sge_integration_benchmark import SGEBenchmarkRunner

        runner = SGEBenchmarkRunner()
        scorecard = runner.run_all()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_scorecard.json"
            data = generate_scorecard(scorecard, output_path=path)

            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert data == loaded
            assert loaded["total_scenarios"] == 4
            assert "summary" in loaded
            assert "pass_rate" in loaded["summary"]
            assert loaded["summary"]["all_passed"] is True
