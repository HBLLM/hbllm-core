from __future__ import annotations

"""SGE Integration Benchmark Harness.

Standardised benchmark that measures the performance of the integrated
SGE × HCIR cognitive loop across multiple domains.  All operations are
deterministic and LLM-free.

Usage::

    from benchmarks.sge_integration_benchmark import SGEBenchmarkRunner
    runner = SGEBenchmarkRunner()
    scorecard = runner.run_all()
"""

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge


@dataclass
class BenchmarkMetrics:
    """Metrics for a single benchmark scenario."""

    scenario_name: str
    steps: int = 0
    entities_tracked: int = 0
    concepts_formed: int = 0
    concept_splits: int = 0
    concept_merges: int = 0
    features_extracted: int = 0
    claims_committed: int = 0
    surprise_alerts: int = 0
    actions_authorised: int = 0
    actions_rejected: int = 0
    outcomes_verified: int = 0
    total_time_ms: float = 0.0
    mean_step_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


@dataclass
class BenchmarkScorecard:
    """Aggregated results from all benchmark scenarios."""

    timestamp: str = ""
    scenarios: list[BenchmarkMetrics] = field(default_factory=list)
    total_scenarios: int = 0
    passed_scenarios: int = 0
    failed_scenarios: int = 0
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_scenarios": self.total_scenarios,
            "passed": self.passed_scenarios,
            "failed": self.failed_scenarios,
            "total_time_ms": round(self.total_time_ms, 2),
            "scenarios": [
                {
                    "name": s.scenario_name,
                    "passed": s.passed,
                    "steps": s.steps,
                    "entities_tracked": s.entities_tracked,
                    "concepts_formed": s.concepts_formed,
                    "features_extracted": s.features_extracted,
                    "surprise_alerts": s.surprise_alerts,
                    "actions_authorised": s.actions_authorised,
                    "actions_rejected": s.actions_rejected,
                    "mean_step_time_ms": round(s.mean_step_time_ms, 2),
                    "errors": s.errors,
                }
                for s in self.scenarios
            ],
        }


def _make_engine() -> tuple[AutonomousEpistemicEngine, CorticalGroundingBridge]:
    """Factory: creates a fresh engine with grounding bridge attached."""
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    sys = build_cognitive_system(CognitiveSystemConfig(world_id="sge_bench", enable_logging=False))
    assert sys.bridge is not None
    return sys.engine, sys.bridge


class SGEBenchmarkRunner:
    """Runs all SGE integration benchmark scenarios and generates a scorecard."""

    def run_all(self) -> BenchmarkScorecard:
        from benchmarks.scenarios.adversarial_surprise import AdversarialSurpriseScenario
        from benchmarks.scenarios.grid_spatial import GridSpatialScenario
        from benchmarks.scenarios.language_grounding import LanguageGroundingScenario
        from benchmarks.scenarios.robotics_manipulation import RoboticsManipulationScenario

        scenarios = [
            GridSpatialScenario(),
            RoboticsManipulationScenario(),
            LanguageGroundingScenario(),
            AdversarialSurpriseScenario(),
        ]

        scorecard = BenchmarkScorecard(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        for scenario in scenarios:
            metrics = scenario.run()
            scorecard.scenarios.append(metrics)

        scorecard.total_scenarios = len(scorecard.scenarios)
        scorecard.passed_scenarios = sum(1 for s in scorecard.scenarios if s.passed)
        scorecard.failed_scenarios = scorecard.total_scenarios - scorecard.passed_scenarios
        scorecard.total_time_ms = sum(s.total_time_ms for s in scorecard.scenarios)

        return scorecard
