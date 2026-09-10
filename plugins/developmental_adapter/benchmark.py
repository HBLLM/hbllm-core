"""Developmental Benchmark Runner for Milestone A23.5.

Executes randomized comparative evaluation across the five cohorts for:
- Experiment E1: Active Interventional Causal Discovery under Fixed Confounder
- Experiment E2: Causal Variable Invariance across Randomized Confounded Worlds
Outputs publication-grade empirical distributions and discrete histograms.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from .cohorts import (
    ActiveDevelopmentalHCIRCohort,
    MatureHCIRCohort,
    NeuralLearnerCohort,
    PassiveDevelopmentalHCIRCohort,
    ScriptedCohort,
)
from .environment import BabyWorldEnvironment
from .metrics import DevelopmentalMetricsTracker

logger = logging.getLogger(__name__)


def run_a23_5_benchmark(
    n_trials: int = 15,
    max_interventions: int = 20,
    seed_base: int = 100,
    scenario: str = "confounded_train_world",
) -> dict[str, Any]:
    """Execute A23.5 causal discovery benchmarks across five comparative cohorts."""
    tracker = DevelopmentalMetricsTracker()
    env = BabyWorldEnvironment()

    cohort_classes = [
        ScriptedCohort,
        NeuralLearnerCohort,
        MatureHCIRCohort,
        ActiveDevelopmentalHCIRCohort,
        PassiveDevelopmentalHCIRCohort,
    ]

    for cohort_cls in cohort_classes:
        cohort_name = cohort_cls.__name__
        logger.info("Evaluating %s across %d trials on %s...", cohort_name, n_trials, scenario)

        for trial_idx in range(n_trials):
            seed = seed_base + trial_idx
            cohort = cohort_cls(seed=seed)
            env.reset(scenario=scenario)

            trial_result = cohort.run_causal_discovery_trial(
                env=env,
                max_interventions=max_interventions,
            )
            tracker.record_result(trial_result)

    summaries = tracker.aggregate_by_cohort()
    ascii_table = tracker.format_comparison_table()
    discrete_table = tracker.format_discrete_distribution_table()

    exportable_data = {
        "experiment": f"A23.5_{scenario}",
        "scenario": scenario,
        "trials_per_cohort": n_trials,
        "max_interventions": max_interventions,
        "cohorts": {
            cid: {
                "discovery_rate": s.discovery_rate,
                "median_n_tau": s.median_n_tau,
                "mean_n_tau": s.mean_n_tau,
                "iqr_n_tau": list(s.iqr_n_tau),
                "bootstrap_ci_95": list(s.ci_95_n_tau),
                "discrete_distribution": {
                    str(k): v for k, v in s.discrete_n_tau_distribution.items()
                },
                "failures_count": s.failures_count,
                "mean_wasted_interventions": s.mean_wasted_interventions,
                "mean_false_hypotheses": s.mean_false_hypotheses,
                "level1_train_accuracy": s.level1_train_accuracy,
                "level2_unseen_entities_accuracy": s.level2_unseen_entities_accuracy,
                "level3_unseen_world_accuracy": s.level3_unseen_world_accuracy,
                "mean_brier_score": s.mean_brier_score,
                "all_n_tau": s.all_n_tau,
            }
            for cid, s in summaries.items()
        },
    }

    return {
        "data": exportable_data,
        "ascii_table": ascii_table,
        "discrete_table": discrete_table,
        "tracker": tracker,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Run A23.5 Developmental Causal Discovery Benchmark"
    )
    parser.add_argument("--trials", type=int, default=15, help="Number of trials per cohort")
    parser.add_argument(
        "--max-interventions", type=int, default=20, help="Max interventions per trial"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="confounded_train_world",
        choices=[
            "confounded_train_world",
            "randomized_confounded_world",
            "friction_confounded_world",
        ],
        help="Environment scenario to evaluate",
    )
    args = parser.parse_args()

    results = run_a23_5_benchmark(
        n_trials=args.trials,
        max_interventions=args.max_interventions,
        scenario=args.scenario,
    )
    print("\n" + "=" * 85)
    print(f"A23.5: CAUSAL DISCOVERY BENCHMARK — Scenario: {args.scenario}")
    print("=" * 85)
    print(results["ascii_table"])
    print("\n" + results["discrete_table"])


if __name__ == "__main__":
    main()
