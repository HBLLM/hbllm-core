"""Unit tests for the Five-Cohort Comparison and Statistical Tracking."""

from __future__ import annotations

from plugins.developmental_adapter.cohorts import (
    ActiveDevelopmentalHCIRCohort,
    MatureHCIRCohort,
    NeuralLearnerCohort,
    PassiveDevelopmentalHCIRCohort,
    ScriptedCohort,
)
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.metrics import DevelopmentalMetricsTracker


def test_five_cohort_individual_runs():
    env = BabyWorldEnvironment(seed=42)

    c_a = ScriptedCohort(seed=42)
    c_b = NeuralLearnerCohort(seed=42)
    c_c = MatureHCIRCohort(seed=42)
    c_d = ActiveDevelopmentalHCIRCohort(seed=42)
    c_e = PassiveDevelopmentalHCIRCohort(seed=42)

    res_a = c_a.run_causal_discovery_trial(env)
    res_b = c_b.run_causal_discovery_trial(env)
    res_c = c_c.run_causal_discovery_trial(env)
    res_d = c_d.run_causal_discovery_trial(env)
    res_e = c_e.run_causal_discovery_trial(env)

    # Both Scripted and Mature should achieve 100%
    assert res_a.identified_causal_rule is True
    assert res_b is not None
    assert res_c.identified_causal_rule is True

    # Active Developmental HCIR should discover the true mass causal rule
    assert res_d.identified_causal_rule is True
    assert res_d.level2_unseen_entities_accuracy == 1.0
    assert res_d.level3_unseen_world_accuracy == 1.0

    # Key scientific comparison: Active HCIR should waste fewer interventions than Passive HCIR
    assert res_d.interventions_wasted <= res_e.interventions_wasted


def test_metrics_tracker_aggregation_and_ci():
    tracker = DevelopmentalMetricsTracker()
    env = BabyWorldEnvironment(seed=42)

    # Run 3 trials for Active and Passive cohorts
    for s in [101, 102, 103]:
        c_d = ActiveDevelopmentalHCIRCohort(seed=s)
        c_e = PassiveDevelopmentalHCIRCohort(seed=s)
        tracker.record_result(c_d.run_causal_discovery_trial(env))
        tracker.record_result(c_e.run_causal_discovery_trial(env))

    summaries = tracker.aggregate_by_cohort()
    assert "Cohort_D_Active_Developmental_HCIR" in summaries
    assert "Cohort_E_Passive_Developmental_HCIR" in summaries

    summary_d = summaries["Cohort_D_Active_Developmental_HCIR"]
    assert summary_d.trials_count == 3
    assert summary_d.discovery_rate == 1.0
    assert summary_d.ci_95_n_tau[0] <= summary_d.ci_95_n_tau[1]

    table = tracker.format_comparison_table()
    assert "Active_Developmental_HCIR" in table
    assert "Passive_Developmental_HCIR" in table
