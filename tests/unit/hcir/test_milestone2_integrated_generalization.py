"""Milestone 2 Unit Test Suite: Integrated World-Model Generalization.

Validates the five scientific pillars of Milestone 2:
1. Independent Reproduction & Evidence Generation: Verifies pinned commit reproducibility and SHA-256 evidence generation.
2. End-to-End Cognitive Closed Loop: Verifies full perception -> graph -> induction -> rollout -> revision -> action/abstention.
3. Unseen Mechanism Family Evaluation: Verifies generalization across 6 disjoint mechanism families.
4. Continual Learning & Catastrophic Forgetting Immunity: Verifies forward transfer speedup and 100% backward retention.
5. Baseline Comparisons & Static ARC Isolation: Verifies comparative advantage over baselines and zero test ground-truth leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.evaluation.generalization_harness import (
    AblationRegime,
)
from experiments.benchmarks.evaluation.run_milestone2_evaluation import (
    Milestone2Evaluator,
)


def test_milestone2_pillar1_independent_reproduction():
    """Verify Pillar 1: 160 capabilities and 640 dimensional gate outcomes are fully reconciled."""
    evaluator = Milestone2Evaluator()
    res = evaluator.execute_pillar_1_reproduction()

    assert res["total_capabilities"] == 160
    assert res["total_dimensional_gate_checks"] == 640
    assert res["gate_1_implementation"] == "160/160"
    assert res["gate_2_runtime_integration"] == "160/160"
    assert res["gate_3_empirical_generalization"] == "160/160"
    assert res["gate_4_architectural_integrity"] == "160/160"
    assert res["fully_4d_verified_capabilities"] == "160/160"
    assert res["reconciliation_status"] == "100.0% RECONCILED"

    for p in [1, 2, 3, 4]:
        p_data = res["phase_breakdown"][f"Phase_{p}"]
        assert p_data["fully_4d_verified"] == p_data["capabilities"]
        assert p_data["pass_rate_pct"] == 100.0


def test_milestone2_pillar2_end_to_end_cognitive_closed_loop():
    """Verify Pillar 2: Raw visual grid flows through segmentation, graph, induction, rollout, and safety gate."""
    evaluator = Milestone2Evaluator()
    res = evaluator.execute_pillar_2_closed_loop()

    # Solvable scenario
    solvable = res["scenario_a_solvable"]
    assert solvable["perceptual_fixations_detected"] >= 1
    assert solvable["scene_graph_nodes"] >= 1
    assert solvable["simulation_exact_match"] is True
    assert solvable["safety_gate_passed"] is True
    assert solvable["decision_action"] == "DISPATCH_PREDICTION"

    # Ambiguous scenario (calibrated abstention)
    underspec = res["scenario_b_underspecified"]
    assert underspec["ambiguity_detected"] is True
    assert underspec["ambiguity_entropy_bits"] > 0.0
    assert underspec["safety_gate_blocked"] is True
    assert underspec["decision_action"] == "CALIBRATED_ABSTENTION"


def test_milestone2_pillar3_unseen_mechanism_family_evaluation():
    """Verify Pillar 3: Disjoint independent task families tested with Level 3 representation expansion."""
    evaluator = Milestone2Evaluator()
    res = evaluator.execute_pillar_3_unseen_mechanisms(seed=42)

    assert res["total_independent_tasks"] == 56  # 50 suite + 6 cellular automata
    assert res["total_solved_exact_match"] >= 50
    assert res["overall_solve_rate_pct"] >= 85.0

    # Every family must have non-zero representation
    families = res["family_breakdown"]
    assert "gravity_settle" in families
    assert "diagonal_ray_cast" in families
    assert "interior_infill" in families
    assert "alternating_pattern_extrapolate" in families
    assert "component_size_rank" in families
    assert "cellular_automaton_local_rule" in families

    for fam, stats in families.items():
        assert stats["exact_matches"] >= 1
        ci_low, ci_high = stats["wilson_95_ci"]
        assert 0.0 <= ci_low <= ci_high <= 1.0


def test_milestone2_pillar4_continual_learning_and_zero_forgetting():
    """Verify Pillar 4: Forward transfer speedup and zero catastrophic forgetting across sequential tasks."""
    evaluator = Milestone2Evaluator()
    res = evaluator.execute_pillar_4_continual_learning()

    assert res["forward_transfer_analogy_verified"] is True
    assert res["backward_retention_accuracy_pct"] == 100.0
    assert res["catastrophic_forgetting_detected"] is False
    assert res["continual_non_destructive_update"] is True


def test_milestone2_pillar5_baseline_comparisons_and_arc_gating():
    """Verify Pillar 5: Full HCIR outperforms baselines, and test ground truth is strictly isolated."""
    evaluator = Milestone2Evaluator()
    res = evaluator.execute_pillar_5_baseline_comparisons()

    regimes = res["ablation_regimes"]
    # Reference baseline must achieve 0%
    assert regimes[AblationRegime.REFERENCE_BASELINE.value]["accuracy_pct"] == 0.0

    # Full HCIR must achieve 100% on the independent evaluation manifest
    assert regimes[AblationRegime.FULL_HCIR.value]["accuracy_pct"] == 100.0

    # Strict ordering: Full HCIR > Atomic Only >= DFS Greedy > Heuristic Geometry > Reference Baseline
    assert (
        regimes[AblationRegime.FULL_HCIR.value]["solved_tasks"]
        > regimes[AblationRegime.ATOMIC_ONLY.value]["solved_tasks"]
    )
    assert (
        regimes[AblationRegime.ATOMIC_ONLY.value]["solved_tasks"]
        >= regimes[AblationRegime.DFS_GREEDY.value]["solved_tasks"]
    )
    assert (
        regimes[AblationRegime.DFS_GREEDY.value]["solved_tasks"]
        > regimes[AblationRegime.HEURISTIC_GEOMETRY.value]["solved_tasks"]
    )
    assert (
        regimes[AblationRegime.HEURISTIC_GEOMETRY.value]["solved_tasks"]
        > regimes[AblationRegime.REFERENCE_BASELINE.value]["solved_tasks"]
    )

    # Static ARC gating
    assert res["static_arc_gating_verified"] is True
    assert res["zero_test_ground_truth_leakage"] is True
    assert res["active_probing_restricted_to_interactive"] is True


def test_milestone2_full_pipeline_evidence_generation(tmp_path: Path):
    """Verify full Milestone 2 end-to-end execution and machine-readable report generation."""
    evaluator = Milestone2Evaluator(output_dir=tmp_path)
    report = evaluator.run_all_pillars()

    assert report["milestone"] == "Milestone 2: Integrated World-Model Generalization"
    assert "integrity_sha256" in report
    assert len(report["integrity_sha256"]) == 64
    assert report["overall_status"] == "MILESTONE_2_COMPLETED_AND_VERIFIED"

    saved_file = tmp_path / "milestone2_evidence_report.json"
    assert saved_file.exists()
    with open(saved_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["integrity_sha256"] == report["integrity_sha256"]
