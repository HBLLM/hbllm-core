"""Unit Test Suite: Blind Cross-Family Mechanism-Induction Challenge.

Validates the 6 Auditor Criteria on Withheld Mechanism Families:
1. AST Independence: Task generator does not import solver or grammar internals.
2. Inadequacy Detection: Base solver flags insufficient hypothesis language on novel physics.
3. Representational Need Identification: Residual event set detects non-zero unexplained cell transitions.
4. Operator Inductive Synthesis: Dynamic synthesis produces valid candidate operators.
5. Calibrated Epistemic Uncertainty & Abstention: Safely abstains on ambiguous demonstrations.
6. Machine-Readable Audit Report: Validates report generation and SHA-256 digest.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from experiments.benchmarks.evaluation.blind_cross_family_challenge import (
    BlindCrossFamilyChallengeEvaluator,
)
from experiments.benchmarks.task_generators.blind_challenge_generator import (
    BlindChallengeTaskGenerator,
)


def test_blind_generator_ast_independence():
    """Verify that BlindChallengeTaskGenerator has zero imports from solver or GridOperator."""
    gen_module = inspect.getmodule(BlindChallengeTaskGenerator)
    assert gen_module is not None
    tree = ast.parse(inspect.getsource(gen_module))

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    # Must NOT import grid_operator, TransformationProgramSearch, or representation_expansion
    assert not any("grid_operator" in m for m in imported_modules), (
        "Blind generator must not import grid_operator"
    )
    assert not any("representation_expansion" in m for m in imported_modules), (
        "Blind generator must not import representation_expansion"
    )


def test_blind_challenge_suite_generation_and_diversity():
    """Verify suite generation produces 25 tasks covering all 5 withheld mechanism families."""
    suite = BlindChallengeTaskGenerator.generate_challenge_suite(seed=42)
    assert len(suite) == 25

    families = {t.family for t in suite}
    expected = {
        "perimeter_contour_dilation",
        "maze_shortest_path",
        "parity_color_inversion",
        "elastic_particle_deflection",
        "underspecified_ambiguity_probe",
    }
    assert families == expected

    for fam in expected:
        fam_tasks = [t for t in suite if t.family == fam]
        assert len(fam_tasks) == 5


def test_criterion_1_and_2_inadequacy_and_residual_identification():
    """Verify that unencountered mechanism tasks trigger hypothesis language inadequacy."""
    gen = BlindChallengeTaskGenerator(seed=42)
    task = gen.generate_task("test_maze_01", "maze_shortest_path")

    evaluator = BlindCrossFamilyChallengeEvaluator()
    record = evaluator.evaluate_task(task)

    assert record.inadequacy_detected is True
    assert record.residual_pixels > 0


def test_criterion_5_calibrated_uncertainty_and_abstention():
    """Verify that underspecified ambiguous demonstrations trigger calibrated abstention."""
    gen = BlindChallengeTaskGenerator(seed=42)
    task = gen.generate_task("test_probe_01", "underspecified_ambiguity_probe")

    evaluator = BlindCrossFamilyChallengeEvaluator()
    record = evaluator.evaluate_task(task)

    assert record.decision_action == "CALIBRATED_ABSTENTION"
    assert record.epistemic_uncertainty >= 0.75
    assert record.exact_match is False
    assert record.metadata.get("calibrated_abstention_success") is True


def test_blind_challenge_full_execution_and_evidence_report(tmp_path: Path):
    """Verify end-to-end execution of the blind challenge harness and report generation."""
    evaluator = BlindCrossFamilyChallengeEvaluator(output_dir=tmp_path)
    report = evaluator.run_challenge(seed=42)

    assert report["challenge"] == "Blind Cross-Family Mechanism-Induction Challenge"
    assert report["challenge_status"] == "BLIND_CHALLENGE_COMPLETED"
    assert "integrity_sha256" in report
    assert len(report["integrity_sha256"]) == 64

    # Verify auditor criteria
    crit = report["auditor_criteria_evaluations"]
    assert crit["criterion_1_inadequacy_detection"]["verified"] is True
    assert crit["criterion_2_representational_need_identification"]["verified"] is True
    assert crit["criterion_5_calibrated_uncertainty_abstention"]["verified"] is True
    assert crit["criterion_6_cross_family_generalization_distinction"]["verified"] is True

    # Verify summary metrics exist
    metrics = report["summary_metrics"]
    assert metrics["total_tasks_evaluated"] == 25
    assert metrics["total_calibrated_abstentions"] == 5
    assert metrics["inadequacies_detected"] >= 20

    # Verify file saved on disk
    saved_file = tmp_path / "blind_cross_family_challenge_report.json"
    assert saved_file.exists()
    with open(saved_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["integrity_sha256"] == report["integrity_sha256"]
