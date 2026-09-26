"""Unit tests for Downstream Cognitive Reasoning Benchmarks (ARC & HCIR) (Milestone A24)."""

from __future__ import annotations

from plugins.developmental_adapter.downstream_benchmarks import DownstreamBenchmarkRunner
from plugins.developmental_adapter.school import CognitiveSchool


def test_arc_topological_battery() -> None:
    """Verify ARC-style topological boundary filling, obstacle clearance, and tool reach."""
    school = CognitiveSchool(seed=42)
    student = school.student
    school.teacher.conduct_kindergarten(student)

    # Prime student with affordances & causal rule
    student.substrate.affordances["box"] = ["CONTAINER", "GRASPABLE"]
    student.substrate.affordances["ball"] = ["ROLLABLE", "GRASPABLE", "SLIDABLE"]
    student.substrate.affordances["block"] = ["GRASPABLE", "SLIDABLE"]
    student.substrate.affordances["tool"] = ["TOOL", "EXTENDS_REACH", "GRASPABLE"]
    student.substrate.causal_rules.append(
        {
            "rule_id": "rule_push_mass",
            "precondition": {"property": "mass_sensation", "operator": "<", "value": 3.0},
            "action": "push",
            "consequence": "MOVES",
            "empirical_support_count": 6,
        }
    )

    runner = DownstreamBenchmarkRunner(student)
    results = runner.run_arc_battery()

    assert len(results) == 3
    for r in results:
        assert r.is_success is True
        assert r.score == 1.0
        assert r.brier_error <= 0.05


def test_hcir_relational_transfer_battery() -> None:
    """Verify zero-shot schema lifting and transfer to formal HCIR industrial CognitiveGraphs."""
    school = CognitiveSchool(seed=42)
    student = school.student
    school.teacher.conduct_kindergarten(student)

    student.substrate.causal_rules.append(
        {
            "rule_id": "rule_push_mass",
            "precondition": {"property": "mass_sensation", "operator": "<", "value": 3.0},
            "action": "push",
            "consequence": "MOVES",
            "empirical_support_count": 8,
        }
    )

    runner = DownstreamBenchmarkRunner(student)
    results = runner.run_hcir_transfer_battery()

    assert len(results) == 2
    for r in results:
        assert r.is_success is True
        assert r.score >= 0.60
        assert r.brier_error <= 0.05
        assert r.details["is_applicable"] is True


def test_counterfactual_epistemic_battery() -> None:
    """Verify non-hallucination calibration and falsification resistance on counterfactual queries."""
    school = CognitiveSchool(seed=42)
    student = school.student

    # Accurate physical affordances (blocks do NOT roll)
    student.substrate.affordances["block"] = ["GRASPABLE", "SLIDABLE"]
    student.substrate.affordances["ball"] = ["ROLLABLE", "GRASPABLE", "SLIDABLE"]
    student.substrate.causal_rules.append(
        {
            "rule_id": "rule_push_mass",
            "precondition": {"property": "mass_sensation", "operator": "<", "value": 3.0},
            "action": "push",
            "consequence": "MOVES",
            "empirical_support_count": 5,
        }
    )

    runner = DownstreamBenchmarkRunner(student)
    results = runner.run_counterfactual_battery()

    assert len(results) == 2
    for r in results:
        assert r.is_success is True
        assert r.score == 1.0
        assert r.brier_error <= 0.05


def test_full_downstream_benchmark_suite_and_markdown_report() -> None:
    """Verify full end-to-end execution of all 3 batteries and report generation."""
    school = CognitiveSchool(seed=42)
    student = school.student
    school.teacher.conduct_kindergarten(student)

    # Establish full learned substrate state
    student.substrate.affordances["box"] = ["CONTAINER", "GRASPABLE"]
    student.substrate.affordances["ball"] = ["ROLLABLE", "GRASPABLE", "SLIDABLE"]
    student.substrate.affordances["block"] = ["GRASPABLE", "SLIDABLE"]
    student.substrate.affordances["tool"] = ["TOOL", "EXTENDS_REACH", "GRASPABLE"]
    student.substrate.causal_rules.append(
        {
            "rule_id": "rule_push_mass",
            "precondition": {"property": "mass_sensation", "operator": "<", "value": 3.0},
            "action": "push",
            "consequence": "MOVES",
            "empirical_support_count": 10,
        }
    )

    runner = DownstreamBenchmarkRunner(student)
    report = runner.run_all_benchmarks()

    assert report.total_tasks == 7
    assert report.tasks_passed == 7
    assert report.overall_score == 1.0
    assert report.mean_brier_score <= 0.05

    # Check Markdown report
    md = report.format_markdown()
    assert "# Downstream Cognitive Reasoning Benchmark Report" in md
    assert "**Overall Reasoning Score**: 100.0%" in md
    assert "**Tasks Passed**: 7/7" in md
    assert "ARC-Topological" in md
    assert "HCIR-Transfer" in md
    assert "Counterfactual-Epistemics" in md
