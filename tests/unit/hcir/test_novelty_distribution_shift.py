"""Milestone M4.3 Verification Suite: Novel-Rule, Distribution-Shift, and Interactive Transfer.

Validates that HBLLM/HCIR visual and interactive world modeling generalizes
beyond anticipated operators, frozen manifests, and static action semantics:
1. Unseen Combinations: Procedural combinatorial compositions (depth 2 and 3).
2. Distribution Shifts: Grid dimension scaling (5x5 -> 15x15), palette shifts.
3. Ambiguous Demonstrations: Multi-hypothesis disambiguation via subsequent evidence.
4. Novel Operators: Discovering withheld transformation primitives (perimeter outline) from observations.
5. Interactive Transfer: Rapid adaptation to permuted action alphabets with retained transition topology.
"""

from __future__ import annotations

from experiments.benchmarks.evaluation.novelty_distribution_shift import (
    NoveltyAndDistributionShiftSuite,
)


def test_unseen_combinations_procedural_manifest():
    """Verify that procedurally generated unfamiliar compositions achieve 100% exact match."""
    results = NoveltyAndDistributionShiftSuite.test_unseen_combinations(seed=123, num_tasks=5)
    assert len(results) == 5
    for r in results:
        assert r.passed, f"Failed on unseen procedural task: {r.description}"
        assert r.exact_match, f"Non-exact match: {r.description}"
        assert r.pixel_accuracy == 1.0


def test_dimension_scale_shift_invariance():
    """Verify that transformation rules induced on 5x5 grids apply exactly to 15x15 grids."""
    res = NoveltyAndDistributionShiftSuite.test_dimension_scale_shift()
    assert res.passed, f"Dimension scale shift failed: {res.description}"
    assert res.exact_match
    assert res.pixel_accuracy == 1.0


def test_palette_disjoint_shift_invariance():
    """Verify that transformation rules generalize across palette variations."""
    res = NoveltyAndDistributionShiftSuite.test_palette_disjoint_shift()
    assert res.passed, f"Palette shift failed: {res.description}"
    assert res.exact_match
    assert res.pixel_accuracy == 1.0


def test_ambiguous_demonstration_disambiguation():
    """Verify that ambiguous demonstrations maintain multiple candidates and disambiguate upon evidence."""
    res = NoveltyAndDistributionShiftSuite.test_ambiguous_demonstration_disambiguation()
    assert res["demo1_has_multiple_survivors"], (
        "Single demonstration should have multiple valid hypotheses"
    )
    assert res["resolved_via_demo2"], (
        "Second demonstration must disambiguate and refute the invalid hypothesis"
    )
    assert res["spurious_rejected_count"] >= 1, "Must actively reject candidate that failed demo 2"


def test_novel_operator_discovery_from_observations():
    """Verify that a transformation family withheld during development is discovered from observations."""
    res = NoveltyAndDistributionShiftSuite.test_novel_operator_discovery()
    assert res.passed, f"Novel operator discovery failed: {res.description}"
    assert res.exact_match
    assert res.pixel_accuracy == 1.0
    assert res.details.get("is_dynamically_synthesized"), (
        "Must be flagged as dynamically synthesized"
    )


def test_interactive_transfer_under_action_permutation():
    """Verify that learned spatial world model adapts to permuted action semantics via epistemic surprise."""
    res = NoveltyAndDistributionShiftSuite.test_interactive_action_permutation_transfer()
    assert res["transfer_adapted"], "Interactive transfer failed to adapt to permuted action"
    assert res["observations_recorded"] >= 2
    assert res["retained_goal_position"] == (5, 5)
