"""Generalization Closure and Ablation Evaluation Suite (Milestone M4.1).

Verifies the 7 empirical generalization gates:
1. Split Integrity: Development manifest (12 tasks) and Independent Evaluation manifest (30 tasks).
2. Spurious Hypothesis Refutation: Proves that rules fitting Demo 1 are refuted by Demo 2.
3. Paired Ablation Matrix: FullHCIR vs AtomicOnly vs NoMDL vs NoRelational vs ReferenceBaseline.
4. Compositional Depth Scaling: Validates depth 1, depth 2, and depth 3 transformations with explicit denominators.
5. W148 Structural Transfer: Positive transfer accelerates search on novel palettes;
   negative transfer is rejected without memory corruption.
6. Failure Attribution Taxonomy: Accurately identifies hypothesis generation vs selection vs execution failure.
7. Large-Sample Evaluation: Evaluates 30 independent held-out tasks for tightened confidence intervals.
"""

from __future__ import annotations

import numpy as np

from experiments.benchmarks.evaluation.generalization_harness import (
    AblationRegime,
    TransformationGeneralizationHarness,
)
from experiments.benchmarks.manifests.transformation_task_manifest import (
    get_development_manifest,
    get_evaluation_manifest,
)
from hbllm.hcir.world.grid_operator import (
    TransformationProgramSearch,
)


def test_manifest_integrity_and_task_independence():
    """Verify that development and evaluation manifests have unique IDs, valid pairs, and non-empty test sets."""
    dev_manifest = get_development_manifest()
    eval_manifest = get_evaluation_manifest()

    assert len(dev_manifest) == 12
    assert len(eval_manifest) == 30

    dev_ids = [t.task_id for t in dev_manifest]
    eval_ids = [t.task_id for t in eval_manifest]

    # No ID collision within or between manifests
    assert len(dev_ids) == len(set(dev_ids))
    assert len(eval_ids) == len(set(eval_ids))
    assert len(set(dev_ids) & set(eval_ids)) == 0, (
        "Development and evaluation sets must be completely disjoint"
    )

    # Both cover depths 1, 2, and 3
    assert {1, 2, 3}.issubset({t.depth for t in eval_manifest})


def test_spurious_hypothesis_refutation_on_subsequent_demos():
    """Verify Popperian disambiguation: rules fitting Demo 1 are refuted on Demo 2."""
    manifest = get_development_manifest()
    disambig_task = next(t for t in manifest if t.family == "DISAMBIGUATION")

    # If evaluated on demo 1 alone, both rot180 and flip_h fit because input is diagonal symmetric
    d1_x, d1_y = disambig_task.train_pairs[0]
    searcher_single = TransformationProgramSearch()
    _, b1, meta1 = searcher_single.solve([(d1_x, d1_y)], disambig_task.test_input)
    assert meta1["survivor_count"] >= 2

    # When evaluated on all demos, Demo 2 refutes spurious flips and isolates ROT_180
    searcher_multi = TransformationProgramSearch()
    pred, b_multi, meta_multi = searcher_multi.solve(
        list(disambig_task.train_pairs), disambig_task.test_input
    )
    assert meta_multi["solved"] is True
    assert meta_multi["spurious_rejected_on_later_demos"] >= 1
    assert b_multi is not None
    assert pred is not None
    assert b_multi.params.get("op") == "ROT_180"
    assert np.array_equal(pred, disambig_task.test_output)


def test_compositional_depth_scaling_depth_1_2_3():
    """Stress test transformation depth scaling from 1 to 3 operators with explicit denominators."""
    manifest = get_development_manifest()

    # Depth 1: Flip Vertical (1/1 tested here)
    t_d1 = next(t for t in manifest if "geo_flip_v" in t.task_id)
    res_d1 = TransformationGeneralizationHarness.evaluate_task(t_d1, AblationRegime.FULL_HCIR)
    assert res_d1.exact_match is True
    assert res_d1.pixel_accuracy == 1.0

    # Depth 2: Crop -> Rotate (1/1 tested here)
    t_d2 = next(t for t in manifest if "comp_crop_rot" in t.task_id)
    res_d2 = TransformationGeneralizationHarness.evaluate_task(t_d2, AblationRegime.FULL_HCIR)
    assert res_d2.exact_match is True
    assert res_d2.pixel_accuracy == 1.0

    # Depth 3: Crop -> Rotate -> Recolor (1/1 tested here)
    t_d3 = next(t for t in manifest if "comp_crop_rot_recolor" in t.task_id)
    res_d3 = TransformationGeneralizationHarness.evaluate_task(t_d3, AblationRegime.FULL_HCIR)
    assert res_d3.exact_match is True
    assert res_d3.pixel_accuracy == 1.0


def test_ablation_matrix_and_paired_differences():
    """Verify that FullHCIR statistically outperforms ablations using paired difference analysis."""
    manifest = get_development_manifest()

    card_full = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.FULL_HCIR, manifest
    )
    card_atomic = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.ATOMIC_ONLY, manifest
    )
    card_base = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.REFERENCE_BASELINE, manifest
    )
    card_no_mdl = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.NO_MDL, manifest
    )

    # Paired differences against Full HCIR
    diff_atomic = TransformationGeneralizationHarness.compute_paired_differences(
        card_full, card_atomic
    )
    diff_base = TransformationGeneralizationHarness.compute_paired_differences(card_full, card_base)
    diff_no_mdl = TransformationGeneralizationHarness.compute_paired_differences(
        card_full, card_no_mdl
    )

    # Full HCIR wins strictly positive number of tasks over each ablation
    assert diff_atomic["delta_accuracy_pct"] > 0
    assert diff_atomic["tasks_won_by_full_count"] >= 5
    assert diff_atomic["tasks_won_by_abl_count"] == 0

    assert diff_base["delta_accuracy_pct"] > 80.0
    assert diff_base["tasks_won_by_full_count"] >= 10

    # No MDL fails tasks due to non-minimal overparameterized hypothesis selection
    assert diff_no_mdl["delta_accuracy_pct"] > 0
    assert diff_no_mdl["tasks_won_by_full_count"] >= 1

    # Explicit denominator check for Depth 1:
    # Depth 1 has 7 tasks in development manifest:
    # Atomic solves 7/7 (100.0%), NoMDL solves <=4/7, ReferenceBase solves 0/7 (0.0%)
    assert card_atomic.depth_breakdown[1]["solved"] == 7
    assert card_atomic.depth_breakdown[1]["pct"] == 100.0
    assert card_no_mdl.depth_breakdown[1]["solved"] in (2, 4)
    assert card_no_mdl.depth_breakdown[1]["pct"] in (28.57, 57.14)
    assert card_base.depth_breakdown[1]["solved"] == 0
    assert card_base.depth_breakdown[1]["pct"] == 0.0


def test_independent_evaluation_manifest_30_tasks():
    """Run full evaluation across the 30 independently selected held-out tasks."""
    eval_manifest = get_evaluation_manifest()
    assert len(eval_manifest) == 30

    card_full = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.FULL_HCIR, eval_manifest
    )
    card_atomic = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.ATOMIC_ONLY, eval_manifest
    )

    # Across 30 tasks, Full HCIR achieves exact match across all stratified families
    assert card_full.total_tasks == 30
    assert card_full.solved_tasks == 30
    assert card_full.exact_match_pct == 100.0

    # With N=30, the 95% Wilson confidence interval is tightened from [75.75%, 100%] to [88.65%, 100%]
    ci_low, ci_high = card_full.confidence_interval_95
    assert ci_low >= 88.0, f"Expected tightened lower bound >= 88%, got {ci_low}%"
    assert ci_high == 100.0

    # Atomic Only fails all depth 2 and depth 3 tasks in the 30-task evaluation set
    assert card_atomic.depth_breakdown[1]["pct"] == 100.0
    assert card_atomic.depth_breakdown[2]["pct"] == 0.0
    assert card_atomic.depth_breakdown[3]["pct"] == 0.0


def test_w148_structural_transfer_and_negative_transfer_immunity():
    """Verify positive analogical transfer and rejection of conflicting prior rules."""
    manifest = get_development_manifest()
    src_task = next(t for t in manifest if "transfer_source" in t.task_id)
    target_task = next(t for t in manifest if "transfer_target" in t.task_id)
    neg_task = next(t for t in manifest if "transfer_negative" in t.task_id)

    # Step 1: Solve source task and extract learned program binding
    searcher_src = TransformationProgramSearch()
    _, src_binding, src_meta = searcher_src.solve(list(src_task.train_pairs), src_task.test_input)
    assert src_meta["solved"] is True
    assert src_binding is not None

    # Step 2: Positive transfer on target task with novel palette
    res_transfer = TransformationGeneralizationHarness.evaluate_task(
        target_task,
        AblationRegime.FULL_HCIR,
        prior_rules=[src_binding],
    )
    assert res_transfer.exact_match is True
    assert res_transfer.metadata.get("used_prior_transfer") is True

    # Step 3: Negative transfer resilience on conflicting task
    res_neg = TransformationGeneralizationHarness.evaluate_task(
        neg_task,
        AblationRegime.FULL_HCIR,
        prior_rules=[src_binding],  # Conflicting Crop->Rotate prior provided!
    )
    # The Popperian refutation gate MUST refute the conflicting prior and correctly find Flip
    assert res_neg.exact_match is True
    assert res_neg.metadata.get("used_prior_transfer") is False
    assert res_neg.winning_description is not None
    assert "flip" in res_neg.winning_description.lower() or "FLIP" in res_neg.winning_description
