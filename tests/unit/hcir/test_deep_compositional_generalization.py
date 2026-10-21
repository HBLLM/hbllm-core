"""Deep Compositional Generalization and Exploratory Stress Test Suite (Milestone M4.2).

Validates high-order composition beyond 2 operators:
1. Extended Depth-3 Evaluation: 5 diverse multi-stage pipelines combining previously unexercised operator pairs.
2. Exploratory Depth-4 Stress Test: 4-stage pipelines (e.g. Crop ∘ Rotate ∘ Translate ∘ Recolor).
3. Ablation Verification: Proves AtomicOnly collapses to 0.0% on depth-3 and depth-4 tasks.
4. Strengthened Baseline Comparisons: Heuristic Geometry and DFS Greedy.
"""

from __future__ import annotations

import numpy as np

from experiments.benchmarks.evaluation.generalization_harness import (
    AblationRegime,
    TransformationGeneralizationHarness,
)
from experiments.benchmarks.evaluation.scaled_evaluation_suite import (
    DeepCompositionBenchmark,
)
from experiments.benchmarks.manifests.deep_composition_manifest import (
    get_depth3_extended_manifest,
    get_depth4_exploratory_manifest,
    get_depth5_exploratory_manifest,
)
from hbllm.hcir.world.grid_operator import TransformationProgramSearch


def test_depth3_extended_compositional_manifest_5_tasks():
    """Verify that Full HCIR solves 100% of the 5 independently constructed depth-3 pipelines."""
    d3_tasks = get_depth3_extended_manifest()
    assert len(d3_tasks) == 5

    solved_count = 0
    for task in d3_tasks:
        searcher = TransformationProgramSearch(max_depth=3, use_mdl=True)
        pred, binding, meta = searcher.solve(list(task.train_pairs), task.test_input)
        assert meta["solved"] is True
        assert binding is not None
        assert pred is not None
        assert np.array_equal(pred, task.test_output), f"Task {task.task_id} prediction failed"
        solved_count += 1

    assert solved_count == 5


def test_depth4_exploratory_stress_test_manifest():
    """Exploratory stress test: Verify that the engine composes 4 sequential operator stages."""
    d4_tasks = get_depth4_exploratory_manifest()
    assert len(d4_tasks) == 4

    for task in d4_tasks:
        searcher = TransformationProgramSearch(max_depth=4, use_mdl=True)
        pred, binding, meta = searcher.solve(list(task.train_pairs), task.test_input)
        assert meta["solved"] is True
        assert binding is not None
        assert pred is not None
        assert np.array_equal(pred, task.test_output), (
            f"Depth 4 task {task.task_id} prediction failed"
        )


def test_depth5_exploratory_stress_test_manifest():
    """Exploratory stress test: Verify that the engine composes 5 sequential operator stages."""
    d5_tasks = get_depth5_exploratory_manifest()
    assert len(d5_tasks) == 2

    for task in d5_tasks:
        searcher = TransformationProgramSearch(max_depth=5, use_mdl=True)
        pred, binding, meta = searcher.solve(list(task.train_pairs), task.test_input)
        assert meta["solved"] is True
        assert binding is not None
        assert pred is not None
        assert np.array_equal(pred, task.test_output), (
            f"Depth 5 task {task.task_id} prediction failed"
        )


def test_atomic_only_collapses_on_depth3_4_5():
    """Verify that AtomicOnly fails 100% of depth-3, depth-4, and depth-5 tasks."""
    all_deep_tasks = (
        get_depth3_extended_manifest()
        + get_depth4_exploratory_manifest()
        + get_depth5_exploratory_manifest()
    )
    assert len(all_deep_tasks) == 11

    for task in all_deep_tasks:
        searcher = TransformationProgramSearch(max_depth=1, use_mdl=True)
        pred, binding, meta = searcher.solve(list(task.train_pairs), task.test_input)
        # Single operators cannot compose multi-stage transformations
        if pred is not None and pred.shape == task.test_output.shape:
            assert not np.array_equal(pred, task.test_output), (
                f"Atomic search unexpectedly solved deep task {task.task_id}"
            )
        else:
            assert not meta["solved"] or pred is None or pred.shape != task.test_output.shape


def test_procedural_depth_scaling_benchmark_d1_to_d5():
    """Stage 2 Validation: Verify procedural generalization across depths D=1..5 with per-depth scores."""
    scorecards = DeepCompositionBenchmark.evaluate_depth_scaling(
        seed=42, max_depth=5, tasks_per_depth=3
    )
    assert len(scorecards) == 5

    for d in range(1, 6):
        card = scorecards[d]
        assert card.depth == d
        assert card.total_tasks == 3
        assert card.solved_tasks == 3, f"Depth {d} failed: {card.solved_tasks}/3 solved"
        assert card.exact_match_pct == 100.0
        assert card.mean_latency_ms >= 0.0


def test_strengthened_baselines_on_deep_composition_manifest():
    """Compare Full HCIR against Heuristic Geometry and DFS Greedy on deep compositional tasks."""
    d3_tasks = get_depth3_extended_manifest()

    card_full = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.FULL_HCIR, d3_tasks
    )
    card_heur = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.HEURISTIC_GEOMETRY, d3_tasks
    )
    card_dfs = TransformationGeneralizationHarness.evaluate_regime(
        AblationRegime.DFS_GREEDY, d3_tasks
    )

    assert card_full.solved_tasks == 5
    assert card_full.exact_match_pct == 100.0

    # Heuristic Geometry cannot compose and fails all depth 3 tasks
    assert card_heur.solved_tasks == 0

    # DFS Greedy explores depth-3 compositions first, so it solves pure depth-3 tasks
    assert card_dfs.solved_tasks == 5
    # But Full HCIR selects lower-complexity winning descriptions via MDL (mean complexity < DFS Greedy)
    full_mean_c = np.mean([r.winning_complexity for r in card_full.results if r.winning_complexity])
    dfs_mean_c = np.mean([r.winning_complexity for r in card_dfs.results if r.winning_complexity])
    assert full_mean_c < dfs_mean_c, (
        f"Full HCIR should yield simpler solutions ({full_mean_c}) than DFS Greedy ({dfs_mean_c})"
    )
