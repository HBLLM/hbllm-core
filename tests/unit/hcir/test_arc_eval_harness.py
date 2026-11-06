"""Unit tests for ARC Evaluation Harness & Rule Induction Engine.

Verifies:
1. Candidate rule generation and Popperian refutation across training demonstrations.
2. Simplicity / MDL ranking selecting the minimal valid rule.
3. End-to-end test prediction and held-out generalization on geometric, color, and crop tasks.
4. Strict separation of training consistency from held-out generalization (overfitting detection).
5. Standardized failure categorization (FITTING_FAILED, OVERFITTING_FALSE_SELECTION).
6. Suite aggregation with 95% Wilson confidence intervals.
"""

from __future__ import annotations

import numpy as np

from experiments.benchmarks.evaluation.eval_harness import (
    ARCEvaluationHarness,
    ARCTask,
    DemonstrationTask,
    FailureCategory,
)
from hbllm.hcir.world.rule_induction import (
    AffineOp,
    AffineRule,
    ColorMappingRule,
    PopperianRefutationGate,
    SimplicityRanker,
)


def test_popperian_refutation_gate() -> None:
    """Verify that candidate rules failing ANY demonstration pair are immediately refuted."""
    # Demo 1: flip horizontal
    x1 = np.array([[1, 2], [3, 4]], dtype=int)
    y1 = np.array([[3, 4], [1, 2]], dtype=int)  # flipped horizontally (up-down)

    # Demo 2: flip horizontal
    x2 = np.array([[5, 6, 7], [8, 9, 0]], dtype=int)
    y2 = np.array([[8, 9, 0], [5, 6, 7]], dtype=int)

    candidates = [
        AffineRule(AffineOp.FLIP_H),  # Correct rule
        AffineRule(AffineOp.FLIP_V),  # Fails demo 1
        AffineRule(AffineOp.ROT_90),  # Fails demo 1
        ColorMappingRule({1: 3}),  # Fails demo 1
    ]

    gate = PopperianRefutationGate()
    survivors = gate.evaluate_and_filter(candidates, [(x1, y1), (x2, y2)])

    assert len(survivors) == 1
    assert survivors[0].rule_id == "affine_flip_h"
    assert len(gate.falsifications) >= 3


def test_simplicity_ranker_mdl() -> None:
    """Verify that SimplicityRanker sorts rules by Kolmogorov / MDL complexity."""
    rule_simple = AffineRule(AffineOp.ROT_180)  # complexity 1.0
    rule_complex = ColorMappingRule({1: 2, 3: 4, 5: 6, 7: 8})  # complexity 2.0

    ranked = SimplicityRanker.rank_survivors([rule_complex, rule_simple])
    assert ranked[0].rule_id == rule_simple.rule_id
    assert ranked[1].rule_id == rule_complex.rule_id


def test_harness_solve_geometric_reflection() -> None:
    """End-to-end evaluation of geometric horizontal reflection task."""
    d1_x = np.array([[1, 0, 2], [0, 3, 0]], dtype=int)
    d1_y = np.array([[0, 3, 0], [1, 0, 2]], dtype=int)

    d2_x = np.array([[4, 5], [6, 7], [8, 9]], dtype=int)
    d2_y = np.array([[8, 9], [6, 7], [4, 5]], dtype=int)

    test_x = np.array([[1, 1], [2, 2]], dtype=int)
    test_y = np.array([[2, 2], [1, 1]], dtype=int)

    task = ARCTask(
        task_id="geometric_reflection",
        train_pairs=[(d1_x, d1_y), (d2_x, d2_y)],
        test_input=test_x,
        test_output=test_y,
    )

    harness = ARCEvaluationHarness()
    record = harness.evaluate_task(task)

    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is True
    assert record.test_pixel_accuracy == 1.0
    assert record.failure_category is None
    assert record.selected_rule_id == "affine_flip_h"


def test_harness_solve_color_remapping() -> None:
    """End-to-end evaluation of deterministic palette substitution task."""
    d1_x = np.array([[1, 2], [2, 1]], dtype=int)
    d1_y = np.array([[3, 4], [4, 3]], dtype=int)

    d2_x = np.array([[1, 1, 2], [2, 0, 1]], dtype=int)
    d2_y = np.array([[3, 3, 4], [4, 0, 3]], dtype=int)

    test_x = np.array([[2, 2], [1, 0]], dtype=int)
    test_y = np.array([[4, 4], [3, 0]], dtype=int)

    task = ARCTask(
        task_id="color_remapping",
        train_pairs=[(d1_x, d1_y), (d2_x, d2_y)],
        test_input=test_x,
        test_output=test_y,
    )

    harness = ARCEvaluationHarness()
    record = harness.evaluate_task(task)

    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is True
    assert record.test_pixel_accuracy == 1.0
    assert record.failure_category is None


def test_harness_solve_crop_bounding_box() -> None:
    """End-to-end evaluation of non-background bounding box crop task."""
    # 5x5 grid with 2x2 object
    d1_x = np.zeros((5, 5), dtype=int)
    d1_x[1:3, 2:4] = 2
    d1_y = np.full((2, 2), 2, dtype=int)

    # 6x6 grid with 3x2 object
    d2_x = np.zeros((6, 6), dtype=int)
    d2_x[2:5, 1:3] = 4
    d2_y = np.full((3, 2), 4, dtype=int)

    # Test: 7x7 grid with 2x3 object
    test_x = np.zeros((7, 7), dtype=int)
    test_x[3:5, 1:4] = 5
    test_y = np.full((2, 3), 5, dtype=int)

    task = ARCTask(
        task_id="crop_bbox",
        train_pairs=[(d1_x, d1_y), (d2_x, d2_y)],
        test_input=test_x,
        test_output=test_y,
    )

    harness = ARCEvaluationHarness()
    record = harness.evaluate_task(task)

    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is True
    assert record.test_pixel_accuracy == 1.0
    assert record.selected_rule_id == "crop_bbox_non_bg"


def test_harness_distinguishes_fitting_from_generalization() -> None:
    """Verify that fitting training demos does NOT imply test generalization (Overfitting test)."""
    # Suppose training demos are symmetric under multiple transformations (e.g. Identity and 180 Rotation)
    # 2x2 grid of all 1s: Identity AND Rot180 fit 100% of training demos!
    d1_x = np.ones((2, 2), dtype=int)
    d1_y = np.ones((2, 2), dtype=int)

    # But the held-out test is asymmetric: [[1, 2], [3, 4]]
    # Ground truth expects 180 rotation: [[4, 3], [2, 1]]
    test_x = np.array([[1, 2], [3, 4]], dtype=int)
    test_y = np.array([[4, 3], [2, 1]], dtype=int)

    task = ARCTask(
        task_id="ambiguous_symmetry_task",
        train_pairs=[(d1_x, d1_y)],
        test_input=test_x,
        test_output=test_y,
    )

    harness = ARCEvaluationHarness()
    record = harness.evaluate_task(task)

    # Both Identity (cost 0.1) and Rot180 (cost 1.0) fit training demos.
    # Identity is chosen by SimplicityRanker (0.1 < 1.0).
    # Identity on test_x yields [[1, 2], [3, 4]], which does NOT match test_y!
    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is False
    # Harness MUST catch that training passed but test failed!
    assert record.failure_category in (
        FailureCategory.AMBIGUOUS_SURVIVORS,
        FailureCategory.OVERFITTING_FALSE_SELECTION,
    )


def test_harness_evaluate_suite_summary() -> None:
    """Verify suite aggregation, Wilson score interval, and failure distribution."""
    # Task 1: solvable
    t1_x = np.array([[1, 2], [3, 4]], dtype=int)
    t1_y = np.array([[3, 4], [1, 2]], dtype=int)
    task1 = ARCTask(
        task_id="t1",
        train_pairs=[(t1_x, t1_y)],
        test_input=t1_x,
        test_output=t1_y,
    )

    # Task 2: insolvable under current minimal generator (arbitrary random noise)
    t2_x = np.array([[1, 2], [3, 4]], dtype=int)
    t2_y = np.array([[9, 8], [7, 6]], dtype=int)
    t2_x2 = np.array([[2, 3], [4, 5]], dtype=int)
    t2_y2 = np.array([[0, 1], [2, 3]], dtype=int)
    task2 = ARCTask(
        task_id="t2_unsolvable",
        train_pairs=[(t2_x, t2_y), (t2_x2, t2_y2)],
        test_input=t2_x,
        test_output=t2_y,
    )

    harness = ARCEvaluationHarness()
    summary = harness.evaluate_suite([task1, task2])

    assert summary.total_tasks == 2
    assert summary.train_perfect_fit_count == 1
    assert summary.test_exact_solve_count == 1
    assert summary.test_exact_solve_rate == 0.5
    assert 0.0 <= summary.confidence_interval_95[0] < summary.confidence_interval_95[1] <= 1.0
    assert FailureCategory.FITTING_FAILED.value in summary.failure_distribution


def test_scaling_induction_w054() -> None:
    """Verify W054: Parameterized 2x scaling induction and execution."""
    d1_x = np.array([[1, 2], [3, 4]], dtype=int)
    d1_y = np.repeat(np.repeat(d1_x, 2, axis=0), 2, axis=1)

    d2_x = np.array([[5, 0], [0, 6]], dtype=int)
    d2_y = np.repeat(np.repeat(d2_x, 2, axis=0), 2, axis=1)

    test_x = np.array([[7, 8], [9, 1]], dtype=int)
    test_y = np.repeat(np.repeat(test_x, 2, axis=0), 2, axis=1)

    task = ARCTask(
        task_id="w054_scale_2x",
        train_pairs=[(d1_x, d1_y), (d2_x, d2_y)],
        test_input=test_x,
        test_output=test_y,
    )

    harness = ARCEvaluationHarness()
    record = harness.evaluate_task(task)

    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is True
    assert record.test_pixel_accuracy == 1.0
    assert "scaling_2x2" in (record.selected_rule_id or "")


def test_duplication_and_kaleidoscope_w056() -> None:
    """Verify W056: 2x2 lattice duplication with alternating reflections."""
    d1_x = np.array([[1, 2], [3, 0]], dtype=int)
    # 2x2 duplication with horizontal and vertical flips
    top = np.hstack([d1_x, np.fliplr(d1_x)])
    bot = np.hstack([np.flipud(d1_x), np.flipud(np.fliplr(d1_x))])
    d1_y = np.vstack([top, bot])

    test_x = np.array([[4, 5], [6, 7]], dtype=int)
    test_top = np.hstack([test_x, np.fliplr(test_x)])
    test_bot = np.hstack([np.flipud(test_x), np.flipud(np.fliplr(test_x))])
    test_y = np.vstack([test_top, test_bot])

    task = ARCTask(
        task_id="w056_kaleidoscope_duplication",
        train_pairs=[(d1_x, d1_y)],
        test_input=test_x,
        test_output=test_y,
    )

    harness = ARCEvaluationHarness()
    record = harness.evaluate_task(task)

    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is True
    assert record.test_pixel_accuracy == 1.0
    assert "duplicate_2x2" in (record.selected_rule_id or "")


def test_motif_autocorrelation_and_tiling_w036_w106() -> None:
    """Verify W036 & W106: Autocorrelation period discovery and 2D wallpaper tiling."""
    unit = np.array([[1, 2], [2, 1]], dtype=int)
    # Tile 3x3 times -> 6x6 canvas
    y1 = np.tile(unit, (3, 3))
    x1 = np.zeros((6, 6), dtype=int)
    x1[:2, :2] = unit  # prompt provides motif kernel in corner

    y2 = np.tile(unit, (4, 4))
    x2 = np.zeros((8, 8), dtype=int)
    x2[:2, :2] = unit

    test_y = np.tile(unit, (5, 5))
    test_x = np.zeros((10, 10), dtype=int)
    test_x[:2, :2] = unit

    task = ARCTask(
        task_id="w036_w106_wallpaper_tiling",
        train_pairs=[(x1, y1), (x2, y2)],
        test_input=test_x,
        test_output=test_y,
    )

    harness = ARCEvaluationHarness()
    record = harness.evaluate_task(task)

    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is True
    assert record.test_pixel_accuracy == 1.0
    assert "tiling_p2x2" in (record.selected_rule_id or "")


def test_relational_graph_matcher_w038() -> None:
    """Verify W038: Constraint-aware relational graph matching and step-budget bounding."""
    from hbllm.hcir.world.object_state_graph import (
        RelationalEdge,
        RelationalGraphMatcher,
        RelationalNode,
    )

    # Source graph: 3 objects in triangle: Key (color 1), Switch (color 2), Door (color 3)
    # Key is adjacent to Switch; Switch is connected to Door.
    src_nodes = [
        RelationalNode("k1", color=1, area=4, centroid=(1.0, 1.0), bounding_box=(0, 2, 0, 2)),
        RelationalNode("s1", color=2, area=4, centroid=(1.0, 5.0), bounding_box=(0, 2, 4, 6)),
        RelationalNode("d1", color=3, area=9, centroid=(5.0, 5.0), bounding_box=(4, 7, 4, 7)),
    ]
    src_edges = [
        RelationalEdge("k1", "s1", "adjacent"),
        RelationalEdge("s1", "d1", "triggers"),
    ]

    # Target graph: includes a distractor (color 1, wrong connection) and correct objects
    tgt_nodes = [
        RelationalNode(
            "k_distractor", color=1, area=4, centroid=(9.0, 9.0), bounding_box=(8, 10, 8, 10)
        ),
        RelationalNode("k_true", color=1, area=4, centroid=(0.0, 0.0), bounding_box=(0, 2, 0, 2)),
        RelationalNode("s_true", color=2, area=4, centroid=(0.0, 4.0), bounding_box=(0, 2, 3, 5)),
        RelationalNode("d_true", color=3, area=9, centroid=(4.0, 4.0), bounding_box=(3, 6, 3, 6)),
    ]
    tgt_edges = [
        RelationalEdge("k_distractor", "d_true", "adjacent"),  # Wrong edge!
        RelationalEdge("k_true", "s_true", "adjacent"),
        RelationalEdge("s_true", "d_true", "triggers"),
    ]

    matcher = RelationalGraphMatcher(max_steps=1000)
    mapping = matcher.find_mapping(src_nodes, src_edges, tgt_nodes, tgt_edges)

    assert mapping is not None
    assert mapping["k1"] == "k_true"
    assert mapping["s1"] == "s_true"
    assert mapping["d1"] == "d_true"


def test_morphological_deformation_tracker_w059() -> None:
    """Verify W059: Euler characteristic chi, IoU, and UUID preservation across extrusion."""
    from hbllm.hcir.world.spatiotemporal_tracker import (
        MorphologicalDeformationTracker,
        MorphologicalDeformationType,
        MorphologicalEntity,
    )

    # 1. Topological Invariant Test: 3x3 solid block has chi = 1 (0 holes)
    solid_block = {(r, c) for r in range(3) for c in range(3)}
    assert MorphologicalDeformationTracker.compute_euler_characteristic(solid_block) == 1

    # Hollow 3x3 box (1 hole) has chi = 0
    hollow_box = set(solid_block) - {(1, 1)}
    assert MorphologicalDeformationTracker.compute_euler_characteristic(hollow_box) == 0

    # 2. Continuous identity tracking across non-rigid extrusion
    pe = MorphologicalEntity(
        entity_id="entity_alpha",
        feature_id=4,
        cells={(2, 2), (2, 3)},
        centroid=(2.0, 2.5),
        bounding_box=(2, 3, 2, 4),
    )
    # Extruded into length 5 ray
    ce = MorphologicalEntity(
        entity_id="entity_temp_beta",
        feature_id=4,
        cells={(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)},
        centroid=(2.0, 4.0),
        bounding_box=(2, 3, 2, 7),
    )

    tracker = MorphologicalDeformationTracker()
    mapping, records = tracker.match_entities([pe], [ce], iou_threshold=0.15)

    assert mapping.get("entity_alpha") == "entity_temp_beta"
    assert len(records) == 1
    assert records[0].deformation_type == MorphologicalDeformationType.EXTRUSION


def test_falsifiable_cross_task_analogy_w148() -> None:
    """Verify W148: Falsifiable schema transfer with structural checks and empirical refutation."""
    from hbllm.hcir.world.cortex_episodic import (
        AnalogyStatus,
        FalsifiableAnalogyEngine,
    )

    engine = FalsifiableAnalogyEngine(refutation_threshold=1)

    # Transfer schema: Game TU93 (Key -> Door) to Game AR25 (Switch -> Barrier)
    morphism = engine.propose_analogy(
        source_domain="TU93",
        target_domain="AR25",
        predicate_mapping={"Key": "Switch", "Door": "Barrier"},
    )

    # Check 1: Structural Compatibility
    assert (
        engine.validate_structural_compatibility(morphism, {"Switch", "Barrier", "Avatar"}) is True
    )
    # If target environment is missing "Barrier", it should be rejected immediately
    morphism_bad = engine.propose_analogy(
        source_domain="TU93",
        target_domain="AT44",
        predicate_mapping={"Key": "Switch", "Door": "Barrier"},
    )
    assert engine.validate_structural_compatibility(morphism_bad, {"Switch", "Avatar"}) is False
    assert morphism_bad.status == AnalogyStatus.REJECTED

    # Check 2: Empirical Observation Testing
    analogy_id = "TU93__to__AR25"
    # Observation confirms predicted state transition
    confirmed_step = engine.record_empirical_observation(
        analogy_id, predicted_mutation="CLEAR_BARRIER", actual_mutation="CLEAR_BARRIER"
    )
    assert confirmed_step is True
    assert engine.is_analogy_active(analogy_id) is True

    # Counterexample contradicts hypothesis -> Popperian rejection
    engine.record_empirical_observation(
        analogy_id, predicted_mutation="CLEAR_BARRIER", actual_mutation="NO_EFFECT"
    )
    assert engine.is_analogy_active(analogy_id) is False
    assert morphism.status == AnalogyStatus.REJECTED


def test_arc_adapter_plugin_conversion() -> None:
    """Verify plugins/arc_agi_adapter converts official ARC JSON schema into core DemonstrationTask."""
    from plugins.arc_agi_adapter.arc_evaluator import (
        ARCBenchmarkRunner,
        ARCTaskAdapter,
    )

    arc_json_dict = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 4], [1, 2]]},
            {"input": [[5, 6], [7, 8]], "output": [[7, 8], [5, 6]]},
        ],
        "test": [{"input": [[9, 0], [1, 2]], "output": [[1, 2], [9, 0]]}],
    }

    task = ARCTaskAdapter.from_arc_dict(arc_json_dict, task_id="arc_001_sample")
    assert isinstance(task, DemonstrationTask)
    assert len(task.train_pairs) == 2
    assert task.test_input.shape == (2, 2)
    assert task.test_output is not None and task.test_output.shape == (2, 2)

    runner = ARCBenchmarkRunner()
    record = runner.evaluate_task(task)
    assert record.train_fit_rate == 1.0
    assert record.test_exact_match is True
    assert record.test_pixel_accuracy == 1.0
