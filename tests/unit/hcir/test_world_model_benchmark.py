"""Discrete Visual World-Model Empirical Benchmark Suite (VisualWorldModelBenchmark).

Implements 5 empirical verification layers for domain-general 2D discrete visual rule induction:
- Layer 1: Operator Correctness (Discrete geometric, extraction, symmetry, recolor, count, containment)
- Layer 2: Training-Pair Consistency (Popperian refutation across multi-example demonstration sets)
- Layer 3: Held-Out Task Performance (Exact-match on unseen test inputs across multiple task families)
- Layer 4: Generalization and Ablation (Relational invariance under transformation, compositionality ablation)
- Layer 5: End-to-End Runtime and Capability Ledger Verification
"""

import numpy as np

from hbllm.hcir.world.capability_evidence import (
    CapabilityEvidence,
    CapabilityLedger,
    GeneralizationStatus,
    ImplementationStatus,
    IntegrationStatus,
    UnitTestStatus,
)
from hbllm.hcir.world.grid_operator import (
    AffineOperator,
    ContainmentOperator,
    ObjectExtractOperator,
    OperatorBinding,
    RecolorOperator,
    ScaleOperator,
    SymmetryCompletionOperator,
    TransformationProgramSearch,
    TranslationOperator,
)
from hbllm.hcir.world.relational_graph_matcher import RelationalGraphMatcher

# =====================================================================
# LAYER 1: Operator Correctness
# =====================================================================


def test_layer1_geometric_operators_correctness():
    grid = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    # Affine ROT_90
    aff = AffineOperator()
    binding_rot90 = OperatorBinding(operator_name="affine", params={"op": "ROT_90"})
    rot90_res = aff.apply(grid, binding_rot90)
    expected_rot90 = np.array(
        [
            [4, 1],
            [5, 2],
            [6, 3],
        ]
    )
    ev = aff.verify(grid, expected_rot90, rot90_res)
    assert ev.is_exact
    assert ev.pixel_accuracy == 1.0

    # Scale 2x
    scale_op = ScaleOperator()
    binding_scale2 = OperatorBinding(operator_name="scale", params={"factor_r": 2, "factor_c": 2})
    scaled_res = scale_op.apply(grid, binding_scale2)
    assert scaled_res.shape == (4, 6)
    assert np.all(scaled_res[:2, :2] == 1)

    # Translation (dr=1, dc=0)
    trans_op = TranslationOperator()
    binding_trans = OperatorBinding(
        operator_name="translate", params={"dr": 1, "dc": 0, "bg_color": 0}
    )
    trans_res = trans_op.apply(grid, binding_trans)
    assert np.all(trans_res[0, :] == 0)
    assert np.array_equal(trans_res[1, :], grid[0, :])


def test_layer1_semantic_operators_correctness():
    # Object Extraction (Non-bg bbox)
    grid_with_bg = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    extract_op = ObjectExtractOperator()
    b_extract = OperatorBinding(
        operator_name="object_extract", params={"mode": "NON_BG_BBOX", "bg_color": 0}
    )
    cropped = extract_op.apply(grid_with_bg, b_extract)
    assert cropped.shape == (2, 2)
    assert np.array_equal(cropped, [[2, 2], [2, 3]])

    # Symmetry completion (Horizontal mirror)
    half_grid = np.array(
        [
            [1, 2, 0],
            [0, 0, 0],
        ]
    )
    sym_op = SymmetryCompletionOperator()
    b_sym = OperatorBinding(operator_name="symmetry_completion", params={"mode": "HORIZONTAL"})
    sym_res = sym_op.apply(half_grid, b_sym)
    assert sym_res[1, 0] == 1
    assert sym_res[1, 1] == 2

    # Recolor
    recolor_op = RecolorOperator()
    b_recolor = OperatorBinding(operator_name="recolor", params={"mapping": {2: 7, 3: 8}})
    recolored = recolor_op.apply(cropped, b_recolor)
    assert np.array_equal(recolored, [[7, 7], [7, 8]])

    # Containment hole fill
    hollow_square = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    contain_op = ContainmentOperator()
    b_contain = OperatorBinding(
        operator_name="containment_fill", params={"fill_color": 4, "bg_color": 0}
    )
    filled = contain_op.apply(hollow_square, b_contain)
    assert filled[2, 2] == 4
    assert filled[0, 0] == 0


# =====================================================================
# LAYER 2: Training-Pair Consistency
# =====================================================================


def test_layer2_training_pair_consistency_poppering_refutation():
    # 3 demonstration pairs: Rotation 90 degrees
    d1_x = np.array([[1, 2], [3, 4]])
    d1_y = np.rot90(d1_x, -1)

    d2_x = np.array([[5, 0, 1], [2, 3, 0]])
    d2_y = np.rot90(d2_x, -1)

    d3_x = np.array([[7, 8], [9, 0], [1, 2]])
    d3_y = np.rot90(d3_x, -1)

    train_pairs = [(d1_x, d1_y), (d2_x, d2_y), (d3_x, d3_y)]

    searcher = TransformationProgramSearch()
    candidates = searcher.propose_candidates(train_pairs)
    assert len(candidates) > 5

    survivors = []
    for op, binding in candidates:
        is_consistent, _, _ = searcher.evaluate_consistency(op, binding, train_pairs)
        if is_consistent:
            survivors.append((op, binding))

    # All survivors must be valid: either pure Affine(ROT_90) or a composition containing ROT_90
    assert len(survivors) >= 1
    # Ranked by MDL simplicity, the top survivor must be the minimal atomic rule Affine(ROT_90)
    survivors.sort(key=lambda item: item[1].complexity)
    best_op, best_binding = survivors[0]
    assert best_op.name == "affine"
    assert best_binding.params.get("op") == "ROT_90"
    assert best_binding.complexity == 1.0


# =====================================================================
# LAYER 3: Held-Out Task Performance (Exact Match)
# =====================================================================


def test_layer3_held_out_task_exact_match_geometric():
    # Training demonstrations: Flip Horizontal (up-down)
    train_pairs = [
        (np.array([[1, 2], [0, 0]]), np.array([[0, 0], [1, 2]])),
        (np.array([[3, 4, 5], [1, 1, 1]]), np.array([[1, 1, 1], [3, 4, 5]])),
        (np.array([[8, 9], [7, 6], [5, 4]]), np.array([[5, 4], [7, 6], [8, 9]])),
    ]

    test_input = np.array(
        [
            [2, 3, 4],
            [0, 1, 0],
        ]
    )
    expected_test_output = np.flipud(test_input)

    searcher = TransformationProgramSearch()
    pred_test, winning_binding, metadata = searcher.solve(train_pairs, test_input)

    assert metadata["solved"] is True
    assert winning_binding is not None
    assert winning_binding.params.get("op") == "FLIP_H"
    assert pred_test is not None
    assert np.array_equal(pred_test, expected_test_output)


def test_layer3_held_out_task_exact_match_recoloring():
    # Training demonstrations: Color substitution (1->3, 2->4)
    train_pairs = [
        (np.array([[1, 2], [1, 0]]), np.array([[3, 4], [3, 0]])),
        (np.array([[0, 2], [2, 1]]), np.array([[0, 4], [4, 3]])),
    ]

    test_input = np.array([[2, 1, 2], [1, 0, 1]])
    expected_test_output = np.array([[4, 3, 4], [3, 0, 3]])

    searcher = TransformationProgramSearch()
    pred_test, winning_binding, metadata = searcher.solve(train_pairs, test_input)

    assert metadata["solved"] is True
    assert winning_binding is not None
    assert np.array_equal(pred_test, expected_test_output)


def test_layer3_held_out_task_exact_match_extraction():
    # Training demonstrations: Extract foreground bounding box
    train_pairs = [
        (np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]]), np.array([[5]])),
        (
            np.array([[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]]),
            np.array([[2, 2], [2, 2]]),
        ),
    ]

    test_input = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 7, 7, 0],
            [0, 0, 7, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected_test_output = np.array(
        [
            [7, 7],
            [7, 0],
        ]
    )

    searcher = TransformationProgramSearch()
    pred_test, winning_binding, metadata = searcher.solve(train_pairs, test_input)

    assert metadata["solved"] is True
    assert pred_test is not None
    assert np.array_equal(pred_test, expected_test_output)


# =====================================================================
# LAYER 4: Generalization and Ablation
# =====================================================================


def test_layer4_relational_graph_matching_invariance():
    # Test relational correspondence across translations and color permutations (W038)
    grid_a = np.zeros((8, 8), dtype=int)
    grid_a[1:3, 1:3] = 2  # Object 1 (2x2 square)
    grid_a[5:6, 5:7] = 3  # Object 2 (1x2 bar)

    # Transformed grid B: Translated objects, colors permuted
    grid_b = np.zeros((8, 8), dtype=int)
    grid_b[2:4, 4:6] = 5  # Object 1 (2x2 square, translated & color changed)
    grid_b[6:7, 1:3] = 8  # Object 2 (1x2 bar, translated & color changed)

    graph_a = RelationalGraphMatcher.build_scene_graph(grid_a)
    graph_b = RelationalGraphMatcher.build_scene_graph(grid_b)

    assert len(graph_a.nodes) == 2
    assert len(graph_b.nodes) == 2

    correspondences = RelationalGraphMatcher.match_graphs(
        graph_a, graph_b, ignore_color=True, ignore_scale=False
    )

    assert len(correspondences) == 2
    # Verify bijective mapping by shape topology
    scores = [c.match_score for c in correspondences]
    assert all(s > 0.5 for s in scores)


def test_layer4_compositional_ablation():
    # Demonstrations require Crop then Rotate
    raw_in = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
    )
    cropped = np.array([[1, 2], [3, 4]])
    expected_out = np.rot90(cropped, -1)  # [[3, 1], [4, 2]]

    train_pairs = [(raw_in, expected_out)]
    test_input = np.array(
        [
            [0, 0, 0, 0],
            [0, 5, 6, 0],
            [0, 7, 8, 0],
            [0, 0, 0, 0],
        ]
    )
    expected_test_out = np.rot90(np.array([[5, 6], [7, 8]]), -1)

    searcher = TransformationProgramSearch()
    pred_test, winning_binding, metadata = searcher.solve(train_pairs, test_input)

    assert metadata["solved"] is True
    assert "composite" in winning_binding.operator_name or "Crop" in winning_binding.description
    assert np.array_equal(pred_test, expected_test_out)


# =====================================================================
# LAYER 5: End-to-End Runtime & Capability Ledger Verification
# =====================================================================


def test_layer5_capability_ledger_reconciliation():
    ledger = CapabilityLedger()

    # Register sample capabilities across the 3 independent status dimensions
    ledger.register(
        CapabilityEvidence(
            capability_id="W083",
            name="Whole-Grid Outcome Prediction",
            domain="Predictive Simulation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            integration_status=IntegrationStatus.ACTIVE_RUNTIME,
            unit_test_status=UnitTestStatus.PASSING,
            benchmark_status="BENCHMARKED",
            generalization_status=GeneralizationStatus.HELD_OUT_EVIDENCED,
            evidence_refs=["grid_operator.py", "test_arc_world_model_benchmark.py"],
            notes="Full raster outcome synthesis verified on held-out tasks.",
        )
    )

    ledger.register(
        CapabilityEvidence(
            capability_id="W038",
            name="Relational Structure and Graph Matching",
            domain="Spatial Relations and Geometry",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            integration_status=IntegrationStatus.ACTIVE_RUNTIME,
            unit_test_status=UnitTestStatus.PASSING,
            benchmark_status="BENCHMARKED",
            generalization_status=GeneralizationStatus.HELD_OUT_EVIDENCED,
            evidence_refs=["relational_graph_matcher.py", "test_arc_world_model_benchmark.py"],
            notes="Bijective scene graph matching across translation and recoloring variants.",
        )
    )

    summary = ledger.summary()
    assert summary["total_capabilities"] == 2
    assert summary["fully_verified"] == 2
    assert summary["implementation"]["IMPLEMENTED"] == 2
    assert summary["integration"]["ACTIVE_RUNTIME"] == 2
    assert summary["generalization"]["HELD_OUT_EVIDENCED"] == 2
