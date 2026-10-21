"""Unit Test Suite: Milestone 3 Phase 1 Global Invariant Induction.

Validates the 5 Core Abstraction Induction Properties:
1. AST Independence: Task generator does not import solver or GridOperator modules.
2. Parity Invariant Discovery: Induces N % 2 object-count and pixel-count rules.
3. Ternary Modular Generalization: Induces N % 3 cycle without binary hardcoding.
4. Out-of-Construction Generalization: Validates synthesized operators on unseen inputs.
5. Epistemic Rejection: Returns None when demonstrations exhibit no consistent invariant.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from experiments.benchmarks.task_generators.global_invariant_generator import (
    GlobalInvariantTaskGenerator,
)
from hbllm.hcir.world.global_invariant_induction import (
    GlobalScalarPropertyExtractor,
    InductiveGlobalInvariantSynthesizer,
)


def test_global_invariant_generator_ast_independence():
    """Verify that GlobalInvariantTaskGenerator does not import solver or GridOperator."""
    gen_module = inspect.getmodule(GlobalInvariantTaskGenerator)
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

    assert not any("grid_operator" in m for m in imported_modules)
    assert not any("representation_expansion" in m for m in imported_modules)
    assert not any("global_invariant_induction" in m for m in imported_modules)


def test_property_extractor_correctness():
    """Verify connected component object count and pixel count extraction."""
    grid = np.zeros((10, 10), dtype=int)
    # Object 1 (2x2 block at 0,0)
    grid[0:2, 0:2] = 1
    # Object 2 (cross at 5,5)
    grid[5, 6] = 2
    grid[6, 5:8] = 2
    grid[7, 6] = 2

    assert GlobalScalarPropertyExtractor.extract_object_count(grid) == 2
    assert GlobalScalarPropertyExtractor.extract_pixel_count(grid) == 4 + 5  # 9
    assert GlobalScalarPropertyExtractor.extract_color_count(grid) == 2


def test_inductive_parity_discovery_and_heldout_application():
    """Verify synthesis of N % 2 rule from demonstration residuals."""
    # Demo 1: 3 objects -> recolor to 5
    d1_x = np.zeros((10, 10), dtype=int)
    d1_x[1, 1] = 1
    d1_x[3, 3] = 1
    d1_x[5, 5] = 1
    d1_y = np.zeros((10, 10), dtype=int)
    d1_y[d1_x != 0] = 5

    # Demo 2: 2 objects -> identity (remain 1)
    d2_x = np.zeros((10, 10), dtype=int)
    d2_x[2, 2] = 1
    d2_x[7, 7] = 1
    d2_y = np.copy(d2_x)

    train_pairs = [(d1_x, d1_y), (d2_x, d2_y)]
    synthesized_op = InductiveGlobalInvariantSynthesizer.synthesize_from_demonstrations(train_pairs)

    assert synthesized_op is not None
    assert synthesized_op.property_name == "object_count"
    assert synthesized_op.modulus == 2

    # Test query 1: 5 objects (odd) -> should recolor to 5
    test_odd = np.zeros((10, 10), dtype=int)
    for i in range(5):
        test_odd[i, i] = 1
    binding = synthesized_op.propose({}, {"train_pairs": train_pairs})[0]
    pred_odd = synthesized_op.apply(test_odd, binding)
    assert np.all(pred_odd[test_odd != 0] == 5)

    # Test query 2: 4 objects (even) -> should remain 1
    test_even = np.zeros((10, 10), dtype=int)
    for i in range(4):
        test_even[i, i] = 1
    pred_even = synthesized_op.apply(test_even, binding)
    assert np.array_equal(pred_even, test_even)


def test_ternary_modulo_3_generalization():
    """Verify synthesis of N % 3 ternary cycle rule without binary hardcoding."""
    # Demo 1: 1 object (% 3 == 1) -> recolor to 2
    d1_x = np.zeros((8, 8), dtype=int)
    d1_x[1, 1] = 1
    d1_y = np.zeros((8, 8), dtype=int)
    d1_y[1, 1] = 2

    # Demo 2: 2 objects (% 3 == 2) -> recolor to 3
    d2_x = np.zeros((8, 8), dtype=int)
    d2_x[1, 1] = 1
    d2_x[3, 3] = 1
    d2_y = np.zeros((8, 8), dtype=int)
    d2_y[d2_x != 0] = 3

    # Demo 3: 3 objects (% 3 == 0) -> identity
    d3_x = np.zeros((8, 8), dtype=int)
    d3_x[1, 1] = 1
    d3_x[3, 3] = 1
    d3_x[5, 5] = 1
    d3_y = np.copy(d3_x)

    train_pairs = [(d1_x, d1_y), (d2_x, d2_y), (d3_x, d3_y)]
    synthesized_op = InductiveGlobalInvariantSynthesizer.synthesize_from_demonstrations(train_pairs)

    assert synthesized_op is not None
    assert synthesized_op.modulus == 3

    # Test on unseen 4 objects (4 % 3 == 1) -> must map to color 2
    test_4 = np.zeros((8, 8), dtype=int)
    for i in range(4):
        test_4[i, i] = 1
    binding = synthesized_op.propose({}, {"train_pairs": train_pairs})[0]
    pred_4 = synthesized_op.apply(test_4, binding)
    assert np.all(pred_4[test_4 != 0] == 2)


def test_epistemic_rejection_on_inconsistent_residuals():
    """Verify that synthesizer safely returns None when demonstrations violate modular laws."""
    # Demo 1: 2 objects -> recolored to 5
    d1_x = np.zeros((8, 8), dtype=int)
    d1_x[1, 1] = 1
    d1_x[3, 3] = 1
    d1_y = np.zeros((8, 8), dtype=int)
    d1_y[d1_x != 0] = 5

    # Demo 2: 2 objects -> recolored to 7 (contradiction: identical input structure, conflicting outputs)
    d2_x = np.zeros((8, 8), dtype=int)
    d2_x[2, 2] = 1
    d2_x[4, 4] = 1
    d2_y = np.zeros((8, 8), dtype=int)
    d2_y[d2_x != 0] = 7

    train_pairs = [(d1_x, d1_y), (d2_x, d2_y)]
    op = InductiveGlobalInvariantSynthesizer.synthesize_from_demonstrations(train_pairs)
    assert op is None
