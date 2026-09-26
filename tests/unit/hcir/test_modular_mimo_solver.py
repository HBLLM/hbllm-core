"""Unit tests for Modular MIMO Linear System Identifier and Solver."""

from __future__ import annotations

import numpy as np

from plugins.arc_agi_adapter.inductive_learner import (
    CoupledMIMOIdentifier,
    DynamicPermutationSolver,
)


def test_solve_gf2_binary_linear_system() -> None:
    """Verify solve_modular_linear_system over GF(2) toggles."""
    A = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=int)
    b = np.array([1, 0, 1], dtype=int)
    x = DynamicPermutationSolver.solve_modular_linear_system(A, b, modulus=2)
    assert x is not None
    assert np.array_equal((A @ x) % 2, b)


def test_solve_modular_linear_system_mod4() -> None:
    """Verify solve_modular_linear_system on coupled dials with modulus 4."""
    # Action 1 rotates dial 0 and 2
    # Action 2 rotates dial 0 and 1
    # Action 3 rotates dial 1 and 2
    A = np.array([[1, 2, 0], [0, 1, 1], [1, 0, 1]], dtype=int)
    expected_x = np.array([1, 2, 3], dtype=int)
    b = (A @ expected_x) % 4

    x = DynamicPermutationSolver.solve_modular_linear_system(A, b, modulus=4)
    assert x is not None
    assert np.array_equal((A @ x) % 4, b)


def test_coupled_mimo_identifier_plan() -> None:
    """Verify CoupledMIMOIdentifier learns transition vectors empirically and solves plan."""
    identifier = CoupledMIMOIdentifier(num_variables=3, modulus=4)

    # State transition for action 1: dial 0 += 1, dial 2 += 1
    s0 = np.array([0, 0, 0])
    s1 = np.array([1, 0, 1])
    identifier.register_transition(action=1, pre_state=s0, post_state=s1)

    # State transition for action 2: dial 0 += 2, dial 1 += 1
    s2 = np.array([2, 1, 0])
    identifier.register_transition(action=2, pre_state=s0, post_state=s2)

    # State transition for action 3: dial 1 += 1, dial 2 += 1
    s3 = np.array([0, 1, 1])
    identifier.register_transition(action=3, pre_state=s0, post_state=s3)

    # Target state: [1, 1, 0] (requires 1x action 1, 2x action 2, 3x action 3)
    target = np.array([1, 1, 0])
    plan = identifier.solve_plan(current_state=s0, target_state=target, available_actions=[1, 2, 3])

    assert len(plan) == 6
    assert plan.count(1) == 1
    assert plan.count(2) == 2
    assert plan.count(3) == 3
