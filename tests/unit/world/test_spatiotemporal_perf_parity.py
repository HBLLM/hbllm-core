"""Parity tests for the vectorised hot paths in spatiotemporal_tracker.

``PhasePrecessionEstimator.estimate_periods_batch`` must be semantically
identical to running the scalar ``estimate_period`` on every column, and the
rasterised ``MorphologicalDeformationTracker.compute_euler_characteristic``
must match a brute-force V - E + F count over the cubical cell complex.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.world.spatiotemporal_tracker import (
    MorphologicalDeformationTracker,
    PhasePrecessionEstimator,
)


def _scalar_periods(
    series_matrix: np.ndarray, max_period: int, min_match_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise reference: run ``estimate_period`` on each column, mapping None -> 0."""
    periods = np.zeros(series_matrix.shape[1], dtype=np.int64)
    confidences = np.zeros(series_matrix.shape[1], dtype=np.float64)
    for j in range(series_matrix.shape[1]):
        period, confidence = PhasePrecessionEstimator.estimate_period(
            series_matrix[:, j].tolist(),
            max_period=max_period,
            min_match_ratio=min_match_ratio,
        )
        periods[j] = period if period is not None else 0
        confidences[j] = confidence
    return periods, confidences


def _assert_period_parity(series_matrix: np.ndarray, max_period: int = 8) -> None:
    expected_t, expected_conf = _scalar_periods(series_matrix, max_period, 0.85)
    got_t, got_conf = PhasePrecessionEstimator.estimate_periods_batch(
        series_matrix, max_period=max_period
    )
    assert got_t.tolist() == expected_t.tolist()
    assert got_conf == pytest.approx(expected_conf)


def test_estimate_periods_batch_matches_scalar_on_random_series() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.integers(0, 4, size=(12, 50))
    _assert_period_parity(matrix)

    short = rng.integers(0, 4, size=(5, 9))
    _assert_period_parity(short)


def test_estimate_periods_batch_matches_scalar_on_periodic_series() -> None:
    n = 24
    series = [
        [i % 2 for i in range(n)],  # period 2
        [i % 3 for i in range(n)],  # period 3
        [i % 4 for i in range(n)],  # period 4
        [(i // 2) % 3 for i in range(n)],  # period 6
        [0 if i % 4 < 2 else 1 for i in range(n)],  # period 4, 50% duty cycle
        [7] * n,  # constant: matches every lag at ratio 1.0
        [i % 3 + (i // 8) for i in range(n)],  # drifting: no clean periodicity
    ]
    matrix = np.asarray(series, dtype=np.int64).T
    _assert_period_parity(matrix)

    periods, confidences = PhasePrecessionEstimator.estimate_periods_batch(matrix)
    assert periods.tolist()[:5] == [2, 3, 4, 6, 4]
    assert confidences[:5] == pytest.approx([1.0] * 5)


def _reference_euler_characteristic(cells: set[tuple[int, int]]) -> int:
    """Brute-force chi = V - E + F over the cubical complex of occupied cells."""
    vertices: set[tuple[int, int]] = set()
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in cells:
        vertices.update(((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)))
        for a, b in (
            ((r, c), (r, c + 1)),  # top horizontal edge
            ((r + 1, c), (r + 1, c + 1)),  # bottom horizontal edge
            ((r, c), (r + 1, c)),  # left vertical edge
            ((r, c + 1), (r + 1, c + 1)),  # right vertical edge
        ):
            edges.add((a, b) if a <= b else (b, a))
    return len(vertices) - len(edges) + len(cells)


def test_euler_characteristic_matches_bruteforce_on_random_masks() -> None:
    rng = np.random.default_rng(0)
    for _ in range(25):
        h, w = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        mask = rng.integers(0, 2, size=(h, w)).astype(bool)
        cells = {(int(rc[0]), int(rc[1])) for rc in np.argwhere(mask)}
        assert MorphologicalDeformationTracker.compute_euler_characteristic(
            cells
        ) == _reference_euler_characteristic(cells)


def test_euler_characteristic_structured_cases() -> None:
    cases = {
        "empty": set(),
        "single": {(0, 0)},
        "solid_block": {(r, c) for r in range(4) for c in range(4)},
        "ring_one_hole": {(r, c) for r in range(3) for c in range(3)} - {(1, 1)},
        "two_disconnected_cells": {(0, 0), (3, 3)},
        "shifted_negative_origin": {(r - 7, c + 11) for r in range(3) for c in range(2)},
    }
    expected_chi = {
        "empty": 0,
        "single": 1,
        "solid_block": 1,
        "ring_one_hole": 0,
        "two_disconnected_cells": 2,
        "shifted_negative_origin": 1,
    }
    for name, cells in cases.items():
        reference = _reference_euler_characteristic(cells)
        assert reference == expected_chi[name], name
        assert MorphologicalDeformationTracker.compute_euler_characteristic(cells) == reference, (
            name
        )
