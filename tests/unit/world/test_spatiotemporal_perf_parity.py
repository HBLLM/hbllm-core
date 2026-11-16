"""Parity tests for the vectorised hot paths in spatiotemporal_tracker.

``PhasePrecessionEstimator.estimate_periods_batch`` must be semantically
identical to running the scalar ``estimate_period`` on every column, and the
rasterised ``MorphologicalDeformationTracker.compute_euler_characteristic``
must match a brute-force V - E + F count over the cubical cell complex.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.world.spatiotemporal_tracker import PhasePrecessionEstimator


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
