"""Tests for the Utility Calibration Framework."""

from __future__ import annotations

import pytest

from hbllm.brain.evaluation.utility_calibrator import UtilityCalibrator


@pytest.fixture
def temp_calibrator(tmp_path):
    """Provides a UtilityCalibrator with a temporary DB."""
    return UtilityCalibrator(data_dir=tmp_path)


def test_record_and_get_traces(temp_calibrator):
    trace = temp_calibrator.record_trace(
        trace_id="test_t1",
        decision_point="planner:expand",
        predicted_utility=0.8,
        actual_outcome=0.6,
        metadata={"step": 1},
    )

    assert trace is not None
    assert trace.trace_id == "test_t1"
    assert abs(trace.prediction_error - 0.2) < 1e-6

    traces = temp_calibrator.get_traces()
    assert len(traces) == 1
    assert traces[0].trace_id == "test_t1"
    assert traces[0].predicted_utility == 0.8
    assert traces[0].actual_outcome == 0.6
    assert traces[0].metadata == {"step": 1}


def test_average_error_calculation(temp_calibrator):
    temp_calibrator.record_trace(
        trace_id="t1",
        decision_point="planner:expand",
        predicted_utility=0.8,
        actual_outcome=0.6,  # error = 0.2
    )
    temp_calibrator.record_trace(
        trace_id="t2",
        decision_point="planner:expand",
        predicted_utility=0.5,
        actual_outcome=0.7,  # error = -0.2 (abs = 0.2)
    )

    avg_err = temp_calibrator.get_average_error()
    assert abs(avg_err - 0.2) < 1e-6

    # Test filtering by decision point
    avg_err_filtered = temp_calibrator.get_average_error("planner:expand")
    assert abs(avg_err_filtered - 0.2) < 1e-6

    avg_err_miss = temp_calibrator.get_average_error("non_existent")
    assert avg_err_miss == 0.0
