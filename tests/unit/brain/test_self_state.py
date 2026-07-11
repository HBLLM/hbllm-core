"""Tests for Introspective SelfStateEngine."""

from __future__ import annotations

from hbllm.brain.self_model.self_state import EpistemicCalibrationTracker, ToolReliabilityTracker


def test_tool_ewma_reliability():
    """Ensure EWMA reliably tracks tool successes and failures."""
    tracker = ToolReliabilityTracker(alpha=0.2)

    assert tracker.get_reliability("test_tool") == 0.8

    # Simulate a failure
    tracker.record_execution("test_tool", success=False)

    # EWMA formula: (0.2 * 0) + (0.8 * 0.8) = 0.64
    assert round(tracker.get_reliability("test_tool"), 2) == 0.64

    # Simulate success
    tracker.record_execution("test_tool", success=True)

    # EWMA formula: (0.2 * 1) + (0.8 * 0.64) = 0.2 + 0.512 = 0.712
    assert round(tracker.get_reliability("test_tool"), 3) == 0.712


def test_epistemic_calibration():
    """Ensure Epistemic calibration updates via verification."""
    tracker = EpistemicCalibrationTracker()

    assert tracker.get_calibration("simulation.physics") == 0.7

    # Match success
    tracker.record_verification("simulation.physics", True, True, match=True)

    # EWMA formula: (0.15 * 1.0) + (0.85 * 0.7) = 0.15 + 0.595 = 0.745
    assert round(tracker.get_calibration("simulation.physics"), 3) == 0.745
