"""
Tests for Multimodal Vector Projector.

Verifies projection fallback logic, zero-padding/truncation, and sensor sorting.
"""

from __future__ import annotations

import pytest

from hbllm.perception.vector_projector import MultimodalProjector


def test_projector_fallback_padding() -> None:
    """Verify fallback padding/truncation to llm_dim (default 4096)."""
    projector = MultimodalProjector(llm_dim=10)

    # 1. Padding: input shorter than target
    input_short = [1.0, 2.0, 3.0]
    padded = projector.project_vision(input_short)
    assert len(padded) == 10
    assert padded[:3] == [1.0, 2.0, 3.0]
    assert padded[3:] == [0.0] * 7

    # 2. Truncation: input longer than target
    input_long = list(range(15))
    truncated = projector.project_vision(input_long)
    assert len(truncated) == 10
    assert truncated == list(range(10))

    # 3. Exact: input matches target
    input_exact = list(range(10))
    exact = projector.project_vision(input_exact)
    assert exact == input_exact


def test_projector_sensor_sorting() -> None:
    """Verify sensor readings are sorted alphabetically before projection."""
    projector = MultimodalProjector(llm_dim=5)

    readings = {"temp": 24.5, "humidity": 60.0, "pressure": 1013.25}
    projected = projector.project_sensor(readings)

    # Sorted order of keys: 'humidity' (60.0), 'pressure' (1013.25), 'temp' (24.5)
    # Expected fallback padding to size 5:
    assert len(projected) == 5
    assert projected[:3] == [60.0, 1013.25, 24.5]
    assert projected[3:] == [0.0, 0.0]
