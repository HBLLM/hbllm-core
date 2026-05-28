"""
Tests for ChangeDetector Rust extension.

Verifies perceptual hash change detection, threshold limits, and caching.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image


def make_test_image(
    color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (64, 64)
) -> bytes:
    """Helper to generate PNG image bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_change_detector_integration() -> None:
    """Test the ChangeDetector class exposed by the Rust extension."""
    try:
        import hbllm_perception_rs  # type: ignore
    except ImportError:
        pytest.skip("hbllm_perception_rs Rust extension not compiled/installed yet.")

    # Initialize with default threshold
    detector = hbllm_perception_rs.ChangeDetector(threshold=5)
    assert detector.get_threshold() == 5

    img1 = make_test_image((255, 255, 255))
    img2 = make_test_image((255, 255, 255))
    img3 = make_test_image((0, 0, 0))

    # First frame is always considered changed
    assert detector.is_changed(img1) is True

    # Identical frame is unchanged
    assert detector.is_changed(img2) is False

    # Different frame is changed
    assert detector.is_changed(img3) is True

    # Resetting should make the next frame changed
    detector.reset()
    assert detector.is_changed(img3) is True
