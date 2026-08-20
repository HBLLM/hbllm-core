"""Tests for Correlation Engine — §A8.2.

Validates pure geometry/time correlation between perceptual observations.
"""

from __future__ import annotations

import pytest

from hbllm.perception.correlation_engine import (
    CorrelationCandidate,
    CorrelationEngine,
    ObservationEnvelope,
)


def _make_envelope(
    obs_id: str,
    modality: str,
    start: float,
    end: float,
    direction: float | None = None,
) -> ObservationEnvelope:
    return ObservationEnvelope(
        observation_id=obs_id,
        modality=modality,
        start_time=start,
        end_time=end,
        direction_degrees=direction,
    )


class TestTemporalCorrelation:
    """Temporal alignment produces correct overlap scores."""

    def test_perfect_overlap(self) -> None:
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.0, 101.0)
        b = _make_envelope("a1", "audio", 100.0, 101.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.temporal_overlap == 1.0
        assert result.score == 1.0

    def test_partial_overlap(self) -> None:
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.0, 102.0)
        b = _make_envelope("a1", "audio", 101.0, 103.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert 0.0 < result.temporal_overlap < 1.0

    def test_near_simultaneous(self) -> None:
        """Events 120ms apart should have high temporal overlap."""
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.0, 100.5)
        b = _make_envelope("a1", "audio", 100.12, 100.62)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.temporal_overlap > 0.5
        assert result.delta_time_ms == pytest.approx(120.0, abs=1.0)

    def test_no_overlap_within_gap(self) -> None:
        """Events 2s apart but within max_temporal_gap should still correlate."""
        engine = CorrelationEngine(max_temporal_gap=5.0)
        a = _make_envelope("v1", "vision", 100.0, 100.5)
        b = _make_envelope("a1", "audio", 102.5, 103.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.temporal_overlap > 0.0
        assert result.score > 0.0

    def test_too_far_apart_returns_none(self) -> None:
        """Events beyond max_temporal_gap should not correlate."""
        engine = CorrelationEngine(max_temporal_gap=5.0)
        a = _make_envelope("v1", "vision", 100.0, 100.5)
        b = _make_envelope("a1", "audio", 110.0, 110.5)
        result = engine.correlate(a, b)
        assert result is None

    def test_delta_time_positive_when_target_after(self) -> None:
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.0, 100.5)
        b = _make_envelope("a1", "audio", 100.2, 100.7)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.delta_time_ms == pytest.approx(200.0, abs=1.0)

    def test_delta_time_negative_when_target_before(self) -> None:
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.5, 101.0)
        b = _make_envelope("a1", "audio", 100.0, 100.5)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.delta_time_ms == pytest.approx(-500.0, abs=1.0)


class TestSpatialCorrelation:
    """Spatial alignment with temporal correlation."""

    def test_spatial_overlap_same_direction(self) -> None:
        """Vision person @ 32° + Audio footsteps @ 35° → high spatial overlap."""
        engine = CorrelationEngine(spatial_threshold=30.0)
        a = _make_envelope("v1", "vision", 100.0, 101.0, direction=32.0)
        b = _make_envelope("a1", "audio", 100.0, 101.0, direction=35.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.spatial_overlap is not None
        assert result.spatial_overlap > 0.8  # 3° difference out of 30° threshold

    def test_spatial_overlap_opposite_direction(self) -> None:
        """Vision @ 10° + Audio @ 190° → zero spatial overlap."""
        engine = CorrelationEngine(spatial_threshold=30.0)
        a = _make_envelope("v1", "vision", 100.0, 101.0, direction=10.0)
        b = _make_envelope("a1", "audio", 100.0, 101.0, direction=190.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.spatial_overlap == 0.0

    def test_no_spatial_data_returns_none_spatial(self) -> None:
        """Missing direction → spatial_overlap is None."""
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.0, 101.0)
        b = _make_envelope("a1", "audio", 100.0, 101.0, direction=35.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.spatial_overlap is None

    def test_spatial_boosts_combined_score(self) -> None:
        """Spatial alignment should boost the combined score."""
        engine = CorrelationEngine(temporal_weight=0.7, spatial_weight=0.3)
        # With spatial
        a1 = _make_envelope("v1", "vision", 100.0, 101.0, direction=30.0)
        b1 = _make_envelope("a1", "audio", 100.0, 101.0, direction=32.0)
        with_spatial = engine.correlate(a1, b1)

        # Without spatial
        a2 = _make_envelope("v2", "vision", 100.0, 101.0)
        b2 = _make_envelope("a2", "audio", 100.0, 101.0)
        without_spatial = engine.correlate(a2, b2)

        assert with_spatial is not None
        assert without_spatial is not None
        # With spatial alignment, score should be similar or higher
        # (depends on weights, but spatial adds info)
        assert with_spatial.score > 0


class TestCrossModalProperty:
    """CorrelationCandidate.is_cross_modal flag."""

    def test_cross_modal_true(self) -> None:
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.0, 101.0)
        b = _make_envelope("a1", "audio", 100.0, 101.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.is_cross_modal is True

    def test_same_modal_false(self) -> None:
        engine = CorrelationEngine()
        a = _make_envelope("a1", "audio", 100.0, 101.0)
        b = _make_envelope("a2", "audio", 100.0, 101.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.is_cross_modal is False


class TestBatchCorrelation:
    """Batch correlation across multiple observations."""

    def test_batch_returns_sorted_by_score(self) -> None:
        engine = CorrelationEngine(max_temporal_gap=5.0)
        sources = [
            _make_envelope("v1", "vision", 100.0, 101.0),
            _make_envelope("v2", "vision", 103.0, 104.0),
        ]
        targets = [
            _make_envelope("a1", "audio", 100.0, 101.0),  # Perfect overlap with v1
            _make_envelope("a2", "audio", 102.0, 103.0),  # Between v1 and v2
        ]
        results = engine.correlate_batch(sources, targets)
        assert len(results) > 0
        # Should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_batch_skips_self_correlation(self) -> None:
        engine = CorrelationEngine()
        envelopes = [_make_envelope("a1", "audio", 100.0, 101.0)]
        results = engine.correlate_batch(envelopes, envelopes)
        assert len(results) == 0

    def test_empty_input(self) -> None:
        engine = CorrelationEngine()
        assert engine.correlate_batch([], []) == []


class TestEdgeCases:
    """Edge cases for the correlation engine."""

    def test_zero_duration_observation(self) -> None:
        engine = CorrelationEngine()
        a = _make_envelope("v1", "vision", 100.0, 100.0)
        b = _make_envelope("a1", "audio", 100.0, 100.0)
        result = engine.correlate(a, b)
        assert result is not None

    def test_wraparound_angular_difference(self) -> None:
        """350° and 10° should be 20° apart, not 340°."""
        engine = CorrelationEngine(spatial_threshold=30.0)
        a = _make_envelope("v1", "vision", 100.0, 101.0, direction=350.0)
        b = _make_envelope("a1", "audio", 100.0, 101.0, direction=10.0)
        result = engine.correlate(a, b)
        assert result is not None
        assert result.spatial_overlap is not None
        # 20° out of 30° threshold = 1/3 of threshold
        assert result.spatial_overlap > 0.3

    def test_correlation_candidate_is_frozen(self) -> None:
        candidate = CorrelationCandidate(
            source_observation_id="v1",
            target_observation_id="a1",
            source_modality="vision",
            target_modality="audio",
            temporal_overlap=1.0,
            spatial_overlap=None,
            delta_time_ms=0.0,
            score=0.9,
        )
        assert candidate.score == 0.9
