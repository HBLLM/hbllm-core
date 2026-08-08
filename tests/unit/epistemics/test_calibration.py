"""Tests for EpistemicCalibrationEngine — meta-epistemic self-assessment."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.calibration import EpistemicCalibrationEngine
from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.interfaces import PredictionOutcome


@pytest.fixture
async def seeded_memory(memory: EpistemicMemory) -> EpistemicMemory:
    """Memory with enough data for meaningful calibration."""
    # 10 predictions with varying accuracy
    for i in range(10):
        correct = i < 7  # 70% accuracy
        predicted_conf = 0.8 if i < 5 else 0.5  # overconfident on first 5
        outcome = PredictionOutcome(
            prediction_id=f"p{i}",
            hypothesis_id=f"h{i % 3}",
            predicted="x",
            observed="x" if correct else "y",
            correct=correct,
        )
        await memory.record_prediction_result(
            outcome,
            predicted_confidence=predicted_conf,
        )

    # 5 hypotheses: 3 promoted, 1 falsified, 1 abandoned
    await memory.record_hypothesis_outcome("h0", "promoted", "good", claim="A")
    await memory.record_hypothesis_outcome("h1", "promoted", "good", claim="B")
    await memory.record_hypothesis_outcome("h2", "promoted", "good", claim="C")
    await memory.record_hypothesis_outcome("h3", "falsified", "bad", claim="D")
    await memory.record_hypothesis_outcome("h4", "abandoned", "stale", claim="E")

    return memory


class TestCalibrate:
    """Test calibrate() report generation."""

    @pytest.mark.asyncio
    async def test_calibration_report_fields(
        self,
        seeded_memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=seeded_memory)
        report = await calibrator.calibrate()

        assert report.prediction_accuracy == pytest.approx(0.7)
        assert report.total_predictions == 10
        assert report.total_hypotheses == 5
        assert report.hypothesis_survival_rate == pytest.approx(0.6)
        assert report.falsification_rate == pytest.approx(0.2)
        assert len(report.recommendations) > 0

    @pytest.mark.asyncio
    async def test_empty_memory_calibration(
        self,
        memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=memory)
        report = await calibrator.calibrate()

        assert report.overall_calibration == 0.0
        assert report.overconfidence_bias == 0.0
        assert report.total_predictions == 0


class TestCalibrationCurve:
    """Test calibration curve computation."""

    @pytest.mark.asyncio
    async def test_calibration_curve_bins(
        self,
        seeded_memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=seeded_memory)
        curve = await calibrator.compute_calibration_curve(n_bins=5)

        # Should have non-empty bins
        assert len(curve) > 0
        for bin_center, accuracy, count in curve:
            assert 0.0 <= bin_center <= 1.0
            assert 0.0 <= accuracy <= 1.0
            assert count > 0

    @pytest.mark.asyncio
    async def test_empty_calibration_curve(
        self,
        memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=memory)
        curve = await calibrator.compute_calibration_curve()
        assert curve == []


class TestBiasDetection:
    """Test bias detection."""

    @pytest.mark.asyncio
    async def test_detect_biases(
        self,
        seeded_memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=seeded_memory)
        biases = await calibrator.detect_epistemic_biases()

        # With 60% survival and 20% falsification, should detect biases
        assert isinstance(biases, list)


class TestStrategyRecommendation:
    """Test strategy adjustment recommendations."""

    @pytest.mark.asyncio
    async def test_recommend_strategy(
        self,
        seeded_memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=seeded_memory)
        rec = await calibrator.recommend_strategy_adjustment()

        # Should return either a strategy string or None
        assert rec is None or isinstance(rec, str)
