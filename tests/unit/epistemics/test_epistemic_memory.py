"""Tests for EpistemicMemory — universal reasoning history."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.interfaces import ConfidenceSnapshot, PredictionOutcome


class TestRecordMethods:
    """Test all record_* methods."""

    @pytest.mark.asyncio
    async def test_record_hypothesis_outcome(self, memory: EpistemicMemory) -> None:
        await memory.record_hypothesis_outcome(
            "h1",
            "falsified",
            "Evidence contradicted",
            claim="X causes Y",
            program_id="prog1",
        )
        history = await memory.get_hypothesis_history()
        assert len(history) == 1
        assert history[0]["outcome"] == "falsified"
        assert history[0]["claim"] == "X causes Y"

    @pytest.mark.asyncio
    async def test_record_prediction_result(self, memory: EpistemicMemory) -> None:
        outcome = PredictionOutcome(
            prediction_id="p1",
            hypothesis_id="h1",
            predicted="increase",
            observed="increase",
            correct=True,
            confidence_delta=0.1,
        )
        await memory.record_prediction_result(outcome, predicted_confidence=0.7)
        accuracy = await memory.get_prediction_accuracy()
        assert accuracy == 1.0

    @pytest.mark.asyncio
    async def test_record_evidence_retraction(self, memory: EpistemicMemory) -> None:
        await memory.record_evidence_retraction(
            "ev1",
            "Data fabrication detected",
            quality_score=0.9,
            bias_flags=["fabrication"],
        )
        # Verify via total counts
        counts = await memory.get_total_counts()
        assert counts["evidence_history"] == 1

    @pytest.mark.asyncio
    async def test_snapshot_belief_confidence(self, memory: EpistemicMemory) -> None:
        snap = ConfidenceSnapshot(
            belief_id="b1",
            derived_confidence=0.75,
            evidence_quality=0.8,
            reproducibility=0.7,
        )
        await memory.snapshot_belief_confidence("b1", snap)
        trajectory = await memory.get_confidence_trajectory("b1")
        assert len(trajectory) == 1
        assert trajectory[0].derived_confidence == 0.75

    @pytest.mark.asyncio
    async def test_record_unknown_resolved(self, memory: EpistemicMemory) -> None:
        await memory.record_unknown_resolved(
            "u1",
            "Answered by experiment E3",
            question="Why does X happen?",
            program_id="prog1",
        )
        history = await memory.get_unknown_history(status="resolved")
        assert len(history) == 1
        assert history[0]["resolution"] == "Answered by experiment E3"


class TestQueryMethods:
    """Test query methods."""

    @pytest.mark.asyncio
    async def test_prediction_accuracy_multiple(self, memory: EpistemicMemory) -> None:
        for i, correct in enumerate([True, True, False, True]):
            outcome = PredictionOutcome(
                prediction_id=f"p{i}",
                hypothesis_id="h1",
                predicted="x",
                observed="y" if not correct else "x",
                correct=correct,
            )
            await memory.record_prediction_result(outcome)
        accuracy = await memory.get_prediction_accuracy()
        assert accuracy == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_prediction_accuracy_empty(self, memory: EpistemicMemory) -> None:
        accuracy = await memory.get_prediction_accuracy()
        assert accuracy == 0.0

    @pytest.mark.asyncio
    async def test_get_hypothesis_history_filter(self, memory: EpistemicMemory) -> None:
        await memory.record_hypothesis_outcome("h1", "falsified", "bad", claim="A")
        await memory.record_hypothesis_outcome("h2", "promoted", "good", claim="B")
        await memory.record_hypothesis_outcome("h3", "falsified", "bad", claim="C")

        falsified = await memory.get_hypothesis_history(outcome="falsified")
        assert len(falsified) == 2

        promoted = await memory.get_hypothesis_history(outcome="promoted")
        assert len(promoted) == 1

    @pytest.mark.asyncio
    async def test_confidence_trajectory_ordering(self, memory: EpistemicMemory) -> None:
        for i in range(3):
            snap = ConfidenceSnapshot(
                belief_id="b1",
                derived_confidence=0.5 + i * 0.1,
                timestamp=1000.0 + i,
            )
            await memory.snapshot_belief_confidence("b1", snap)

        traj = await memory.get_confidence_trajectory("b1")
        assert len(traj) == 3
        assert traj[0].derived_confidence < traj[2].derived_confidence

    @pytest.mark.asyncio
    async def test_survival_and_falsification_rate(self, memory: EpistemicMemory) -> None:
        await memory.record_hypothesis_outcome("h1", "promoted", "good")
        await memory.record_hypothesis_outcome("h2", "falsified", "bad")
        await memory.record_hypothesis_outcome("h3", "abandoned", "stale")
        await memory.record_hypothesis_outcome("h4", "promoted", "good")

        survival = await memory.get_hypothesis_survival_rate()
        assert survival == pytest.approx(0.5)

        falsification = await memory.get_falsification_rate()
        assert falsification == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_calibration_data(self, memory: EpistemicMemory) -> None:
        for i, correct in enumerate([True, False, True]):
            outcome = PredictionOutcome(
                prediction_id=f"p{i}",
                hypothesis_id="h1",
                predicted="x",
                observed="x" if correct else "y",
                correct=correct,
            )
            await memory.record_prediction_result(
                outcome,
                predicted_confidence=0.7 + i * 0.05,
                domain="physics",
            )
        data = await memory.get_calibration_data(domain="physics")
        assert len(data) == 3
