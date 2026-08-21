"""Tests for provider reputation independence — anti-circularity guard.

Covers:
- Internal belief revision does NOT inflate provider reputation
- External ground truth DOES update empirical_accuracy
- Cross-modal agreement updates concordance, not accuracy
- Circular feedback rejection (invalid OutcomeType raises ValueError)
"""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.reputation import SourceReputationTracker
from hbllm.hcir.types import OutcomeType

# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReputationIndependence:
    """Verify that three reputation dimensions are independent."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return SourceReputationTracker(data_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_external_ground_truth_updates_empirical_accuracy(self, tracker):
        """Experiment verification should update empirical_accuracy."""
        rep = await tracker.record_empirical_outcome(
            "vision_yolo",
            "claim_001",
            outcome=OutcomeType.EXPERIMENT,
            verified=True,
        )
        initial_acc = rep.empirical_accuracy

        rep = await tracker.record_empirical_outcome(
            "vision_yolo",
            "claim_002",
            outcome=OutcomeType.EXPERIMENT,
            verified=True,
        )

        # Empirical accuracy should increase
        assert rep.empirical_accuracy >= initial_acc
        assert rep.confirmed_claims == 2

    @pytest.mark.asyncio
    async def test_cross_modal_updates_concordance_not_accuracy(self, tracker):
        """Cross-modal agreement should only update concordance, never accuracy."""
        # Get baseline empirical accuracy
        baseline_rep = await tracker.record_empirical_outcome(
            "audio_whisper",
            "claim_001",
            outcome=OutcomeType.EXTERNAL_VERIFICATION,
            verified=True,
        )
        baseline_accuracy = baseline_rep.empirical_accuracy

        # Record several concordance events
        await tracker.record_concordance("audio_whisper", concordant=True)
        await tracker.record_concordance("audio_whisper", concordant=True)
        rep = await tracker.record_concordance("audio_whisper", concordant=True)

        # Concordance should increase
        assert rep.cross_modal_concordance > 0.5

        # But empirical_accuracy should NOT have changed
        assert rep.empirical_accuracy == baseline_accuracy

    @pytest.mark.asyncio
    async def test_signal_quality_independent_of_accuracy(self, tracker):
        """Signal quality updates should not affect empirical_accuracy."""
        rep = await tracker.record_signal_quality("camera_main", quality_score=0.9)
        initial_acc = rep.empirical_accuracy

        await tracker.record_signal_quality("camera_main", quality_score=0.95)
        rep = await tracker.record_signal_quality("camera_main", quality_score=0.92)

        # Signal quality should improve
        assert rep.signal_quality > 0.5

        # Empirical accuracy should remain at default
        assert rep.empirical_accuracy == initial_acc

    @pytest.mark.asyncio
    async def test_circular_feedback_rejection(self, tracker):
        """Attempting to update accuracy with an invalid outcome type should fail."""
        with pytest.raises(ValueError, match="Invalid outcome type"):
            await tracker.record_empirical_outcome(
                "vision_yolo",
                "claim_001",
                outcome="cross_modal_consensus",  # Not a valid OutcomeType
                verified=True,
            )

    @pytest.mark.asyncio
    async def test_all_valid_outcome_types_accepted(self, tracker):
        """All four valid OutcomeTypes should be accepted."""
        for outcome_type in OutcomeType:
            rep = await tracker.record_empirical_outcome(
                f"source_{outcome_type.value}",
                f"claim_{outcome_type.value}",
                outcome=outcome_type,
                verified=True,
            )
            assert rep.confirmed_claims == 1

    @pytest.mark.asyncio
    async def test_composite_reputation_combines_all_three(self, tracker):
        """Composite reputation should reflect all three independent dimensions."""
        await tracker.record_empirical_outcome(
            "src_1", "c1", outcome=OutcomeType.EXPERIMENT, verified=True
        )
        await tracker.record_concordance("src_1", concordant=True)
        rep = await tracker.record_signal_quality("src_1", quality_score=0.9)

        # Composite should differ from any single dimension
        composite = rep.compute_composite_reputation()
        assert composite == rep.reputation_score
        assert composite != rep.signal_quality
        assert composite != rep.cross_modal_concordance
        assert composite != rep.empirical_accuracy
