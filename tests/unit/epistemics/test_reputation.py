"""Tests for SourceReputationTracker — source trust tracking."""

from __future__ import annotations

import tempfile

import pytest

from hbllm.brain.epistemics.reputation import SourceReputationTracker


@pytest.fixture
def tracker() -> SourceReputationTracker:
    with tempfile.TemporaryDirectory() as td:
        yield SourceReputationTracker(data_dir=td)


class TestRecordOutcome:
    """Test outcome recording and reputation updates."""

    @pytest.mark.asyncio
    async def test_record_confirmed_raises_reputation(
        self,
        tracker: SourceReputationTracker,
    ) -> None:
        await tracker.record_outcome("src1", claim_id="c1", confirmed=True, domain="science")
        rep = await tracker.get_reputation("src1")
        assert rep >= 0.5  # Confirmed should keep or raise reputation

    @pytest.mark.asyncio
    async def test_record_refuted_lowers_reputation(
        self,
        tracker: SourceReputationTracker,
    ) -> None:
        # First make some confirmed to establish baseline
        for i in range(3):
            await tracker.record_outcome("src2", claim_id=f"c{i}", confirmed=True)
        rep_before = await tracker.get_reputation("src2")

        await tracker.record_outcome("src2", claim_id="c_bad", confirmed=False)
        rep_after = await tracker.get_reputation("src2")
        assert rep_after <= rep_before

    @pytest.mark.asyncio
    async def test_unknown_source_returns_default(
        self,
        tracker: SourceReputationTracker,
    ) -> None:
        rep = await tracker.get_reputation("nonexistent")
        assert rep == 0.5  # Default reputation


class TestSourceDetails:
    """Test source detail retrieval."""

    @pytest.mark.asyncio
    async def test_get_source_details(
        self,
        tracker: SourceReputationTracker,
    ) -> None:
        await tracker.record_outcome("src1", claim_id="c1", confirmed=True, domain="physics")
        await tracker.record_outcome("src1", claim_id="c2", confirmed=True, domain="physics")
        await tracker.record_outcome("src1", claim_id="c3", confirmed=False, domain="physics")

        details = await tracker.get_source_details("src1")
        assert details is not None
        assert details.total_claims == 3
        assert details.confirmed_claims == 2


class TestTopAndUnreliable:
    """Test ranking sources."""

    @pytest.mark.asyncio
    async def test_top_sources(
        self,
        tracker: SourceReputationTracker,
    ) -> None:
        # Create sources with different reputations
        for i in range(5):
            await tracker.record_outcome("good_src", claim_id=f"g{i}", confirmed=True)
        for i in range(5):
            await tracker.record_outcome("bad_src", claim_id=f"b{i}", confirmed=False)

        top = await tracker.get_top_sources(limit=2)
        assert len(top) <= 2
        if len(top) == 2:
            # Returns list of tuples (source_id, score)
            assert top[0][1] >= top[1][1]

    @pytest.mark.asyncio
    async def test_unreliable_sources(
        self,
        tracker: SourceReputationTracker,
    ) -> None:
        for i in range(5):
            await tracker.record_outcome("bad_src", claim_id=f"b{i}", confirmed=False)

        unreliable = await tracker.get_unreliable_sources(threshold=0.4)
        # Returns list of tuples (source_id, score)
        source_ids = [s[0] for s in unreliable]
        assert "bad_src" in source_ids
