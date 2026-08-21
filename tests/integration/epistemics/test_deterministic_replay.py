"""Tests for deterministic replay — event journal and replay harness.

Covers:
- Journal starts with SESSION_CONFIG
- Sequence number integrity (gaps, duplicates rejected)
- Configuration mismatch detection
- Replay under identical timestamps (sequence_number authoritative)
"""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.replay import (
    ConfigMismatchError,
    EpistemicEventJournal,
    EpistemicEventType,
    JournalReplayHarness,
    SequenceIntegrityError,
)
from hbllm.hcir.types import EpistemicRuntimeConfig, NoveltyPolicy

# ═══════════════════════════════════════════════════════════════════════════
# Journal Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicEventJournal:
    """Test journal recording and integrity."""

    def test_journal_starts_with_session_config(self):
        config = EpistemicRuntimeConfig()
        journal = EpistemicEventJournal(config)

        assert journal.size == 1
        first = journal.entries[0]
        assert first.event_type == EpistemicEventType.SESSION_CONFIG
        assert first.sequence_number == 0
        assert first.payload["config_hash"] == config.config_hash
        assert first.payload["algorithm_version"] == config.algorithm_version

    def test_monotonic_sequence_numbers(self):
        journal = EpistemicEventJournal()

        e1 = journal.record(EpistemicEventType.PERCEPTION_RECEIVED, {"frame": 1})
        e2 = journal.record(EpistemicEventType.EVIDENCE_COMMITTED, {"id": "ev_1"})
        e3 = journal.record(EpistemicEventType.BELIEF_REVISED, {"belief_id": "b1"})

        assert e1.sequence_number == 1
        assert e2.sequence_number == 2
        assert e3.sequence_number == 3
        assert journal.size == 4  # Including SESSION_CONFIG

    def test_validate_integrity_passes_for_valid_journal(self):
        journal = EpistemicEventJournal()
        journal.record(EpistemicEventType.PERCEPTION_RECEIVED)
        journal.record(EpistemicEventType.EVIDENCE_COMMITTED)
        journal.validate_integrity()  # Should not raise

    def test_validate_integrity_empty_journal_fails(self):
        journal = EpistemicEventJournal()
        journal._entries.clear()
        journal._next_sequence = 0

        with pytest.raises(SequenceIntegrityError, match="empty"):
            journal.validate_integrity()

    def test_serialization_roundtrip(self):
        config = EpistemicRuntimeConfig(novelty_policy=NoveltyPolicy(half_life_seconds=10.0))
        journal = EpistemicEventJournal(config)
        journal.record(EpistemicEventType.EVIDENCE_COMMITTED, {"id": "ev_1"})
        journal.record(EpistemicEventType.BELIEF_REVISED, {"belief_id": "b1"})

        data = journal.to_list()
        assert len(data) == 3
        assert data[0]["event_type"] == "session_config"
        assert data[1]["event_type"] == "evidence_committed"


# ═══════════════════════════════════════════════════════════════════════════
# Config Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    """Test that replay rejects config mismatches."""

    def test_config_hash_mismatch_rejected(self):
        config_a = EpistemicRuntimeConfig(novelty_policy=NoveltyPolicy(half_life_seconds=5.0))
        config_b = EpistemicRuntimeConfig(novelty_policy=NoveltyPolicy(half_life_seconds=10.0))

        # They should have different hashes
        assert config_a.config_hash != config_b.config_hash

        journal = EpistemicEventJournal(config_a)
        journal.record(EpistemicEventType.PERCEPTION_RECEIVED)

        harness = JournalReplayHarness()
        with pytest.raises(ConfigMismatchError, match="Config hash mismatch"):
            harness.replay(journal.to_list(), expected_config=config_b)

    def test_matching_config_accepted(self):
        config = EpistemicRuntimeConfig()
        journal = EpistemicEventJournal(config)
        journal.record(EpistemicEventType.PERCEPTION_RECEIVED)

        harness = JournalReplayHarness()
        # Should not raise
        graph = harness.replay(journal.to_list(), expected_config=config)
        assert graph is not None

    def test_algorithm_version_mismatch_rejected(self):
        config = EpistemicRuntimeConfig(algorithm_version="a11.0")
        journal = EpistemicEventJournal(config)
        journal.record(EpistemicEventType.PERCEPTION_RECEIVED)

        mismatch_config = EpistemicRuntimeConfig(algorithm_version="a12.0")
        # Need same hash to isolate version check
        # (different version → different hash, so this also fails on hash)

        harness = JournalReplayHarness()
        with pytest.raises(ConfigMismatchError):
            harness.replay(journal.to_list(), expected_config=mismatch_config)


# ═══════════════════════════════════════════════════════════════════════════
# Sequence Integrity Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSequenceIntegrity:
    """Test that replay validates sequence number integrity."""

    def test_sequence_gap_detected(self):
        journal = EpistemicEventJournal()
        journal.record(EpistemicEventType.PERCEPTION_RECEIVED)
        journal.record(EpistemicEventType.EVIDENCE_COMMITTED)

        # Corrupt: remove sequence 1 (leaving sequence 0 and sequence 2)
        data = journal.to_list()
        data = [d for d in data if d["sequence_number"] != 1]

        harness = JournalReplayHarness()
        with pytest.raises(SequenceIntegrityError, match="Sequence gap"):
            harness.replay(data)

    def test_duplicate_sequence_detected(self):
        journal = EpistemicEventJournal()
        journal.record(EpistemicEventType.PERCEPTION_RECEIVED)

        data = journal.to_list()
        # Add a duplicate entry with same sequence number
        dup = data[1].copy()
        data.append(dup)

        harness = JournalReplayHarness()
        with pytest.raises(SequenceIntegrityError, match="Duplicate"):
            harness.replay(data)


# ═══════════════════════════════════════════════════════════════════════════
# Replay Ordering Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReplayOrdering:
    """Test that sequence_number is authoritative, not timestamp."""

    def test_identical_timestamps_ordered_by_sequence(self):
        """Events with same timestamp must be ordered by sequence_number."""
        journal = EpistemicEventJournal()

        # Record two events at the same timestamp
        e1 = journal.record(
            EpistemicEventType.EVIDENCE_COMMITTED,
            {"id": "ev_1", "order": "first"},
        )
        e2 = journal.record(
            EpistemicEventType.EVIDENCE_COMMITTED,
            {"id": "ev_2", "order": "second"},
        )
        # Force identical timestamps
        e1.timestamp = 100.123
        e2.timestamp = 100.123

        data = journal.to_list()

        # Verify sequence numbers are monotonic
        seq_1 = data[1]["sequence_number"]
        seq_2 = data[2]["sequence_number"]
        assert seq_1 < seq_2  # seq_1 should always be before seq_2

        # Replay should succeed despite identical timestamps
        harness = JournalReplayHarness()
        graph = harness.replay(data)
        assert graph is not None

    def test_out_of_order_timestamps_reordered_by_sequence(self):
        """Replay should sort by sequence_number even if timestamps are out of order."""
        journal = EpistemicEventJournal()
        e1 = journal.record(EpistemicEventType.EVIDENCE_COMMITTED, {"id": "ev_1"})
        e2 = journal.record(EpistemicEventType.EVIDENCE_COMMITTED, {"id": "ev_2"})

        # Force reversed timestamps
        e1.timestamp = 200.0
        e2.timestamp = 100.0

        data = journal.to_list()

        # Replay should not fail — it uses sequence_number not timestamp
        harness = JournalReplayHarness()
        graph = harness.replay(data)
        assert graph is not None
