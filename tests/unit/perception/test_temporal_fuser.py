"""Tests for TemporalFuser — cross-time perception event correlation."""

import time

import pytest

from hbllm.perception.temporal_fuser import (
    BUILTIN_PATTERNS,
    FusedSequence,
    PerceptionSnapshot,
    SequencePattern,
    TemporalFuser,
)


@pytest.fixture
def fuser():
    return TemporalFuser(window_s=60.0)


def test_builtin_patterns_loaded():
    """Built-in patterns are defined and have expected properties."""
    assert len(BUILTIN_PATTERNS) >= 8
    names = {p.name for p in BUILTIN_PATTERNS}
    assert "person_entered" in names
    assert "fire_danger" in names
    assert "glass_break_intrusion" in names


def test_single_event_no_sequence(fuser):
    """A single event should not trigger any sequences."""
    e = PerceptionSnapshot(event_type="iot.door", sub_type="opened", room="front")
    sequences = fuser.ingest(e)
    assert len(sequences) == 0


def test_person_entered_detection(fuser):
    """Door opened + motion detected triggers person_entered."""
    e1 = PerceptionSnapshot(event_type="iot.door", sub_type="opened", room="front")
    e2 = PerceptionSnapshot(event_type="iot.motion", sub_type="detected", room="hallway")
    fuser.ingest(e1)
    sequences = fuser.ingest(e2)
    assert len(sequences) >= 1
    assert sequences[0].pattern_name == "person_entered"
    assert "front" in sequences[0].narrative


def test_fire_danger_detection():
    """Temperature spike + smoke alarm triggers fire_danger."""
    fuser = TemporalFuser(window_s=180.0)
    e1 = PerceptionSnapshot(event_type="iot.temperature", sub_type="high", room="kitchen")
    e2 = PerceptionSnapshot(event_type="audio.ambient", sub_type="smoke_detector", room="kitchen")
    fuser.ingest(e1)
    sequences = fuser.ingest(e2)
    assert len(sequences) >= 1
    assert sequences[0].pattern_name == "fire_danger"


def test_glass_break_intrusion():
    """Glass breaking + motion triggers intrusion alert."""
    fuser = TemporalFuser(window_s=60.0)
    e1 = PerceptionSnapshot(event_type="audio.ambient", sub_type="glass_breaking", room="window")
    e2 = PerceptionSnapshot(event_type="iot.motion", sub_type="detected", room="window")
    fuser.ingest(e1)
    sequences = fuser.ingest(e2)
    assert len(sequences) >= 1
    assert sequences[0].pattern_name == "glass_break_intrusion"


def test_cooldown_prevents_duplicate(fuser):
    """Same pattern shouldn't fire again within cooldown period."""
    e1 = PerceptionSnapshot(event_type="iot.door", sub_type="opened", room="front")
    e2 = PerceptionSnapshot(event_type="iot.motion", sub_type="detected", room="front")
    fuser.ingest(e1)
    seq1 = fuser.ingest(e2)
    assert len(seq1) >= 1

    # Immediately try again
    e3 = PerceptionSnapshot(event_type="iot.door", sub_type="opened", room="front")
    e4 = PerceptionSnapshot(event_type="iot.motion", sub_type="detected", room="front")
    fuser.ingest(e3)
    seq2 = fuser.ingest(e4)
    assert len(seq2) == 0  # Cooldown prevents re-fire


def test_window_pruning():
    """Events outside the window are pruned."""
    fuser = TemporalFuser(window_s=5.0)
    old = PerceptionSnapshot(
        event_type="iot.door",
        sub_type="opened",
        timestamp=time.time() - 10.0,
    )
    fuser.ingest(old)
    assert fuser.stats()["events_in_window"] == 0


def test_unrelated_events_no_match(fuser):
    """Unrelated events don't trigger patterns."""
    e1 = PerceptionSnapshot(event_type="iot.light", sub_type="on")
    e2 = PerceptionSnapshot(event_type="iot.light", sub_type="off")
    fuser.ingest(e1)
    sequences = fuser.ingest(e2)
    assert len(sequences) == 0


def test_get_recent_events(fuser):
    """Recent events are retrievable for introspection."""
    for i in range(5):
        fuser.ingest(PerceptionSnapshot(event_type=f"test.{i}", sub_type="x"))
    events = fuser.get_recent_events(limit=3)
    assert len(events) == 3


def test_stats(fuser):
    """Stats returns expected telemetry."""
    fuser.ingest(PerceptionSnapshot(event_type="test", sub_type="x"))
    s = fuser.stats()
    assert s["events_processed"] == 1
    assert s["pattern_count"] == len(BUILTIN_PATTERNS)


def test_fused_sequence_to_dict():
    """FusedSequence.to_dict produces valid output."""
    seq = FusedSequence(
        sequence_id="test", pattern_name="fire_danger", narrative="Fire!", confidence=0.95
    )
    d = seq.to_dict()
    assert d["pattern_name"] == "fire_danger"
    assert d["confidence"] == 0.95


def test_custom_pattern():
    """Custom patterns can be added to the fuser."""
    custom = SequencePattern(
        name="test_pattern",
        steps=[
            {"event_type": "custom.a"},
            {"event_type": "custom.b"},
        ],
        max_window_s=30.0,
        narrative_template="Custom event in {room} ({duration:.0f}s).",
    )
    fuser = TemporalFuser(patterns=[custom])
    e1 = PerceptionSnapshot(event_type="custom.a", room="lab")
    e2 = PerceptionSnapshot(event_type="custom.b", room="lab")
    fuser.ingest(e1)
    seqs = fuser.ingest(e2)
    assert len(seqs) == 1
    assert seqs[0].pattern_name == "test_pattern"
