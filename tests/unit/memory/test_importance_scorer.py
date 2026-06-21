"""Tests for ImportanceScorer — Ebbinghaus-inspired memory decay."""

import time

import pytest

from hbllm.memory.importance_scorer import ImportanceConfig, ImportanceScorer, ScoredMemory


@pytest.fixture
def scorer():
    return ImportanceScorer()


def test_fresh_memory_high_importance(scorer):
    """A fresh, frequently accessed memory has high importance."""
    m = ScoredMemory(
        memory_id="m1",
        raw_importance=1.0,
        access_count=10,
        created_at=time.time() - 3600,  # 1 hour old
        last_accessed=time.time() - 60,
        memory_type="episodic",
    )
    scored = scorer.score(m)
    assert scored.computed_importance > 1.5
    assert not scored.should_archive


def test_old_unaccessed_memory_decays(scorer):
    """An old memory with no access decays significantly."""
    m = ScoredMemory(
        memory_id="m2",
        raw_importance=0.5,
        access_count=0,
        created_at=time.time() - 86400 * 90,  # 90 days old
        memory_type="episodic",
    )
    scored = scorer.score(m)
    assert scored.computed_importance < 0.2
    assert scored.should_archive


def test_emotional_memory_decays_slower(scorer):
    """Emotional memories decay slower than neutral ones."""
    now = time.time()
    base = ScoredMemory(
        memory_id="neutral",
        raw_importance=1.0,
        created_at=now - 86400 * 30,
        emotional_weight=0.0,
        memory_type="episodic",
    )
    emotional = ScoredMemory(
        memory_id="emotional",
        raw_importance=1.0,
        created_at=now - 86400 * 30,
        emotional_weight=0.8,
        memory_type="episodic",
    )
    scorer.score(base, now)
    scorer.score(emotional, now)
    assert emotional.computed_importance > base.computed_importance


def test_procedural_memories_have_longer_halflife():
    """Procedural memories have a longer half-life than episodic."""
    config = ImportanceConfig()
    assert config.procedural_half_life > config.episodic_half_life


def test_access_reinforcement_increases_importance(scorer):
    """More access count increases importance."""
    m0 = ScoredMemory(memory_id="a", raw_importance=1.0, access_count=0, created_at=time.time())
    m10 = ScoredMemory(memory_id="b", raw_importance=1.0, access_count=10, created_at=time.time())
    scorer.score(m0)
    scorer.score(m10)
    assert m10.computed_importance > m0.computed_importance


def test_reinforcement_cap(scorer):
    """Reinforcement bonus is capped at max_reinforcement."""
    m = ScoredMemory(memory_id="x", raw_importance=1.0, access_count=1000, created_at=time.time())
    scored = scorer.score(m)
    # Should not be astronomically high
    assert scored.computed_importance < 10.0


def test_rank_memories(scorer):
    """rank_memories returns memories sorted by importance (highest first)."""
    now = time.time()
    memories = [
        ScoredMemory(memory_id="low", raw_importance=0.1, created_at=now - 86400 * 60),
        ScoredMemory(memory_id="high", raw_importance=1.0, access_count=5, created_at=now),
        ScoredMemory(memory_id="mid", raw_importance=0.5, created_at=now - 86400 * 10),
    ]
    ranked = scorer.rank_memories(memories, now)
    assert ranked[0].memory_id == "high"
    assert ranked[-1].memory_id == "low"


def test_consolidate_splits_keep_and_archive(scorer):
    """consolidate returns separate keep and archive lists."""
    now = time.time()
    memories = [
        ScoredMemory(memory_id="keep", raw_importance=1.0, access_count=5, created_at=now),
        ScoredMemory(memory_id="archive", raw_importance=0.1, created_at=now - 86400 * 120),
    ]
    keep, archive = scorer.consolidate(memories, now)
    keep_ids = {m.memory_id for m in keep}
    assert "keep" in keep_ids


def test_importance_never_below_floor(scorer):
    """Importance never drops below the floor value."""
    m = ScoredMemory(
        memory_id="ancient",
        raw_importance=0.001,
        access_count=0,
        created_at=time.time() - 86400 * 365,  # 1 year old
        memory_type="episodic",
    )
    scored = scorer.score(m)
    assert scored.computed_importance >= scorer.config.floor


def test_custom_config():
    """Custom configuration overrides defaults."""
    config = ImportanceConfig(episodic_half_life=7.0, archival_threshold=0.5)
    scorer = ImportanceScorer(config)
    assert scorer.config.episodic_half_life == 7.0
    assert scorer.config.archival_threshold == 0.5


def test_stats(scorer):
    """Stats returns configuration summary."""
    s = scorer.stats()
    assert "half_lives" in s
    assert s["half_lives"]["episodic"] == 30.0
