"""Tests for EpisodicMemory — conversation turn storage and retrieval."""

import pytest
from hbllm.memory.episodic import EpisodicMemory


@pytest.fixture
def ep_mem(tmp_path):
    return EpisodicMemory(db_path=tmp_path / "test_episodic.db")


def test_store_and_retrieve(ep_mem):
    """Store turns and retrieve them in chronological order."""
    ep_mem.store_turn("s1", "user", "Hello")
    ep_mem.store_turn("s1", "assistant", "Hi there!")

    turns = ep_mem.retrieve_recent("s1")
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"


def test_session_isolation(ep_mem):
    """Turns in different sessions don't mix."""
    ep_mem.store_turn("s1", "user", "Session 1 message")
    ep_mem.store_turn("s2", "user", "Session 2 message")

    turns_s1 = ep_mem.retrieve_recent("s1")
    turns_s2 = ep_mem.retrieve_recent("s2")
    assert len(turns_s1) == 1
    assert len(turns_s2) == 1
    assert "Session 1" in turns_s1[0]["content"]


def test_tenant_isolation(ep_mem):
    """Turns are scoped to tenant_id."""
    ep_mem.store_turn("s1", "user", "Tenant A", tenant_id="a")
    ep_mem.store_turn("s1", "user", "Tenant B", tenant_id="b")

    assert len(ep_mem.retrieve_recent("s1", tenant_id="a")) == 1
    assert len(ep_mem.retrieve_recent("s1", tenant_id="b")) == 1


def test_clear_session(ep_mem):
    """Clear all turns in a session."""
    ep_mem.store_turn("s1", "user", "msg1")
    ep_mem.store_turn("s1", "user", "msg2")
    ep_mem.store_turn("s2", "user", "other session")

    deleted = ep_mem.clear_session("s1")
    assert deleted == 2
    assert len(ep_mem.retrieve_recent("s1")) == 0
    assert len(ep_mem.retrieve_recent("s2")) == 1


def test_retrieve_limit(ep_mem):
    """Limit controls how many turns are returned."""
    for i in range(20):
        ep_mem.store_turn("s1", "user", f"Message {i}")

    turns = ep_mem.retrieve_recent("s1", limit=5)
    assert len(turns) == 5


def test_domain_tag(ep_mem):
    """Turns can be tagged with a domain."""
    ep_mem.store_turn("s1", "assistant", "Here is your code", domain="coding")
    ep_mem.store_turn("s1", "assistant", "The answer is 42", domain="math")

    turns = ep_mem.retrieve_recent("s1")
    assert turns[0]["domain"] == "coding"
    assert turns[1]["domain"] == "math"


def test_search_by_content(ep_mem):
    """Search across sessions by content keyword."""
    ep_mem.store_turn("s1", "user", "How to sort a list in Python?")
    ep_mem.store_turn("s2", "user", "What is the weather today?")
    ep_mem.store_turn("s3", "user", "Python decorators explained")

    results = ep_mem.search_by_content("Python")
    assert len(results) == 2


def test_retrieve_by_domain(ep_mem):
    """Retrieve turns by domain across sessions."""
    ep_mem.store_turn("s1", "assistant", "Code snippet 1", domain="coding")
    ep_mem.store_turn("s2", "assistant", "Code snippet 2", domain="coding")
    ep_mem.store_turn("s3", "assistant", "Math result", domain="math")

    coding = ep_mem.retrieve_by_domain("coding")
    assert len(coding) == 2
    math = ep_mem.retrieve_by_domain("math")
    assert len(math) == 1


def test_stats(ep_mem):
    """Session and turn count stats."""
    ep_mem.store_turn("s1", "user", "a")
    ep_mem.store_turn("s1", "assistant", "b")
    ep_mem.store_turn("s2", "user", "c")

    assert ep_mem.get_turn_count() == 3
    assert ep_mem.get_session_count() == 2


def test_cleanup_old_turns(ep_mem):
    """Cleanup removes only old turns."""
    ep_mem.store_turn("s1", "user", "recent message")
    # Cleanup with 0 days would remove everything
    deleted = ep_mem.cleanup_old_turns(days=0)
    assert deleted == 1
    assert ep_mem.get_turn_count() == 0


def test_empty_retrieve(ep_mem):
    """Retrieving from empty DB returns empty list."""
    assert ep_mem.retrieve_recent("nonexistent") == []
    assert ep_mem.search_by_content("anything") == []
    assert ep_mem.retrieve_by_domain("coding") == []
