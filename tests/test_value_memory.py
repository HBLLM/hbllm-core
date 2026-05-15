import pytest_asyncio
"""Tests for Value/Reward Memory — preference tracking."""

import pytest

from hbllm.memory.value_memory import ValueMemory


@pytest_asyncio.fixture
async def val_mem(tmp_path):
    mem = ValueMemory(db_path=tmp_path / "test.db")
    await mem.init_db()
    yield mem
    await mem.close()


@pytest.mark.asyncio
async def test_record_and_query_reward(val_mem):
    """Record a reward and query preferences for the topic."""
    await val_mem.record_reward("t1", "response_style", "formal_tone", 0.8)
    await val_mem.record_reward("t1", "response_style", "casual_tone", -0.5)

    prefs = await val_mem.get_preference("t1", "response_style")
    assert "formal_tone" in prefs
    assert "casual_tone" in prefs
    assert prefs["formal_tone"] > prefs["casual_tone"]


@pytest.mark.asyncio
async def test_tenant_isolation(val_mem):
    """Rewards are scoped to tenant_id."""
    await val_mem.record_reward("t1", "style", "verbose", 0.9)
    await val_mem.record_reward("t2", "style", "concise", 0.9)

    prefs_t1 = await val_mem.get_preference("t1", "style")
    assert "verbose" in prefs_t1
    assert "concise" not in prefs_t1


@pytest.mark.asyncio
async def test_reward_clamping(val_mem):
    """Rewards are clamped to [-1, 1]."""
    await val_mem.record_reward("t1", "test", "extreme", 5.0)
    await val_mem.record_reward("t1", "test", "negative", -3.0)

    prefs = await val_mem.get_preference("t1", "test")
    assert prefs["extreme"] <= 1.0
    assert prefs["negative"] >= -1.0


@pytest.mark.asyncio
async def test_top_preferences(val_mem):
    """Top preferences returns ranked (topic, action) pairs."""
    await val_mem.record_reward("t1", "style", "formal", 0.9)
    await val_mem.record_reward("t1", "style", "formal", 0.8)
    await val_mem.record_reward("t1", "length", "short", 0.5)
    await val_mem.record_reward("t1", "domain", "coding", -0.3)

    top = await val_mem.get_top_preferences("t1", top_k=3)
    assert len(top) == 3
    assert top[0]["avg_reward"] >= top[1]["avg_reward"]


@pytest.mark.asyncio
async def test_recency_weighting(val_mem):
    """More recent signals should carry more weight."""
    # Old signal: positive
    await val_mem.record_reward("t1", "topic", "action", 1.0)
    # New signal: negative (should weigh more)
    await val_mem.record_reward("t1", "topic", "action", -1.0)

    prefs = await val_mem.get_preference("t1", "topic")
    # The weighted average should lean negative due to recency
    assert prefs["action"] < 0


@pytest.mark.asyncio
async def test_signal_count(val_mem):
    """Total signal count per tenant."""
    await val_mem.record_reward("t1", "a", "b", 0.5)
    await val_mem.record_reward("t1", "c", "d", 0.5)
    await val_mem.record_reward("t2", "e", "f", 0.5)

    assert await val_mem.get_signal_count("t1") == 2
    assert await val_mem.get_signal_count("t2") == 1


@pytest.mark.asyncio
async def test_empty_preferences(val_mem):
    """No rewards returns empty dict."""
    assert await val_mem.get_preference("nonexistent", "topic") == {}
    assert await val_mem.get_top_preferences("nonexistent") == []
