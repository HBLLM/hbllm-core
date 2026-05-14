import asyncio
import time

import pytest

from hbllm.memory.cache import CacheEntry, ResponseCache


@pytest.mark.asyncio
async def test_cache_put_get():
    cache = ResponseCache(max_size=10, ttl_seconds=60.0)
    await cache.put("key1", "value1")

    entry = await cache.get("key1")
    assert entry is not None
    assert entry.value == "value1"


@pytest.mark.asyncio
async def test_cache_miss():
    cache = ResponseCache(max_size=10, ttl_seconds=60.0)
    entry = await cache.get("missing")
    assert entry is None


@pytest.mark.asyncio
async def test_cache_eviction_lru():
    cache = ResponseCache(max_size=2, ttl_seconds=60.0)
    await cache.put("k1", "v1")
    await cache.put("k2", "v2")
    await cache.put("k3", "v3")  # Should evict k1

    assert await cache.get("k1") is None
    assert (await cache.get("k2")).value == "v2"
    assert (await cache.get("k3")).value == "v3"


@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    cache = ResponseCache(max_size=10, ttl_seconds=0.1)
    await cache.put("k1", "v1")

    time.sleep(0.15)

    assert await cache.get("k1") is None


@pytest.mark.asyncio
async def test_cache_invalidation():
    cache = ResponseCache(max_size=10, ttl_seconds=60.0)
    await cache.put("prefix_1", "v1")
    await cache.put("prefix_2", "v2")
    await cache.put("other", "v3")

    removed = cache.invalidate("prefix_")
    assert removed == 2

    assert await cache.get("prefix_1") is None
    assert await cache.get("prefix_2") is None
    assert (await cache.get("other")).value == "v3"


def test_cache_stats():
    cache = ResponseCache(max_size=5, ttl_seconds=60.0)
    stats = cache.stats()
    assert stats["max_size"] == 5
    assert stats["size"] == 0
    assert stats["default_ttl"] == 60.0
