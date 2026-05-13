"""
Intelligent Response Cache for HBLLM Core.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class CacheEntry:
    def __init__(self, key: str, value: Any, ttl: float | None = None):
        self.key = key
        self.value = value
        self.timestamp = time.monotonic()
        self.ttl = ttl

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.monotonic() - self.timestamp) > self.ttl


class ResponseCache:
    """
    LRU Cache for responses with TTL support.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheEntry | None:
        async with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry

    async def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        async with self._lock:
            ttl_to_use = ttl if ttl is not None else self.ttl_seconds
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = CacheEntry(key, value, ttl_to_use)

            if len(self._cache) > self.max_size:
                # remove oldest
                self._cache.popitem(last=False)

    async def get_similar(self, query: str, threshold: float = 0.85) -> CacheEntry | None:
        # Semantic similarity matching would go here.
        # For now, falls back to exact match or None.
        return await self.get(query)

    def invalidate(self, pattern: str) -> int:
        """Invalidate entries by simple substring matching."""
        to_remove = [k for k in self._cache.keys() if pattern in k]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.ttl_seconds,
        }
