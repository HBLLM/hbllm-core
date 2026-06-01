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
    def __init__(self, key: str, value: Any, ttl: float | None = None, embedding: Any = None):
        self.key = key
        self.value = value
        self.timestamp = time.monotonic()
        self.ttl = ttl
        self.embedding = embedding

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.monotonic() - self.timestamp) > self.ttl


class ResponseCache:
    """
    LRU Cache for responses with TTL and Semantic Cosine Similarity support.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600, embedder: Any = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.embedder = embedder
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

            embedding = None
            if self.embedder is not None:
                try:
                    # SemanticMemory._encode expects a list of texts and returns a stacked array
                    # We grab the first index for our single query key
                    embedding = self.embedder._encode([key])[0]
                except Exception as e:
                    logger.debug("Failed to pre-compute cache entry embedding: %s", e)

            self._cache[key] = CacheEntry(key, value, ttl_to_use, embedding=embedding)

            if len(self._cache) > self.max_size:
                # remove oldest
                self._cache.popitem(last=False)

    async def get_similar(self, query: str, threshold: float = 0.88) -> CacheEntry | None:
        # Check exact match first
        exact_entry = await self.get(query)
        if exact_entry is not None:
            return exact_entry

        if self.embedder is None:
            return None

        try:
            import numpy as np

            # Compute query embedding
            query_emb = self.embedder._encode([query])[0]
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                return None

            best_entry = None
            max_sim = -1.0

            async with self._lock:
                for k, entry in list(self._cache.items()):
                    if entry.is_expired:
                        del self._cache[k]
                        continue

                    if entry.embedding is None:
                        continue

                    entry_norm = np.linalg.norm(entry.embedding)
                    if entry_norm == 0:
                        continue

                    # Vectorized cosine similarity
                    sim = np.dot(query_emb, entry.embedding) / (query_norm * entry_norm + 1e-9)

                    if sim > max_sim:
                        max_sim = sim
                        best_entry = entry

                if max_sim >= threshold and best_entry is not None:
                    logger.info("🎯 Semantic Cache Hit! (Similarity = %.2f)", max_sim)
                    self._cache.move_to_end(best_entry.key)
                    return best_entry
        except Exception as e:
            logger.warning("Semantic Cache Similarity lookup failed: %s", e)

        return None

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
