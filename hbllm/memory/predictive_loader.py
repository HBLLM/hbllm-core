"""
Predictive Memory Loader — Anticipatory memory pre-fetching.

Uses ``CognitivePredictors.query`` to anticipate which memories the
system will need next, and pre-fetches them into a warm cache before
they are explicitly requested.

This reduces retrieval latency by converting cache misses into cache
hits for predictable access patterns.

Architecture::

    User: "Write a unit test for the auth module"
        ↓
    CognitivePredictors.query.predict()
        → {"testing": 0.6, "debugging": 0.3, ...}
        ↓
    PredictiveLoader.prefetch(predicted_domains)
        → Pre-fetch testing + debugging memories into cache
        ↓
    Later: MemoryNode.retrieve("testing")
        → Cache HIT (already pre-fetched)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Cache entry
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CacheEntry:
    """A prefetched memory cache entry.

    Attributes:
        domain: The predicted domain this entry covers.
        results: Prefetched search results.
        timestamp: When this entry was created.
        hit_count: Number of times this entry was used.
    """

    domain: str
    results: list[Any]
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    ttl: float = 300.0  # Time-to-live in seconds

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl


# ═══════════════════════════════════════════════════════════════════════════
# PredictiveLoader
# ═══════════════════════════════════════════════════════════════════════════


class PredictiveLoader:
    """Anticipatory memory pre-fetcher using Markov predictions.

    Monitors the prediction distribution from ``CognitivePredictors``
    and pre-fetches memories for likely next queries.

    Args:
        fetch_fn: Async callable that retrieves memories for a domain.
            Signature: ``async def fetch(domain: str) -> list[Any]``
        confidence_threshold: Minimum prediction probability to trigger prefetch.
        max_cache_size: Maximum number of cached domains.
        cache_ttl: Time-to-live for cache entries in seconds.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str], Coroutine[Any, Any, list[Any]]] | None = None,
        confidence_threshold: float = 0.15,
        max_cache_size: int = 10,
        cache_ttl: float = 300.0,
    ) -> None:
        self._fetch_fn = fetch_fn
        self._confidence_threshold = confidence_threshold
        self._max_cache_size = max_cache_size
        self._cache_ttl = cache_ttl

        self._cache: dict[str, CacheEntry] = {}
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._prefetch_count: int = 0

    async def prefetch(self, predictions: dict[str, float]) -> int:
        """Pre-fetch memories for predicted domains.

        Only fetches domains above the confidence threshold that
        are not already cached.

        Args:
            predictions: Domain → probability distribution.

        Returns:
            Number of new domains prefetched.
        """
        if not self._fetch_fn:
            return 0

        fetched = 0
        # Sort by probability (descending)
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        for domain, prob in sorted_predictions:
            if prob < self._confidence_threshold:
                break

            # Skip if already cached and not expired
            if domain in self._cache and not self._cache[domain].is_expired:
                continue

            # Evict oldest if cache is full
            if len(self._cache) >= self._max_cache_size:
                self._evict_oldest()

            try:
                results = await self._fetch_fn(domain)
                self._cache[domain] = CacheEntry(
                    domain=domain,
                    results=results,
                    ttl=self._cache_ttl,
                )
                fetched += 1
                self._prefetch_count += 1
                logger.debug(
                    "Prefetched %d results for domain '%s' (p=%.2f)",
                    len(results),
                    domain,
                    prob,
                )
            except Exception as e:
                logger.warning("Prefetch failed for domain '%s': %s", domain, e)

        return fetched

    def get(self, domain: str) -> list[Any] | None:
        """Retrieve prefetched results for a domain.

        Args:
            domain: The domain to look up.

        Returns:
            Cached results, or None if not prefetched or expired.
        """
        self._total_requests += 1
        entry = self._cache.get(domain)

        if entry is None or entry.is_expired:
            return None

        entry.hit_count += 1
        self._cache_hits += 1
        return entry.results

    def invalidate(self, domain: str) -> None:
        """Remove a domain from the cache.

        Args:
            domain: The domain to invalidate.
        """
        self._cache.pop(domain, None)

    def clear(self) -> None:
        """Clear the entire prefetch cache."""
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (hits / total requests)."""
        if self._total_requests == 0:
            return 0.0
        return self._cache_hits / self._total_requests

    def stats(self) -> dict[str, Any]:
        """Loader statistics."""
        return {
            "cache_size": len(self._cache),
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "hit_rate": round(self.hit_rate, 3),
            "prefetch_count": self._prefetch_count,
        }

    def _evict_oldest(self) -> None:
        """Evict the oldest or least-used cache entry."""
        if not self._cache:
            return
        # Evict expired entries first
        expired = [k for k, v in self._cache.items() if v.is_expired]
        if expired:
            del self._cache[expired[0]]
            return
        # Then evict least recently used (lowest hit_count)
        lru_key = min(self._cache, key=lambda k: self._cache[k].hit_count)
        del self._cache[lru_key]
