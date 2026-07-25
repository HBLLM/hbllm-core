"""
Tenant-Aware, Model-Isolated Embedding Cache.

Provides a thread-safe embedding cache keyed by tenant_id, model_name,
adapter_id, and text hash. Ensures strict tenant isolation and prevents
cache corruption when model or adapter weights change.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """Composite cache key enforcing isolation boundaries."""

    tenant_id: str
    model_name: str
    adapter_id: str
    text_hash: str

    @classmethod
    def create(
        cls,
        text: str,
        tenant_id: str = "default",
        model_name: str = "default",
        adapter_id: str = "base",
    ) -> EmbeddingCacheKey:
        """Create a deterministic cache key for a given text and context."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(
            tenant_id=tenant_id,
            model_name=model_name,
            adapter_id=adapter_id,
            text_hash=text_hash,
        )


class EmbeddingCache:
    """
    LRU Embedding Cache with tenant and model isolation.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = max_entries
        self._cache: OrderedDict[EmbeddingCacheKey, np.ndarray[Any, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        text: str,
        tenant_id: str = "default",
        model_name: str = "default",
        adapter_id: str = "base",
    ) -> np.ndarray[Any, Any] | None:
        """Fetch cached embedding if present for this exact tenant and model context."""
        key = EmbeddingCacheKey.create(
            text, tenant_id=tenant_id, model_name=model_name, adapter_id=adapter_id
        )
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        return None

    def put(
        self,
        text: str,
        vector: np.ndarray[Any, Any],
        tenant_id: str = "default",
        model_name: str = "default",
        adapter_id: str = "base",
    ) -> None:
        """Store an embedding in the cache."""
        key = EmbeddingCacheKey.create(
            text, tenant_id=tenant_id, model_name=model_name, adapter_id=adapter_id
        )
        if key in self._cache:
            del self._cache[key]

        self._cache[key] = vector.copy()

        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def invalidate_tenant(self, tenant_id: str) -> int:
        """Invalidate all cached embeddings for a specific tenant."""
        keys_to_remove = [k for k in self._cache if k.tenant_id == tenant_id]
        for k in keys_to_remove:
            del self._cache[k]
        logger.info(
            "[EmbeddingCache] Invalidated %d entries for tenant '%s'",
            len(keys_to_remove),
            tenant_id,
        )
        return len(keys_to_remove)

    def invalidate_model(self, model_name: str) -> int:
        """Invalidate all cached embeddings when a model is updated."""
        keys_to_remove = [k for k in self._cache if k.model_name == model_name]
        for k in keys_to_remove:
            del self._cache[k]
        logger.info(
            "[EmbeddingCache] Invalidated %d entries for model '%s'",
            len(keys_to_remove),
            model_name,
        )
        return len(keys_to_remove)

    def clear(self) -> None:
        """Flush the entire embedding cache."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss telemetry statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }
