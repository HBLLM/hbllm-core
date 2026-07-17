"""
Memory Service — Versioned facade wrapping all HBLLM memory backends.

HBLLM has 9+ memory backends, each with their own API. The Memory Service
provides a unified, versioned API that insulates callers from backend
details and enables:

    1. Unified search across multiple memory types
    2. Backend swapping without changing callers
    3. Caching layer for hot-path queries
    4. Per-tenant memory isolation
    5. Statistics aggregation

Memory backends wrapped:

    ┌─────────────────────────────────────────────────────────────────┐
    │                       MemoryService v3                         │
    ├─────────────────┬───────────────┬──────────────────────────────┤
    │   Declarative   │  Experiential │       Structural             │
    │                 │               │                              │
    │ • Semantic      │ • Episodic    │ • KnowledgeGraph             │
    │ • Procedural    │ • ConvThread  │ • BeliefGraph                │
    │ • Value         │ • GoalMemory  │ • SpatialMemory              │
    │ • LatentCluster │               │ • TemporalPatterns           │
    └─────────────────┴───────────────┴──────────────────────────────┘

Usage::

    from hbllm.memory.memory_service import MemoryService

    service = MemoryService(data_dir="data")
    await service.init()

    # Unified search across all backends
    results = await service.search("HBLLM plugin architecture", tenant_id="default")

    # Store to a specific backend
    await service.store("episodic", content="User asked about plugins", tenant_id="default")

    # Cross-memory recall with ranking
    context = await service.recall(query="What did we discuss?", tenant_id="default", top_k=5)
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Memory Types (superset of existing MemoryType enum)
# ═══════════════════════════════════════════════════════════════════════════


class MemoryBackend(StrEnum):
    """All available memory backends."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    VALUE = "value"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    BELIEF_GRAPH = "belief_graph"
    CONVERSATION = "conversation"
    GOAL = "goal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    LATENT_CLUSTER = "latent_cluster"


# ═══════════════════════════════════════════════════════════════════════════
# Unified Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryResult:
    """A result from any memory backend, normalized for cross-backend ranking."""

    backend: MemoryBackend
    content: str
    score: float = 0.0  # Relevance score (0.0–1.0)
    memory_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    tenant_id: str = "default"

    @property
    def age_hours(self) -> float:
        if self.timestamp == 0:
            return 0.0
        return (time.time() - self.timestamp) / 3600


# ═══════════════════════════════════════════════════════════════════════════
# LRU Cache
# ═══════════════════════════════════════════════════════════════════════════


class _QueryCache:
    """Simple LRU cache for search results."""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 60.0) -> None:
        self._cache: OrderedDict[str, tuple[float, list[MemoryResult]]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> list[MemoryResult] | None:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        cached_at, results = entry
        if time.time() - cached_at > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        self._cache.move_to_end(key)
        return results

    def put(self, key: str, results: list[MemoryResult]) -> None:
        self._cache[key] = (time.time(), results)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, prefix: str = "") -> int:
        if not prefix:
            count = len(self._cache)
            self._cache.clear()
            return count
        keys = [k for k in self._cache if k.startswith(prefix)]
        for k in keys:
            del self._cache[k]
        return len(keys)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Memory Service
# ═══════════════════════════════════════════════════════════════════════════


class MemoryService:
    """Versioned facade wrapping all HBLLM memory backends.

    Provides a unified API for store, retrieve, search, and recall
    operations across 11 memory backends. Includes LRU caching and
    per-tenant isolation.

    Backends are lazily registered — the service works with whatever
    backends are available. Missing backends are gracefully skipped.
    """

    API_VERSION = "3.0"

    def __init__(
        self,
        data_dir: str = "data",
        cache_size: int = 100,
        cache_ttl: float = 60.0,
    ) -> None:
        self._data_dir = data_dir
        self._backends: dict[MemoryBackend, Any] = {}
        self._cache = _QueryCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._store_count = 0
        self._search_count = 0

    # ── Backend Registration ─────────────────────────────────────────

    def register_backend(self, backend_type: MemoryBackend, instance: Any) -> None:
        """Register a memory backend instance.

        Args:
            backend_type: Which backend slot this fills.
            instance: The backend instance (e.g., SemanticMemory, EpisodicMemory).
        """
        self._backends[backend_type] = instance
        logger.info("Registered memory backend: %s", backend_type.value)

    def has_backend(self, backend_type: MemoryBackend) -> bool:
        return backend_type in self._backends

    @property
    def available_backends(self) -> list[str]:
        return [b.value for b in self._backends]

    async def init(self) -> None:
        """Initialize the service (backends should already be registered)."""
        logger.info(
            "MemoryService v%s initialized (%d backends: %s)",
            self.API_VERSION,
            len(self._backends),
            ", ".join(self.available_backends),
        )

    # ── Store ────────────────────────────────────────────────────────

    async def store(
        self,
        backend: str | MemoryBackend,
        *,
        content: str,
        tenant_id: str = "default",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Store content in a specific memory backend.

        Args:
            backend: Backend name or enum.
            content: Content to store.
            tenant_id: Tenant scope.
            metadata: Optional metadata.

        Returns:
            Memory ID from the backend, or None if backend unavailable.
        """
        backend_enum = MemoryBackend(backend) if isinstance(backend, str) else backend
        instance = self._backends.get(backend_enum)
        if instance is None:
            logger.debug("Backend '%s' not registered — skipping store", backend_enum.value)
            return None

        self._store_count += 1

        # Invalidate cache for this tenant
        self._cache.invalidate(f"{tenant_id}:")

        # Delegate to the backend's store method
        try:
            if hasattr(instance, "store"):
                result = await instance.store(
                    content, tenant_id=tenant_id, metadata=metadata or {}, **kwargs
                )
                return str(result) if result else None
            elif hasattr(instance, "add"):
                result = instance.add(content, tenant_id=tenant_id, metadata=metadata or {}, **kwargs)
                return str(result) if result else None
        except Exception as e:
            logger.warning("Store to '%s' failed: %s", backend_enum.value, e)

        return None

    # ── Search ───────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        backends: list[str | MemoryBackend] | None = None,
        top_k: int = 10,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> list[MemoryResult]:
        """Search across one or more memory backends.

        Results from all queried backends are merged and ranked by score.

        Args:
            query: Search query.
            tenant_id: Tenant scope.
            backends: Specific backends to search (None = all available).
            top_k: Maximum results.
            use_cache: Whether to use the query cache.

        Returns:
            Merged, ranked list of MemoryResult.
        """
        self._search_count += 1

        # Check cache
        cache_key = f"{tenant_id}:{query}:{top_k}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine which backends to query
        target_backends: list[MemoryBackend]
        if backends:
            target_backends = [
                MemoryBackend(b) if isinstance(b, str) else b for b in backends
            ]
        else:
            target_backends = list(self._backends.keys())

        # Query each backend
        all_results: list[MemoryResult] = []
        for backend_type in target_backends:
            instance = self._backends.get(backend_type)
            if instance is None:
                continue

            try:
                backend_results = await self._query_backend(
                    backend_type, instance, query, tenant_id, top_k, **kwargs
                )
                all_results.extend(backend_results)
            except Exception as e:
                logger.debug("Search in '%s' failed: %s", backend_type.value, e)

        # Sort by score descending, take top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        results = all_results[:top_k]

        # Cache results
        if use_cache and results:
            self._cache.put(cache_key, results)

        return results

    # ── Recall (Context-building convenience) ────────────────────────

    async def recall(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        top_k: int = 5,
        max_tokens: int = 2000,
    ) -> str:
        """Recall relevant context for prompt injection.

        Convenience method that searches across all backends and
        formats results into a compact text block.

        Args:
            query: What to recall.
            tenant_id: Tenant scope.
            top_k: Max results.
            max_tokens: Approximate token limit for output.

        Returns:
            Formatted context string.
        """
        results = await self.search(query, tenant_id=tenant_id, top_k=top_k)
        if not results:
            return ""

        lines: list[str] = []
        char_budget = max_tokens * 4  # ~4 chars per token estimate

        for r in results:
            entry = f"[{r.backend.value}] {r.content}"
            if len("\n".join(lines)) + len(entry) > char_budget:
                break
            lines.append(entry)

        return "\n".join(lines)

    # ── Cache Management ─────────────────────────────────────────────

    def invalidate_cache(self, tenant_id: str = "") -> int:
        """Invalidate cached search results."""
        prefix = f"{tenant_id}:" if tenant_id else ""
        return self._cache.invalidate(prefix)

    # ── Statistics ───────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all backends."""
        return {
            "api_version": self.API_VERSION,
            "backends_registered": len(self._backends),
            "backends": self.available_backends,
            "total_stores": self._store_count,
            "total_searches": self._search_count,
            "cache": self._cache.stats,
        }

    # ── Internal ─────────────────────────────────────────────────────

    async def _query_backend(
        self,
        backend_type: MemoryBackend,
        instance: Any,
        query: str,
        tenant_id: str,
        top_k: int,
        **kwargs: Any,
    ) -> list[MemoryResult]:
        """Query a single backend and normalize results."""
        results: list[MemoryResult] = []

        # Try the most common memory API methods
        if hasattr(instance, "search"):
            raw = await instance.search(query, tenant_id=tenant_id, top_k=top_k, **kwargs)
            if raw:
                for item in raw:
                    results.append(self._normalize_result(backend_type, item, tenant_id))

        elif hasattr(instance, "retrieve"):
            raw = await instance.retrieve(query, tenant_id=tenant_id, limit=top_k, **kwargs)
            if raw:
                for item in raw:
                    results.append(self._normalize_result(backend_type, item, tenant_id))

        elif hasattr(instance, "query"):
            raw = instance.query(query, tenant_id=tenant_id, limit=top_k, **kwargs)
            if raw:
                for item in raw:
                    results.append(self._normalize_result(backend_type, item, tenant_id))

        return results

    def _normalize_result(
        self,
        backend_type: MemoryBackend,
        item: Any,
        tenant_id: str,
    ) -> MemoryResult:
        """Normalize a backend-specific result into MemoryResult."""
        # Handle dict-like results
        if isinstance(item, dict):
            return MemoryResult(
                backend=backend_type,
                content=str(item.get("content", item.get("text", str(item)))),
                score=float(item.get("score", item.get("similarity", 0.5))),
                memory_id=str(item.get("id", "")),
                metadata=item.get("metadata", {}),
                timestamp=float(item.get("timestamp", item.get("created_at", 0))),
                tenant_id=tenant_id,
            )

        # Handle objects with common attributes
        if hasattr(item, "content"):
            return MemoryResult(
                backend=backend_type,
                content=str(item.content),
                score=float(getattr(item, "score", getattr(item, "similarity", 0.5))),
                memory_id=str(getattr(item, "id", "")),
                metadata=getattr(item, "metadata", {}),
                timestamp=float(getattr(item, "timestamp", getattr(item, "created_at", 0))),
                tenant_id=tenant_id,
            )

        # Fallback: treat as string
        return MemoryResult(
            backend=backend_type,
            content=str(item),
            score=0.5,
            tenant_id=tenant_id,
        )
