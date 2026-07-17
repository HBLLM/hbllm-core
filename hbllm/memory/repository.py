"""
Memory Repository — Abstract base for HBLLM memory backends.

Defines the common interface that all memory types (episodic, semantic,
procedural, value) must implement. ``MemoryNode`` and ``MemoryService``
interact exclusively through this interface, remaining agnostic to
storage implementation.

The repository pattern separates **memory semantics** (what kind of
memory) from **storage mechanics** (how it's persisted).

Architecture::

    MemoryService / MemoryNode
        │
        ▼
    MemoryRepository (ABC)
        │
    ┌───┼───┬───┐
    ▼   ▼   ▼   ▼
    Episodic
    Semantic
    Procedural
    Value

Usage::

    class SemanticMemory(MemoryRepository):
        @property
        def memory_type(self) -> MemoryType:
            return MemoryType.SEMANTIC

        async def store(self, content, **kwargs) -> str: ...
        async def retrieve(self, memory_id, **kwargs) -> dict | None: ...
        async def search(self, query, **kwargs) -> list[dict]: ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from hbllm.memory.interface import MemoryType
from hbllm.memory.memcube import MemCube, MemoryEventStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# MemoryProjection — folds events into MemCube state
# ═══════════════════════════════════════════════════════════════════════════


class MemoryProjection:
    """Projection layer that folds event streams into MemCube state.

    This layer owns the ``fold()`` operation. Neither ``MemoryNode``
    nor individual repositories expose fold directly — it belongs
    to the projection layer.

    Args:
        event_store: The underlying event store.
    """

    def __init__(self, event_store: MemoryEventStore) -> None:
        self._store = event_store

    async def project(self, memory_id: str) -> MemCube | None:
        """Project the current state of a memory by folding its events.

        Args:
            memory_id: The memory identifier.

        Returns:
            Current MemCube state, or None if no events exist.
        """
        cube = await self._store.fold(memory_id)
        return cube

    async def project_many(self, memory_ids: list[str]) -> list[MemCube]:
        """Project multiple memories.

        Args:
            memory_ids: List of memory identifiers.

        Returns:
            List of MemCubes (excludes None results).
        """
        results: list[MemCube] = []
        for mid in memory_ids:
            cube = await self.project(mid)
            if cube is not None:
                results.append(cube)
        return results

    @property
    def event_store(self) -> MemoryEventStore:
        """Access to the underlying event store."""
        return self._store


# ═══════════════════════════════════════════════════════════════════════════
# MemoryRepository — abstract base
# ═══════════════════════════════════════════════════════════════════════════


class MemoryRepository(ABC):
    """Abstract base for all memory repositories.

    Every memory backend (episodic, semantic, procedural, value)
    implements this interface. ``MemoryService`` and ``MemoryNode``
    interact with repositories exclusively through these methods.

    Lifecycle::

        initialize()  →  store / search / retrieve  →  shutdown()

    The ``health()`` and ``compact()`` hooks support distributed
    monitoring and maintenance without backend-specific knowledge.
    """

    # ── Identity ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """The cognitive domain this repository serves."""
        ...

    # ── Lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Start the backend (create tables, connect, warm caches).

        Called once at system startup. Default is a no-op — override
        in backends that need initialisation.
        """

    async def shutdown(self) -> None:
        """Gracefully stop the backend (flush, close connections).

        Called once during system shutdown. Default is a no-op.
        """

    async def health(self) -> dict[str, Any]:
        """Report backend health for monitoring.

        Returns a dict with at minimum ``{"status": "ok" | "degraded" | "error"}``.
        Default returns ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    async def compact(self) -> dict[str, Any]:
        """Run maintenance / compression / WAL checkpoint.

        Called periodically (e.g. by SleepNode). Default is a no-op.

        Returns:
            Summary of work done (e.g. ``{"freed_bytes": 0}``).
        """
        return {}

    # ── Core CRUD ────────────────────────────────────────────────────

    @abstractmethod
    async def store(self, content: str, tenant_id: str = "default", **kwargs: Any) -> str:
        """Store new content and return its memory ID.

        Args:
            content: The content to store.
            tenant_id: Multi-tenant isolation key.
            **kwargs: Type-specific parameters.

        Returns:
            The assigned memory ID.
        """
        ...

    @abstractmethod
    async def retrieve(
        self, memory_id: str, tenant_id: str = "default", **kwargs: Any
    ) -> Any | None:
        """Retrieve a single memory by ID.

        Args:
            memory_id: The memory identifier.
            tenant_id: Multi-tenant isolation key.

        Returns:
            Memory data (format is backend-specific), or None if not found.
        """
        ...

    @abstractmethod
    async def search(
        self, query: str, tenant_id: str = "default", **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search memories matching a query.

        Args:
            query: Search query string.
            tenant_id: Multi-tenant isolation key.
            **kwargs: Type-specific search parameters (top_k, etc.).

        Returns:
            List of result dicts with at minimum ``content`` and ``score`` keys.
        """
        ...

    async def delete(self, memory_id: str, tenant_id: str = "default", **kwargs: Any) -> bool:
        """Delete a memory by ID.

        Default implementation is a no-op. Override in backends that
        support deletion.

        Returns:
            True if the memory was deleted.
        """
        return False

    async def forget(self, memory_id: str, tenant_id: str = "default", **kwargs: Any) -> bool:
        """Mark a memory as forgotten (soft-delete / archival).

        Default implementation delegates to ``delete()``.

        Returns:
            True if the memory was forgotten.
        """
        return await self.delete(memory_id, tenant_id=tenant_id, **kwargs)

    # ── Observability ────────────────────────────────────────────────

    async def stats(self, tenant_id: str = "default") -> dict[str, Any]:
        """Get repository statistics.

        Default implementation returns empty stats. Override to provide
        backend-specific metrics.
        """
        return {"memory_type": self.memory_type.value}
