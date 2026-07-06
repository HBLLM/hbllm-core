"""
Memory Repository — Abstract base for event-sourced memory repositories.

Defines the common interface that all memory types (episodic, semantic,
procedural, value) must implement. ``MemoryNode`` interacts exclusively
through this interface, remaining agnostic to storage implementation.

The repository pattern separates **memory semantics** (what kind of
memory) from **storage mechanics** (how it's persisted). Internally,
repositories delegate to ``MemoryEventStore`` for event-sourced storage
and use ``MemoryProjection`` to fold events into ``MemCube`` state.

Architecture::

    MemoryNode
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
        │
        ▼
    MemoryProjection
        │
        ▼
    MemoryEventStore

Usage::

    class SemanticMemory(MemoryRepository):
        async def store(self, content, **kwargs) -> str: ...
        async def retrieve(self, memory_id, **kwargs) -> MemCube | None: ...
        async def search(self, query, **kwargs) -> list[MemCube]: ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

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

    Each memory type (episodic, semantic, procedural, value)
    implements this interface. ``MemoryNode`` interacts with
    repositories exclusively through these methods.

    Subclasses are thin semantic wrappers around event-sourced
    storage — they handle domain-specific logic (search ranking,
    skill matching, preference aggregation) while delegating
    persistence to ``MemoryEventStore`` via ``MemoryProjection``.
    """

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
    ) -> MemCube | None:
        """Retrieve a single memory by ID.

        Args:
            memory_id: The memory identifier.
            tenant_id: Multi-tenant isolation key.

        Returns:
            Current MemCube state, or None if not found.
        """
        ...

    @abstractmethod
    async def search(self, query: str, tenant_id: str = "default", **kwargs: Any) -> list[MemCube]:
        """Search memories matching a query.

        Args:
            query: Search query string.
            tenant_id: Multi-tenant isolation key.
            **kwargs: Type-specific search parameters (top_k, etc.).

        Returns:
            List of matching MemCubes, ranked by relevance.
        """
        ...

    async def forget(self, memory_id: str, tenant_id: str = "default", **kwargs: Any) -> bool:
        """Mark a memory as forgotten.

        Default implementation is a no-op. Subclasses that support
        forgetting should override.

        Args:
            memory_id: The memory identifier.
            tenant_id: Multi-tenant isolation key.

        Returns:
            True if the memory was forgotten.
        """
        return False

    async def stats(self, tenant_id: str = "default") -> dict[str, Any]:
        """Get repository statistics.

        Default implementation returns empty stats.
        """
        return {}
