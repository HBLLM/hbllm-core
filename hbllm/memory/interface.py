"""
Unified Memory Interface for HBLLM Core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class MemoryType(StrEnum):
    """Cognitive memory domains.

    These represent genuine memory types in the cognitive architecture.
    Storage backends, indexing layers, and reasoning systems (e.g.
    KnowledgeGraph, BeliefGraph, LatentCluster) are NOT memory types
    and should not be added here.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    VALUE = "value"
    GOAL = "goal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    KNOWLEDGE_GRAPH = "knowledge_graph"


@dataclass
class SearchResult:
    """Unified search result format across all memory types."""

    memory_type: MemoryType
    id: str
    content: Any
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedMemoryInterface(Protocol):
    """
    Protocol defining the standard interface for all memory interactions.
    """

    async def store(self, memory_type: MemoryType, data: Any, **kwargs: Any) -> str:
        """Store new data in the specified memory type."""
        ...

    async def retrieve(self, memory_type: MemoryType, query: Any, **kwargs: Any) -> list[Any]:
        """Retrieve exact matches or structured data from the specified memory."""
        ...

    async def search(
        self, query: str, memory_types: list[MemoryType] | None = None, **kwargs: Any
    ) -> list[SearchResult]:
        """
        Search across one or multiple memory types using a unified query.
        Results are normalized into SearchResult objects.
        """
        ...

    async def forget(self, memory_type: MemoryType, **kwargs: Any) -> int:
        """Remove or archive data from the specified memory."""
        ...

    async def stats(self, tenant_id: str) -> dict[str, Any]:
        """Get usage statistics for the memory systems."""
        ...
