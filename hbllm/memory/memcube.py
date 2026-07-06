"""
MemCube — Event-sourced memory primitives.

Memory is **never overwritten**.  Current state = ``fold(events)``.

Every mutation (store, reinforce, correct, merge, link, forget) is an
append-only ``MemoryEvent``.  The ``MemCube`` is reconstructed by folding
the event stream.

This provides:
    - **Explainability**: Full audit trail for every belief
    - **Rollback**: Undo any mutation by replaying without it
    - **Sleep replay**: Consolidation replays events for compression
    - **Debugging**: Trace how a memory evolved over time

Architecture::

    MemoryNode.store(content)
        → MemoryEvent(CREATED)
        → MemoryEventStore.append(event)
        → MemCube = fold(events)

    MemoryNode.update(id, correction)
        → MemoryEvent(CORRECTED)
        → MemoryEventStore.append(event)
        → MemCube = fold(events)

    SleepNode.consolidate()
        → replay events → compress → MERGED events

Usage::

    from hbllm.memory.memcube import (
        MemCube, MemoryEvent, MemoryEventStore, MemoryEventType, MemoryType,
    )

    store = MemoryEventStore(db_path)
    await store.initialize()

    # Create a memory
    event = MemoryEvent.create(
        memory_id="mem_001",
        content="User prefers dark mode",
        memory_type=MemoryType.SEMANTIC,
        source_node="perception",
        tenant_id="default",
    )
    await store.append(event)

    # Fold to current state
    cube = await store.fold("mem_001")
    assert cube.content == "User prefers dark mode"

    # Later: correct it
    correction = MemoryEvent.correct(
        memory_id="mem_001",
        old_content="User prefers dark mode",
        new_content="User prefers light mode",
        reason="User explicitly stated preference change",
        source_node="workspace",
    )
    await store.append(correction)

    cube = await store.fold("mem_001")
    assert cube.content == "User prefers light mode"
    assert cube.version == 2
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Memory Types
# ═══════════════════════════════════════════════════════════════════════════


class MemoryType(StrEnum):
    """Semantic categories for memory cubes."""

    EPISODIC = "episodic"  # Events, interactions, conversations
    SEMANTIC = "semantic"  # Facts, knowledge, beliefs
    PROCEDURAL = "procedural"  # Skills, how-to, recipes
    VALUE = "value"  # User preferences, constraints
    GOAL = "goal"  # Active and completed goals


# ═══════════════════════════════════════════════════════════════════════════
# MemCube — Lightweight memory core
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MemCube:
    """Lightweight core memory object.

    Heavy parts (embeddings, metadata, relationships) are lazy-loaded
    via reference IDs.  The cube itself is cheap to create and fold.

    Attributes:
        id: Unique memory identifier.
        content: The memory content (text).
        memory_type: Category of memory.
        tenant_id: Multi-tenant isolation key.
        created_at: Epoch timestamp of creation.
        version: Monotonically increasing version from fold.
        importance: Importance score [0.0, 1.0] for prioritization.
        access_count: Number of times this memory was retrieved.
        last_accessed: Timestamp of last retrieval.
        forgotten: If True, this memory is soft-deleted.
        tags: Optional tags for categorization.
        source_node: The node that originally created this memory.
    """

    id: str
    content: str
    memory_type: MemoryType
    tenant_id: str = "default"
    created_at: float = 0.0
    version: int = 1
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    forgotten: bool = False
    tags: list[str] = field(default_factory=list)
    source_node: str = ""

    # Lazy references — loaded on demand
    _metadata_ref: str | None = None
    _embedding_ref: str | None = None
    _relationship_ref: str | None = None
    _belief_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence and logging."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "version": self.version,
            "importance": round(self.importance, 4),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "forgotten": self.forgotten,
            "tags": self.tags,
            "source_node": self.source_node,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemCube:
        """Deserialize from dict."""
        return cls(
            id=d["id"],
            content=d["content"],
            memory_type=MemoryType(d["memory_type"]),
            tenant_id=d.get("tenant_id", "default"),
            created_at=d.get("created_at", 0.0),
            version=d.get("version", 1),
            importance=d.get("importance", 0.5),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed", 0.0),
            forgotten=d.get("forgotten", False),
            tags=d.get("tags", []),
            source_node=d.get("source_node", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Memory Events — append-only event log
# ═══════════════════════════════════════════════════════════════════════════


class MemoryEventType(StrEnum):
    """Types of mutations that can happen to a memory."""

    CREATED = "created"
    REINFORCED = "reinforced"
    CORRECTED = "corrected"
    MERGED = "merged"
    LINKED = "linked"
    IMPORTANCE_CHANGED = "importance_changed"
    FORGOTTEN = "forgotten"  # Soft delete
    RESTORED = "restored"  # Undo forgotten
    ACCESSED = "accessed"  # Read access (for recency tracking)
    TAGGED = "tagged"  # Tags modified


@dataclass
class MemoryEvent:
    """A single append-only event in a memory's lifecycle.

    Current state = ``fold(events)``.  Events are never modified
    after creation — only new events are appended.

    Attributes:
        id: Unique event identifier.
        memory_id: The MemCube this event modifies.
        event_type: Type of mutation.
        timestamp: When the event occurred.
        source_node: Which cognitive node caused this event.
        tenant_id: Multi-tenant isolation key.
        payload: Event-type-specific data.
    """

    id: str
    memory_id: str
    event_type: MemoryEventType
    timestamp: float
    source_node: str
    tenant_id: str = "default"
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source_node": self.source_node,
            "tenant_id": self.tenant_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEvent:
        """Deserialize from dict."""
        return cls(
            id=d["id"],
            memory_id=d["memory_id"],
            event_type=MemoryEventType(d["event_type"]),
            timestamp=d["timestamp"],
            source_node=d["source_node"],
            tenant_id=d.get("tenant_id", "default"),
            payload=d.get("payload", {}),
        )

    # ── Factory methods ──────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        memory_id: str,
        content: str,
        memory_type: MemoryType | str,
        source_node: str,
        tenant_id: str = "default",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> MemoryEvent:
        """Create a CREATED event for a new memory."""
        return cls(
            id=_gen_id("evt"),
            memory_id=memory_id,
            event_type=MemoryEventType.CREATED,
            timestamp=time.time(),
            source_node=source_node,
            tenant_id=tenant_id,
            payload={
                "content": content,
                "memory_type": str(memory_type),
                "importance": importance,
                "tags": tags or [],
            },
        )

    @classmethod
    def reinforce(
        cls,
        memory_id: str,
        source_node: str,
        by_query: str = "",
        strength: float = 0.1,
    ) -> MemoryEvent:
        """Create a REINFORCED event (memory was relevant to a query)."""
        return cls(
            id=_gen_id("evt"),
            memory_id=memory_id,
            event_type=MemoryEventType.REINFORCED,
            timestamp=time.time(),
            source_node=source_node,
            payload={"by_query": by_query, "strength": strength},
        )

    @classmethod
    def correct(
        cls,
        memory_id: str,
        old_content: str,
        new_content: str,
        reason: str,
        source_node: str,
    ) -> MemoryEvent:
        """Create a CORRECTED event (content was updated)."""
        return cls(
            id=_gen_id("evt"),
            memory_id=memory_id,
            event_type=MemoryEventType.CORRECTED,
            timestamp=time.time(),
            source_node=source_node,
            payload={
                "old_content": old_content,
                "new_content": new_content,
                "reason": reason,
            },
        )

    @classmethod
    def merge(
        cls,
        memory_id: str,
        merged_with: str,
        merged_content: str,
        strategy: str,
        source_node: str,
    ) -> MemoryEvent:
        """Create a MERGED event (two memories combined)."""
        return cls(
            id=_gen_id("evt"),
            memory_id=memory_id,
            event_type=MemoryEventType.MERGED,
            timestamp=time.time(),
            source_node=source_node,
            payload={
                "merged_with": merged_with,
                "merged_content": merged_content,
                "strategy": strategy,
            },
        )

    @classmethod
    def forget(cls, memory_id: str, source_node: str, reason: str = "") -> MemoryEvent:
        """Create a FORGOTTEN event (soft delete)."""
        return cls(
            id=_gen_id("evt"),
            memory_id=memory_id,
            event_type=MemoryEventType.FORGOTTEN,
            timestamp=time.time(),
            source_node=source_node,
            payload={"reason": reason},
        )

    @classmethod
    def access(cls, memory_id: str, source_node: str, query: str = "") -> MemoryEvent:
        """Create an ACCESSED event (read tracking)."""
        return cls(
            id=_gen_id("evt"),
            memory_id=memory_id,
            event_type=MemoryEventType.ACCESSED,
            timestamp=time.time(),
            source_node=source_node,
            payload={"query": query},
        )

    @classmethod
    def change_importance(
        cls,
        memory_id: str,
        old_importance: float,
        new_importance: float,
        source_node: str,
        reason: str = "",
    ) -> MemoryEvent:
        """Create an IMPORTANCE_CHANGED event."""
        return cls(
            id=_gen_id("evt"),
            memory_id=memory_id,
            event_type=MemoryEventType.IMPORTANCE_CHANGED,
            timestamp=time.time(),
            source_node=source_node,
            payload={
                "old_importance": old_importance,
                "new_importance": new_importance,
                "reason": reason,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# MemoryEventStore — in-memory append-only event log
# ═══════════════════════════════════════════════════════════════════════════


class MemoryEventStore:
    """Append-only event store for memory lifecycle events.

    Stores events in-memory with optional SQLite persistence.
    Supports fold (reconstruct current state from events),
    history (full audit trail), and compaction.

    Args:
        db_path: Path to SQLite database for persistence.
            If None, events are stored in-memory only.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._events: dict[str, list[MemoryEvent]] = {}  # memory_id → events
        self._db_path = db_path
        self._total_events = 0

    async def initialize(self) -> None:
        """Initialize the event store (create tables if persistent)."""
        # SQLite persistence can be added later — in-memory for M2
        logger.debug("MemoryEventStore initialized (in-memory)")

    async def append(self, event: MemoryEvent) -> None:
        """Append an event to the store.

        Events are immutable after appending. Never modify an event.

        Args:
            event: The memory event to append.
        """
        self._events.setdefault(event.memory_id, [])
        self._events[event.memory_id].append(event)
        self._total_events += 1

        logger.debug(
            "MemoryEvent: %s on %s by %s",
            event.event_type.value,
            event.memory_id,
            event.source_node,
        )

    async def get_events(self, memory_id: str) -> list[MemoryEvent]:
        """Get all events for a memory, ordered by timestamp.

        Args:
            memory_id: The memory to get events for.

        Returns:
            Ordered list of events (oldest first).
        """
        events = self._events.get(memory_id, [])
        return sorted(events, key=lambda e: e.timestamp)

    async def fold(self, memory_id: str) -> MemCube | None:
        """Reconstruct current MemCube state from event history.

        Replays all events in order to build the current state.
        Returns None if no events exist for this memory.

        Args:
            memory_id: The memory to reconstruct.

        Returns:
            The current MemCube state, or None if not found.
        """
        events = await self.get_events(memory_id)
        if not events:
            return None

        # Find the CREATED event (must be first)
        created = events[0]
        if created.event_type != MemoryEventType.CREATED:
            logger.warning(
                "Memory %s has no CREATED event (first is %s)",
                memory_id,
                created.event_type.value,
            )
            return None

        # Build initial state from CREATED event
        payload = created.payload
        cube = MemCube(
            id=memory_id,
            content=payload.get("content", ""),
            memory_type=MemoryType(payload.get("memory_type", "episodic")),
            tenant_id=created.tenant_id,
            created_at=created.timestamp,
            version=1,
            importance=payload.get("importance", 0.5),
            tags=payload.get("tags", []),
            source_node=created.source_node,
        )

        # Apply subsequent events
        for event in events[1:]:
            cube = _apply_event(cube, event)

        return cube

    async def get_history(self, memory_id: str) -> list[MemoryEvent]:
        """Get the full audit trail for a memory.

        Same as ``get_events`` but named explicitly for explainability.

        Args:
            memory_id: The memory to get history for.

        Returns:
            Full event history, oldest first.
        """
        return await self.get_events(memory_id)

    async def get_all_memory_ids(self, tenant_id: str | None = None) -> list[str]:
        """Get all known memory IDs, optionally filtered by tenant.

        Args:
            tenant_id: If set, only return memories for this tenant.

        Returns:
            List of memory IDs.
        """
        if tenant_id is None:
            return list(self._events.keys())

        result = []
        for mid, events in self._events.items():
            if events and events[0].tenant_id == tenant_id:
                result.append(mid)
        return result

    async def get_recent_events(
        self,
        since: float,
        tenant_id: str | None = None,
        event_types: list[MemoryEventType] | None = None,
    ) -> list[MemoryEvent]:
        """Get events since a timestamp, for sleep consolidation.

        Args:
            since: Only return events after this timestamp.
            tenant_id: Optional tenant filter.
            event_types: Optional filter by event type.

        Returns:
            Matching events, oldest first.
        """
        result: list[MemoryEvent] = []
        for events in self._events.values():
            for event in events:
                if event.timestamp < since:
                    continue
                if tenant_id and event.tenant_id != tenant_id:
                    continue
                if event_types and event.event_type not in event_types:
                    continue
                result.append(event)
        result.sort(key=lambda e: e.timestamp)
        return result

    async def compact(self, memory_id: str) -> MemCube | None:
        """Compact events into a snapshot + discard old events.

        Used during deep sleep to reduce event store size.
        Creates a single CREATED event from the current folded state,
        replacing the full history.

        Args:
            memory_id: The memory to compact.

        Returns:
            The compacted MemCube, or None if not found.
        """
        cube = await self.fold(memory_id)
        if cube is None:
            return None

        # Replace event history with a single CREATED event
        snapshot_event = MemoryEvent.create(
            memory_id=memory_id,
            content=cube.content,
            memory_type=cube.memory_type,
            source_node="sleep_compaction",
            tenant_id=cube.tenant_id,
            importance=cube.importance,
            tags=cube.tags,
        )
        # Preserve original creation time
        snapshot_event.timestamp = cube.created_at

        self._events[memory_id] = [snapshot_event]

        logger.debug("Compacted memory %s (events reduced to 1)", memory_id)
        return cube

    def stats(self) -> dict[str, Any]:
        """Event store statistics."""
        return {
            "total_memories": len(self._events),
            "total_events": self._total_events,
            "avg_events_per_memory": (round(self._total_events / max(1, len(self._events)), 1)),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Event folding — apply events to build current state
# ═══════════════════════════════════════════════════════════════════════════


def _apply_event(cube: MemCube, event: MemoryEvent) -> MemCube:
    """Apply a single event to a MemCube, producing the next version.

    Non-mutating (creates a new cube with incremented version).
    """
    p = event.payload

    if event.event_type == MemoryEventType.REINFORCED:
        strength = p.get("strength", 0.1)
        new_importance = min(1.0, cube.importance + strength * 0.1)
        return MemCube(
            id=cube.id,
            content=cube.content,
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version + 1,
            importance=new_importance,
            access_count=cube.access_count,
            last_accessed=cube.last_accessed,
            forgotten=cube.forgotten,
            tags=cube.tags,
            source_node=cube.source_node,
        )

    elif event.event_type == MemoryEventType.CORRECTED:
        return MemCube(
            id=cube.id,
            content=p.get("new_content", cube.content),
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version + 1,
            importance=cube.importance,
            access_count=cube.access_count,
            last_accessed=cube.last_accessed,
            forgotten=cube.forgotten,
            tags=cube.tags,
            source_node=cube.source_node,
        )

    elif event.event_type == MemoryEventType.MERGED:
        return MemCube(
            id=cube.id,
            content=p.get("merged_content", cube.content),
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version + 1,
            importance=cube.importance,
            access_count=cube.access_count,
            last_accessed=cube.last_accessed,
            forgotten=cube.forgotten,
            tags=cube.tags,
            source_node=cube.source_node,
        )

    elif event.event_type == MemoryEventType.FORGOTTEN:
        return MemCube(
            id=cube.id,
            content=cube.content,
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version + 1,
            importance=cube.importance,
            access_count=cube.access_count,
            last_accessed=cube.last_accessed,
            forgotten=True,
            tags=cube.tags,
            source_node=cube.source_node,
        )

    elif event.event_type == MemoryEventType.RESTORED:
        return MemCube(
            id=cube.id,
            content=cube.content,
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version + 1,
            importance=cube.importance,
            access_count=cube.access_count,
            last_accessed=cube.last_accessed,
            forgotten=False,
            tags=cube.tags,
            source_node=cube.source_node,
        )

    elif event.event_type == MemoryEventType.ACCESSED:
        return MemCube(
            id=cube.id,
            content=cube.content,
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version,
            importance=cube.importance,
            access_count=cube.access_count + 1,
            last_accessed=event.timestamp,
            forgotten=cube.forgotten,
            tags=cube.tags,
            source_node=cube.source_node,
        )

    elif event.event_type == MemoryEventType.IMPORTANCE_CHANGED:
        return MemCube(
            id=cube.id,
            content=cube.content,
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version + 1,
            importance=p.get("new_importance", cube.importance),
            access_count=cube.access_count,
            last_accessed=cube.last_accessed,
            forgotten=cube.forgotten,
            tags=cube.tags,
            source_node=cube.source_node,
        )

    elif event.event_type == MemoryEventType.TAGGED:
        new_tags = list(set(cube.tags + p.get("add_tags", [])))
        for tag in p.get("remove_tags", []):
            if tag in new_tags:
                new_tags.remove(tag)
        return MemCube(
            id=cube.id,
            content=cube.content,
            memory_type=cube.memory_type,
            tenant_id=cube.tenant_id,
            created_at=cube.created_at,
            version=cube.version + 1,
            importance=cube.importance,
            access_count=cube.access_count,
            last_accessed=cube.last_accessed,
            forgotten=cube.forgotten,
            tags=new_tags,
            source_node=cube.source_node,
        )

    # Unknown event type — pass through
    return cube


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════


def _gen_id(prefix: str = "mem") -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def gen_memory_id() -> str:
    """Generate a new memory ID."""
    return _gen_id("mem")
