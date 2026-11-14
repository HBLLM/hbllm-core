"""Three-Layer Memory Store and Versioned Knowledge Provenance for A22.

Enforces strict separation across three memory tiers:
1. FAST_EPISODIC: Immediate high-resolution interaction traces and perceptions.
2. SLOW_CONSOLIDATED: Durable, compacted concepts, schemas, lexicon, and competence profiles.
3. IMMUTABLE_PROVENANCE: Authoritative append-only event log and ground-truth evidence.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryLayer(str, Enum):
    """The three authoritative memory tiers."""

    FAST_EPISODIC = "fast_episodic"  # Transient interaction buffer
    SLOW_CONSOLIDATED = "slow_consolidated"  # Durable generalized knowledge
    IMMUTABLE_PROVENANCE = "immutable_provenance"  # Authoritative ground-truth event history


@dataclass
class ImmutableEvent:
    """An append-only ground-truth interaction record in the immutable layer."""

    event_id: str = field(default_factory=lambda: f"iev_{uuid.uuid4().hex[:8]}")
    domain: str = ""
    action_type: str = ""
    action_parameters: dict[str, Any] = field(default_factory=dict)
    pre_state_snapshot: dict[str, Any] = field(default_factory=dict)
    post_state_snapshot: dict[str, Any] = field(default_factory=dict)
    prediction_made: dict[str, Any] = field(default_factory=dict)
    actual_outcome: bool = True
    prediction_error: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodicTrace:
    """A raw interaction episode held in the fast episodic buffer."""

    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:8]}")
    event_id: str = ""  # Foreign key to immutable event
    domain: str = ""
    context_props: dict[str, Any] = field(default_factory=dict)
    actions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    prediction_error: float = 0.0
    is_success: bool = True
    salience_score: float = 0.50
    timestamp: float = field(default_factory=time.time)


@dataclass
class VersionedKnowledgeRecord:
    """A versioned consolidated schema, concept, or rule with explicit provenance."""

    knowledge_id: str
    knowledge_type: str  # "schema", "concept", "lexical_sense", "competence"
    revision: int = 1
    supersedes_revision: int | None = None
    revision_reason: str = "initial_induction"
    content: dict[str, Any] = field(default_factory=dict)
    source_event_ids: list[str] = field(default_factory=list)  # Pointers to ImmutableEvent IDs
    confidence: float = 0.75
    created_at: float = field(default_factory=time.time)


class DualStoreMemory:
    """Coordinates the Fast Episodic Buffer, Slow Consolidated Repertoire, and Immutable Log."""

    def __init__(self) -> None:
        self.immutable_log: dict[str, ImmutableEvent] = {}
        self.fast_buffer: list[EpisodicTrace] = []
        self.slow_store: dict[str, VersionedKnowledgeRecord] = {}  # knowledge_id -> latest revision
        self.revision_history: dict[
            str, list[VersionedKnowledgeRecord]
        ] = {}  # knowledge_id -> [r1, r2, ...]

    def append_immutable_event(self, event: ImmutableEvent) -> str:
        """Commit an append-only event into the authoritative immutable log."""
        self.immutable_log[event.event_id] = event
        return event.event_id

    def buffer_episodic_trace(self, trace: EpisodicTrace) -> None:
        """Push an episodic trace into the fast episodic buffer."""
        self.fast_buffer.append(trace)

    def commit_consolidated_knowledge(
        self,
        knowledge_id: str,
        knowledge_type: str,
        content: dict[str, Any],
        source_event_ids: list[str],
        reason: str = "sleep_consolidation",
        confidence: float = 0.80,
    ) -> VersionedKnowledgeRecord:
        """Store or revise consolidated knowledge with monotonic revisioning and provenance pointers."""
        prev_rev = self.slow_store.get(knowledge_id)
        next_rev_num = (prev_rev.revision + 1) if prev_rev else 1

        record = VersionedKnowledgeRecord(
            knowledge_id=knowledge_id,
            knowledge_type=knowledge_type,
            revision=next_rev_num,
            supersedes_revision=prev_rev.revision if prev_rev else None,
            revision_reason=reason,
            content=content,
            source_event_ids=list(dict.fromkeys(source_event_ids)),
            confidence=confidence,
        )

        self.slow_store[knowledge_id] = record
        if knowledge_id not in self.revision_history:
            self.revision_history[knowledge_id] = []
        self.revision_history[knowledge_id].append(record)
        return record

    def clear_fast_buffer(self) -> int:
        """Flush the fast episodic buffer post-consolidation. Returns cleared count."""
        count = len(self.fast_buffer)
        self.fast_buffer.clear()
        return count

    def reconstruct_knowledge_justification(self, knowledge_id: str) -> list[ImmutableEvent]:
        """Audit trail: Reconstruct exact immutable ground-truth events that justify a consolidated knowledge record."""
        record = self.slow_store.get(knowledge_id)
        if not record:
            return []
        return [
            self.immutable_log[eid] for eid in record.source_event_ids if eid in self.immutable_log
        ]
