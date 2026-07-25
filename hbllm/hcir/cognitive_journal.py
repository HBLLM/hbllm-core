"""
Cognitive Journal — high-volume, append-only record of every cognitive event.

The journal captures **every** raw event that flows through the system.
It is the lowest layer of the event-sourced architecture:

    Nodes → Bus → Normalizer → Journal → Event Log → Projection → Workspace

The journal is:
    - **Append-only**: Events are never modified or deleted.
    - **High-volume**: Millions of entries per session.
    - **Rarely queried directly**: Used for replay, auditing, and debugging.
    - **Hash-chained**: Backed by ``SqliteEventStore`` for tamper detection.

The journal is NOT the same as the Cognitive Event Log.  The event log
only stores meaningful state transitions (``GoalCreated``, ``DecisionMade``).
The journal stores everything, including heartbeats, intermediate routing,
and raw sensor ticks.

Usage::

    from hbllm.hcir.stores import InMemoryEventStore
    journal = CognitiveJournal(InMemoryEventStore())
    seq = journal.record(event)
    events = list(journal.replay(from_seq=0))
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.semantic_normalizer import CognitiveEventKind
from hbllm.hcir.stores import EventType, GraphEvent, IEventStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Event — the canonical event record
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CognitiveEvent:
    """An immutable cognitive event record with full provenance.

    This is the canonical event representation used by both the journal
    and the event log.  It carries normalized kind, provenance fields,
    and the raw payload.

    Attributes:
        id: Unique event identifier.
        kind: Canonical event kind from ``SemanticNormalizer``.
        timestamp: Wall-clock timestamp (epoch seconds).
        author: Node ID or subsystem that produced this event.
        tenant_id: Tenant isolation boundary.
        session_id: Conversation/session trace ID.
        goal_id: Parent goal that caused this event.
        trace_id: End-to-end request trace ID.
        model_used: LLM model involved (if any).
        confidence: Confidence score of the event producer [0.0, 1.0].
        reason: Human-readable justification.
        source_node: Originating network node ID.
        logical_time: Lamport-style logical clock.
        generation: Reasoning generation / depth.
        attention_epoch: Attention recomputation epoch.
        reflection_cycle: Reflection cycle index.
        data: Raw event payload (serializable dict).
        raw_topic: Original bus topic before normalization.
    """

    id: str = field(default_factory=lambda: f"cev_{uuid.uuid4().hex[:12]}")
    kind: CognitiveEventKind = CognitiveEventKind.OBSERVATION_RECEIVED
    timestamp: float = field(default_factory=time.time)
    author: str = ""
    tenant_id: str = "default"

    # ── Traceability ─────────────────────────────────────────────────
    session_id: str = ""
    goal_id: str = ""
    trace_id: str = ""
    model_used: str = ""
    confidence: float = 1.0
    reason: str = ""
    source_node: str = ""

    # ── Cognitive Time ───────────────────────────────────────────────
    logical_time: int = 0
    generation: int = 0
    attention_epoch: int = 0
    reflection_cycle: int = 0

    # ── Payload ──────────────────────────────────────────────────────
    data: dict[str, Any] = field(default_factory=dict)
    raw_topic: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Journal
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveJournal:
    """Append-only journal of every raw cognitive event.

    Backed by ``IEventStore`` with hash-chaining for tamper detection.
    Never queried in hot paths.  Used for:

        - Full replay / time travel
        - Compliance auditing
        - Offline analytics
        - Deterministic test replay

    Usage::

        journal = CognitiveJournal(InMemoryEventStore())
        seq = journal.record(cognitive_event)
        events = list(journal.replay(from_seq=0))
    """

    def __init__(self, store: IEventStore) -> None:
        self._store = store
        self._logical_clock: int = 0

    @property
    def latest_sequence(self) -> int:
        """Return the sequence number of the most recent journal entry."""
        return self._store.latest_sequence()

    @property
    def logical_clock(self) -> int:
        """Return the current logical clock value."""
        return self._logical_clock

    def advance_clock(self, remote_time: int = 0) -> int:
        """Advance the logical clock (Lamport clock semantics).

        Args:
            remote_time: Logical time from a remote event (for sync).

        Returns:
            The new logical clock value.
        """
        self._logical_clock = max(self._logical_clock, remote_time) + 1
        return self._logical_clock

    def record(self, event: CognitiveEvent) -> int:
        """Append a cognitive event to the journal.

        The event is converted to a ``GraphEvent`` and appended to the
        underlying ``IEventStore``.  The logical clock is advanced.

        Args:
            event: The cognitive event to record.

        Returns:
            The sequence number assigned to this event.
        """
        self._logical_clock = max(self._logical_clock, event.logical_time) + 1
        seq = self._store.latest_sequence() + 1

        # Map CognitiveEventKind → EventType for storage
        event_type = _kind_to_event_type(event.kind)

        graph_event = GraphEvent(
            sequence=seq,
            event_type=event_type,
            timestamp=event.timestamp,
            author=event.author,
            data={
                "cognitive_event_id": event.id,
                "kind": event.kind.value,
                "tenant_id": event.tenant_id,
                "model_used": event.model_used,
                "confidence": event.confidence,
                "reason": event.reason,
                "source_node": event.source_node,
                "attention_epoch": event.attention_epoch,
                "reflection_cycle": event.reflection_cycle,
                "raw_topic": event.raw_topic,
                **event.data,
            },
            session_id=event.session_id,
            goal_id=event.goal_id,
            trace_id=event.trace_id,
            logical_time=self._logical_clock,
            generation=event.generation,
        )
        self._store.append(graph_event)

        logger.debug(
            "Journal: seq=%d kind=%s author=%s trace=%s",
            seq,
            event.kind.value,
            event.author,
            event.trace_id,
        )
        return seq

    def replay(
        self,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> Iterator[CognitiveEvent]:
        """Replay events from the journal in sequence order.

        Args:
            from_seq: Start replaying from this sequence (inclusive).
            to_seq: Stop at this sequence (inclusive).  ``None`` = latest.

        Yields:
            ``CognitiveEvent`` objects in sequence order.
        """
        graph_events = self._store.get_events(
            from_sequence=from_seq,
            to_sequence=to_seq,
        )
        for ge in graph_events:
            yield _graph_event_to_cognitive_event(ge)

    def replay_by_trace(self, trace_id: str) -> list[CognitiveEvent]:
        """Retrieve all events for a given request trace.

        This scans the full journal — use the ``CognitiveEventLog``
        for efficient trace queries.
        """
        results: list[CognitiveEvent] = []
        for ge in self._store.get_events():
            if ge.trace_id == trace_id:
                results.append(_graph_event_to_cognitive_event(ge))
        return results

    def count(self) -> int:
        """Return the total number of events in the journal."""
        return self._store.latest_sequence()


# ═══════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from CognitiveEventKind → closest EventType for storage.
# Many cognitive events don't have a direct EventType equivalent,
# so we fall back to a generic type.
_KIND_TO_EVENT_TYPE: dict[CognitiveEventKind, EventType] = {
    CognitiveEventKind.GOAL_CREATED: EventType.GOAL_CREATED,
    CognitiveEventKind.GOAL_COMPLETED: EventType.GOAL_COMPLETED,
    CognitiveEventKind.GOAL_ABANDONED: EventType.GOAL_ABANDONED,
    CognitiveEventKind.GOAL_BLOCKED: EventType.GOAL_BLOCKED,
    CognitiveEventKind.OBSERVATION_RECEIVED: EventType.PERCEPTION_RECEIVED,
    CognitiveEventKind.BELIEF_UPDATED: EventType.BELIEF_UPDATED,
    CognitiveEventKind.BELIEF_REVISED: EventType.BELIEF_REVISED,
    CognitiveEventKind.PREDICTION_MADE: EventType.PREDICTION_MADE,
    CognitiveEventKind.PREDICTION_VERIFIED: EventType.PREDICTION_VERIFIED,
    CognitiveEventKind.PREDICTION_ERROR: EventType.PREDICTION_ERROR,
    CognitiveEventKind.DECISION_MADE: EventType.DECISION_MADE,
    CognitiveEventKind.ACTION_PLANNED: EventType.ACTION_PLANNED,
    CognitiveEventKind.ACTION_EXECUTED: EventType.ACTION_EXECUTED,
    CognitiveEventKind.ACTION_RESULT: EventType.ACTION_RESULT,
    CognitiveEventKind.CAPABILITY_INVOKED: EventType.CAPABILITY_INVOKED,
    CognitiveEventKind.MEMORY_STORED: EventType.MEMORY_STORED,
    CognitiveEventKind.MEMORY_RECALLED: EventType.MEMORY_RECALLED,
    CognitiveEventKind.MEMORY_CONSOLIDATED: EventType.MEMORY_CONSOLIDATED,
    CognitiveEventKind.SKILL_LEARNED: EventType.SKILL_LEARNED,
    CognitiveEventKind.GOVERNANCE_EVALUATED: EventType.GOVERNANCE_EVALUATED,
    CognitiveEventKind.GOVERNANCE_BLOCKED: EventType.GOVERNANCE_BLOCKED,
    CognitiveEventKind.COGNITIVE_STATE_CHANGED: EventType.COGNITIVE_STATE_CHANGED,
    CognitiveEventKind.ATTENTION_SHIFTED: EventType.ATTENTION_SHIFTED,
    CognitiveEventKind.PERCEPTION_RECEIVED: EventType.PERCEPTION_RECEIVED,
    CognitiveEventKind.LEARNING_EVENT: EventType.LEARNING_EVENT,
}


def _kind_to_event_type(kind: CognitiveEventKind) -> EventType:
    """Map a CognitiveEventKind to the closest EventType for storage."""
    return _KIND_TO_EVENT_TYPE.get(kind, EventType.NODE_MODIFIED)


def _graph_event_to_cognitive_event(ge: GraphEvent) -> CognitiveEvent:
    """Reconstruct a CognitiveEvent from a stored GraphEvent."""
    data = dict(ge.data)
    kind_str = data.pop("kind", "observation.received")
    try:
        kind = CognitiveEventKind(kind_str)
    except ValueError:
        kind = CognitiveEventKind.OBSERVATION_RECEIVED

    return CognitiveEvent(
        id=data.pop("cognitive_event_id", f"cev_replay_{ge.sequence}"),
        kind=kind,
        timestamp=ge.timestamp,
        author=ge.author,
        tenant_id=data.pop("tenant_id", "default"),
        session_id=ge.session_id,
        goal_id=ge.goal_id,
        trace_id=ge.trace_id,
        model_used=data.pop("model_used", ""),
        confidence=data.pop("confidence", 1.0),
        reason=data.pop("reason", ""),
        source_node=data.pop("source_node", ""),
        logical_time=ge.logical_time,
        generation=ge.generation,
        attention_epoch=data.pop("attention_epoch", 0),
        reflection_cycle=data.pop("reflection_cycle", 0),
        data=data,
        raw_topic=data.pop("raw_topic", ""),
    )
