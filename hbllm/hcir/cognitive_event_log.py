"""
Cognitive Event Log — filtered log of meaningful cognitive state transitions.

While the ``CognitiveJournal`` records every raw event (millions per session),
the event log only stores events that represent actual state changes:

    GoalCreated, DecisionMade, PredictionVerified, SkillLearned, BeliefUpdated

NOT:

    Heartbeats, intermediate routing, raw sensor ticks

This dramatically reduces replay cost.  The event log is the primary
queryable interface for reasoning about past cognitive activity:

    - ``query_by_trace(trace_id)`` — end-to-end request tracing
    - ``query_by_goal(goal_id)`` — all events related to a goal
    - ``query_by_session(session_id)`` — all events in a session
    - ``query_by_kind(kind)`` — all events of a specific type

Usage::

    log = CognitiveEventLog(InMemoryEventStore())
    was_recorded = log.record_if_significant(event)
    trace_events = log.query_by_trace("trace_abc123")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from hbllm.hcir.cognitive_journal import CognitiveEvent
from hbllm.hcir.semantic_normalizer import CognitiveEventKind
from hbllm.hcir.stores import GraphEvent, IEventStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Significant Event Filter
# ═══════════════════════════════════════════════════════════════════════════

# These event kinds represent meaningful cognitive state transitions
# that are worth persisting in the queryable event log.
# Everything else goes to the journal only.
_SIGNIFICANT_EVENTS: frozenset[CognitiveEventKind] = frozenset(
    {
        # Directives
        CognitiveEventKind.GOAL_CREATED,
        CognitiveEventKind.GOAL_COMPLETED,
        CognitiveEventKind.GOAL_ABANDONED,
        CognitiveEventKind.GOAL_BLOCKED,
        # Epistemology
        CognitiveEventKind.BELIEF_UPDATED,
        CognitiveEventKind.BELIEF_REVISED,
        CognitiveEventKind.PREDICTION_MADE,
        CognitiveEventKind.PREDICTION_VERIFIED,
        CognitiveEventKind.PREDICTION_ERROR,
        # Execution
        CognitiveEventKind.DECISION_MADE,
        CognitiveEventKind.ACTION_PLANNED,
        CognitiveEventKind.ACTION_EXECUTED,
        CognitiveEventKind.ACTION_RESULT,
        CognitiveEventKind.CAPABILITY_INVOKED,
        # Memory
        CognitiveEventKind.MEMORY_STORED,
        CognitiveEventKind.MEMORY_CONSOLIDATED,
        CognitiveEventKind.SKILL_LEARNED,
        # Governance
        CognitiveEventKind.GOVERNANCE_EVALUATED,
        CognitiveEventKind.GOVERNANCE_BLOCKED,
        # Cognitive state
        CognitiveEventKind.COGNITIVE_STATE_CHANGED,
        CognitiveEventKind.ATTENTION_SHIFTED,
        # Learning
        CognitiveEventKind.LEARNING_EVENT,
        # World model
        CognitiveEventKind.WORLD_STATE_UPDATED,
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Event Log
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveEventLog:
    """Filtered, queryable log of meaningful cognitive state transitions.

    Receives all events from the journal, but only persists events
    that represent actual state changes.

    The log maintains in-memory indices for fast trace, goal, and
    session queries.  For production deployments, these indices should
    be backed by SQLite indexes on the event store.

    Usage::

        log = CognitiveEventLog(InMemoryEventStore())
        was_recorded = log.record_if_significant(event)
        trace_events = log.query_by_trace("trace_abc123")
    """

    def __init__(
        self,
        store: IEventStore,
        significant_events: frozenset[CognitiveEventKind] | None = None,
    ) -> None:
        self._store = store
        self._significant = significant_events or _SIGNIFICANT_EVENTS

        # In-memory indices for fast queries.
        # Keys are trace_id / goal_id / session_id, values are lists
        # of sequence numbers.
        self._trace_index: dict[str, list[int]] = {}
        self._goal_index: dict[str, list[int]] = {}
        self._session_index: dict[str, list[int]] = {}
        self._kind_index: dict[CognitiveEventKind, list[int]] = {}

    @property
    def latest_sequence(self) -> int:
        """Return the sequence number of the most recent log entry."""
        return self._store.latest_sequence()

    # ── Recording ────────────────────────────────────────────────────

    def record_if_significant(self, event: CognitiveEvent) -> bool:
        """Record the event if it represents a meaningful state transition.

        Args:
            event: The cognitive event to evaluate.

        Returns:
            ``True`` if the event was recorded, ``False`` if filtered out.
        """
        if event.kind not in self._significant:
            return False

        seq = self._store.latest_sequence() + 1

        from hbllm.hcir.cognitive_journal import _kind_to_event_type

        graph_event = GraphEvent(
            sequence=seq,
            event_type=_kind_to_event_type(event.kind),
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
            logical_time=event.logical_time,
            generation=event.generation,
        )
        self._store.append(graph_event)

        # Update in-memory indices
        self._update_indices(event, seq)

        logger.debug(
            "EventLog: seq=%d kind=%s author=%s trace=%s",
            seq,
            event.kind.value,
            event.author,
            event.trace_id,
        )
        return True

    def add_significant_kind(self, kind: CognitiveEventKind) -> None:
        """Add a new event kind to the significant events filter."""
        self._significant = self._significant | frozenset({kind})

    def remove_significant_kind(self, kind: CognitiveEventKind) -> None:
        """Remove an event kind from the significant events filter."""
        self._significant = self._significant - frozenset({kind})

    # ── Querying ─────────────────────────────────────────────────────

    def query_by_trace(self, trace_id: str) -> list[CognitiveEvent]:
        """Retrieve all significant events for a given request trace.

        Args:
            trace_id: End-to-end request trace ID.

        Returns:
            List of events in sequence order.
        """
        sequences = self._trace_index.get(trace_id, [])
        return self._load_events_by_sequences(sequences)

    def query_by_goal(self, goal_id: str) -> list[CognitiveEvent]:
        """Retrieve all significant events related to a goal.

        Args:
            goal_id: The HCIR goal node ID.

        Returns:
            List of events in sequence order.
        """
        sequences = self._goal_index.get(goal_id, [])
        return self._load_events_by_sequences(sequences)

    def query_by_session(self, session_id: str) -> list[CognitiveEvent]:
        """Retrieve all significant events in a session.

        Args:
            session_id: Conversation/session ID.

        Returns:
            List of events in sequence order.
        """
        sequences = self._session_index.get(session_id, [])
        return self._load_events_by_sequences(sequences)

    def query_by_kind(
        self,
        kind: CognitiveEventKind,
        limit: int = 100,
    ) -> list[CognitiveEvent]:
        """Retrieve the most recent events of a specific kind.

        Args:
            kind: The canonical event kind to filter by.
            limit: Maximum number of events to return.

        Returns:
            List of events in sequence order (most recent last).
        """
        sequences = self._kind_index.get(kind, [])
        # Return the last `limit` entries
        recent = sequences[-limit:] if len(sequences) > limit else sequences
        return self._load_events_by_sequences(recent)

    def replay(
        self,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> Iterator[CognitiveEvent]:
        """Replay significant events from the log in sequence order.

        Args:
            from_seq: Start replaying from this sequence (inclusive).
            to_seq: Stop at this sequence (inclusive).

        Yields:
            ``CognitiveEvent`` objects in sequence order.
        """
        graph_events = self._store.get_events(
            from_sequence=from_seq,
            to_sequence=to_seq,
        )
        from hbllm.hcir.cognitive_journal import _graph_event_to_cognitive_event

        for ge in graph_events:
            yield _graph_event_to_cognitive_event(ge)

    def count(self) -> int:
        """Return the total number of significant events in the log."""
        return self._store.latest_sequence()

    # ── Internal ─────────────────────────────────────────────────────

    def _update_indices(self, event: CognitiveEvent, seq: int) -> None:
        """Update in-memory query indices."""
        if event.trace_id:
            self._trace_index.setdefault(event.trace_id, []).append(seq)
        if event.goal_id:
            self._goal_index.setdefault(event.goal_id, []).append(seq)
        if event.session_id:
            self._session_index.setdefault(event.session_id, []).append(seq)
        self._kind_index.setdefault(event.kind, []).append(seq)

    def _load_events_by_sequences(self, sequences: list[int]) -> list[CognitiveEvent]:
        """Load CognitiveEvents for a list of sequence numbers."""
        if not sequences:
            return []

        from hbllm.hcir.cognitive_journal import _graph_event_to_cognitive_event

        # Load all events in the range and filter to exact sequences
        all_events = self._store.get_events(
            from_sequence=sequences[0],
            to_sequence=sequences[-1],
        )
        seq_set = set(sequences)
        return [_graph_event_to_cognitive_event(ge) for ge in all_events if ge.sequence in seq_set]
