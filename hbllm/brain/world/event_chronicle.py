"""Event Chronicle — foundational temporal primitive for A13 world model.

Every change to the world is recorded as an immutable event, creating a
complete history that A12's TemporalOperator and CausalOperator can reason over.

The EventChronicle does NOT own HCIR state.  It writes EventNode entries
and CAUSES edges into the CognitiveGraph via transaction proposals.
It provides query interfaces over the event timeline.

Architecture::

    Entity state change
        ↓
    EventChronicle.record()
        ↓
    EventNode + HCIREdge(CAUSES) written to CognitiveGraph
        ↓
    Timeline queries available for A12 operators
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EventNode,
    HCIREdge,
    HCIREdgeType,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Event Kinds — well-known event type constants
# ═══════════════════════════════════════════════════════════════════════════


class WorldEventKind:
    """Well-known event kinds for world-model state changes."""

    # Entity lifecycle events
    ENTITY_DISCOVERED = "entity_discovered"
    ENTITY_TRACKED = "entity_tracked"
    ENTITY_OCCLUDED = "entity_occluded"
    ENTITY_RE_IDENTIFIED = "entity_re_identified"
    ENTITY_FORGOTTEN = "entity_forgotten"

    # Property change events
    PROPERTY_CHANGED = "property_changed"
    PROPERTY_ADDED = "property_added"
    PROPERTY_REMOVED = "property_removed"

    # Spatial relation events
    RELATION_ESTABLISHED = "relation_established"
    RELATION_CHANGED = "relation_changed"
    RELATION_REMOVED = "relation_removed"

    # Location events
    LOCATION_CHANGED = "location_changed"
    ENTERED_REGION = "entered_region"
    LEFT_REGION = "left_region"


# ═══════════════════════════════════════════════════════════════════════════
# Chronicle Event — rich event descriptor
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChronicleEvent:
    """Rich event descriptor for world-model state changes.

    This is the input to ``EventChronicle.record()``.  The chronicle
    converts it into an HCIR ``EventNode`` with proper provenance.
    """

    event_kind: str
    subject_entity_id: str
    timestamp: float = field(default_factory=time.time)
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cause_event_id: str | None = None  # ID of the event that caused this one


# ═══════════════════════════════════════════════════════════════════════════
# Sequence Pattern — detected recurring event sequences
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SequencePattern:
    """A detected recurring temporal pattern in the event timeline."""

    pattern_id: str
    event_kinds: tuple[str, ...]  # Ordered sequence of event kinds
    subject_entity_ids: tuple[str, ...]  # Entities involved
    occurrence_count: int = 0
    avg_interval_s: float = 0.0  # Average time between pattern occurrences


# ═══════════════════════════════════════════════════════════════════════════
# Event Chronicle
# ═══════════════════════════════════════════════════════════════════════════


class EventChronicle:
    """Event-sourced timeline for world-model state changes.

    Maintains an immutable chronicle of all world-model events as HCIR
    EventNodes.  Provides timeline queries and sequence detection for
    A12's TemporalOperator and CausalOperator.

    **HCIR invariant:** EventChronicle writes ``EventNode`` and
    ``HCIREdge(CAUSES)`` entries into the ``CognitiveGraph``.
    It never mutates entity or world state directly.

    Usage::

        chronicle = EventChronicle(graph)

        # Record a state change
        event_id = chronicle.record(ChronicleEvent(
            event_kind=WorldEventKind.PROPERTY_CHANGED,
            subject_entity_id="entity_17",
            state_before={"color": "red"},
            state_after={"color": "blue"},
        ))

        # Query timeline
        events = chronicle.events_for_entity("entity_17", since=t0)
        chain = chronicle.causal_chain(event_id)
    """

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph
        # In-memory index: entity_id → list of event node IDs (chronological)
        self._entity_timeline: dict[str, list[str]] = {}
        # In-memory index: event_node_id → ChronicleEvent metadata
        self._event_metadata: dict[str, ChronicleEvent] = {}

    # ── Recording ─────────────────────────────────────────────────────

    def record(self, event: ChronicleEvent) -> str:
        """Record a world-model event in the chronicle.

        Creates an HCIR EventNode and optionally a CAUSES edge if
        ``event.cause_event_id`` is specified.

        Returns:
            The ID of the created EventNode.
        """
        # Build event data payload
        event_data: dict[str, Any] = {
            "subject_entity_id": event.subject_entity_id,
            "state_before": event.state_before,
            "state_after": event.state_after,
        }
        if event.metadata:
            event_data["metadata"] = event.metadata

        # Create HCIR EventNode
        node = EventNode(
            event_kind=event.event_kind,
            event_data=event_data,
            event_timestamp=event.timestamp,
            tags=["world_chronicle", event.event_kind],
        )
        self._graph.add_node(node)

        # Create causal edge if this event has a cause
        if event.cause_event_id is not None:
            cause_node = self._graph.get_node(event.cause_event_id)
            if cause_node is not None:
                edge = HCIREdge(
                    edge_type=HCIREdgeType.CAUSES,
                    sources=[event.cause_event_id],
                    targets=[node.id],
                )
                self._graph.add_edge(edge)

        # Link event to subject entity via AFTER edge
        entity_node = self._graph.get_node(event.subject_entity_id)
        if entity_node is not None:
            edge = HCIREdge(
                edge_type=HCIREdgeType.AFTER,
                sources=[event.subject_entity_id],
                targets=[node.id],
                properties={"relationship": "subject_of_event"},
            )
            self._graph.add_edge(edge)

        # Update in-memory indexes
        if event.subject_entity_id not in self._entity_timeline:
            self._entity_timeline[event.subject_entity_id] = []
        self._entity_timeline[event.subject_entity_id].append(node.id)
        self._event_metadata[node.id] = event

        logger.debug(
            "Chronicle: recorded %s for entity %s (node %s)",
            event.event_kind,
            event.subject_entity_id,
            node.id,
        )

        return node.id

    # ── Timeline Queries ──────────────────────────────────────────────

    def events_for_entity(
        self,
        entity_id: str,
        since: float | None = None,
        until: float | None = None,
        event_kind: str | None = None,
    ) -> list[EventNode]:
        """Query events for a specific entity within an optional time window.

        Args:
            entity_id: The entity to query events for.
            since: Optional start timestamp (inclusive).
            until: Optional end timestamp (inclusive).
            event_kind: Optional filter by event kind.

        Returns:
            List of EventNodes in chronological order.
        """
        event_ids = self._entity_timeline.get(entity_id, [])
        results: list[EventNode] = []

        for eid in event_ids:
            node = self._graph.get_node(eid)
            if node is None or not isinstance(node, EventNode):
                continue

            # Time window filter
            if since is not None and node.event_timestamp < since:
                continue
            if until is not None and node.event_timestamp > until:
                continue

            # Event kind filter
            if event_kind is not None and node.event_kind != event_kind:
                continue

            results.append(node)

        return results

    def all_events(
        self,
        since: float | None = None,
        until: float | None = None,
    ) -> list[EventNode]:
        """Return all chronicle events, optionally filtered by time window.

        Returns events in chronological order.
        """
        results: list[EventNode] = []

        for event_ids in self._entity_timeline.values():
            for eid in event_ids:
                node = self._graph.get_node(eid)
                if node is None or not isinstance(node, EventNode):
                    continue
                if since is not None and node.event_timestamp < since:
                    continue
                if until is not None and node.event_timestamp > until:
                    continue
                results.append(node)

        # De-duplicate (same event could appear via different entities)
        seen: set[str] = set()
        unique: list[EventNode] = []
        for ev in results:
            if ev.id not in seen:
                seen.add(ev.id)
                unique.append(ev)

        # Sort chronologically
        unique.sort(key=lambda e: e.event_timestamp)
        return unique

    def latest_event_for_entity(
        self,
        entity_id: str,
        event_kind: str | None = None,
    ) -> EventNode | None:
        """Return the most recent event for an entity."""
        events = self.events_for_entity(entity_id, event_kind=event_kind)
        return events[-1] if events else None

    # ── Causal Chain Queries ──────────────────────────────────────────

    def causal_chain(
        self,
        event_id: str,
        direction: str = "backward",
        max_depth: int = 10,
    ) -> list[EventNode]:
        """Trace the causal chain from an event.

        Args:
            event_id: Starting event node ID.
            direction: "backward" traces causes, "forward" traces effects.
            max_depth: Maximum chain depth to prevent infinite loops.

        Returns:
            List of EventNodes forming the causal chain.
        """
        chain: list[EventNode] = []
        visited: set[str] = set()
        current_id = event_id

        for _ in range(max_depth):
            if current_id in visited:
                break
            visited.add(current_id)

            node = self._graph.get_node(current_id)
            if node is None or not isinstance(node, EventNode):
                break

            chain.append(node)

            # Follow causal edges
            if direction == "backward":
                # Find events that CAUSE this one
                edges = self._graph.edges_to(current_id)
                cause_edges = [e for e in edges if e.edge_type == HCIREdgeType.CAUSES]
                if not cause_edges:
                    break
                current_id = cause_edges[0].sources[0]
            else:
                # Find events caused by this one
                edges = self._graph.edges_from(current_id)
                effect_edges = [e for e in edges if e.edge_type == HCIREdgeType.CAUSES]
                if not effect_edges:
                    break
                current_id = effect_edges[0].targets[0]

        return chain

    # ── Sequence Detection ────────────────────────────────────────────

    def detect_sequences(
        self,
        entity_id: str,
        min_occurrences: int = 2,
        window_size: int = 3,
    ) -> list[SequencePattern]:
        """Detect recurring event sequences for an entity.

        Scans the timeline with a sliding window to find repeated
        event-kind sequences.

        Args:
            entity_id: Entity to analyze.
            min_occurrences: Minimum repetitions to count as a pattern.
            window_size: Size of the sliding window (number of events).

        Returns:
            List of detected sequence patterns.
        """
        events = self.events_for_entity(entity_id)
        if len(events) < window_size:
            return []

        # Build sliding window sequences
        sequence_occurrences: dict[tuple[str, ...], list[float]] = {}

        for i in range(len(events) - window_size + 1):
            window = events[i : i + window_size]
            kinds = tuple(e.event_kind for e in window)
            timestamp = window[0].event_timestamp

            if kinds not in sequence_occurrences:
                sequence_occurrences[kinds] = []
            sequence_occurrences[kinds].append(timestamp)

        # Filter by min_occurrences
        patterns: list[SequencePattern] = []
        for kinds, timestamps in sequence_occurrences.items():
            if len(timestamps) >= min_occurrences:
                # Compute average interval
                intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
                avg_interval = sum(intervals) / len(intervals) if intervals else 0.0

                patterns.append(
                    SequencePattern(
                        pattern_id=f"seq_{'_'.join(k[:8] for k in kinds)}",
                        event_kinds=kinds,
                        subject_entity_ids=(entity_id,),
                        occurrence_count=len(timestamps),
                        avg_interval_s=avg_interval,
                    )
                )

        return patterns

    # ── Statistics ────────────────────────────────────────────────────

    @property
    def total_events(self) -> int:
        """Total number of recorded events."""
        return len(self._event_metadata)

    @property
    def tracked_entities(self) -> set[str]:
        """Set of entity IDs that have events in the chronicle."""
        return set(self._entity_timeline.keys())
