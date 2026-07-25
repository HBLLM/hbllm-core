"""
Storage Abstractions — HCIR Graph and Event persistence interfaces.

The graph never owns persistence.  Storage is delegated to abstract
``IGraphStore`` and ``IEventStore`` interfaces, allowing future
backends (SQLite, Redis, Neo4j, DuckDB, remote) without changing
HCIR internals.

Design:
    Graph → IGraphStore → SQLite / Memory / Neo4j / Remote
    Events → IEventStore → SQLite / Redis / Distributed
    Snapshots are derived from events, never stored independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIRNode

# ═══════════════════════════════════════════════════════════════════════════
# Event Store
# ═══════════════════════════════════════════════════════════════════════════


class EventType(StrEnum):
    """Types of events in the append-only event log."""

    NODE_ADDED = "node_added"
    NODE_MODIFIED = "node_modified"
    NODE_REMOVED = "node_removed"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    SNAPSHOT_CREATED = "snapshot_created"

    # Kernel telemetry events
    TRANSACTION_COMMITTED = "transaction_committed"
    TRANSACTION_REJECTED = "transaction_rejected"
    TRANSACTION_COMPENSATED = "transaction_compensated"
    CAPABILITY_BOUND = "capability_bound"
    BUDGET_EXCEEDED = "budget_exceeded"
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_FINISHED = "simulation_finished"
    ROLLBACK = "rollback"
    POLICY_VIOLATION = "policy_violation"

    # ── Cognitive Domain Events (Phase 1) ────────────────────────────
    # Perception
    PERCEPTION_RECEIVED = "perception_received"

    # Memory
    MEMORY_STORED = "memory_stored"
    MEMORY_RECALLED = "memory_recalled"
    MEMORY_CONSOLIDATED = "memory_consolidated"

    # Planning / Directives
    GOAL_CREATED = "goal_created"
    GOAL_COMPLETED = "goal_completed"
    GOAL_ABANDONED = "goal_abandoned"
    GOAL_BLOCKED = "goal_blocked"

    # Execution
    DECISION_MADE = "decision_made"
    ACTION_PLANNED = "action_planned"
    ACTION_EXECUTED = "action_executed"
    ACTION_RESULT = "action_result"
    CAPABILITY_INVOKED = "capability_invoked"

    # Epistemology
    BELIEF_UPDATED = "belief_updated"
    BELIEF_REVISED = "belief_revised"
    PREDICTION_MADE = "prediction_made"
    PREDICTION_VERIFIED = "prediction_verified"
    PREDICTION_ERROR = "prediction_error"

    # Governance
    GOVERNANCE_EVALUATED = "governance_evaluated"
    GOVERNANCE_BLOCKED = "governance_blocked"

    # Learning
    SKILL_LEARNED = "skill_learned"
    LEARNING_EVENT = "learning_event"

    # Cognitive state
    COGNITIVE_STATE_CHANGED = "cognitive_state_changed"
    ATTENTION_SHIFTED = "attention_shifted"


@dataclass(frozen=True)
class GraphEvent:
    """An immutable event in the append-only log.

    Kernel Invariant #5: Event logs are write-once, append-only,
    and read-only after creation.
    """

    sequence: int  # Monotonically increasing sequence number
    event_type: EventType
    timestamp: float
    author: str  # Node or service that produced this event
    data: dict[str, Any] = field(default_factory=dict)
    # For graph mutations, data contains serialized node/edge payloads
    # For kernel events, data contains event-specific metadata

    # ── Optional traceability (Phase 1) ──────────────────────────────
    session_id: str = ""
    goal_id: str = ""
    trace_id: str = ""
    logical_time: int = 0
    generation: int = 0


class IEventStore(ABC):
    """Abstract append-only event store.

    Snapshots are derived from events.  The event store is the
    source of truth for state reconstruction.
    """

    @abstractmethod
    def append(self, event: GraphEvent) -> None:
        """Append an event to the log.  Must be idempotent for replays."""
        ...

    @abstractmethod
    def get_events(
        self,
        from_sequence: int = 0,
        to_sequence: int | None = None,
        event_types: list[EventType] | None = None,
    ) -> list[GraphEvent]:
        """Retrieve events in sequence order, optionally filtered."""
        ...

    @abstractmethod
    def latest_sequence(self) -> int:
        """Return the sequence number of the most recent event."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all events (for testing only)."""
        ...


class InMemoryEventStore(IEventStore):
    """In-memory event store backed by a Python list.

    Suitable for single-process development and testing.
    """

    def __init__(self) -> None:
        self._events: list[GraphEvent] = []
        self._sequence: int = 0

    def append(self, event: GraphEvent) -> None:
        self._events.append(event)
        self._sequence = max(self._sequence, event.sequence)

    def get_events(
        self,
        from_sequence: int = 0,
        to_sequence: int | None = None,
        event_types: list[EventType] | None = None,
    ) -> list[GraphEvent]:
        results: list[GraphEvent] = []
        for ev in self._events:
            if ev.sequence < from_sequence:
                continue
            if to_sequence is not None and ev.sequence > to_sequence:
                break
            if event_types and ev.event_type not in event_types:
                continue
            results.append(ev)
        return results

    def latest_sequence(self) -> int:
        return self._sequence

    def clear(self) -> None:
        self._events.clear()
        self._sequence = 0


# ═══════════════════════════════════════════════════════════════════════════
# Graph Store
# ═══════════════════════════════════════════════════════════════════════════


class IGraphStore(ABC):
    """Abstract persistence layer for the CognitiveGraph.

    The graph operates in-memory; the store provides save/load
    semantics for durability across restarts.
    """

    @abstractmethod
    def save_graph(self, graph: CognitiveGraph) -> None:
        """Persist the full graph state."""
        ...

    @abstractmethod
    def load_graph(self) -> CognitiveGraph:
        """Load the full graph state from storage."""
        ...

    @abstractmethod
    def save_node(self, node: HCIRNode) -> None:
        """Persist a single node (upsert semantics)."""
        ...

    @abstractmethod
    def save_edge(self, edge: HCIREdge) -> None:
        """Persist a single edge (upsert semantics)."""
        ...

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node from persistent storage."""
        ...

    @abstractmethod
    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge from persistent storage."""
        ...


class InMemoryGraphStore(IGraphStore):
    """In-memory graph store — stores a deep copy of the graph.

    Useful for testing and single-process development.
    """

    def __init__(self) -> None:
        self._stored_nodes: dict[str, HCIRNode] = {}
        self._stored_edges: dict[str, HCIREdge] = {}

    def save_graph(self, graph: CognitiveGraph) -> None:
        self._stored_nodes = {n.id: n.model_copy(deep=True) for n in graph.all_nodes()}
        self._stored_edges = {e.id: e.model_copy(deep=True) for e in graph.all_edges()}

    def load_graph(self) -> CognitiveGraph:
        graph = CognitiveGraph()
        for node in self._stored_nodes.values():
            graph.add_node(node.model_copy(deep=True))
        for edge in self._stored_edges.values():
            graph.add_edge(edge.model_copy(deep=True))
        return graph

    def save_node(self, node: HCIRNode) -> None:
        self._stored_nodes[node.id] = node.model_copy(deep=True)

    def save_edge(self, edge: HCIREdge) -> None:
        self._stored_edges[edge.id] = edge.model_copy(deep=True)

    def delete_node(self, node_id: str) -> None:
        self._stored_nodes.pop(node_id, None)

    def delete_edge(self, edge_id: str) -> None:
        self._stored_edges.pop(edge_id, None)
