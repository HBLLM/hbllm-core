"""
Snapshot Manager — event-sourced graph versioning.

State is derived from the event log, not stored as monolithic blobs.
Rollback = replay events up to target index.
Fork = diverge the event stream from a snapshot point.

    State_N = State_Base + Σ Event_i  (for i in Base..N)

Snapshots are lightweight bookmarks into the event stream that
record the sequence number and a content hash for fast equality
checks and branch identification.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIRNode
from hbllm.hcir.stores import EventType, GraphEvent, IEventStore


@dataclass(frozen=True)
class Snapshot:
    """A versioned bookmark into the event stream.

    Immutable after creation (Kernel Invariant #5).
    """

    version: int
    sequence: int  # Event log sequence number at this point
    content_hash: str  # SHA-256 hash of graph state at this version
    timestamp: float
    branch: str = "main"
    parent_version: int | None = None  # For forked branches


class SnapshotManager:
    """Manages event-sourced graph versioning.

    The SnapshotManager does NOT own the event store.  It receives
    an ``IEventStore`` instance and creates snapshots as bookmarks.

    Usage::

        mgr = SnapshotManager(event_store)
        snap = mgr.create_snapshot(graph, branch="main")
        # Later...
        mgr.rollback(graph, target_version=snap.version)
    """

    def __init__(self, event_store: IEventStore) -> None:
        self._event_store = event_store
        self._snapshots: dict[int, Snapshot] = {}
        self._current_version = 0
        self._branches: dict[str, int] = {"main": 0}  # branch → latest version

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def branches(self) -> dict[str, int]:
        return dict(self._branches)

    def record_node_added(self, node: HCIRNode, author: str = "system") -> None:
        """Record a node addition event."""
        seq = self._event_store.latest_sequence() + 1
        self._event_store.append(GraphEvent(
            sequence=seq,
            event_type=EventType.NODE_ADDED,
            timestamp=time.time(),
            author=author,
            data={"node_id": node.id, "node_type": node.node_type, "node_data": node.model_dump()},
        ))

    def record_node_modified(
        self, node_id: str, changes: dict, author: str = "system"
    ) -> None:
        """Record a node modification event."""
        seq = self._event_store.latest_sequence() + 1
        self._event_store.append(GraphEvent(
            sequence=seq,
            event_type=EventType.NODE_MODIFIED,
            timestamp=time.time(),
            author=author,
            data={"node_id": node_id, "changes": changes},
        ))

    def record_node_removed(self, node_id: str, author: str = "system") -> None:
        """Record a node removal event."""
        seq = self._event_store.latest_sequence() + 1
        self._event_store.append(GraphEvent(
            sequence=seq,
            event_type=EventType.NODE_REMOVED,
            timestamp=time.time(),
            author=author,
            data={"node_id": node_id},
        ))

    def record_edge_added(self, edge: HCIREdge, author: str = "system") -> None:
        """Record an edge addition event."""
        seq = self._event_store.latest_sequence() + 1
        self._event_store.append(GraphEvent(
            sequence=seq,
            event_type=EventType.EDGE_ADDED,
            timestamp=time.time(),
            author=author,
            data={"edge_id": edge.id, "edge_data": edge.model_dump()},
        ))

    def record_edge_removed(self, edge_id: str, author: str = "system") -> None:
        """Record an edge removal event."""
        seq = self._event_store.latest_sequence() + 1
        self._event_store.append(GraphEvent(
            sequence=seq,
            event_type=EventType.EDGE_REMOVED,
            timestamp=time.time(),
            author=author,
            data={"edge_id": edge_id},
        ))

    def record_kernel_event(
        self, event_type: EventType, data: dict, author: str = "kernel"
    ) -> None:
        """Record a kernel telemetry event."""
        seq = self._event_store.latest_sequence() + 1
        self._event_store.append(GraphEvent(
            sequence=seq,
            event_type=event_type,
            timestamp=time.time(),
            author=author,
            data=data,
        ))

    def create_snapshot(
        self, graph: CognitiveGraph, branch: str = "main"
    ) -> Snapshot:
        """Create a snapshot bookmark at the current event log position."""
        self._current_version += 1
        content_hash = self._hash_graph(graph)
        parent = self._branches.get(branch)

        snap = Snapshot(
            version=self._current_version,
            sequence=self._event_store.latest_sequence(),
            content_hash=content_hash,
            timestamp=time.time(),
            branch=branch,
            parent_version=parent,
        )
        self._snapshots[snap.version] = snap
        self._branches[branch] = snap.version
        return snap

    def get_snapshot(self, version: int) -> Snapshot | None:
        return self._snapshots.get(version)

    def fork_branch(
        self, graph: CognitiveGraph, new_branch: str, from_branch: str = "main"
    ) -> Snapshot:
        """Create a new branch from an existing branch's latest snapshot."""
        snap = self.create_snapshot(graph, branch=new_branch)
        return snap

    def get_events_since(self, snapshot: Snapshot) -> list[GraphEvent]:
        """Get all events after a snapshot's sequence position."""
        return self._event_store.get_events(from_sequence=snapshot.sequence + 1)

    @staticmethod
    def _hash_graph(graph: CognitiveGraph) -> str:
        """Compute a deterministic hash of the graph state."""
        hasher = hashlib.sha256()
        # Hash nodes in sorted order for determinism
        for node in sorted(graph.all_nodes(), key=lambda n: n.id):
            hasher.update(node.id.encode())
            hasher.update(node.node_type.encode())
        for edge in sorted(graph.all_edges(), key=lambda e: e.id):
            hasher.update(edge.id.encode())
            hasher.update(edge.edge_type.encode())
        return hasher.hexdigest()
