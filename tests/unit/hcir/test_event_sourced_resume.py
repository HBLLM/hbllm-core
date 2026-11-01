"""
Unit tests for WP7: Kernel Event Bus bounded ring buffer & HCIRWorkspaceState event-sourced resume.
"""

from __future__ import annotations

import pytest

from hbllm.hcir.graph import (
    CognitiveCategory,
    HCIREdge,
    HCIREdgeType,
    HCIRNode,
)
from hbllm.hcir.kernel.event_bus import KernelEvent, KernelEventBus
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_kernel_event_bus_bounded_history():
    """Verify KernelEventBus uses an O(1) bounded ring buffer that does not leak memory.

    Publishing 1,200 events to a 1,000-sized bus maintains exactly 1,000 events.
    """
    history_limit = 1000
    bus = KernelEventBus(history_size=history_limit)

    total_published = 1200
    for i in range(total_published):
        bus.publish(
            KernelEvent(
                event_type="telemetry.test",
                source="test_runner",
                data={"index": i},
            )
        )

    assert bus.event_count == total_published
    assert len(bus._history) == history_limit
    stats = bus.stats()
    assert stats["history_size"] == history_limit
    assert stats["total_events"] == total_published

    # Verify oldest events (0 to 199) were evicted and newest (200 to 1199) are present
    recent_events = bus.get_history(last_n=history_limit)
    assert len(recent_events) == history_limit
    assert recent_events[0].data["index"] == 200
    assert recent_events[-1].data["index"] == 1199


def test_workspace_resume_from_event():
    """Verify HCIRWorkspaceState.resume_from_event(seq) accurately restores point-in-time state."""
    ws = HCIRWorkspaceState()

    # Step 1: Add Node A
    node_a = HCIRNode(
        id="node_a",
        node_type="concept",
        category=CognitiveCategory.MEMORY,
        properties={"val": 1},
    )
    ws.snapshot_manager.record_node_added(node_a, author="test")
    ws.graph.add_node(node_a)

    # Step 2: Add Node B
    node_b = HCIRNode(
        id="node_b",
        node_type="concept",
        category=CognitiveCategory.MEMORY,
        properties={"val": 2},
    )
    ws.snapshot_manager.record_node_added(node_b, author="test")
    ws.graph.add_node(node_b)

    # Step 3: Add Edge A -> B
    edge_ab = HCIREdge(
        id="edge_ab",
        sources=["node_a"],
        targets=["node_b"],
        edge_type=HCIREdgeType.CAUSES,
        weight=1.0,
    )
    ws.snapshot_manager.record_edge_added(edge_ab, author="test")
    ws.graph.add_edge(edge_ab)

    # Sequence after step 3
    seq_at_step_3 = ws.event_store.latest_sequence()

    # Step 4: Modify Node A
    ws.snapshot_manager.record_node_modified(
        node_id="node_a",
        changes={"properties": {"val": 99}},
        author="test",
    )
    node_a.properties["val"] = 99
    ws.graph.upsert_node(node_a)

    # Step 5: Add Node C
    node_c = HCIRNode(
        id="node_c",
        node_type="concept",
        category=CognitiveCategory.MEMORY,
        properties={"val": 3},
    )
    ws.snapshot_manager.record_node_added(node_c, author="test")
    ws.graph.add_node(node_c)

    # Step 6: Remove Edge A -> B
    ws.snapshot_manager.record_edge_removed("edge_ab", author="test")
    ws.graph.remove_edge("edge_ab")

    # Current state verification
    assert ws.graph.node_count == 3
    assert ws.graph.edge_count == 0
    assert ws.graph.get_node("node_a").properties["val"] == 99

    # Now Resume from Step 3 (before Node A modification, before Node C, edge still present)
    resumed_ws = ws.resume_from_event(seq_at_step_3)

    assert resumed_ws.graph.node_count == 2
    assert resumed_ws.graph.has_node("node_a")
    assert resumed_ws.graph.has_node("node_b")
    assert not resumed_ws.graph.has_node("node_c")
    assert resumed_ws.graph.edge_count == 1
    assert resumed_ws.graph.has_edge("edge_ab")
    assert resumed_ws.graph.get_node("node_a").properties["val"] == 1


def test_workspace_resume_from_event_not_found():
    """Verify resume_from_event raises ValueError for non-existent sequence/id."""
    ws = HCIRWorkspaceState()
    with pytest.raises(ValueError, match="not found in event store"):
        ws.resume_from_event(9999)
