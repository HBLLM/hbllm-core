"""Tests for WorkspaceNode error recovery and graceful fallback."""

import pytest
import asyncio
import time

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.workspace_node import WorkspaceNode


@pytest.mark.asyncio
async def test_workspace_fallback_on_zero_thoughts():
    """When no thoughts arrive by deadline, workspace sends a fallback response."""
    bus = InProcessBus()
    await bus.start()
    
    workspace = WorkspaceNode(node_id="ws_test")
    await workspace.start(bus)
    
    fallback_received = []
    
    async def output_handler(msg: Message):
        fallback_received.append(msg)
    
    await bus.subscribe("sensory.output", output_handler)
    
    # Send a query update with a very short deadline
    query_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        session_id="s1",
        topic="workspace.update",
        payload={"text": "What is 2+2?"},
    )
    await bus.publish("workspace.update", query_msg)
    
    # Wait for the 4s deadline + some buffer
    await asyncio.sleep(5.0)
    
    # The workspace should have sent a fallback response
    assert len(fallback_received) >= 1
    fallback_text = fallback_received[0].payload.get("text", "")
    assert "rephrase" in fallback_text.lower() or "unable" in fallback_text.lower() or "wasn't able" in fallback_text.lower()
    assert fallback_received[0].payload.get("source") == "workspace_fallback"
    
    await workspace.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_workspace_absolute_deadline_cap():
    """Workspace has a hard 30s cap regardless of deadline extensions."""
    bus = InProcessBus()
    await bus.start()
    
    workspace = WorkspaceNode(node_id="ws_test")
    await workspace.start(bus)
    
    query_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        session_id="s1",
        topic="workspace.update",
        payload={"text": "test query"},
    )
    await bus.publish("workspace.update", query_msg)
    await asyncio.sleep(0.1)
    
    # Verify the absolute deadline is set
    board = list(workspace.blackboards.values())[0]
    assert "absolute_deadline" in board
    assert board["absolute_deadline"] > time.time()
    assert board["absolute_deadline"] <= time.time() + 31  # ~30s from now
    
    await workspace.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_workspace_cleanup_after_commit():
    """Workspace cleans up blackboard memory after committing to decision."""
    bus = InProcessBus()
    await bus.start()
    
    workspace = WorkspaceNode(node_id="ws_test")
    await workspace.start(bus)
    
    # Capture decision messages
    decisions = []
    async def decision_handler(msg: Message):
        decisions.append(msg)
    await bus.subscribe("decision.evaluate", decision_handler)
    
    # Send query
    query_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        session_id="s1",
        topic="workspace.update",
        payload={"text": "Hello"},
    )
    await bus.publish("workspace.update", query_msg)
    await asyncio.sleep(0.1)
    
    corr_id = list(workspace.blackboards.keys())[0]
    
    # Post a thought
    thought_msg = Message(
        type=MessageType.EVENT,
        source_node_id="intuition_01",
        tenant_id="t1",
        session_id="s1",
        topic="workspace.thought",
        payload={"type": "intuition", "confidence": 0.9, "content": "Hello there!"},
        correlation_id=corr_id,
    )
    await bus.publish("workspace.thought", thought_msg)
    
    # Wait for consensus
    await asyncio.sleep(5.0)
    
    # Blackboard should be cleaned up
    assert corr_id not in workspace.blackboards
    
    await workspace.stop()
    await bus.stop()
