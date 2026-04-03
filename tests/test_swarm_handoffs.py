"""Tests for Multi-Agent Swarm Handoffs (Horizon 2)."""

import asyncio
import time

import pytest

from hbllm.brain.router_node import RouterNode
from hbllm.brain.workspace_node import WorkspaceNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


async def _poll_until(condition_fn, timeout=5.0, interval=0.1):
    start = time.time()
    while time.time() - start < timeout:
        if condition_fn():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_native_swarm_transfer():
    """
    Test that an initial query routed to one specialty can be gracefully
    bounced to another specialty without losing its correlation_id.
    """
    bus = InProcessBus()
    await bus.start()

    workspace = WorkspaceNode(node_id="workspace_swarm", thinking_deadline=1.0)
    router = RouterNode(node_id="router_swarm", default_domain="general", llm=None, use_vectors=False)

    await workspace.start(bus)
    await router.start(bus)

    updates_received = []
    async def track_updates(msg):
        updates_received.append(msg)
    await bus.subscribe("workspace.update", track_updates)

    try:
        corr_id = "swarm_session_123"

        # 1. Initial query drops on Router
        initial_msg = Message(
            type=MessageType.QUERY,
            source_node_id="user",
            topic="router.query",
            payload={"text": "I need help with a poem about math", "domain_hint": "general"},
            correlation_id=corr_id
        )
        await bus.publish("router.query", initial_msg)

        # Wait for Router to bounce it to Workspace
        assert await _poll_until(lambda: len(updates_received) == 1)

        first_update = updates_received[0]
        assert first_update.correlation_id == corr_id

        # Ensure Workspace picked it up
        assert await _poll_until(lambda: len(workspace.blackboards) == 1)

        # 2. General Agent reads it, realizes it's better for the Poetry/Creative module
        handoff_thought = Message(
            type=MessageType.EVENT,
            source_node_id="agent_general",
            topic="workspace.thought",
            payload={
                "type": "swarm_transfer",
                "confidence": 1.0,
                "content": "creative_writer", # The target target_domain
            },
            correlation_id=corr_id
        )
        await bus.publish("workspace.thought", handoff_thought)

        # Wait for the Workspace to clear the board and Router to republish it
        assert await _poll_until(lambda: len(updates_received) == 2, timeout=2.0)

        # The swarm integration cleared the local blackboard and transferred it
        # Eventually the new update brings it BACK to the workspace as a new board
        assert await _poll_until(lambda: len(workspace.blackboards) == 1, timeout=2.0)

        # 3. Verify the newly bounced query payload
        second_update = updates_received[1]
        assert second_update.correlation_id == corr_id # Must NOT drop connection
        assert second_update.payload["domain_hint"] == "creative_writer"
        assert "SWARM TRANSFER" in second_update.payload["text"]
        assert "agent_general" in second_update.payload["text"]
    finally:
        await workspace.stop()
        await router.stop()
        await bus.stop()
