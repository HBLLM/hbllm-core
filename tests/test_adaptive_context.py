"""
Tests for Adaptive Context Window compression logic in MCTS Planner and Workspace nodes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hbllm.brain.planner_node import PlannerNode, ThoughtGraph, ThoughtNode
from hbllm.brain.workspace_node import WorkspaceNode
from hbllm.network.messages import Message, MessageType


@pytest.mark.asyncio
async def test_planner_compresses_long_step():
    """Verify PlannerNode._compress_text accurately trims the middle of huge steps."""
    planner = PlannerNode(node_id="test_planner")
    planner._bus = MagicMock()

    # 5000 'A's + 5000 'B's
    massive_text = ("A" * 5000) + ("B" * 5000)
    assert len(massive_text) == 10000

    compressed = planner._compress_text(massive_text, max_chars=2000)

    # Needs to be roughly 2000 chars + the text of the omission log
    assert 2000 < len(compressed) < 2100

    # Since max_chars=2000, half=1000
    assert compressed.startswith("A" * 1000)
    assert compressed.endswith("B" * 1000)
    assert "[... 8000 characters dynamically omitted" in compressed

@pytest.mark.asyncio
async def test_planner_trajectory_middle_out_truncation():
    """Verify PlannerNode truncates deep histories (e.g., depth 10) by dropping middle steps."""
    planner = PlannerNode(node_id="test_planner")
    planner._bus = MagicMock()

    # Mock bus to capture the domain.general.query prompt
    query_payload = None
    async def mock_request(topic, msg, timeout=30.0):
        nonlocal query_payload
        query_payload = msg.payload
        return Message(type=MessageType.QUERY, payload={"text": "Mock Response"})

    planner.request = mock_request

    graph = ThoughtGraph()

    # Create a deeply nested parent history
    dict(
        id="mock_parent",
        node="internal_monologue",
        content="Step 10 Content",
        score=1.0,
        trajectory=["Step 1 Base"] + [f"Intermediate Step {i}" for i in range(2, 10)]
    )

    # We must construct a real ThoughtNode for the _refine_thought signature
    parent_node = ThoughtNode()
    parent_node.node = "test"
    parent_node.content = "Step 10 Content"
    parent_node.trajectory_history = ["Step 1 Base"] + [f"Intermediate Step {i}" for i in range(2, 10)]

    await planner._refine_thought(graph, parent_node, "How to X?", 1)

    assert query_payload is not None
    prompt = query_payload["text"]

    # Check bounded properties (MAX_CONTEXT_STEPS = 6)
    # total steps in trajectory = 9 history + 1 current = 10
    # Steps kept: 1, 6, 7, 8, 9, 10.

    assert "Step 1: Step 1 Base" in prompt
    assert "[... 4 intermediate logic steps dynamically omitted to preserve context bounds ...]" in prompt
    assert "Step 6: Intermediate Step 6" in prompt
    assert "Step 10: Step 10 Content" in prompt

    # Middle steps should be erased
    assert "Step 2: Intermediate Step 2" not in prompt
    assert "Step 3:" not in prompt

@pytest.mark.asyncio
async def test_workspace_compresses_dpo_payload():
    """Verify WorkspaceNode compresses the generated DPO payload in _emit_training_feedback."""
    workspace = WorkspaceNode(node_id="test_workspace")
    workspace._bus = MagicMock()

    published_msg = None
    async def mock_publish(topic, msg):
        nonlocal published_msg
        published_msg = msg

    workspace._bus.publish = mock_publish

    board = {
        "original_query": {"text": "Make me a sandwich"},
        "tenant_id": "test",
        "session_id": "test-session"
    }

    massive_thought = {
        "content": ("X" * 6000)
    }

    await workspace._emit_training_feedback(board, massive_thought, rating=1)

    assert published_msg is not None
    response_payload = published_msg.payload["response"]

    # Default compression bound for Workspace is 4000
    assert 4000 < len(response_payload) < 4100
    assert "[... 2000 characters dynamically omitted" in response_payload
