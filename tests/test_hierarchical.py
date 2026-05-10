import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
import websockets
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.uplink_node import UplinkNode
from hbllm.serving.api import app as core_app


@pytest.fixture
def bus():
    b = InProcessBus()
    return b


@pytest.mark.asyncio
async def test_uplink_node_registration():
    """Test that UplinkNode connects and registers capabilities upstream."""
    mock_ws = AsyncMock()
    mock_ws.__aiter__.return_value = []  # Empty stream for read loop

    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_ws

        node = UplinkNode(
            node_id="test_uplink",
            upstream_url="ws://mock.url",
            tenant_id="tenant1",
            user_id="user1",
            device_id="device1",
            local_tools=["local_tool_1", "mcp.local_mcp_tool"],
        )

        # Trigger on_start directly to avoid needing a full bus lifecycle
        await node.on_start()

        # Check that it connected
        mock_connect.assert_called_once()
        assert "tenant_id=tenant1" in mock_connect.call_args[0][0]

        # Check that it sent capabilities
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_capabilities"
        assert "local_tool_1" in sent_data["tools"]

        await node.on_stop()


@pytest.mark.asyncio
async def test_uplink_node_tool_routing(bus):
    """Test that UplinkNode routes upstream tool_calls to the local bus."""
    mock_ws = AsyncMock()

    # Simulate receiving a tool call from upstream
    mock_ws.__aiter__.return_value = [
        json.dumps(
            {
                "type": "tool_call",
                "correlation_id": "test_corr_123",
                "tool_name": "local_tool_1",
                "args": {"param": "value"},
            }
        )
    ]

    # Create a local responder for the tool
    async def mock_tool_handler(msg: Message):
        result_msg = Message(
            type=MessageType.EVENT,
            source_node_id="local_tool_1",
            tenant_id=msg.tenant_id,
            topic="action.tool.local_tool_1.response",
            correlation_id=msg.correlation_id,
            payload={"status": "success", "result": "mocked_result"},
        )
        # Using the bus's internal response mechanism
        # For InProcessBus, request() creates a future in _requests
        if msg.correlation_id in bus._requests:
            bus._requests[msg.correlation_id].set_result(result_msg)

    await bus.subscribe("action.tool.local_tool_1", mock_tool_handler)

    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_ws

        node = UplinkNode(
            node_id="test_uplink",
            upstream_url="ws://mock.url",
            tenant_id="tenant1",
            user_id="user1",
            device_id="device1",
        )

        # Start the node correctly using the base class method which injects the bus
        await node.start(bus)

        # Wait a tiny bit for async tasks to settle
        await asyncio.sleep(0.1)

        # Check that the result was sent back upstream
        assert mock_ws.send.call_count == 1
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "tool_result"
        assert sent_data["correlation_id"] == "test_corr_123"
        assert sent_data["result"]["status"] == "success"

        await node.on_stop()


def test_sync_endpoints():
    """Test the memory sync endpoints in the Core API."""
    client = TestClient(core_app)

    # Need a mock brain in _state for the endpoints
    from hbllm.serving.api import _state

    mock_brain = type("MockBrain", (), {"bus": AsyncMock()})()
    _state["brain"] = mock_brain

    # Create a valid JWT token to bypass auth middleware
    import os

    import jwt

    auth_mw = next(
        (
            m
            for m in getattr(core_app, "user_middleware", [])
            if hasattr(m, "kwargs") and "secret_key" in m.kwargs
        ),
        None,
    )
    secret_key = auth_mw.kwargs.get("secret_key", "test_secret") if auth_mw else "test_secret"

    # Reload auth middleware config if needed, or just mock it.
    token = jwt.encode(
        {"tenant_id": "tenant1", "user_id": "user1", "device_id": "device1"},
        secret_key,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}", "X-Tenant-ID": "tenant1"}

    # Test Episodic Sync
    resp = client.post(
        "/v1/sync/episodic",
        json={"memories": [{"content": "Hello", "role": "user"}]},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["synced"] == 1

    # Test Semantic Sync
    resp = client.post(
        "/v1/sync/semantic",
        json={"knowledge_items": [{"fact": "The sky is blue"}]},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["synced"] == 1
