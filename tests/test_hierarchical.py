import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.uplink_node import UplinkNode
from hbllm.serving.api import app as core_app


@pytest.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_uplink_node_registration():
    """Test that UplinkNode connects and registers capabilities upstream."""
    mock_ws = AsyncMock()
    mock_ws.__aiter__.return_value = []  # Empty stream for read loop

    with patch("websockets.connect") as mock_connect:
        mock_ws.__aenter__.return_value = mock_ws
        mock_connect.return_value = mock_ws

        node = UplinkNode(
            node_id="test_uplink",
            upstream_url="ws://mock.url",
            tenant_id="tenant1",
            user_id="user1",
            device_id="device1",
            local_tools=["local_tool_1", "mcp.local_mcp_tool"],
        )

        # Start the node with a mock bus to initialize self._bus
        mock_bus = AsyncMock()
        await node.start(mock_bus)

        # Give the background connect_loop a moment to run
        await asyncio.sleep(0.1)

        # Check that it connected
        mock_connect.assert_called()
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
        return Message(
            type=MessageType.EVENT,
            source_node_id="local_tool_1",
            tenant_id=msg.tenant_id,
            topic="action.tool.local_tool_1.response",
            correlation_id=msg.correlation_id,
            payload={"status": "success", "result": "mocked_result"},
        )

    await bus.subscribe("action.tool.local_tool_1", mock_tool_handler)

    with patch("websockets.connect") as mock_connect:
        mock_ws.__aenter__.return_value = mock_ws
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

        # Wait for the result to be sent back upstream (up to 3 seconds)
        for _ in range(30):
            if mock_ws.send.called:
                break
            await asyncio.sleep(0.1)

        # Check that the result was sent back upstream
        mock_ws.send.assert_called()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "tool_result"
        assert sent_data["correlation_id"] == "test_corr_123"
        assert sent_data["result"]["status"] == "success"

        await node.on_stop()


def test_sync_endpoints():
    """Test the memory sync endpoints in the Core API."""
    import os

    os.environ["HBLLM_JWT_SECRET"] = "test_secret_key_for_jwt_testing_32ch"

    client = TestClient(core_app)

    # Need a mock brain in _state for the endpoints
    from hbllm.serving.api import _state

    mock_brain = type("MockBrain", (), {"bus": AsyncMock()})()
    _state["brain"] = mock_brain

    # Create a valid JWT token to bypass auth middleware
    import jwt

    # We dynamically extract the actual secret key used by JWTAuthMiddleware
    secret_key = None
    for middleware in core_app.user_middleware:
        if middleware.cls.__name__ == "JWTAuthMiddleware":
            secret_key = middleware.kwargs.get("secret_key")

    if not secret_key:
        secret_key = "test_secret_key_for_jwt_testing_32ch"

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
