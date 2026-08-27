"""Unit tests for the pure ASGI RESTTransport."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hbllm.network.session import MessageRole, SessionMessage
from hbllm.network.transports.rest import RESTTransport


@pytest.fixture
def mock_gateway() -> MagicMock:
    gw = MagicMock()
    gw.active_session_count = 3
    gw.handle_inbound = AsyncMock(return_value="sess_123")
    gw.register_transport = MagicMock()
    gw.unregister_transport = MagicMock()
    return gw


@pytest.mark.asyncio
async def test_rest_transport_health_endpoint(mock_gateway: MagicMock) -> None:
    transport = RESTTransport(mock_gateway)
    await transport.start()

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/health",
    }

    sent_messages = []

    async def receive() -> dict[str, Any]:
        return {"body": b"", "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        sent_messages.append(msg)

    await transport(scope, receive, send)

    assert len(sent_messages) == 2
    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 200

    body = json.loads(sent_messages[1]["body"].decode("utf-8"))
    assert body["status"] == "healthy"
    assert body["transport"] == "rest"
    assert body["active_sessions"] == 3


@pytest.mark.asyncio
async def test_rest_transport_chat_endpoint_success(mock_gateway: MagicMock) -> None:
    transport = RESTTransport(mock_gateway)
    await transport.start()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat",
    }

    request_payload = json.dumps({"text": "Hello world"}).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"body": request_payload, "more_body": False}

    sent_messages = []

    async def send(msg: dict[str, Any]) -> None:
        sent_messages.append(msg)

    async def mock_handle_inbound(**kwargs: Any) -> str:
        resp_msg = SessionMessage.from_text(
            role=MessageRole.ASSISTANT,
            text="Hello human!",
        )
        await transport._on_response("sess_123", resp_msg)
        return "sess_123"

    mock_gateway.handle_inbound = mock_handle_inbound
    await transport(scope, receive, send)

    assert len(sent_messages) == 2
    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 200

    body = json.loads(sent_messages[1]["body"].decode("utf-8"))
    assert body["session_id"] == "sess_123"
    assert body["text"] == "Hello human!"
    assert "message_id" in body


@pytest.mark.asyncio
async def test_rest_transport_not_found(mock_gateway: MagicMock) -> None:
    transport = RESTTransport(mock_gateway)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/unknown",
    }
    sent_messages = []

    async def receive() -> dict[str, Any]:
        return {"body": b"", "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        sent_messages.append(msg)

    await transport(scope, receive, send)
    assert sent_messages[0]["status"] == 404
