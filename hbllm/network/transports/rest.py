"""
REST Transport — HTTP API transport for HBLLM.

A thin, pure ASGI transport adapter that exposes REST endpoints
and bridges HTTP requests to the Gateway. Contains zero cognitive logic
and zero external framework dependencies (pure ASGI 3.0).

Architecture::

    HTTP Client (curl, browser, mobile app)
        ↓
    REST Transport  (this module / ASGI App)
        ↓
    Gateway.handle_inbound()
        ↓
    ConversationBus → Executive

Usage::

    from hbllm.network.transports.rest import RESTTransport
    from hbllm.network.gateway import Gateway

    gateway = Gateway(bus)
    rest = RESTTransport(gateway, host="0.0.0.0", port=8000)
    await rest.start()  # Starts the HTTP server
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from hbllm.network.session import (
    SessionMessage,
    TransportType,
)

logger = logging.getLogger(__name__)

REST_TRANSPORT_ID = "rest-api"


# ═══════════════════════════════════════════════════════════════════════════
# REST Transport (Pure ASGI Application)
# ═══════════════════════════════════════════════════════════════════════════


class RESTTransport:
    """Pure ASGI HTTP REST API transport for HBLLM.

    Exposes ``POST /v1/chat`` and ``GET /v1/health`` endpoints
    that bridge HTTP requests to the Gateway.

    This adapter:
      1. Parses JSON request bodies asynchronously over ASGI.
      2. Sends them to the Gateway via handle_inbound().
      3. Collects the response via a registered callback.
      4. Returns the response as a JSON HTTP response.

    Contains NO cognitive logic and NO FastAPI coupling.
    """

    def __init__(
        self,
        gateway: Any,  # Gateway
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._gateway = gateway
        self._host = host
        self._port = port

        # Pending response futures: request_id → asyncio.Future
        self._pending: dict[str, asyncio.Future[str]] = {}

        self._server_task: asyncio.Task[None] | None = None
        self._started = False

    @property
    def app(self) -> RESTTransport:
        """Return the ASGI callable application."""
        return self

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Register with Gateway and mark transport active."""
        self._gateway.register_transport(REST_TRANSPORT_ID, self._on_response)
        self._started = True
        logger.info("REST transport registered with Gateway on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the HTTP server and unregister."""
        self._gateway.unregister_transport(REST_TRANSPORT_ID)
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
        # Cancel any pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result("")
        self._pending.clear()
        self._started = False
        logger.info("REST transport stopped")

    # ── Pure ASGI 3.0 Interface ──────────────────────────────────────────

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """Handle incoming ASGI connection."""
        if scope["type"] != "http":
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()

        if path == "/v1/health" and method == "GET":
            await self._handle_health(send)
        elif path == "/v1/chat" and method == "POST":
            await self._handle_chat(receive, send)
        else:
            await self._send_json(send, 404, {"error": "Not Found", "path": path})

    async def _handle_health(self, send: Any) -> None:
        """Handle GET /v1/health."""
        active_count = getattr(self._gateway, "active_session_count", 0)
        await self._send_json(
            send,
            200,
            {
                "status": "healthy",
                "transport": "rest",
                "active_sessions": active_count,
            },
        )

    async def _handle_chat(self, receive: Any, send: Any) -> None:
        """Handle POST /v1/chat."""
        body_bytes = b""
        more_body = True
        while more_body:
            message = await receive()
            body_bytes += message.get("body", b"")
            more_body = message.get("more_body", False)

        try:
            data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            await self._send_json(send, 400, {"error": "Invalid JSON payload"})
            return

        text = data.get("text", "")
        if not text:
            await self._send_json(send, 400, {"error": "Field 'text' is required"})
            return

        tenant_id = data.get("tenant_id", "default")
        user_id = data.get("user_id", "default")
        device_id = data.get("device_id", "default")
        workspace_id = data.get("workspace_id", "default")

        # Create a future for this request
        request_id = str(uuid.uuid4())
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        try:
            session_id = await self._gateway.handle_inbound(
                transport_type=TransportType.REST,
                transport_id=REST_TRANSPORT_ID,
                tenant_id=tenant_id,
                user_id=user_id,
                device_id=device_id,
                workspace_id=workspace_id,
                text=text,
            )

            # Wait for cognitive response (timeout: 120s)
            response_text = await asyncio.wait_for(future, timeout=120.0)

            await self._send_json(
                send,
                200,
                {
                    "session_id": session_id,
                    "text": response_text,
                    "message_id": str(uuid.uuid4()),
                },
            )
        except asyncio.TimeoutError:
            await self._send_json(send, 504, {"error": "Response timed out"})
        except Exception as e:
            logger.error("Error processing REST chat request: %s", e)
            await self._send_json(send, 500, {"error": "Internal server error"})
        finally:
            self._pending.pop(request_id, None)

    async def _send_json(self, send: Any, status_code: int, data: dict[str, Any]) -> None:
        """Send an HTTP JSON response over ASGI."""
        body = json.dumps(data).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("utf-8")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )

    # ── Response Callback ────────────────────────────────────────────────

    async def _on_response(self, session_id: str, message: SessionMessage) -> None:
        """Called by the Gateway when the Brain responds."""
        response_text = message.text
        for _request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_result(response_text)
                break

    # ── Server Runner ────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Start the uvicorn ASGI server (blocking)."""
        try:
            import uvicorn

            config = uvicorn.Config(
                self,
                host=self._host,
                port=self._port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError:
            logger.error(
                "uvicorn not installed — cannot start REST server. "
                "Install with: pip install uvicorn"
            )
