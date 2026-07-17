"""
REST Transport — HTTP API transport for HBLLM.

A thin adapter that exposes a REST endpoint and bridges HTTP
requests to the Gateway. Contains zero cognitive logic.

Architecture::

    HTTP Client (curl, browser, mobile app)
        ↓
    REST Transport  (this module)
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
# REST Transport
# ═══════════════════════════════════════════════════════════════════════════


class RESTTransport:
    """HTTP REST API transport for HBLLM.

    Exposes ``POST /v1/chat`` and ``POST /v1/sessions`` endpoints
    that bridge HTTP requests to the Gateway.

    This adapter:
      1. Parses JSON request bodies.
      2. Sends them to the Gateway via handle_inbound().
      3. Collects the response via a registered callback.
      4. Returns the response as a JSON HTTP response.

    Contains NO cognitive logic.
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
        self._app: Any = None  # FastAPI app (lazily created)
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Register with Gateway and start the HTTP server."""
        self._gateway.register_transport(REST_TRANSPORT_ID, self._on_response)
        self._app = self._create_app()
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

    # ── App Factory ──────────────────────────────────────────────────────

    def _create_app(self) -> Any:
        """Create the FastAPI application with chat endpoints.

        Returns None if FastAPI is not installed.
        """
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel as PydanticModel
            from pydantic import Field
        except ImportError:
            logger.warning(
                "FastAPI not installed — REST transport endpoints unavailable. "
                "Install with: pip install fastapi uvicorn"
            )
            return None

        app = FastAPI(
            title="HBLLM REST API",
            description="REST transport gateway for the HBLLM Cognitive OS",
            version="1.0.0",
        )

        # ── Request/Response Models ──────────────────────────────────────

        class ChatRequest(PydanticModel):
            text: str
            tenant_id: str = "default"
            user_id: str = "default"
            device_id: str = "default"
            workspace_id: str = "default"
            session_id: str | None = None

        class ChatResponse(PydanticModel):
            session_id: str
            text: str
            message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

        # ── Endpoints ────────────────────────────────────────────────────

        @app.post("/v1/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest) -> ChatResponse:
            """Send a message and receive a response."""
            # Create a future for this request
            request_id = str(uuid.uuid4())
            future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
            self._pending[request_id] = future

            try:
                session_id = await self._gateway.handle_inbound(
                    transport_type=TransportType.REST,
                    transport_id=REST_TRANSPORT_ID,
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    device_id=request.device_id,
                    workspace_id=request.workspace_id,
                    text=request.text,
                )

                # Wait for cognitive response
                response_text = await asyncio.wait_for(future, timeout=120.0)

                return ChatResponse(
                    session_id=session_id,
                    text=response_text,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Response timed out")
            finally:
                self._pending.pop(request_id, None)

        @app.get("/v1/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "healthy",
                "transport": "rest",
                "active_sessions": self._gateway.active_session_count,
            }

        return app

    # ── Response Callback ────────────────────────────────────────────────

    async def _on_response(self, session_id: str, message: SessionMessage) -> None:
        """Called by the Gateway when the Brain responds.

        Resolves the oldest pending future for now. In production,
        this should correlate by session_id.
        """
        response_text = message.text
        # Resolve any pending futures
        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_result(response_text)
                break

    # ── Server Runner ────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Start the uvicorn server (blocking)."""
        if self._app is None:
            logger.error("Cannot serve — FastAPI app not created")
            return

        try:
            import uvicorn

            config = uvicorn.Config(
                self._app,
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
