"""
Gateway — Central dispatcher between Transports and the Cognitive Core.

The Gateway is the single choke-point where external transport events
are normalized into ConversationSessions and published onto the
ConversationBus. Responses from the Brain flow back through the Gateway
to the originating transport.

Architecture::

    Transport  ──┐
    Transport  ──┤──►  Gateway  ──►  ConversationBus  ──►  Executive
    Transport  ──┘         ▲
                           │
              Response  ◄──┘ (routed back via TransportContext)

Responsibilities:
    - Accept inbound messages from any transport.
    - Create or resume ConversationSessions.
    - Validate tenant/user identity.
    - Publish normalized session messages onto the ConversationBus.
    - Route Brain responses back to the correct transport.
    - Manage session lifecycle (idle timeout, migration, cleanup).

The Gateway MUST NOT contain any cognitive logic. It is infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from hbllm.network.messages import Message, MessageType, Priority
from hbllm.network.session import (
    ContentItem,
    ContentModality,
    ConversationSession,
    MessageRole,
    SessionMessage,
    SessionState,
    TransportContext,
    TransportType,
)

logger = logging.getLogger(__name__)

# Type alias for transport response callbacks
ResponseCallback = Callable[[str, SessionMessage], Coroutine[Any, Any, None]]


# ═══════════════════════════════════════════════════════════════════════════
# Gateway Configuration
# ═══════════════════════════════════════════════════════════════════════════


class GatewayConfig:
    """Configuration for the Gateway."""

    def __init__(
        self,
        *,
        session_idle_timeout_s: float = 1800.0,  # 30 minutes
        session_max_history: int = 200,
        enable_session_persistence: bool = True,
        max_concurrent_sessions: int = 1000,
    ) -> None:
        self.session_idle_timeout_s = session_idle_timeout_s
        self.session_max_history = session_max_history
        self.enable_session_persistence = enable_session_persistence
        self.max_concurrent_sessions = max_concurrent_sessions


# ═══════════════════════════════════════════════════════════════════════════
# Gateway
# ═══════════════════════════════════════════════════════════════════════════


class Gateway:
    """Central dispatcher between transports and the cognitive MessageBus.

    Usage::

        from hbllm.network.bus import InProcessBus
        from hbllm.network.gateway import Gateway

        bus = InProcessBus()
        gateway = Gateway(bus)
        await gateway.start()

        # Transports call this to deliver messages:
        response = await gateway.handle_inbound(
            transport_type=TransportType.CLI,
            transport_id="cli-main",
            tenant_id="default",
            user_id="dumith",
            text="What's the weather?",
        )

    The Gateway maintains a session table keyed by
    (tenant_id, user_id, transport_id).
    """

    def __init__(
        self,
        bus: Any,  # MessageBus protocol
        config: GatewayConfig | None = None,
    ) -> None:
        self._bus = bus
        self._config = config or GatewayConfig()

        # Active sessions: key = (tenant_id, user_id, transport_id)
        self._sessions: dict[tuple[str, str, str], ConversationSession] = {}

        # Transport response callbacks: transport_id → callback
        self._transport_callbacks: dict[str, ResponseCallback] = {}

        # Background tasks
        self._cleanup_task: asyncio.Task[None] | None = None
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the gateway and subscribe to brain response topics."""
        if self._started:
            return

        # Subscribe to responses from the Brain
        await self._bus.subscribe("session.response", self._handle_brain_response)

        # Start background session cleanup
        self._cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        self._started = True
        logger.info("Gateway started — accepting transport connections")

    async def stop(self) -> None:
        """Stop the gateway and clean up all sessions."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all active sessions
        for session in list(self._sessions.values()):
            session.close()

        self._sessions.clear()
        self._transport_callbacks.clear()
        self._started = False
        logger.info("Gateway stopped — all sessions closed")

    # ── Transport Registration ───────────────────────────────────────────

    def register_transport(
        self,
        transport_id: str,
        callback: ResponseCallback,
    ) -> None:
        """Register a transport's response callback.

        Transports call this at startup so the Gateway can route
        Brain responses back to the correct transport.

        Args:
            transport_id: Unique identifier for the transport instance.
            callback: Async function(session_id, SessionMessage) called
                      when the Brain produces a response for this transport.
        """
        self._transport_callbacks[transport_id] = callback
        logger.debug("Registered transport callback: %s", transport_id)

    def unregister_transport(self, transport_id: str) -> None:
        """Remove a transport's response callback."""
        self._transport_callbacks.pop(transport_id, None)
        logger.debug("Unregistered transport callback: %s", transport_id)

    # ── Inbound Message Handling ─────────────────────────────────────────

    async def handle_inbound(
        self,
        *,
        transport_type: TransportType,
        transport_id: str,
        tenant_id: str = "default",
        user_id: str = "default",
        device_id: str = "default",
        workspace_id: str = "default",
        text: str | None = None,
        content: list[ContentItem] | None = None,
        platform_user_id: str = "",
        channel_id: str = "",
        platform_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Accept an inbound message from any transport.

        Creates or resumes a session, appends the message, and publishes
        it to the ConversationBus for cognitive processing.

        Args:
            transport_type: Which transport is sending this.
            transport_id: Unique transport instance ID.
            tenant_id: Tenant scope for multi-tenant isolation.
            user_id: Authenticated user ID.
            device_id: Originating device.
            workspace_id: Active workspace context.
            text: Text content (convenience for text-only messages).
            content: Full multimodal content items.
            platform_user_id: Platform-specific user identifier.
            channel_id: Platform-specific channel/room.
            platform_metadata: Additional platform-specific data.

        Returns:
            The session ID that was used/created.
        """
        # 1. Resolve or create session
        session_key = (tenant_id, user_id, transport_id)
        session = self._sessions.get(session_key)

        if session is None or not session.is_active:
            session = self._create_session(
                tenant_id=tenant_id,
                user_id=user_id,
                device_id=device_id,
                workspace_id=workspace_id,
                transport_type=transport_type,
                transport_id=transport_id,
                platform_user_id=platform_user_id,
                channel_id=channel_id,
                platform_metadata=platform_metadata or {},
            )
            self._sessions[session_key] = session

        # 2. Build the message
        if content:
            items = content
        elif text:
            items = [ContentItem(modality=ContentModality.TEXT, text=text)]
        else:
            items = [ContentItem(modality=ContentModality.TEXT, text="")]

        user_msg = SessionMessage(
            role=MessageRole.USER,
            content=items,
        )
        session.add_message(user_msg)

        # 3. Publish to ConversationBus
        bus_message = Message(
            type=MessageType.QUERY,
            source_node_id="gateway",
            target_node_id=None,  # Broadcast to router
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            device_id=session.device_id,
            session_id=session.id,
            topic="session.message",
            payload={
                "session_id": session.id,
                "workspace_id": session.workspace_id,
                "text": user_msg.text,
                "has_media": user_msg.has_media,
                "message_id": user_msg.id,
                "transport_type": transport_type.value,
                "transport_id": transport_id,
                "history": session.to_llm_messages()[-20:],  # Last 20 turns
            },
            priority=Priority.NORMAL,
        )
        await self._bus.publish("session.message", bus_message)

        logger.debug(
            "Gateway dispatched message from %s/%s [session=%s]",
            transport_type.value,
            user_id,
            session.id[:8],
        )
        return session.id

    # ── Response Routing ─────────────────────────────────────────────────

    async def _handle_brain_response(self, message: Message) -> Message | None:
        """Route a Brain response back to the originating transport.

        Called when the Brain publishes to ``session.response``.
        """
        session_id = message.payload.get("session_id", "")
        response_text = message.payload.get("text", "")
        transport_id = message.payload.get("transport_id", "")

        if not transport_id:
            # Look up transport_id from session
            session = self._find_session_by_id(session_id)
            if session:
                transport_id = session.transport.transport_id

        # Append assistant response to session history
        session = self._find_session_by_id(session_id)
        if session and response_text:
            session.add_assistant_text(response_text)

        # Route to transport callback
        callback = self._transport_callbacks.get(transport_id)
        if callback and response_text:
            assistant_msg = SessionMessage.from_text(
                MessageRole.ASSISTANT,
                response_text,
            )
            try:
                await callback(session_id, assistant_msg)
            except Exception:
                logger.exception(
                    "Error routing response to transport %s",
                    transport_id,
                )

        return None

    # ── Session Management ───────────────────────────────────────────────

    def _create_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        device_id: str,
        workspace_id: str,
        transport_type: TransportType,
        transport_id: str,
        platform_user_id: str,
        channel_id: str,
        platform_metadata: dict[str, Any],
    ) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            workspace_id=workspace_id,
            transport=TransportContext(
                transport_type=transport_type,
                transport_id=transport_id,
                platform_user_id=platform_user_id,
                channel_id=channel_id,
                platform_metadata=platform_metadata,
            ),
            state=SessionState.PENDING,
        )
        logger.info(
            "Created session %s for %s/%s via %s",
            session.id[:8],
            tenant_id,
            user_id,
            transport_type.value,
        )
        return session

    def _find_session_by_id(self, session_id: str) -> ConversationSession | None:
        """Look up a session by its ID."""
        for session in self._sessions.values():
            if session.id == session_id:
                return session
        return None

    def get_session(
        self, tenant_id: str, user_id: str, transport_id: str
    ) -> ConversationSession | None:
        """Get an active session by key."""
        return self._sessions.get((tenant_id, user_id, transport_id))

    @property
    def active_session_count(self) -> int:
        """Number of active sessions."""
        return sum(1 for s in self._sessions.values() if s.is_active)

    # ── Background Maintenance ───────────────────────────────────────────

    async def _session_cleanup_loop(self) -> None:
        """Background task to expire idle sessions."""
        while True:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                now = time.time()
                expired_keys: list[tuple[str, str, str]] = []

                for key, session in self._sessions.items():
                    if not session.is_active:
                        expired_keys.append(key)
                        continue
                    elapsed = now - session.last_activity_at.timestamp()
                    if elapsed > self._config.session_idle_timeout_s:
                        session.state = SessionState.EXPIRED
                        expired_keys.append(key)
                        logger.debug(
                            "Session %s expired after %.0fs idle",
                            session.id[:8],
                            elapsed,
                        )

                for key in expired_keys:
                    self._sessions.pop(key, None)

            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error in session cleanup loop")
                await asyncio.sleep(5.0)
