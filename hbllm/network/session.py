"""
ConversationSession — Channel-agnostic session abstraction.

Every user interaction — whether from CLI, REST, Telegram, Discord,
WebSocket, or any future transport — is normalized into a
ConversationSession before reaching the cognitive core.

The Brain never knows whether the user is typing in a terminal or
Slack. All transports produce identical session frames.

Architecture::

    Transport (cli, telegram, rest, ws)
        ↓
    Gateway
        ↓
    ConversationSession  ← this module
        ↓
    ConversationBus
        ↓
    Executive

Bus Topics:
    session.start       → New session opened
    session.message     → User message within a session
    session.end         → Session closed / disconnected
    session.migrate     → Session transferred to another transport
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class TransportType(StrEnum):
    """Identifies the originating transport channel."""

    CLI = "cli"
    REST = "rest"
    WEBSOCKET = "websocket"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    WHATSAPP = "whatsapp"
    SIGNAL = "signal"
    IDE = "ide"  # VSCode, JetBrains, etc.
    INTERNAL = "internal"  # System-generated (daemon, scheduler)
    UNKNOWN = "unknown"


class SessionState(StrEnum):
    """Lifecycle state of a conversation session."""

    PENDING = "pending"  # Created, not yet active
    ACTIVE = "active"  # User is actively interacting
    IDLE = "idle"  # No recent activity (may be consolidated)
    MIGRATING = "migrating"  # Transferring to another transport
    CLOSED = "closed"  # Explicitly ended
    EXPIRED = "expired"  # Timed out


class MessageRole(StrEnum):
    """Role of a message participant."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentModality(StrEnum):
    """Type of content in a session message."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    CODE = "code"
    STRUCTURED = "structured"  # JSON, forms, etc.


# ═══════════════════════════════════════════════════════════════════════════
# Content & Message Models
# ═══════════════════════════════════════════════════════════════════════════


class ContentItem(BaseModel):
    """A single piece of content within a session message.

    Supports multimodal content — text, images, audio, files, etc.
    """

    modality: ContentModality = ContentModality.TEXT
    text: str | None = None
    data: bytes | None = None  # Raw binary (image, audio)
    uri: str | None = None  # External reference (file path, URL)
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class SessionMessage(BaseModel):
    """A single message within a conversation session.

    This is the normalized representation that reaches the Brain.
    Transport-specific formatting (Markdown, Slack blocks, Discord
    embeds) has already been stripped by the transport adapter.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: list[ContentItem] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Convenience: concatenate all text content items."""
        parts = [item.text for item in self.content if item.text]
        return "\n".join(parts)

    @property
    def has_media(self) -> bool:
        """Check if this message contains non-text content."""
        return any(item.modality != ContentModality.TEXT for item in self.content)

    @classmethod
    def from_text(cls, role: MessageRole, text: str, **kwargs: Any) -> SessionMessage:
        """Create a simple text-only message."""
        return cls(
            role=role,
            content=[ContentItem(modality=ContentModality.TEXT, text=text)],
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Transport Context
# ═══════════════════════════════════════════════════════════════════════════


class TransportContext(BaseModel):
    """Metadata about the originating transport.

    Carried along with the session but NEVER inspected by cognitive
    components. Only the Gateway and response routing use this.
    """

    transport_type: TransportType = TransportType.UNKNOWN
    transport_id: str = ""  # Unique transport instance identifier
    channel_id: str = ""  # Platform-specific channel (e.g., Slack channel ID)
    platform_user_id: str = ""  # Platform-specific user ID
    platform_metadata: dict[str, Any] = Field(default_factory=dict)
    reply_callback_id: str | None = None  # For async response routing


# ═══════════════════════════════════════════════════════════════════════════
# ConversationSession
# ═══════════════════════════════════════════════════════════════════════════


class ConversationSession(BaseModel):
    """The canonical session abstraction.

    Every interaction with HBLLM is mediated through a session.
    Sessions are created by transports, dispatched through the Gateway,
    and consumed by the Executive. The Brain sees *only* this object.

    Lifecycle::

        Transport creates ConversationSession
            ↓
        Gateway validates & enriches (tenant, workspace)
            ↓
        ConversationBus publishes to cognitive nodes
            ↓
        Executive processes via Router → Planner → Actions
            ↓
        Response flows back through Gateway → Transport

    Attributes:
        id: Unique session identifier (UUID).
        tenant_id: Multi-tenant isolation scope.
        user_id: Authenticated user within the tenant.
        device_id: Originating device identifier.
        workspace_id: Active workspace context.
        transport: Opaque transport metadata (never read by Brain).
        state: Current session lifecycle state.
        history: Ordered conversation messages (most recent last).
        context: Arbitrary context injected by Gateway or plugins.
        created_at: Session creation timestamp.
        last_activity_at: Last message timestamp (for idle detection).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Identity scope — the Brain uses these for routing & isolation
    tenant_id: str = "default"
    user_id: str = "default"
    device_id: str = "default"
    workspace_id: str = "default"

    # Transport context — opaque to the cognitive core
    transport: TransportContext = Field(default_factory=TransportContext)

    # Session state
    state: SessionState = SessionState.PENDING

    # Conversation content
    history: list[SessionMessage] = Field(default_factory=list)

    # Extensible context bag — plugins, middleware, and Gateway can
    # inject additional context without schema changes.
    context: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Mutators ─────────────────────────────────────────────────────────

    def add_message(self, message: SessionMessage) -> None:
        """Append a message and update activity timestamp."""
        self.history.append(message)
        self.last_activity_at = datetime.now(timezone.utc)
        if self.state == SessionState.PENDING:
            self.state = SessionState.ACTIVE

    def add_user_text(self, text: str) -> SessionMessage:
        """Convenience: create and append a user text message."""
        msg = SessionMessage.from_text(MessageRole.USER, text)
        self.add_message(msg)
        return msg

    def add_assistant_text(self, text: str) -> SessionMessage:
        """Convenience: create and append an assistant text message."""
        msg = SessionMessage.from_text(MessageRole.ASSISTANT, text)
        self.add_message(msg)
        return msg

    def close(self) -> None:
        """Mark the session as closed."""
        self.state = SessionState.CLOSED

    def mark_idle(self) -> None:
        """Mark the session as idle (no recent activity)."""
        if self.state == SessionState.ACTIVE:
            self.state = SessionState.IDLE

    # ── Queries ──────────────────────────────────────────────────────────

    @property
    def latest_user_text(self) -> str | None:
        """Get the text of the most recent user message."""
        for msg in reversed(self.history):
            if msg.role == MessageRole.USER and msg.text:
                return msg.text
        return None

    @property
    def message_count(self) -> int:
        return len(self.history)

    @property
    def is_active(self) -> bool:
        return self.state in (SessionState.ACTIVE, SessionState.PENDING)

    def to_llm_messages(self) -> list[dict[str, str]]:
        """Convert session history to standard LLM message format.

        Returns a list of ``{"role": "...", "content": "..."}`` dicts
        compatible with OpenAI / Anthropic / local model APIs.
        """
        return [{"role": msg.role.value, "content": msg.text} for msg in self.history if msg.text]
