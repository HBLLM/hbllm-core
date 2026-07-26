"""
Execution Payload — rich multimodal content for execution.

Universal across all runtimes. Each runtime serializes this into
its own format. TextRuntime renders messages. VisionRuntime uses
images. BrowserRuntime ignores most of it.

This replaces the old ``payload: str`` approach — payloads aren't
always prompts. Providers accept message arrays, JSON schemas,
tool specs, multimodal objects, and embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PayloadMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None  # For tool messages


@dataclass(frozen=True)
class PayloadImage:
    """An image attachment for multimodal payloads."""

    url: str | None = None
    base64_data: str | None = None
    media_type: str = "image/png"
    detail: str = "auto"  # "auto", "low", "high"


@dataclass(frozen=True)
class PayloadAudio:
    """An audio attachment for speech/audio payloads."""

    url: str | None = None
    base64_data: str | None = None
    media_type: str = "audio/wav"
    duration_ms: int | None = None


@dataclass(frozen=True)
class PayloadDocument:
    """A document attachment (PDF, text, etc.)."""

    content: str = ""
    url: str | None = None
    media_type: str = "text/plain"
    title: str | None = None


@dataclass(frozen=True)
class PayloadTool:
    """A tool specification for function calling."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PayloadAttachment:
    """Generic attachment for future extensibility."""

    name: str
    content_type: str
    data: str = ""  # Base64 or text
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPayload:
    """
    Universal execution payload.

    Each runtime serializes this into its provider-specific format.
    TextRuntime renders messages into chat templates.
    VisionRuntime extracts images.
    ToolRuntime extracts tool specs.

    All fields are immutable tuples for frozen dataclass compatibility.
    """

    messages: tuple[PayloadMessage, ...] = ()
    images: tuple[PayloadImage, ...] = ()
    audio: tuple[PayloadAudio, ...] = ()
    documents: tuple[PayloadDocument, ...] = ()
    tools: tuple[PayloadTool, ...] = ()
    structured_output: dict[str, Any] | None = None
    attachments: tuple[PayloadAttachment, ...] = ()

    @staticmethod
    def from_prompt(prompt: str, system: str | None = None) -> ExecutionPayload:
        """
        Convenience: create a payload from a simple prompt string.

        Args:
            prompt: The user message content.
            system: Optional system message.

        Returns:
            ExecutionPayload with messages populated.
        """
        messages: list[PayloadMessage] = []
        if system:
            messages.append(PayloadMessage(role="system", content=system))
        messages.append(PayloadMessage(role="user", content=prompt))
        return ExecutionPayload(messages=tuple(messages))

    @staticmethod
    def from_messages(
        messages: list[dict[str, str]],
    ) -> ExecutionPayload:
        """
        Convenience: create from a list of message dicts.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.

        Returns:
            ExecutionPayload with messages populated.
        """
        return ExecutionPayload(
            messages=tuple(
                PayloadMessage(
                    role=m["role"],
                    content=m["content"],
                    name=m.get("name"),
                )
                for m in messages
            )
        )

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0

    @property
    def has_audio(self) -> bool:
        return len(self.audio) > 0

    @property
    def has_tools(self) -> bool:
        return len(self.tools) > 0

    @property
    def is_multimodal(self) -> bool:
        return self.has_images or self.has_audio
