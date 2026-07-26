"""
Execution Serializer — serializes ExecutionPayload into provider-specific formats.

NOT PromptBuilder. Payloads aren't always prompts.
Providers accept: message arrays, JSON schemas, tool specs,
multimodal objects, embeddings. This handles all of them.

RESTRICTED TO FORMATTING ONLY.
Does NOT: retrieve memories, decide content, apply style,
perform RAG, or make semantic decisions.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.execution.payload import ExecutionPayload

logger = logging.getLogger(__name__)


class ExecutionSerializer:
    """
    Serializes ExecutionPayload into provider-specific formats.

    Each provider has different format requirements:
        - Local: chat template with special tokens
        - OpenAI: message array with role/content dicts
        - Anthropic: message array with system as top-level param

    This serializer handles all format differences while
    remaining purely mechanical — no semantic decisions.
    """

    def __init__(self) -> None:
        self._serializers: dict[str, Any] = {
            "local": self._serialize_local,
            "openai": self._serialize_openai,
            "anthropic": self._serialize_anthropic,
        }

    def register_format(self, provider: str, serializer: Any) -> None:
        """Register a custom serializer for a provider."""
        self._serializers[provider] = serializer

    async def serialize(
        self,
        payload: ExecutionPayload,
        provider: str,
        model_id: str | None = None,
    ) -> Any:
        """
        Serialize payload into provider-specific format.

        Args:
            payload: The execution payload to serialize.
            provider: Target provider name.
            model_id: Optional model ID for model-specific formatting.

        Returns:
            Provider-specific format (dict, string, etc.)
        """
        serializer = self._serializers.get(provider, self._serialize_default)
        return await serializer(payload, model_id)

    async def serialize_prompt(
        self,
        messages: tuple[tuple[str, str], ...],
        provider: str,
    ) -> str | list[dict[str, str]]:
        """
        Serialize (role, content) tuples into provider format.

        Convenience method for plans that store payload_messages.
        """
        if provider in ("openai", "anthropic"):
            return [{"role": role, "content": content} for role, content in messages]
        # Local: concatenate into a single string
        parts: list[str] = []
        for role, content in messages:
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            else:
                parts.append(content)
        return "\n".join(parts)

    # ── Provider-Specific Serializers ─────────────────────────

    async def _serialize_local(self, payload: ExecutionPayload, model_id: str | None) -> str:
        """Serialize for local model (concatenated chat template)."""
        parts: list[str] = []
        for msg in payload.messages:
            if msg.role == "system":
                parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}")
            else:
                parts.append(msg.content)
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    async def _serialize_openai(
        self, payload: ExecutionPayload, model_id: str | None
    ) -> list[dict[str, Any]]:
        """Serialize for OpenAI API format."""
        messages: list[dict[str, Any]] = []
        for msg in payload.messages:
            message: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                message["name"] = msg.name
            messages.append(message)
        return messages

    async def _serialize_anthropic(
        self, payload: ExecutionPayload, model_id: str | None
    ) -> dict[str, Any]:
        """Serialize for Anthropic API format (system as top-level)."""
        system_content = ""
        messages: list[dict[str, str]] = []
        for msg in payload.messages:
            if msg.role == "system":
                system_content += msg.content + "\n"
            else:
                messages.append({"role": msg.role, "content": msg.content})

        result: dict[str, Any] = {"messages": messages}
        if system_content.strip():
            result["system"] = system_content.strip()
        return result

    async def _serialize_default(
        self, payload: ExecutionPayload, model_id: str | None
    ) -> list[dict[str, str]]:
        """Default serialization: OpenAI-compatible message array."""
        return [{"role": msg.role, "content": msg.content} for msg in payload.messages]
