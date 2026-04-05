"""
Message serialization — converts messages to/from wire format.

Phase 1-2: Uses msgpack for fast binary serialization.
Phase 3+: Can switch to protobuf for cross-language compatibility.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

# msgpack might not have stubs installed
try:
    import msgpack  # type: ignore
except ImportError:
    msgpack = None

from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class Serializer(ABC):
    """Abstract serializer for messages."""

    @abstractmethod
    def serialize(self, message: Message) -> bytes:
        """Serialize a message to bytes."""
        ...

    @abstractmethod
    def deserialize(self, data: bytes) -> Message:
        """Deserialize bytes back to a message."""
        ...


class JsonSerializer(Serializer):
    """JSON-based serialization (human-readable, good for debugging)."""

    def serialize(self, message: Message) -> bytes:
        return message.model_dump_json().encode("utf-8")

    def deserialize(self, data: bytes) -> Message:
        return Message.model_validate_json(data)


class MsgpackSerializer(Serializer):
    """
    Msgpack-based serialization (fast, compact binary format).

    Falls back to JSON if msgpack is not available.
    """

    def __init__(self) -> None:
        self._fallback = JsonSerializer()
        if msgpack is not None:
            self._has_msgpack = True
        else:
            logger.warning("msgpack not installed, falling back to JSON serialization")
            self._has_msgpack = False

    def serialize(self, message: Message) -> bytes:
        if not self._has_msgpack or msgpack is None:
            return self._fallback.serialize(message)

        data = message.model_dump(mode="json")
        return cast(bytes, msgpack.packb(data, use_bin_type=True))

    def deserialize(self, data: bytes) -> Message:
        if not self._has_msgpack or msgpack is None:
            return self._fallback.deserialize(data)

        raw = msgpack.unpackb(data, raw=False)
        return Message.model_validate(raw)


def get_serializer(format_name: str = "json") -> Serializer:
    """Factory function to get a serializer by format name."""
    if format_name == "msgpack":
        return MsgpackSerializer()
    return JsonSerializer()
