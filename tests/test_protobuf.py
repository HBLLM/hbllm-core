"""Tests for ProtobufSerializer and serialization factory."""

from __future__ import annotations

import pytest

from hbllm.network.messages import Message, MessageType
from hbllm.network.serialization import (
    JsonSerializer,
    MsgpackSerializer,
    ProtobufSerializer,
    get_serializer,
)


class TestSerializerFactory:
    def test_json_default(self):
        assert isinstance(get_serializer("json"), JsonSerializer)

    def test_msgpack(self):
        assert isinstance(get_serializer("msgpack"), MsgpackSerializer)

    def test_protobuf(self):
        assert isinstance(get_serializer("protobuf"), ProtobufSerializer)

    def test_unknown_falls_back_to_json(self):
        assert isinstance(get_serializer("unknown"), JsonSerializer)


class TestProtobufSerializer:
    def _make_msg(self) -> Message:
        return Message(
            type=MessageType.EVENT,
            source_node_id="test_src",
            target_node_id="test_tgt",
            topic="test.topic",
            payload={"key": "value", "count": 42},
        )

    def test_roundtrip(self):
        """Should roundtrip via protobuf or JSON fallback."""
        s = ProtobufSerializer()
        msg = self._make_msg()
        data = s.serialize(msg)
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = s.deserialize(data)
        assert restored.source_node_id == "test_src"
        assert restored.payload["key"] == "value"

    def test_multiple_roundtrips(self):
        s = ProtobufSerializer()
        for i in range(5):
            msg = Message(
                type=MessageType.QUERY,
                source_node_id=f"node_{i}",
                topic="batch",
                payload={"idx": i},
            )
            data = s.serialize(msg)
            restored = s.deserialize(data)
            assert restored.source_node_id == f"node_{i}"

    def test_empty_payload(self):
        s = ProtobufSerializer()
        msg = Message(
            type=MessageType.RESPONSE,
            source_node_id="src",
            topic="empty",
            payload={},
        )
        data = s.serialize(msg)
        restored = s.deserialize(data)
        assert restored.topic == "empty"
