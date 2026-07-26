"""Tests for ExecutionPayload and related types."""

from __future__ import annotations

import pytest

from hbllm.execution.payload import (
    ExecutionPayload,
    PayloadImage,
    PayloadMessage,
    PayloadTool,
)


class TestPayloadMessage:
    def test_basic(self) -> None:
        msg = PayloadMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_frozen(self) -> None:
        msg = PayloadMessage(role="user", content="Hello")
        with pytest.raises(AttributeError):
            msg.content = "Bye"  # type: ignore[misc]


class TestExecutionPayload:
    def test_empty(self) -> None:
        payload = ExecutionPayload()
        assert payload.messages == ()
        assert payload.images == ()
        assert not payload.has_images
        assert not payload.has_audio
        assert not payload.is_multimodal

    def test_from_prompt(self) -> None:
        payload = ExecutionPayload.from_prompt("Hello world")
        assert len(payload.messages) == 1
        assert payload.messages[0].role == "user"
        assert payload.messages[0].content == "Hello world"

    def test_from_prompt_with_system(self) -> None:
        payload = ExecutionPayload.from_prompt("Hello", system="Be helpful")
        assert len(payload.messages) == 2
        assert payload.messages[0].role == "system"
        assert payload.messages[0].content == "Be helpful"
        assert payload.messages[1].role == "user"

    def test_from_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        payload = ExecutionPayload.from_messages(messages)
        assert len(payload.messages) == 2
        assert payload.messages[0].role == "system"
        assert payload.messages[1].content == "Hi"

    def test_multimodal_detection(self) -> None:
        payload = ExecutionPayload(
            messages=(PayloadMessage(role="user", content="Describe this"),),
            images=(PayloadImage(url="https://example.com/img.png"),),
        )
        assert payload.has_images
        assert payload.is_multimodal
        assert not payload.has_audio

    def test_has_tools(self) -> None:
        payload = ExecutionPayload(
            tools=(PayloadTool(name="search", description="Search the web"),),
        )
        assert payload.has_tools

    def test_frozen(self) -> None:
        payload = ExecutionPayload.from_prompt("Hello")
        with pytest.raises(AttributeError):
            payload.messages = ()  # type: ignore[misc]
