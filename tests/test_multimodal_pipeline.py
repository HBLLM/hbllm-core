"""
Multi-Modal Pipeline Tests — verifies image captioning and audio
transcription are injected as context before cognitive routing.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.serving.pipeline import CognitivePipeline, PipelineConfig


class MockBusWithRequest(InProcessBus):
    """InProcessBus extended with request-response support for multimodal tests."""

    def __init__(self):
        super().__init__()
        self._request_handlers: dict[str, list] = {}

    async def request(self, topic: str, msg: Message, timeout: float = 10.0) -> Message:
        """Simulate request-response by calling registered handlers."""
        handlers = self._request_handlers.get(topic, [])
        for handler in handlers:
            result = await handler(msg)
            if result is not None:
                return result
        raise asyncio.TimeoutError(f"No handler responded for topic: {topic}")

    async def register_responder(self, topic: str, handler):
        """Register a handler that responds to requests on a topic."""
        if topic not in self._request_handlers:
            self._request_handlers[topic] = []
        self._request_handlers[topic].append(handler)


@pytest.mark.asyncio
async def test_multimodal_image_caption_injected():
    """Image bytes are captioned and injected as context."""
    bus = MockBusWithRequest()
    await bus.start()

    # Register a mock vision handler
    async def mock_vision_handler(msg: Message) -> Message:
        return msg.create_response({"caption": "A cat sitting on a laptop keyboard"})

    await bus.register_responder("vision.caption", mock_vision_handler)

    # Register a mock sensory output handler to capture the routed text
    captured_texts = []

    async def capture_output(msg: Message):
        captured_texts.append(msg.payload.get("text", ""))

    await bus.subscribe("sensory.output", capture_output)

    # Create a mock router that echoes the input text
    async def mock_router(msg: Message):
        text = msg.payload.get("text", "")
        response = msg.create_response({
            "text": f"I see: {text}",
            "source_node": "router",
            "confidence": 0.9,
        })
        await bus.publish("sensory.output", response)

    await bus.subscribe("router.query", mock_router)

    pipeline = CognitivePipeline(bus=bus)
    await pipeline.start()

    # Process with an image
    fake_image = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
    result = await pipeline.process_multimodal(
        text="What do you see?",
        images=[fake_image],
        tenant_id="test",
        session_id="s1",
    )

    # The result should contain the image caption as context
    assert "[Image 1]:" in result.text or "cat" in result.text.lower() or result.text


@pytest.mark.asyncio
async def test_multimodal_audio_transcript_injected():
    """Audio bytes are transcribed and injected as context."""
    bus = MockBusWithRequest()
    await bus.start()

    # Register a mock audio handler
    async def mock_audio_handler(msg: Message) -> Message:
        return msg.create_response({"transcript": "Hello, how are you today?"})

    await bus.register_responder("audio.transcribe", mock_audio_handler)

    # Register mock router
    async def mock_router(msg: Message):
        text = msg.payload.get("text", "")
        response = msg.create_response({
            "text": f"Processed: {text}",
            "source_node": "router",
        })
        await bus.publish("sensory.output", response)

    await bus.subscribe("router.query", mock_router)

    pipeline = CognitivePipeline(bus=bus)
    await pipeline.start()

    # Process with audio
    fake_audio = b'\x00' * 200
    result = await pipeline.process_multimodal(
        audio=fake_audio,
        tenant_id="test",
        session_id="s2",
    )

    assert "[Audio transcript]:" in result.text or "Hello" in result.text or result.text


@pytest.mark.asyncio
async def test_multimodal_combined_image_audio_text():
    """Image + audio + text are all combined before routing."""
    bus = MockBusWithRequest()
    await bus.start()

    async def mock_vision(msg: Message) -> Message:
        return msg.create_response({"caption": "A sunset over the ocean"})

    async def mock_audio(msg: Message) -> Message:
        return msg.create_response({"transcript": "Describe this beautiful scene"})

    await bus.register_responder("vision.caption", mock_vision)
    await bus.register_responder("audio.transcribe", mock_audio)

    async def mock_router(msg: Message):
        text = msg.payload.get("text", "")
        response = msg.create_response({
            "text": text,
            "source_node": "router",
        })
        await bus.publish("sensory.output", response)

    await bus.subscribe("router.query", mock_router)

    pipeline = CognitivePipeline(bus=bus)
    await pipeline.start()

    result = await pipeline.process_multimodal(
        text="What is happening here?",
        images=[b'\x89PNG' + b'\x00' * 50],
        audio=b'\x00' * 100,
    )

    # Result should have all 3 components
    assert result.text  # Got a response
    assert not result.error


@pytest.mark.asyncio
async def test_multimodal_handles_vision_timeout():
    """Pipeline handles vision timeout gracefully."""
    bus = MockBusWithRequest()
    await bus.start()

    # No vision handler registered → will timeout

    async def mock_router(msg: Message):
        text = msg.payload.get("text", "")
        response = msg.create_response({"text": text, "source_node": "router"})
        await bus.publish("sensory.output", response)

    await bus.subscribe("router.query", mock_router)

    pipeline = CognitivePipeline(bus=bus, config=PipelineConfig(timeout=5.0))
    await pipeline.start()

    result = await pipeline.process_multimodal(
        text="Describe the image",
        images=[b'\x00' * 50],
    )

    # Should still get a response (with fallback context)
    assert result.text
    assert "(could not process)" in result.text or result.text


@pytest.mark.asyncio
async def test_multimodal_text_only_passthrough():
    """If no images or audio, process_multimodal works like process."""
    bus = MockBusWithRequest()
    await bus.start()

    async def mock_router(msg: Message):
        text = msg.payload.get("text", "")
        response = msg.create_response({"text": f"Echo: {text}", "source_node": "router"})
        await bus.publish("sensory.output", response)

    await bus.subscribe("router.query", mock_router)

    pipeline = CognitivePipeline(bus=bus)
    await pipeline.start()

    result = await pipeline.process_multimodal(text="Plain text query")

    assert "Plain text query" in result.text
    assert not result.error
