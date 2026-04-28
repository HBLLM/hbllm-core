"""
Cognitive Stream — Async streaming interface for real-time token output.

Platform-agnostic bus-based streaming for any HBLLM brain.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from hbllm.network.bus import MessageBus
from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class CognitiveStream:
    """
    Async iterator that streams tokens from the cognitive pipeline.

    Usage:
        stream = CognitiveStream(bus, correlation_id)
        await stream.start()
        async for chunk in stream:
            print(chunk["text"], end="", flush=True)
    """

    def __init__(self, bus: MessageBus, correlation_id: str, timeout: float = 120.0):
        self._bus = bus
        self._corr_id = correlation_id
        self._timeout = timeout
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._done = False
        self._subscriptions: list = []

    async def _handle_chunk(self, message: Message) -> Message | None:
        """Capture streaming chunks and internal thoughts from the bus."""
        if message.correlation_id != self._corr_id:
            return None

        topic = message.topic
        if topic == "system.thought":
            thought = message.payload.get("text", "")
            if thought:
                await self._queue.put({"type": "thought", "text": thought})

        elif topic == "sensory.stream.chunk":
            chunk = message.payload.get("text", "")
            if chunk:
                await self._queue.put({"type": "token", "text": chunk})

        elif topic in ("sensory.output", "sensory.stream.end"):
            # Final output — signal done
            final = message.payload.get("text", "")
            if final:
                await self._queue.put({"type": "token", "text": final})
            await self._queue.put(None)  # Sentinel
        return None

    async def start(self) -> None:
        """Subscribe to stream events and internal thoughts."""
        topics = [
            "system.thought",
            "sensory.stream.chunk",
            "sensory.stream.end",
            "sensory.output",
        ]
        for topic in topics:
            sub_id = await self._bus.subscribe(topic, self._handle_chunk)
            self._subscriptions.append((topic, sub_id))

    async def stop(self) -> None:
        """Unsubscribe from all topics to prevent subscription leaks."""
        for topic, sub_id in self._subscriptions:
            try:
                if hasattr(self._bus, "unsubscribe"):
                    await self._bus.unsubscribe(topic, sub_id)
            except Exception:
                pass
        self._subscriptions.clear()

    def __aiter__(self) -> AsyncIterator[dict[str, str]]:
        return self

    async def __anext__(self) -> dict[str, str]:
        if self._done:
            raise StopAsyncIteration

        try:
            chunk = await asyncio.wait_for(self._queue.get(), timeout=self._timeout)
        except asyncio.TimeoutError:
            self._done = True
            await self.stop()
            raise StopAsyncIteration

        if chunk is None:
            self._done = True
            await self.stop()
            raise StopAsyncIteration

        return chunk
