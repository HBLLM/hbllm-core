"""
Cognitive Stream — Async streaming interface for real-time token output.

Platform-agnostic bus-based streaming for any HBLLM brain.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from hbllm.network.bus import MessageBus, Subscription
from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class CognitiveStream:
    """
    Async iterator that streams tokens and embodied signals from the cognitive pipeline.

    Usage:
        stream = CognitiveStream(bus, correlation_id)
        await stream.start()
        async for chunk in stream:
            print(chunk, flush=True)
    """

    def __init__(self, bus: MessageBus, correlation_id: str, timeout: float = 120.0):
        self._bus = bus
        self._corr_id = correlation_id
        self._timeout = timeout
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._done = False
        self._subscriptions: list[Subscription] = []

    async def _handle_chunk(self, message: Message) -> Message | None:
        """Capture streaming chunks, internal thoughts, and embodied events from the bus."""
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

        elif topic == "embodied.observation":
            await self._queue.put(
                {
                    "type": "embodied_observation",
                    "domain": message.payload.get("domain", "general"),
                    "step": message.payload.get("step", 0),
                    "data": message.payload.get("observation", message.payload),
                }
            )

        elif topic == "embodied.thought":
            thought = message.payload.get("text", "")
            if thought:
                await self._queue.put(
                    {
                        "type": "embodied_thought",
                        "text": thought,
                        "domain": message.payload.get("domain", "general"),
                    }
                )

        elif topic == "embodied.action":
            action = message.payload.get("action", "")
            await self._queue.put(
                {
                    "type": "embodied_action",
                    "action": action,
                    "domain": message.payload.get("domain", "general"),
                    "payload": message.payload,
                }
            )
        return None

    async def start(self) -> None:
        """Subscribe to stream events, internal thoughts, and embodied interactions."""
        topics = [
            "system.thought",
            "sensory.stream.chunk",
            "sensory.stream.end",
            "sensory.output",
            "embodied.observation",
            "embodied.thought",
            "embodied.action",
        ]
        for topic in topics:
            sub = await self._bus.subscribe(topic, self._handle_chunk)
            self._subscriptions.append(sub)

    async def stop(self) -> None:
        """Unsubscribe from all topics to prevent subscription leaks."""
        for sub in self._subscriptions:
            try:
                if hasattr(self._bus, "unsubscribe"):
                    await self._bus.unsubscribe(sub)
            except Exception as e:
                logger.debug("[Streaming] non-critical error: %s", e)
        self._subscriptions.clear()

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self

    async def __anext__(self) -> dict[str, Any]:
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
