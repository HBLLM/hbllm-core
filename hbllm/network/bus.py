"""
MessageBus — the communication backbone of HBLLM.

Defines the MessageBus protocol and provides InProcessBus as the default
implementation. In Phase 3+, swap to GrpcBus or NatsBus with zero changes
to node code.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable

from hbllm.network.messages import Message
from hbllm.network.tracing import BusMetrics, trace_span

logger = logging.getLogger(__name__)

# Type alias for message handlers
MessageHandler = Callable[[Message], Coroutine[Any, Any, Message | None]]


class Subscription:
    """Represents an active subscription to a topic."""

    def __init__(self, topic: str, handler: MessageHandler, sub_id: str):
        self.topic = topic
        self.handler = handler
        self.id = sub_id
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def cancel(self) -> None:
        self._active = False


@runtime_checkable
class MessageBus(Protocol):
    """
    Protocol defining the MessageBus interface.

    All node communication goes through this interface.
    Implementations can be in-process (async queues), gRPC, NATS, etc.
    """

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a topic (fire-and-forget)."""
        ...

    async def request(self, topic: str, message: Message, timeout: float = 30.0) -> Message:
        """Send a request and wait for a correlated response."""
        ...

    async def subscribe(self, topic: str, handler: MessageHandler) -> Subscription:
        """Subscribe to messages on a topic."""
        ...

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        ...

    async def start(self) -> None:
        """Start the bus."""
        ...

    async def stop(self) -> None:
        """Stop the bus and clean up."""
        ...


class InProcessBus:
    """
    In-process MessageBus using asyncio.

    All messages are delivered via async queues within a single process.
    This is the default for Phase 1-2. In Phase 3+, swap to GrpcBus/NatsBus.
    """

    def __init__(self, queue_size: int = 1000):
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        self._queue: asyncio.Queue[tuple[str, Message]] = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._sub_counter = 0
        self.metrics = BusMetrics()

    async def start(self) -> None:
        """Start the message dispatch loop."""
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("InProcessBus started")

    async def stop(self) -> None:
        """Stop the bus and cancel pending requests."""
        self._running = False

        # Cancel all pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Stop dispatch loop
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        logger.info("InProcessBus stopped")

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to all subscribers of a topic."""
        if not self._running:
            self.metrics.record_drop(topic)
            logger.warning("Bus is not running, message dropped: %s", message.id)
            return

        self.metrics.record_publish(topic)
        await self._queue.put((topic, message))

    async def request(self, topic: str, message: Message, timeout: float = 30.0) -> Message:
        """Send a request and wait for a correlated response."""
        # Create a future for the response
        future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()
        self._pending_requests[message.id] = future

        # Publish the request
        await self.publish(topic, message)

        try:
            # Wait for correlated response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Request {message.id} to topic '{topic}' timed out after {timeout}s"
            )
        finally:
            self._pending_requests.pop(message.id, None)

    async def subscribe(self, topic: str, handler: MessageHandler) -> Subscription:
        """Subscribe a handler to a topic."""
        self._sub_counter += 1
        sub = Subscription(topic=topic, handler=handler, sub_id=f"sub-{self._sub_counter}")
        self._subscriptions[topic].append(sub)
        self.metrics.record_subscribe()
        logger.debug("Subscribed to '%s' (id=%s)", topic, sub.id)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        subscription.cancel()
        subs = self._subscriptions.get(subscription.topic, [])
        self._subscriptions[subscription.topic] = [s for s in subs if s.id != subscription.id]
        self.metrics.record_unsubscribe()
        logger.debug("Unsubscribed '%s' from '%s'", subscription.id, subscription.topic)

    async def _dispatch_loop(self) -> None:
        """Main loop: dequeue messages and dispatch to subscribers."""
        while self._running:
            try:
                topic, message = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            # Check if this is a response to a pending request
            if message.correlation_id and message.correlation_id in self._pending_requests:
                future = self._pending_requests.pop(message.correlation_id)
                if not future.done():
                    future.set_result(message)
                continue

            # Dispatch to topic subscribers
            await self._dispatch_to_subscribers(topic, message)

    async def _dispatch_to_subscribers(self, topic: str, message: Message) -> None:
        """Dispatch a message to all active subscribers for a topic."""
        # Match exact topic and wildcard patterns
        matching_topics = self._get_matching_topics(topic)

        for match_topic in matching_topics:
            for sub in self._subscriptions.get(match_topic, []):
                if not sub.active:
                    continue
                async def _run_handler(s=sub, t=topic, m=message):
                    _start = __import__('time').monotonic()
                    try:
                        with trace_span(f"handle:{t}", {"topic": t, "node": s.id, "msg_id": m.id}):
                            response = await s.handler(m)
                        latency = (__import__('time').monotonic() - _start) * 1000
                        self.metrics.record_delivery(t, latency)
                        # If handler returns a response, route it back
                        if response is not None:
                            if response.correlation_id is None:
                                response.correlation_id = m.id
                            await self.publish(response.topic, response)
                    except Exception:
                        self.metrics.record_error(t)
                        logger.exception(
                            "Error in handler for topic '%s', message %s",
                            t,
                            m.id,
                        )
                        
                asyncio.create_task(_run_handler())

    def _get_matching_topics(self, topic: str) -> list[str]:
        """Get all registered topics that match (exact + wildcard)."""
        matches = []
        for registered_topic in self._subscriptions:
            if registered_topic == topic:
                matches.append(registered_topic)
            elif registered_topic.endswith(".*"):
                prefix = registered_topic[:-2]
                if topic.startswith(prefix):
                    matches.append(registered_topic)
            elif registered_topic == "*":
                matches.append(registered_topic)
        return matches
