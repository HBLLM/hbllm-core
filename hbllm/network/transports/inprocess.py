"""
InProcess Transport — Local async message delivery within a single process.

Wraps the existing InProcessBus dispatch logic into the Transport abstraction.
This is the default transport for local node-to-node communication (zero network
overhead). Used as the "brain-local" pipe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import timezone
from typing import Any

from hbllm.network.bus import MessageHandler, Subscription
from hbllm.network.messages import Message
from hbllm.network.transports.base import Transport, TransportState

logger = logging.getLogger(__name__)


class InProcessTransport(Transport):
    """
    In-process transport using asyncio priority queues.

    All messages are delivered within the same process via async dispatch.
    This is the fastest possible transport (zero serialization, zero network).
    """

    def __init__(
        self,
        transport_id: str = "inprocess_local",
        queue_size: int = 1000,
        max_concurrent_handlers: int = 1000,
    ) -> None:
        super().__init__(transport_id=transport_id, transport_type="inprocess")
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        self._queue: asyncio.PriorityQueue[tuple[int, float, str, Message]] = (
            asyncio.PriorityQueue(maxsize=queue_size)
        )
        self._queue_size = queue_size
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._sub_counter = 0
        self._msg_counter = 0
        self._handler_semaphore = asyncio.Semaphore(max_concurrent_handlers)
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._backpressure_threshold = 0.8

    async def start(self) -> None:
        """Start the in-process dispatch loop."""
        if self._running:
            return
        self._running = True
        self._state = TransportState.CONNECTED
        self.metrics.uptime_start = time.monotonic()
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("InProcessTransport '%s' started", self.transport_id)

    async def stop(self) -> None:
        """Stop the transport and cancel all pending work."""
        self._running = False
        self._state = TransportState.STOPPED

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Cancel active handler tasks
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()

        if self._active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    "InProcessTransport stop timed out waiting for %d tasks",
                    len(self._active_tasks),
                )
        self._active_tasks.clear()

        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        self._subscriptions.clear()
        logger.info("InProcessTransport '%s' stopped", self.transport_id)

    async def send(self, topic: str, message: Message) -> None:
        """Enqueue a message for local dispatch."""
        if not self._running:
            self.metrics.record_drop()
            return

        # Backpressure monitoring
        queue_fullness = self._queue.qsize() / self._queue_size
        if queue_fullness >= self._backpressure_threshold:
            logger.warning(
                "InProcess queue at %.0f%% capacity (%d/%d)",
                queue_fullness * 100,
                self._queue.qsize(),
                self._queue_size,
            )

        self.metrics.record_send()
        self._msg_counter += 1
        priority_key = -message.priority.value
        await self._queue.put((priority_key, float(self._msg_counter), topic, message))

    async def send_request(
        self, topic: str, message: Message, timeout: float = 30.0
    ) -> Message:
        """Send a request and wait for a correlated response."""
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending_requests[message.id] = future
        await self.send(topic, message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except (TimeoutError, asyncio.TimeoutError):
            raise TimeoutError(
                f"InProcess request {message.id} to '{topic}' timed out after {timeout}s"
            )
        finally:
            self._pending_requests.pop(message.id, None)

    async def subscribe(
        self, topic: str, handler: MessageHandler, tenant_id: str | None = None
    ) -> Subscription:
        """Subscribe a handler to a topic."""
        self._sub_counter += 1
        sub = Subscription(
            topic=topic,
            handler=handler,
            sub_id=f"ipc-sub-{self._sub_counter}",
            tenant_id=tenant_id,
        )
        self._subscriptions[topic].append(sub)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        subscription.cancel()
        subs = self._subscriptions[subscription.topic]
        if subscription in subs:
            subs.remove(subscription)

    def has_subscribers(self, topic: str) -> bool:
        """Check if a topic has any active subscribers."""
        return bool(self._subscriptions.get(topic))

    async def _dispatch_loop(self) -> None:
        """Main loop: dequeue messages by priority and dispatch."""
        while self._running:
            try:
                priority_key, _counter, topic, message = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
                continue

            # TTL enforcement
            if message.ttl_seconds is not None:
                age = time.time() - message.timestamp.replace(tzinfo=timezone.utc).timestamp()
                if age > message.ttl_seconds:
                    self.metrics.record_drop()
                    continue

            self.metrics.record_receive()

            # Check for pending request correlation
            if message.correlation_id and message.correlation_id in self._pending_requests:
                pending_f = self._pending_requests.pop(message.correlation_id)
                if not pending_f.done():
                    pending_f.set_result(message)
                continue

            # Dispatch to subscribers
            await self._dispatch_to_subscribers(topic, message)

    async def _dispatch_to_subscribers(self, topic: str, message: Message) -> None:
        """Dispatch a message to all matching subscribers."""
        matching = self._get_matching_topics(topic)

        for match_topic in matching:
            for sub in self._subscriptions.get(match_topic, []):
                if not sub.active:
                    continue
                # Tenant isolation
                if sub.tenant_id and message.tenant_id and sub.tenant_id != message.tenant_id:
                    continue

                async def _run(
                    s: Subscription = sub, t: str = topic, m: Message = message
                ) -> None:
                    async with self._handler_semaphore:
                        start = time.monotonic()
                        try:
                            response = await s.handler(m)
                            latency_ms = (time.monotonic() - start) * 1000
                            self.metrics.record_latency(latency_ms)
                            if response is not None:
                                if response.correlation_id is None:
                                    response.correlation_id = m.id
                                await self.send(response.topic, response)
                        except Exception:
                            self.metrics.record_error()
                            logger.exception(
                                "Error in handler for topic '%s', message %s", t, m.id
                            )

                task = asyncio.create_task(_run())
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

    def _get_matching_topics(self, topic: str) -> list[str]:
        """Match exact topics and wildcards (e.g. 'action.*')."""
        matches = []
        for registered in self._subscriptions:
            if registered == topic:
                matches.append(registered)
            elif registered.endswith(".*"):
                prefix = registered[:-2]
                if topic.startswith(prefix):
                    matches.append(registered)
            elif registered == "*":
                matches.append(registered)
        return matches
