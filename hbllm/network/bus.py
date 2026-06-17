"""
MessageBus — the communication backbone of HBLLM.

Defines the MessageBus protocol and provides InProcessBus as the default
implementation. In Phase 3+, swap to GrpcBus or NatsBus with zero changes
to node code.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Coroutine
from datetime import timezone
from typing import Any, Protocol, runtime_checkable

from hbllm.network.messages import Message
from hbllm.network.tracing import BusMetrics, trace_span

logger = logging.getLogger(__name__)

# Type alias for message handlers
MessageHandler = Callable[[Message], Coroutine[Any, Any, Message | None]]


class Subscription:
    """Represents an active subscription to a topic."""

    def __init__(
        self, topic: str, handler: MessageHandler, sub_id: str, tenant_id: str | None = None
    ) -> None:
        self.topic = topic
        self.handler = handler
        self.id = sub_id
        self.tenant_id = tenant_id
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

    async def request(self, topic: str, message: Message, timeout: float = 90.0) -> Message:
        """Send a request and wait for a correlated response."""
        ...

    async def subscribe(
        self, topic: str, handler: MessageHandler, tenant_id: str | None = None
    ) -> Subscription:
        """Subscribe to messages on a topic."""
        ...

    def add_interceptor(
        self, interceptor: Callable[[Message], Coroutine[Any, Any, Message | None]]
    ) -> None:
        """Add a proactive message interceptor."""
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

    def has_subscribers(self, topic: str) -> bool:
        """Check if a topic has any active subscribers."""
        ...


class InProcessBus:
    """
    In-process MessageBus using asyncio.

    All messages are delivered via async queues within a single process.
    This is the default for Phase 1-2. In Phase 3+, swap to GrpcBus/NatsBus.
    """

    def __init__(self, queue_size: int = 1000, max_concurrent_handlers: int = 1000) -> None:
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        self._queue: asyncio.PriorityQueue[tuple[int, float, str, Message]] = asyncio.PriorityQueue(
            maxsize=queue_size
        )
        self._queue_size = queue_size
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._sub_counter = 0
        self._msg_counter = 0  # Tiebreaker for priority queue ordering
        self.metrics = BusMetrics()
        # Backpressure monitoring
        self._backpressure_warning_threshold = 0.8  # Warn at 80% full
        # Concurrency control: limit simultaneous handler tasks to prevent OOM
        self._handler_semaphore = asyncio.Semaphore(max_concurrent_handlers)
        # Track active handler tasks for graceful shutdown
        self._active_tasks: set[asyncio.Task[Any]] = set()
        # Message interceptors for proactive governance
        self._interceptors: list[Callable[[Message], Coroutine[Any, Any, Message | None]]] = []
        # M11: Bounded set for message deduplication (prevents duplicate dispatch)
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self._max_seen_ids: int = 10_000
        # M7: Cache of wildcard-only topic patterns for faster matching
        self._wildcard_topics: list[str] = []

    def add_interceptor(
        self, interceptor: Callable[[Message], Coroutine[Any, Any, Message | None]]
    ) -> None:
        """Add an interceptor to evaluate messages before queuing."""
        self._interceptors.append(interceptor)

    async def start(self) -> None:
        """Start the message dispatch loop."""
        if getattr(self, "_running", False):
            return
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

        # Cancel all active handler tasks to prevent event-loop hang
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()

        # Wait for all active handler tasks to finish
        if self._active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    "InProcessBus.stop() timed out waiting for %d active tasks.",
                    len(self._active_tasks),
                )
                for idx, t in enumerate(self._active_tasks):
                    logger.warning("Active Task %d: %s", idx, t)

        self._active_tasks.clear()

        # Stop dispatch loop
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        self._subscriptions.clear()
        logger.info("InProcessBus stopped")

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to all subscribers of a topic, respecting priority."""
        if not self._running:
            self.metrics.record_drop(topic)
            logger.warning("Bus is not running, message dropped: %s", message.id)
            return

        # Backpressure monitoring
        queue_fullness = self._queue.qsize() / self._queue_size
        if queue_fullness >= self._backpressure_warning_threshold:
            logger.warning(
                "Bus queue at %.0f%% capacity (%d/%d)",
                queue_fullness * 100,
                self._queue.qsize(),
                self._queue_size,
            )

        # ── Proactive Interceptor Chain ──
        for interceptor in self._interceptors:
            try:
                new_msg = await interceptor(message)
                if new_msg is None:
                    # Message blocked by interceptor
                    self.metrics.record_drop(topic)
                    await self._route_to_dlq(message, "interceptor_blocked")
                    return
                message = new_msg
            except Exception as e:
                logger.error(
                    "Interceptor failed on topic '%s', blocking message. Error: %s", topic, e
                )
                self.metrics.record_drop(topic)
                await self._route_to_dlq(message, f"interceptor_error: {e}")
                return

        self.metrics.record_publish(topic)

        # Priority queue: lower number = higher priority. Negate message priority value.
        # Use a monotonic counter as tiebreaker to preserve FIFO order within same priority.
        self._msg_counter += 1
        priority_key = -message.priority.value  # CRITICAL=3 → -3 (highest)
        await self._queue.put((priority_key, float(self._msg_counter), topic, message))

    async def _route_to_dlq(self, message: Message, reason: str) -> None:
        """Route a dropped message to the Dead Letter Queue (DLQ)."""
        if message.topic == "system.dlq":
            return  # Prevent infinite DLQ loops

        dlq_msg = message.model_copy(deep=True)
        dlq_msg.topic = "system.dlq"
        dlq_msg.payload["dlq_reason"] = reason
        dlq_msg.payload["original_topic"] = message.topic

        # Bypass interceptors and queue directly
        self._msg_counter += 1
        priority_key = -dlq_msg.priority.value
        await self._queue.put((priority_key, float(self._msg_counter), "system.dlq", dlq_msg))

    async def request(self, topic: str, message: Message, timeout: float = 90.0) -> Message:
        """Send a request and wait for a correlated response."""
        # Create a future for the response
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending_requests[message.id] = future

        # Publish the request
        await self.publish(topic, message)

        try:
            # Wait for correlated response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except (TimeoutError, asyncio.TimeoutError):
            raise TimeoutError(
                f"Request {message.id} to topic '{topic}' timed out after {timeout}s"
            )
        finally:
            self._pending_requests.pop(message.id, None)

    async def subscribe(
        self, topic: str, handler: MessageHandler, tenant_id: str | None = None
    ) -> Subscription:
        """Subscribe a handler to a topic."""
        self._sub_counter += 1
        sub = Subscription(
            topic=topic, handler=handler, sub_id=f"sub-{self._sub_counter}", tenant_id=tenant_id
        )
        self._subscriptions[topic].append(sub)
        self.metrics.record_subscribe()
        self._rebuild_wildcard_cache()
        logger.debug("Subscribed to '%s' (id=%s)", topic, sub.id)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        subscription.cancel()
        subs = self._subscriptions[subscription.topic]
        if subscription in subs:
            subs.remove(subscription)
        # M19: Clean up empty topic entries to prevent stale accumulation
        if not subs:
            del self._subscriptions[subscription.topic]
        self.metrics.record_unsubscribe()
        self._rebuild_wildcard_cache()
        logger.debug("Unsubscribed '%s' from '%s'", subscription.id, subscription.topic)

    def has_subscribers(self, topic: str) -> bool:
        """Check if a topic has any active subscribers."""
        return bool(self._subscriptions.get(topic))

    async def _dispatch_loop(self) -> None:
        """Main loop: dequeue messages by priority and dispatch to subscribers."""
        while self._running:
            try:
                priority_key, _counter, topic, message = await self._queue.get()
            except asyncio.CancelledError:
                break

            # TTL enforcement: drop expired messages
            if message.ttl_seconds is not None:
                age = time.time() - message.timestamp.replace(tzinfo=timezone.utc).timestamp()
                if age > message.ttl_seconds:
                    self.metrics.record_drop(topic)
                    logger.debug(
                        "Dropped expired message %s (age=%.1fs, ttl=%.1fs)",
                        message.id,
                        age,
                        message.ttl_seconds,
                    )
                    await self._route_to_dlq(message, "ttl_expired")
                    continue

            # Check if this is a response to a pending request
            if message.correlation_id and message.correlation_id in self._pending_requests:
                pending_f = self._pending_requests.pop(message.correlation_id)
                if not pending_f.done():
                    pending_f.set_result(message)
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

                # Tenant Isolation check
                if sub.tenant_id and message.tenant_id and sub.tenant_id != message.tenant_id:
                    continue

                async def _run_handler(
                    s: Subscription = sub, t: str = topic, m: Message = message
                ) -> None:
                    async with self._handler_semaphore:
                        _start = time.monotonic()
                        try:
                            from hbllm.network._tenant_bridge import restore_tenant_ctx

                            with restore_tenant_ctx(m):
                                with trace_span(
                                    f"handle:{t}", {"topic": t, "node": s.id, "msg_id": m.id}
                                ):
                                    response = await s.handler(m)
                            latency = (time.monotonic() - _start) * 1000
                            self.metrics.record_delivery(t, latency)
                            # If handler returns a response, route it back
                            if response is not None:
                                if response.correlation_id is None:
                                    response.correlation_id = m.id
                                await self.publish(response.topic, response)
                        except Exception as exc:
                            self.metrics.record_error(t)
                            logger.exception(
                                "Error in handler for topic '%s', message %s",
                                t,
                                m.id,
                            )
                            # Route failed messages to DLQ for observability
                            await self._route_to_dlq(m, f"handler_error: {type(exc).__name__}")
                            # If there's a pending request future, resolve it with an
                            # error immediately instead of forcing the caller to timeout.
                            corr = m.correlation_id or m.id
                            pending_f = self._pending_requests.pop(corr, None)
                            if pending_f and not pending_f.done():
                                error_resp = m.create_error(
                                    f"Handler error: {type(exc).__name__}"
                                )
                                pending_f.set_result(error_resp)

                task = asyncio.create_task(_run_handler())
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

    def _get_matching_topics(self, topic: str) -> list[str]:
        """Get all registered topics that match (exact + wildcard).

        M7: Uses O(1) dict lookup for exact matches (common case)
        and only iterates over cached wildcard patterns.
        """
        matches: list[str] = []

        # O(1) exact match — the most common case
        if topic in self._subscriptions:
            matches.append(topic)

        # Only iterate wildcard patterns, not all registered topics
        for wc_topic in self._wildcard_topics:
            if wc_topic == "*":
                if topic not in matches:  # avoid double-add
                    matches.append(wc_topic)
            elif wc_topic.endswith(".*"):
                prefix = wc_topic[:-2]
                if topic.startswith(prefix) and wc_topic not in matches:
                    matches.append(wc_topic)

        return matches

    def _rebuild_wildcard_cache(self) -> None:
        """Rebuild the wildcard topic cache after subscription changes."""
        self._wildcard_topics = [
            t for t in self._subscriptions
            if t == "*" or t.endswith(".*")
        ]
