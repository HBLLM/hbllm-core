"""
NatsBus — NATS-based MessageBus for lightweight distributed messaging.

Implements the MessageBus protocol using NATS for high-throughput
pub/sub with optional JetStream durable subscriptions.

Requires: nats-py
Falls back gracefully if not installed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from hbllm.network.bus import MessageHandler, Subscription
from hbllm.network.messages import Message
from hbllm.network.serialization import JsonSerializer, Serializer
from hbllm.network.tracing import BusMetrics

logger = logging.getLogger(__name__)

try:
    import nats as nats_client
    from nats.aio.client import Client as NatsClient

    _HAS_NATS = True
except ImportError:
    _HAS_NATS = False
    logger.info("nats-py not installed — NatsBus unavailable (pip install nats-py)")


class NatsBus:
    """
    NATS-based MessageBus for lightweight distributed messaging.

    Conforms to the MessageBus protocol. Uses NATS core pub/sub
    with optional JetStream for durable subscriptions.
    """

    def __init__(
        self,
        servers: str | list[str] = "nats://localhost:4222",
        serializer: Serializer | None = None,
        subject_prefix: str = "hbllm",
        use_jetstream: bool = False,
    ) -> None:
        if not _HAS_NATS:
            raise RuntimeError("nats-py required for NatsBus. Install with: pip install nats-py")

        self._servers = [servers] if isinstance(servers, str) else servers
        self._serializer = serializer or JsonSerializer()
        self._prefix = subject_prefix
        self._use_jetstream = use_jetstream

        self._nc: NatsClient | None = None
        self._js: Any = None  # JetStream context
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._nats_subs: dict[str, Any] = {}  # sub_id → NATS subscription
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        self._running = False
        self._sub_counter = 0
        self.metrics = BusMetrics()
        self._interceptors: list[Any] = []

    def add_interceptor(self, interceptor: Any) -> None:
        """Add a proactive message interceptor."""
        self._interceptors.append(interceptor)

    async def start(self) -> None:
        """Connect to NATS server."""
        self._nc = NatsClient()
        await self._nc.connect(
            servers=self._servers,
            reconnect_time_wait=2,
            max_reconnect_attempts=10,
        )

        if self._use_jetstream:
            self._js = self._nc.jetstream()

        self._running = True
        logger.info("NatsBus connected to %s", self._servers)

    async def stop(self) -> None:
        """Disconnect from NATS."""
        self._running = False

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Unsubscribe all NATS subscriptions
        for nats_sub in self._nats_subs.values():
            try:
                await nats_sub.unsubscribe()
            except Exception:
                pass
        self._nats_subs.clear()

        # Close connection
        if self._nc and self._nc.is_connected:
            await self._nc.drain()
            await self._nc.close()
        self._nc = None

        logger.info("NatsBus disconnected")

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a NATS subject."""
        if not self._running or not self._nc:
            self.metrics.record_drop(topic)
            return

        # Run interceptors
        for interceptor in self._interceptors:
            try:
                result = await interceptor(message)
                if result is None:
                    self.metrics.record_drop(topic)
                    return
                message = result
            except Exception as e:
                logger.error("Interceptor failed: %s", e)
                self.metrics.record_drop(topic)
                return

        self.metrics.record_publish(topic)

        # Check for pending request response
        if message.correlation_id and message.correlation_id in self._pending_requests:
            pending = self._pending_requests.pop(message.correlation_id)
            if not pending.done():
                pending.set_result(message)
            return

        # Serialize and publish to NATS
        subject = f"{self._prefix}.{topic}"
        data = self._serializer.serialize(message)
        await self._nc.publish(subject, data)

    async def request(self, topic: str, message: Message, timeout: float = 90.0) -> Message:
        """Send a request and wait for a correlated response via NATS request-reply."""
        if not self._nc:
            raise RuntimeError("NatsBus not started")

        subject = f"{self._prefix}.{topic}"
        data = self._serializer.serialize(message)

        try:
            response = await self._nc.request(subject, data, timeout=timeout)
            return self._serializer.deserialize(response.data)
        except asyncio.TimeoutError:
            raise TimeoutError(f"NATS request to '{topic}' timed out after {timeout}s")

    async def subscribe(
        self, topic: str, handler: MessageHandler, tenant_id: str | None = None
    ) -> Subscription:
        """Subscribe to a NATS subject."""
        self._sub_counter += 1
        sub = Subscription(
            topic=topic,
            handler=handler,
            sub_id=f"nats-sub-{self._sub_counter}",
            tenant_id=tenant_id,
        )
        self._subscriptions[topic].append(sub)
        self.metrics.record_subscribe()

        # Create NATS subscription
        subject = f"{self._prefix}.{topic}"

        async def _nats_handler(msg: Any) -> None:
            try:
                message = self._serializer.deserialize(msg.data)

                # Tenant isolation
                if tenant_id and message.tenant_id and tenant_id != message.tenant_id:
                    return

                start = time.monotonic()
                response = await handler(message)
                latency = (time.monotonic() - start) * 1000
                self.metrics.record_delivery(topic, latency)

                # If handler returns a response and there's a reply subject
                if response is not None and msg.reply:
                    resp_data = self._serializer.serialize(response)
                    await self._nc.publish(msg.reply, resp_data)
            except Exception:
                self.metrics.record_error(topic)
                logger.exception("NATS handler error on '%s'", topic)

        if self._nc:
            # Use wildcard for NATS if topic ends with .*
            nats_subject = subject.replace(".*", ".>") if subject.endswith(".*") else subject
            nats_sub = await self._nc.subscribe(nats_subject, cb=_nats_handler)
            self._nats_subs[sub.id] = nats_sub

        logger.debug("NATS subscribed to '%s' (id=%s)", topic, sub.id)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        subscription.cancel()
        subs = self._subscriptions.get(subscription.topic, [])
        self._subscriptions[subscription.topic] = [s for s in subs if s.id != subscription.id]

        # Unsubscribe from NATS
        nats_sub = self._nats_subs.pop(subscription.id, None)
        if nats_sub:
            await nats_sub.unsubscribe()

        self.metrics.record_unsubscribe()
