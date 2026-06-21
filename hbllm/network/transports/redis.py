"""
Redis Transport — Distributed pub/sub for backend cluster communication.

Wraps the existing RedisBus logic into the Transport abstraction.
This transport connects multiple HBLLM processes/servers via Redis Pub/Sub,
with optional HMAC-SHA256 message signing for cluster security.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections import defaultdict
from datetime import timezone
from typing import Any

from hbllm.network.bus import MessageHandler, Subscription
from hbllm.network.messages import Message
from hbllm.network.transports.base import Transport, TransportState

logger = logging.getLogger(__name__)


class RedisTransport(Transport):
    """
    Redis Pub/Sub transport for distributed cluster communication.

    Nodes in separate processes or on different servers communicate via
    Redis channels. Supports optional HMAC-SHA256 message authentication.
    """

    def __init__(
        self,
        transport_id: str = "redis_cluster",
        redis_url: str | None = None,
        auth_secret: str = "",
    ) -> None:
        super().__init__(transport_id=transport_id, transport_type="redis")
        import os

        self.redis_url: str = redis_url or os.getenv("HBLLM_REDIS_URL", "redis://localhost:6379")
        self.auth_secret = auth_secret
        self.client: Any | None = None
        self.pubsub: Any | None = None

        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._sub_counter = 0
        self._active_tasks: set[asyncio.Task[Any]] = set()

    def _sign(self, payload_json: str) -> str:
        """Compute HMAC-SHA256 signature."""
        return hmac.new(
            self.auth_secret.encode(), payload_json.encode(), hashlib.sha256
        ).hexdigest()[:32]

    def _verify(self, payload_json: str, signature: str) -> bool:
        """Verify a message signature."""
        expected = self._sign(payload_json)
        return hmac.compare_digest(expected, signature)

    async def start(self) -> None:
        """Connect to Redis and start the dispatch loop."""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.error("redis[asyncio] package not installed. RedisTransport cannot start.")
            self._state = TransportState.DISCONNECTED
            return

        self._state = TransportState.CONNECTING
        self.client = aioredis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.client.pubsub()
        await self.pubsub.psubscribe("*")

        self._running = True
        self._state = TransportState.CONNECTED
        self.metrics.uptime_start = time.monotonic()
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("RedisTransport '%s' connected to %s", self.transport_id, self.redis_url)

    async def stop(self) -> None:
        """Stop the transport and clean up."""
        self._running = False
        self._state = TransportState.STOPPED

        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        if self.pubsub:
            await self.pubsub.close()
        if self.client:
            await self.client.aclose()

        logger.info("RedisTransport '%s' stopped", self.transport_id)

    async def send(self, topic: str, message: Message) -> None:
        """Publish a message to Redis."""
        if not self._running or not self.client:
            self.metrics.record_drop()
            return

        payload_json = message.model_dump_json()

        if self.auth_secret:
            sig = self._sign(payload_json)
            wire = f"{sig}|{payload_json}"
        else:
            wire = payload_json

        await self.client.publish(topic, wire)
        self.metrics.record_send(len(wire))

        # Fast-path: resolve local pending requests immediately
        if message.correlation_id and message.correlation_id in self._pending_requests:
            future = self._pending_requests.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)

    async def send_request(self, topic: str, message: Message, timeout: float = 30.0) -> Message:
        """Send a request and wait for a correlated response."""
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending_requests[message.id] = future
        await self.send(topic, message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except (TimeoutError, asyncio.TimeoutError):
            self._pending_requests.pop(message.id, None)
            raise TimeoutError(
                f"Redis request {message.id} to '{topic}' timed out after {timeout}s"
            )

    async def subscribe(
        self, topic: str, handler: MessageHandler, tenant_id: str | None = None
    ) -> Subscription:
        """Subscribe a handler to a topic locally."""
        self._sub_counter += 1
        sub = Subscription(
            topic=topic,
            handler=handler,
            sub_id=f"redis-sub-{self._sub_counter}",
            tenant_id=tenant_id,
        )
        self._subscriptions[topic].append(sub)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a local subscription."""
        subscription.cancel()
        subs = self._subscriptions.get(subscription.topic, [])
        self._subscriptions[subscription.topic] = [s for s in subs if s.id != subscription.id]

    def has_subscribers(self, topic: str) -> bool:
        return len(self._subscriptions.get(topic, [])) > 0

    async def ping(self) -> bool:
        """Health check — ping Redis."""
        if not self.client:
            return False
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def _dispatch_loop(self) -> None:
        """Main loop: receive from Redis Pub/Sub and dispatch locally."""
        if not self.pubsub:
            return

        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            try:
                msg_dict = await self.pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.1
                )
                if msg_dict is None:
                    continue

                backoff = 1.0

                if msg_dict["type"] in ("message", "pmessage"):
                    topic = msg_dict["channel"]
                    raw_data = msg_dict["data"]

                    # HMAC verification
                    if self.auth_secret and "|" in raw_data:
                        sig, payload_json = raw_data.split("|", 1)
                        if not self._verify(payload_json, sig):
                            self.metrics.record_drop()
                            continue
                    else:
                        payload_json = raw_data

                    try:
                        message = Message.model_validate_json(payload_json)
                    except Exception as e:
                        logger.error("RedisTransport failed to parse message: %s", e)
                        continue

                    # TTL enforcement
                    if message.ttl_seconds is not None:
                        age = (
                            time.time() - message.timestamp.replace(tzinfo=timezone.utc).timestamp()
                        )
                        if age > message.ttl_seconds:
                            self.metrics.record_drop()
                            continue

                    self.metrics.record_receive(len(payload_json))

                    # Resolve pending requests
                    if message.correlation_id and message.correlation_id in self._pending_requests:
                        future = self._pending_requests.pop(message.correlation_id)
                        if not future.done():
                            future.set_result(message)
                        continue

                    # Dispatch to local subscribers
                    await self._dispatch_to_subscribers(topic, message)

            except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
                continue
            except Exception as e:
                if "ConnectionError" in str(e.__class__):
                    self.metrics.reconnections += 1
                    self._state = TransportState.RECONNECTING
                    logger.error("Redis connection lost. Reconnecting in %.1fs...", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    try:
                        await self._reconnect()
                    except Exception as e:
                        logger.error("Redis reconnection failed.")
                else:
                    logger.error("RedisTransport dispatch error: %s", e)

    async def _reconnect(self) -> None:
        """Reconnect to Redis after a connection loss."""
        import redis.asyncio as aioredis

        try:
            if self.pubsub:
                await self.pubsub.close()
            if self.client:
                await self.client.aclose()
        except Exception as e:
            logger.debug("[Redis] non-critical error: %s", e)
        self.client = aioredis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.client.pubsub()
        await self.pubsub.psubscribe("*")
        self._state = TransportState.CONNECTED
        logger.info("RedisTransport reconnected to %s", self.redis_url)

    async def _dispatch_to_subscribers(self, topic: str, message: Message) -> None:
        """Dispatch a message to all matching local subscribers."""
        matching = self._get_matching_topics(topic)

        for match_topic in matching:
            for sub in self._subscriptions.get(match_topic, []):
                if not sub.active:
                    continue
                if sub.tenant_id and message.tenant_id and sub.tenant_id != message.tenant_id:
                    continue

                async def _run(s: Subscription = sub, t: str = topic, m: Message = message) -> None:
                    start = time.monotonic()
                    try:
                        from hbllm.network._tenant_bridge import restore_tenant_ctx

                        with restore_tenant_ctx(m):
                            response = await s.handler(m)
                        latency_ms = (time.monotonic() - start) * 1000
                        self.metrics.record_latency(latency_ms)
                        if response is not None:
                            if response.correlation_id is None:
                                response.correlation_id = m.id
                            await self.send(response.topic, response)
                    except Exception:
                        self.metrics.record_error()
                        logger.exception("Error in handler for '%s', msg %s", t, m.id)

                task = asyncio.create_task(_run())
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

    def _get_matching_topics(self, topic: str) -> list[str]:
        """Match exact topics and wildcards."""
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
