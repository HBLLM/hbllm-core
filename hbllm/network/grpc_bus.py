"""
GrpcBus — gRPC-based MessageBus for cross-cluster communication.

Implements the MessageBus protocol using gRPC bidirectional streaming
for distributed node communication across processes or machines.

Requires: grpcio, grpcio-tools
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
    import grpc
    import grpc.aio

    _HAS_GRPC = True
except ImportError:
    _HAS_GRPC = False
    logger.info("grpcio not installed — GrpcBus unavailable (pip install grpcio)")


class GrpcBus:
    """
    gRPC-based MessageBus for cross-process/cross-cluster communication.

    Conforms to the MessageBus protocol. Uses gRPC bidirectional streaming
    for low-latency message delivery between distributed nodes.

    If grpcio is not installed, all operations raise RuntimeError
    with a helpful installation message.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        serializer: Serializer | None = None,
        max_message_size: int = 16 * 1024 * 1024,  # 16 MB
    ) -> None:
        if not _HAS_GRPC:
            raise RuntimeError(
                "grpcio required for GrpcBus. Install with: pip install grpcio grpcio-tools"
            )

        self._host = host
        self._port = port
        self._serializer = serializer or JsonSerializer()
        self._max_message_size = max_message_size

        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        self._running = False
        self._sub_counter = 0
        self._server: Any = None
        self._channel: Any = None
        self.metrics = BusMetrics()

        # Message interceptors
        self._interceptors: list[Any] = []

    def add_interceptor(self, interceptor: Any) -> None:
        """Add a proactive message interceptor."""
        self._interceptors.append(interceptor)

    async def start(self) -> None:
        """Start the gRPC server for receiving messages."""
        self._running = True

        # Start gRPC server
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", self._max_message_size),
                ("grpc.max_send_message_length", self._max_message_size),
            ]
        )

        # Add our generic service handler
        self._server.add_generic_rpc_handlers([_BusServiceHandler(self)])
        self._server.add_insecure_port(f"{self._host}:{self._port}")
        await self._server.start()

        logger.info("GrpcBus server started on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the gRPC server and clean up."""
        self._running = False

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Stop server
        if self._server:
            await self._server.stop(grace=2.0)
            self._server = None

        # Close channel
        if self._channel:
            await self._channel.close()
            self._channel = None

        logger.info("GrpcBus stopped")

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to all local subscribers of a topic."""
        if not self._running:
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

        # Check if response to pending request
        if message.correlation_id and message.correlation_id in self._pending_requests:
            pending = self._pending_requests.pop(message.correlation_id)
            if not pending.done():
                pending.set_result(message)
            return

        # Dispatch to local subscribers
        await self._dispatch_local(topic, message)

    async def request(self, topic: str, message: Message, timeout: float = 90.0) -> Message:
        """Send a request and wait for a correlated response."""
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending_requests[message.id] = future

        await self.publish(topic, message)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            raise TimeoutError(f"gRPC request {message.id} to '{topic}' timed out after {timeout}s")
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
            sub_id=f"grpc-sub-{self._sub_counter}",
            tenant_id=tenant_id,
        )
        self._subscriptions[topic].append(sub)
        self.metrics.record_subscribe()
        logger.debug("gRPC subscribed to '%s' (id=%s)", topic, sub.id)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        subscription.cancel()
        subs = self._subscriptions.get(subscription.topic, [])
        self._subscriptions[subscription.topic] = [s for s in subs if s.id != subscription.id]
        self.metrics.record_unsubscribe()

    async def _dispatch_local(self, topic: str, message: Message) -> None:
        """Dispatch message to matching local subscribers."""
        matching = self._get_matching_topics(topic)
        for match_topic in matching:
            for sub in self._subscriptions.get(match_topic, []):
                if not sub.active:
                    continue
                if sub.tenant_id and message.tenant_id and sub.tenant_id != message.tenant_id:
                    continue

                start = time.monotonic()
                try:
                    response = await sub.handler(message)
                    latency = (time.monotonic() - start) * 1000
                    self.metrics.record_delivery(topic, latency)
                    if response is not None:
                        if response.correlation_id is None:
                            response.correlation_id = message.id
                        await self.publish(response.topic, response)
                except Exception:
                    self.metrics.record_error(topic)
                    logger.exception("gRPC handler error on '%s'", topic)

    def _get_matching_topics(self, topic: str) -> list[str]:
        """Get all registered topics that match (exact + wildcard)."""
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

    async def send_remote(
        self, target_host: str, target_port: int, topic: str, message: Message
    ) -> None:
        """Send a message to a remote GrpcBus instance."""
        channel = grpc.aio.insecure_channel(
            f"{target_host}:{target_port}",
            options=[
                ("grpc.max_receive_message_length", self._max_message_size),
                ("grpc.max_send_message_length", self._max_message_size),
            ],
        )
        try:
            data = self._serializer.serialize(message)
            # Use unary call to the remote bus
            method = channel.unary_unary(
                "/hbllm.Bus/Deliver",
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )
            payload = topic.encode() + b"\x00" + data
            await method(payload)
        finally:
            await channel.close()


class _BusServiceHandler(grpc.GenericRpcHandler):
    """Generic gRPC service handler for the bus."""

    def __init__(self, bus: GrpcBus):
        self._bus = bus
        self._serializer = bus._serializer

    def service(self, handler_call_details: Any) -> Any:
        if handler_call_details.method == "/hbllm.Bus/Deliver":
            return grpc.unary_unary_rpc_method_handler(self._handle_deliver)
        return None

    async def _handle_deliver(self, request: bytes, context: Any) -> bytes:
        """Handle an incoming message delivery."""
        try:
            parts = request.split(b"\x00", 1)
            topic = parts[0].decode()
            message = self._serializer.deserialize(parts[1])
            await self._bus.publish(topic, message)
            return b"OK"
        except Exception as e:
            logger.error("Failed to handle gRPC delivery: %s", e)
            return b"ERROR"
