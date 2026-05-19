"""
WebSocket Transport — Global spine for hierarchical Hub ↔ Edge communication.

Wraps WebSocket client logic (UplinkNode-style) into the Transport abstraction.
This transport connects an edge device to a central Hub's SynapseGateway,
advertises local capabilities upstream, and bridges tool calls bidirectionally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.transports.base import Transport, TransportState

logger = logging.getLogger(__name__)


class WebSocketTransport(Transport):
    """
    WebSocket-based transport for hierarchical Hub ↔ Edge communication.

    Maintains a persistent WebSocket connection to an upstream hub,
    sends messages upstream, and receives tool_call / bridge_message
    commands from the hub.
    """

    def __init__(
        self,
        transport_id: str = "ws_uplink",
        upstream_url: str = "ws://localhost:9833/ws/edge",
        tenant_id: str = "default",
        user_id: str = "default",
        device_id: str = "default",
        auth_token: str | None = None,
        local_capabilities: list[str] | None = None,
        device_tier: str = "server",
    ) -> None:
        super().__init__(transport_id=transport_id, transport_type="websocket")
        self.upstream_url = upstream_url
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.device_id = device_id
        self.auth_token = auth_token
        self.local_capabilities = local_capabilities or []
        self.device_tier = device_tier

        self._ws: Any | None = None
        self._connection_task: asyncio.Task[None] | None = None
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}

    async def start(self) -> None:
        """Start the persistent WebSocket connection loop."""
        self._state = TransportState.CONNECTING
        self.metrics.uptime_start = time.monotonic()
        self._connection_task = asyncio.create_task(self._connection_loop())
        logger.info("WebSocketTransport '%s' starting connection to %s", self.transport_id, self.upstream_url)

    async def stop(self) -> None:
        """Disconnect and clean up."""
        self._state = TransportState.STOPPED

        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("WebSocketTransport '%s' stopped", self.transport_id)

    async def send(self, topic: str, message: Message) -> None:
        """Send a message upstream to the Hub via WebSocket."""
        if not self._ws or self._state != TransportState.CONNECTED:
            self.metrics.record_drop()
            logger.warning("WebSocketTransport not connected, dropping message %s", message.id)
            return

        payload = {
            "type": "bridge_message",
            "msg_type": message.type,
            "topic": topic,
            "payload": message.payload,
            "correlation_id": message.correlation_id or message.id,
        }
        try:
            wire = json.dumps(payload)
            await self._ws.send(wire)
            self.metrics.record_send(len(wire))
        except Exception as e:
            self.metrics.record_error()
            logger.error("WebSocketTransport send failed: %s", e)

    async def send_request(
        self, topic: str, message: Message, timeout: float = 30.0
    ) -> Message:
        """Send a request upstream and wait for a correlated response."""
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending_requests[message.id] = future

        await self.send(topic, message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except (TimeoutError, asyncio.TimeoutError):
            raise TimeoutError(
                f"WebSocket request {message.id} to '{topic}' timed out after {timeout}s"
            )
        finally:
            self._pending_requests.pop(message.id, None)

    def has_subscribers(self, topic: str) -> bool:
        """WebSocket transport itself doesn't manage subscriptions locally."""
        return False

    async def _connection_loop(self) -> None:
        """Persistent connection with exponential backoff."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. WebSocketTransport cannot start.")
            self._state = TransportState.DISCONNECTED
            return

        retry_delay = 1.0
        max_delay = 60.0

        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        url = (
            f"{self.upstream_url}"
            f"?tenant_id={self.tenant_id}"
            f"&user_id={self.user_id}"
            f"&device_id={self.device_id}"
        )

        # Detect websockets version for header kwarg
        connect_kwargs: dict[str, Any] = {"extra_headers": headers}
        try:
            import websockets.version
            if int(websockets.version.version.split(".")[0]) >= 14:
                connect_kwargs = {"additional_headers": headers}
        except Exception:
            pass

        while True:
            try:
                self._state = TransportState.CONNECTING
                async with websockets.connect(url, **connect_kwargs) as ws:
                    self._ws = ws
                    self._state = TransportState.CONNECTED
                    retry_delay = 1.0
                    logger.info(
                        "WebSocketTransport '%s' connected to %s",
                        self.transport_id,
                        self.upstream_url,
                    )

                    # Advertise capabilities
                    if self.local_capabilities:
                        await self._ws.send(
                            json.dumps({
                                "type": "register_capabilities",
                                "tools": self.local_capabilities,
                                "device_tier": self.device_tier,
                            })
                        )

                    await self._read_loop()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.reconnections += 1
                logger.warning("WebSocketTransport connection lost: %s. Retrying in %.0fs...", e, retry_delay)

            self._state = TransportState.RECONNECTING
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)

    async def _read_loop(self) -> None:
        """Listen for messages from the Hub."""
        if not self._ws:
            return

        try:
            async for raw in self._ws:
                start = time.monotonic()
                try:
                    data = json.loads(raw)
                    msg_type = data.get("type")
                    self.metrics.record_receive(len(raw) if isinstance(raw, str) else 0)

                    # Check for pending request correlation
                    correlation_id = data.get("correlation_id")
                    if correlation_id and correlation_id in self._pending_requests:
                        future = self._pending_requests.pop(correlation_id)
                        if not future.done():
                            # Construct a Message from the response
                            msg = Message(
                                type=MessageType.RESPONSE,
                                source_node_id="hub",
                                tenant_id=self.tenant_id,
                                topic=data.get("topic", "hub.response"),
                                payload=data.get("result") or data.get("payload", {}),
                                correlation_id=correlation_id,
                            )
                            future.set_result(msg)
                        continue

                    # Dispatch to handler if set
                    if self._message_handler:
                        topic = data.get("topic", f"hub.{msg_type}")
                        msg = Message(
                            type=data.get("msg_type", MessageType.INSTRUCTION),
                            source_node_id="hub",
                            tenant_id=self.tenant_id,
                            user_id=self.user_id,
                            device_id=self.device_id,
                            topic=topic,
                            payload=data.get("payload", {}),
                            correlation_id=correlation_id,
                        )
                        await self._message_handler(msg)

                    latency_ms = (time.monotonic() - start) * 1000
                    self.metrics.record_latency(latency_ms)

                except json.JSONDecodeError:
                    logger.warning("WebSocketTransport received invalid JSON")
                except Exception as e:
                    logger.error("WebSocketTransport read error: %s", e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("WebSocketTransport read loop ended: %s", e)
