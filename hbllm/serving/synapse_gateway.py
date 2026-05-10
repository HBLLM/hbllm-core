"""
Synapse Gateway — Bi-directional WebSocket communication for edge devices.

Allows edge devices (e.g., mobile apps) to connect, register local capabilities,
and execute tools remotely on behalf of the central Hub's cognition engine.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import orjson
from fastapi import WebSocket, WebSocketDisconnect

from hbllm.network.bus import MessageBus
from hbllm.network.messages import Message, MessageType
from hbllm.security.audit_log import AuditLog

if TYPE_CHECKING:
    from hbllm.actions.tool_registry import RemoteToolNode

logger = logging.getLogger(__name__)


class SynapseGateway:
    """
    Manages WebSocket connections and bridges them to the internal MessageBus.
    Tracks active devices by (tenant_id, user_id, device_id).
    """

    def __init__(self, bus: MessageBus | None = None, audit_log: AuditLog | None = None):
        self.bus = bus
        self.audit_log = audit_log
        # Map: (tenant_id, user_id, device_id) -> WebSocket
        self.active_connections: dict[tuple[str, str, str], WebSocket] = {}
        # Map: (tenant_id, user_id, device_id) -> list[str] (tool names)
        self.device_capabilities: dict[tuple[str, str, str], list[str]] = {}
        # Map: (tenant_id, user_id, device_id) -> list[RemoteToolNode]
        self.device_nodes: dict[tuple[str, str, str], list[RemoteToolNode]] = {}
        # Map: (tenant_id, user_id, device_id) -> list[dict] (outbound messages)
        self._outbound_queues: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        # Set of devices that have connected in this session
        self.known_devices: set[tuple[str, str, str]] = set()

        self._bus_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        """Start listening to the internal MessageBus for outbound edge commands."""
        if not self.bus:
            logger.warning(
                "SynapseGateway started without a MessageBus. Outbound messages won't work."
            )
            return

        await self.bus.subscribe("edge.tool_call", self._handle_outbound_tool_call)
        logger.info("SynapseGateway started and subscribed to edge.tool_call")

    async def stop(self) -> None:
        """Stop the gateway and disconnect all clients."""
        for key in list(self.active_connections.keys()):
            self.disconnect(*key)
        self.active_connections.clear()
        self.device_capabilities.clear()
        self.device_nodes.clear()

    async def connect(
        self,
        websocket: WebSocket,
        tenant_id: str,
        user_id: str,
        device_id: str,
    ) -> None:
        """Accept a new WebSocket connection from an edge device."""
        await websocket.accept()
        key = (tenant_id, user_id, device_id)

        # Close existing connection if reconnecting
        if key in self.active_connections:
            old_ws = self.active_connections[key]
            try:
                await old_ws.close(code=1008, reason="Concurrent login detected")
            except Exception:
                pass

        self.active_connections[key] = websocket
        self.device_capabilities[key] = []
        self.known_devices.add(key)
        logger.info(f"Edge device connected: tenant={tenant_id} user={user_id} device={device_id}")

        if self.audit_log:
            self.audit_log.log(
                action="auth.ws_connect",
                tenant_id=tenant_id,
                user_id=user_id,
                device_id=device_id,
                resource=f"device:{device_id}",
            )

        # Flush any pending outbound messages (QoS 1)
        pending_msgs = self._outbound_queues.pop(key, [])
        if pending_msgs:
            logger.info(f"Flushing {len(pending_msgs)} pending messages to {device_id}")
            for msg in pending_msgs:
                try:
                    await websocket.send_text(orjson.dumps(msg).decode("utf-8"))
                except Exception as e:
                    logger.error(f"Failed to flush message to {device_id}: {e}")
                    self.disconnect(tenant_id, user_id, device_id)
                    break

    def disconnect(self, tenant_id: str, user_id: str, device_id: str) -> None:
        """Handle a disconnected edge device."""
        key = (tenant_id, user_id, device_id)

        ws = self.active_connections.pop(key, None)
        if ws:
            try:
                import asyncio

                asyncio.create_task(ws.close(code=1000, reason="Disconnected"))
            except Exception:
                pass

        self.device_capabilities.pop(key, None)

        nodes = self.device_nodes.pop(key, [])
        for node in nodes:
            try:
                import asyncio

                asyncio.create_task(node.stop())
            except Exception:
                pass

        logger.info(
            f"Edge device disconnected: tenant={tenant_id} user={user_id} device={device_id}"
        )

        if self.audit_log:
            self.audit_log.log(
                action="auth.ws_disconnect",
                tenant_id=tenant_id,
                user_id=user_id,
                device_id=device_id,
                resource=f"device:{device_id}",
            )

    async def broadcast_to_tenant(self, tenant_id: str, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected devices in a tenant."""
        message_str = orjson.dumps(message).decode("utf-8")
        tasks = []
        for key, ws in self.active_connections.items():
            if key[0] == tenant_id:
                tasks.append(ws.send_text(message_str))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to_device(
        self,
        tenant_id: str,
        user_id: str,
        device_id: str,
        message: dict[str, Any],
    ) -> bool:
        """Send a targeted message to a specific device."""
        key = (tenant_id, user_id, device_id)
        ws = self.active_connections.get(key)
        if not ws:
            # Only buffer if the device is known to have connected before in this session
            if key not in self.known_devices:
                logger.warning(f"Targeting unknown device {device_id}, cannot buffer.")
                return False

            # Device offline: buffer the message for when it reconnects
            if key not in self._outbound_queues:
                self._outbound_queues[key] = []

            # Prevent unbounded memory growth (keep last 100 messages)
            if len(self._outbound_queues[key]) < 100:
                self._outbound_queues[key].append(message)
                logger.info(f"Buffered message for disconnected device {device_id}")
                return True
            else:
                logger.warning(f"Outbound queue full for device {device_id}, dropping message")
                return False

        try:
            await ws.send_text(orjson.dumps(message).decode("utf-8"))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to device {device_id}: {e}")
            self.disconnect(tenant_id, user_id, device_id)

            # Re-queue the failed message if not full
            if key not in self._outbound_queues:
                self._outbound_queues[key] = []
            if len(self._outbound_queues[key]) < 100:
                self._outbound_queues[key].append(message)

            return False

    async def _handle_outbound_tool_call(self, message: Message) -> None:
        """Handle outbound tool calls directed to edge devices."""
        if message.type != MessageType.EVENT and message.type != MessageType.QUERY:
            return

        payload = message.payload
        tenant_id = message.tenant_id
        user_id = message.user_id
        device_id = message.device_id

        if not device_id or device_id == "default":
            logger.warning("Received edge tool call without specific device_id")
            return

        # Forward to the specific device
        outbound_msg = {
            "type": "tool_call",
            "correlation_id": message.correlation_id or message.id,
            "tool_name": payload.get("tool_name"),
            "args": payload.get("args", {}),
        }

        success = await self.send_to_device(tenant_id, user_id, device_id, outbound_msg)
        if not success:
            logger.warning(f"Failed to route tool call to disconnected device {device_id}")
            # Optionally publish a failure back to the bus

    async def handle_inbound_message(
        self,
        tenant_id: str,
        user_id: str,
        device_id: str,
        text_data: str,
    ) -> None:
        """Process messages received from an edge device via WebSocket."""
        try:
            data = orjson.loads(text_data)
            msg_type = data.get("type")

            if msg_type == "register_capabilities":
                # The device is telling us what local tools it has
                tools = data.get("tools", [])
                self.device_capabilities[(tenant_id, user_id, device_id)] = tools

                # Stop existing remote nodes for this device if any
                old_nodes = self.device_nodes.get((tenant_id, user_id, device_id), [])
                for node in old_nodes:
                    await node.stop()

                # Create and start a RemoteToolNode for each capability
                new_nodes = []
                from hbllm.actions.tool_registry import RemoteToolNode

                for tool_name in tools:
                    node = RemoteToolNode(tool_name, tenant_id, user_id, device_id)
                    if self.bus:
                        await node.start(self.bus)
                    new_nodes.append(node)

                self.device_nodes[(tenant_id, user_id, device_id)] = new_nodes
                logger.info(f"Device {device_id} registered tools: {tools}")

                if self.audit_log:
                    self.audit_log.log(
                        action="edge.capability_registered",
                        tenant_id=tenant_id,
                        user_id=user_id,
                        device_id=device_id,
                        resource=f"device:{device_id}",
                        details={"tools": tools},
                    )

            elif msg_type == "tool_result":
                # The device finished running a tool
                if self.bus:
                    tool_msg = Message(
                        type=MessageType.EVENT,
                        source_node_id=f"edge_{device_id}",
                        tenant_id=tenant_id,
                        user_id=user_id,
                        device_id=device_id,
                        topic="edge.tool_result",
                        correlation_id=data.get("correlation_id"),
                        payload={
                            "tool_name": data.get("tool_name"),
                            "result": data.get("result"),
                            "error": data.get("error"),
                        },
                    )
                    await self.bus.publish("edge.tool_result", tool_msg)
            else:
                logger.debug(f"Unknown message type from edge device {device_id}: {msg_type}")

        except orjson.JSONDecodeError:
            logger.warning(f"Invalid JSON from device {device_id}: {text_data}")
        except Exception as e:
            logger.error(f"Error handling edge message from {device_id}: {e}")
