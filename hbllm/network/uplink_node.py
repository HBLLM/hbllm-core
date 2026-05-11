"""
Uplink Node — Connects a local HBLLM instance to a remote central Hub.

Allows an edge HBLLM instance (or any Python process) to connect to a
central Core's SynapseGateway. It advertises its local tools to the central Hub,
listens for remote tool_call requests, executes them on the local MessageBus,
and returns the results upstream.

Usage:
  node = UplinkNode(
      node_id="desktop_uplink",
      upstream_url="ws://localhost:9833/ws/edge",
      tenant_id="default",
      user_id="user_1",
      device_id="desktop_1",
      auth_token="jwt_token_here",
      local_tools=["local_file_read", "mcp.some_tool"]
  )
  await node.start(bus)
"""

import asyncio
import json
import logging
from typing import Any

import websockets

from hbllm.network.bus import Subscription
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class UplinkNode(Node):
    """
    Connects to a central SynapseGateway and bridges local tools upstream.
    """

    def __init__(
        self,
        node_id: str,
        upstream_url: str,
        tenant_id: str,
        user_id: str,
        device_id: str,
        auth_token: str | None = None,
        local_tools: list[str] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["uplink_bridge"],
        )
        self.upstream_url = upstream_url
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.device_id = device_id
        self.auth_token = auth_token
        self.local_tools = local_tools or []

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._read_task: asyncio.Task | None = None
        self._subs: list[Subscription] = []
        self._pending_calls: dict[str, str] = {}  # correlation_id -> tool_name

    async def on_start(self) -> None:
        """Start the persistent connection loop."""
        self._read_task = asyncio.create_task(self._connection_loop())
        # Subscribe to local topics that should be forwarded upstream
        sub = await self.bus.subscribe("uplink.send", self._handle_local_outbound)
        self._subs.append(sub)

    async def _handle_local_outbound(self, message: Message) -> None:
        """Forward a message from the local bus to the upstream Hub."""
        if not self._ws:
            return

        payload = {
            "type": "bridge_message",
            "msg_type": message.type,
            "topic": message.topic,
            "payload": message.payload,
            "correlation_id": message.correlation_id or message.id,
        }
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.error("UplinkNode failed to forward message upstream: %s", e)

    async def _handle_bridged_instruction(self, data: dict[str, Any]) -> None:
        """Route a high-level instruction from upstream to the local MessageBus."""
        topic = data.get("topic", "hub.instruction")
        msg_type = data.get("msg_type", MessageType.INSTRUCTION)
        payload = data.get("payload", {})
        correlation_id = data.get("correlation_id")

        logger.debug("UplinkNode routing upstream instruction: %s", topic)

        # Publish to local bus
        bridged_msg = Message(
            type=msg_type,
            source_node_id="hub",
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            device_id=self.device_id,
            topic=topic,
            correlation_id=correlation_id,
            payload=payload,
        )
        await self.bus.publish(topic, bridged_msg)

    async def _connection_loop(self) -> None:
        """Maintains a persistent connection with exponential backoff."""
        retry_delay = 1.0
        max_delay = 60.0

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        url = f"{self.upstream_url}?tenant_id={self.tenant_id}&user_id={self.user_id}&device_id={self.device_id}"

        # Use additional_headers for websockets 14+, extra_headers for <14
        connect_kwargs = {"extra_headers": headers}
        try:
            import websockets.version

            if int(websockets.version.version.split(".")[0]) >= 14:
                connect_kwargs = {"additional_headers": headers}
        except Exception:
            pass

        while True:
            try:
                async with websockets.connect(url, **connect_kwargs) as ws:
                    self._ws = ws
                    logger.info("UplinkNode '%s' connected to %s", self.node_id, self.upstream_url)

                    # Reset retry delay on successful connection
                    retry_delay = 1.0

                    # Register local tools upstream
                    if self.local_tools:
                        await self._ws.send(
                            json.dumps({"type": "register_capabilities", "tools": self.local_tools})
                        )
                        logger.info(
                            "UplinkNode advertised %d tools upstream", len(self.local_tools)
                        )

                    # Blocks here until connection closes or errors
                    await self._read_loop()

            except websockets.exceptions.ConnectionClosed:
                logger.warning(
                    "UplinkNode disconnected from %s. Reconnecting...", self.upstream_url
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("UplinkNode connection failed: %s", e)

            # Wait before reconnecting (exponential backoff)
            logger.info("UplinkNode waiting %d seconds before reconnect...", retry_delay)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)

    async def handle_message(self, message: Message) -> None:
        """Handle messages addressed directly to this node. (Required by base class)"""
        logger.debug("UplinkNode received direct message: %s", message.id)

    async def on_stop(self) -> None:
        """Disconnect from upstream and clean up subscriptions."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            await self._ws.close()

        # Clean up subscriptions
        for sub in self._subs:
            await self.bus.unsubscribe(sub)
        self._subs.clear()
        
        logger.info("UplinkNode '%s' stopped", self.node_id)

    async def _read_loop(self) -> None:
        """Listen for messages from the central Hub."""
        if not self._ws:
            return

        try:
            async for message_text in self._ws:
                try:
                    data = json.loads(message_text)
                    msg_type = data.get("type")

                    if msg_type == "tool_call":
                        await self._handle_upstream_tool_call(data)
                    elif msg_type == "bridge_message":
                        await self._handle_bridged_instruction(data)
                    else:
                        logger.debug("UplinkNode received unknown message: %s", msg_type)

                except json.JSONDecodeError:
                    logger.warning("UplinkNode received invalid JSON: %s", message_text)
                except Exception as e:
                    logger.error("UplinkNode error handling message: %s", e)
        except websockets.exceptions.ConnectionClosed:
            logger.info("UplinkNode disconnected from upstream.")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("UplinkNode read loop error: %s", e)

    async def _handle_upstream_tool_call(self, data: dict[str, Any]) -> None:
        """Route a tool call from upstream to the local MessageBus."""
        correlation_id = data.get("correlation_id", "")
        tool_name = data.get("tool_name", "")
        args = data.get("args", {})

        if not tool_name:
            return

        logger.debug("UplinkNode routing upstream tool call: %s", tool_name)

        # Publish to local bus
        # Topic is usually action.tool.{tool_name} or mcp.{node_id}.{tool_name}
        topic = f"action.tool.{tool_name}"
        if tool_name.startswith("mcp."):
            topic = tool_name

        self._pending_calls[correlation_id] = tool_name

        try:
            # We use request() to wait for the local result
            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=self.tenant_id,
                session_id="uplink_session",
                topic=topic,
                payload={"arguments": args} if tool_name.startswith("mcp.") else args,
            )

            result_msg = await self.bus.request(topic, req_msg, timeout=30.0)

            # Send result back upstream
            await self._send_tool_result(correlation_id, tool_name, result=result_msg.payload)

        except asyncio.TimeoutError:
            await self._send_tool_result(
                correlation_id, tool_name, error="Tool execution timed out"
            )
        except Exception as e:
            await self._send_tool_result(correlation_id, tool_name, error=str(e))
        finally:
            self._pending_calls.pop(correlation_id, None)

    async def _send_tool_result(
        self, correlation_id: str, tool_name: str, result: Any = None, error: str | None = None
    ) -> None:
        """Send a tool execution result back to the central Hub."""
        if not self._ws:
            return

        payload = {
            "type": "tool_result",
            "correlation_id": correlation_id,
            "tool_name": tool_name,
            "result": result,
            "error": error,
        }
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.error("UplinkNode failed to send tool result: %s", e)
