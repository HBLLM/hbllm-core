"""
MCP Client Node — connects to external MCP servers and bridges their tools to the bus.

Any external MCP server (database tools, file system, web scraping, etc.)
can be consumed by HBLLM nodes through this bridge. The tools become
regular MessageBus capabilities that PlannerNode and ExecutionNode can invoke.

Architecture:
  External MCP Server ←stdio/sse→ McpClientNode ←bus→ Other HBLLM Nodes

Usage:
  client = McpClientNode(
      node_id="mcp_db_tools",
      server_command="npx",
      server_args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://..."],
  )
  await client.start(bus)
  # Now other nodes can: bus.request("mcp.mcp_db_tools.query", msg)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from hbllm.network.bus import MessageBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class McpTool:
    """Metadata for a discovered external MCP tool."""

    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class McpClientNode(Node):
    """
    Bridges an external MCP server's tools into the HBLLM MessageBus.

    Lifecycle:
      1. Spawns the MCP server process (stdio transport)
      2. Sends initialize → initialized handshake
      3. Discovers tools via tools/list
      4. Subscribes to bus topics for each tool: mcp.<node_id>.<tool_name>
      5. When a bus message arrives, forwards to MCP server and returns result
    """

    def __init__(
        self,
        node_id: str,
        server_command: str,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["mcp_bridge"],
        )
        self.server_command = server_command
        self.server_args = server_args or []
        self.server_env = server_env

        self._process: asyncio.subprocess.Process | None = None
        self._tools: dict[str, McpTool] = {}
        self._request_id_counter = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None

    @property
    def tools(self) -> dict[str, McpTool]:
        """Get discovered MCP tools."""
        return dict(self._tools)

    async def on_start(self) -> None:
        """Spawn MCP server, handshake, discover tools, subscribe to bus."""
        try:
            await self._spawn_server()
            await self._initialize()
            await self._discover_tools()
            await self._subscribe_to_bus()
            logger.info(
                "McpClientNode '%s' started with %d tools from '%s'",
                self.node_id, len(self._tools), self.server_command,
            )
        except Exception as e:
            logger.error("McpClientNode '%s' failed to start: %s", self.node_id, e)
            await self._cleanup()

    async def on_stop(self) -> None:
        """Shut down the MCP server process."""
        await self._cleanup()
        logger.info("McpClientNode '%s' stopped", self.node_id)

    async def handle_message(self, message: Message) -> Message | None:
        """Handle bus messages by forwarding to the MCP server."""
        tool_name = message.payload.get("tool_name", "")

        if not tool_name:
            # Extract tool name from topic: mcp.<node_id>.<tool_name>
            parts = message.topic.split(".")
            if len(parts) >= 3:
                tool_name = parts[2]

        if tool_name not in self._tools:
            return Message(
                type=MessageType.ERROR,
                source_node_id=self.node_id,
                topic=message.topic,
                correlation_id=message.id,
                tenant_id=message.tenant_id,
                payload={"error": f"Unknown MCP tool: {tool_name}"},
            )

        arguments = message.payload.get("arguments", {})
        result = await self._call_tool(tool_name, arguments)

        return Message(
            type=MessageType.RESPONSE,
            source_node_id=self.node_id,
            topic=message.topic,
            correlation_id=message.id,
            tenant_id=message.tenant_id,
            payload=result,
        )

    # ─── MCP Protocol ─────────────────────────────────────────────────────

    async def _spawn_server(self) -> None:
        """Start the MCP server as a subprocess."""
        env = None
        if self.server_env:
            import os
            env = {**os.environ, **self.server_env}

        self._process = await asyncio.create_subprocess_exec(
            self.server_command,
            *self.server_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        logger.info(
            "Spawned MCP server: %s %s (pid=%s)",
            self.server_command,
            " ".join(self.server_args),
            self._process.pid,
        )

        # Start reading responses
        self._read_task = asyncio.create_task(self._read_loop())

    async def _initialize(self) -> None:
        """Send MCP initialize handshake."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": f"hbllm-{self.node_id}",
                "version": "1.0.0",
            },
        })
        logger.debug("MCP initialize response: %s", result)

        # Send initialized notification
        await self._send_notification("initialized", {})

    async def _discover_tools(self) -> None:
        """Discover tools from the MCP server."""
        result = await self._send_request("tools/list", {})
        tools = result.get("tools", [])

        for tool_def in tools:
            tool = McpTool(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                input_schema=tool_def.get("inputSchema", {}),
            )
            self._tools[tool.name] = tool
            self.capabilities.append(f"mcp.{tool.name}")

        logger.info("Discovered %d MCP tools: %s", len(self._tools), list(self._tools.keys()))

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the external MCP server."""
        try:
            result = await self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
            return {
                "tool": tool_name,
                "content": result.get("content", []),
                "is_error": result.get("isError", False),
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "content": [{"type": "text", "text": str(e)}],
                "is_error": True,
            }

    async def _subscribe_to_bus(self) -> None:
        """Subscribe to bus topics for each discovered tool."""
        for tool_name in self._tools:
            topic = f"mcp.{self.node_id}.{tool_name}"
            await self.bus.subscribe(topic, self.handle_message)
            logger.debug("Subscribed to bus topic: %s", topic)

        # Generic topic for any tool on this bridge
        await self.bus.subscribe(f"mcp.{self.node_id}.*", self.handle_message)

    # ─── JSON-RPC Transport ───────────────────────────────────────────────

    async def _send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP server not running")

        self._request_id_counter += 1
        req_id = self._request_id_counter

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        body = json.dumps(request).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        self._process.stdin.write(header + body)
        await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"MCP request '{method}' timed out after {timeout}s")

    async def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        body = json.dumps(notification).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        self._process.stdin.write(header + body)
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Read JSON-RPC responses from the MCP server's stdout."""
        if not self._process or not self._process.stdout:
            return

        reader = self._process.stdout
        buffer = b""

        while True:
            try:
                # Read until we get a Content-Length header
                line = await reader.readline()
                if not line:
                    break

                line_str = line.decode().strip()

                if line_str.startswith("Content-Length:"):
                    content_length = int(line_str.split(":")[1].strip())
                    # Read empty separator line
                    await reader.readline()
                    # Read body
                    body = await reader.readexactly(content_length)
                    msg = json.loads(body.decode())
                elif line_str:
                    # Try plain JSON
                    try:
                        msg = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue

                # Dispatch response to pending future
                if "id" in msg and msg["id"] in self._pending:
                    future = self._pending.pop(msg["id"])
                    if not future.done():
                        if "error" in msg:
                            future.set_exception(
                                RuntimeError(f"MCP error: {msg['error'].get('message', 'Unknown')}")
                            )
                        else:
                            future.set_result(msg.get("result", {}))

            except asyncio.CancelledError:
                break
            except asyncio.IncompleteReadError:
                break
            except Exception as e:
                logger.error("Error reading MCP response: %s", e)

    async def _cleanup(self) -> None:
        """Clean up subprocess and pending requests."""
        # Cancel pending
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

        # Cancel read task
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._process.kill()
            self._process = None

    async def list_tools(self) -> list[dict]:
        """Return all discovered tools as dicts."""
        return [t.to_dict() for t in self._tools.values()]
