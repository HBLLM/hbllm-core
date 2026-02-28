"""
MCP Server — exposes HBLLM cognitive capabilities as MCP tools.

Allows any MCP-compatible AI client (Claude, Gemini, VS Code, etc.)
to call HBLLM's cognitive pipeline, memory, identity, planning, and
learning systems as standard MCP tools.

Supports two transports:
  - stdio  (default, for local development)
  - sse    (for network deployment)

Usage:
  python -m hbllm.serving.mcp_server
  python -m hbllm.serving.mcp_server --transport sse --port 8100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from typing import Any

from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node

logger = logging.getLogger(__name__)

# ─── JSON-RPC Helpers ─────────────────────────────────────────────────────────

def _jsonrpc_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _jsonrpc_notification(method: str, params: Any = None) -> dict:
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# ─── Tool Definitions ────────────────────────────────────────────────────────

HBLLM_TOOLS = [
    {
        "name": "hbllm_chat",
        "description": "Send a query through the full HBLLM cognitive pipeline (workspace → reasoning → critic → decision). Returns the final response.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The user query to process"},
                "tenant_id": {"type": "string", "description": "Tenant ID for multi-tenant isolation", "default": "default"},
                "session_id": {"type": "string", "description": "Session ID for conversation continuity", "default": "default"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "hbllm_memory_query",
        "description": "Search HBLLM's memory systems (episodic, semantic, procedural). Returns matching memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for memory retrieval"},
                "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural", "all"], "default": "all"},
                "tenant_id": {"type": "string", "default": "default"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hbllm_memory_store",
        "description": "Store a fact or skill in HBLLM's memory systems.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to store"},
                "memory_type": {"type": "string", "enum": ["episodic", "semantic", "skill"], "default": "semantic"},
                "tenant_id": {"type": "string", "default": "default"},
                "metadata": {"type": "object", "description": "Additional metadata", "default": {}},
            },
            "required": ["content"],
        },
    },
    {
        "name": "hbllm_identity_get",
        "description": "Get the current identity/persona profile for a tenant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "default": "default"},
            },
        },
    },
    {
        "name": "hbllm_identity_set",
        "description": "Configure the identity/persona for a tenant (system prompt, goals, constraints, personality).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "default": "default"},
                "persona_name": {"type": "string", "description": "Name of the persona"},
                "system_prompt": {"type": "string", "description": "System prompt for the persona"},
                "goals": {"type": "array", "items": {"type": "string"}, "description": "Persona goals"},
                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Persona constraints"},
            },
        },
    },
    {
        "name": "hbllm_plan",
        "description": "Decompose a complex query into a Graph-of-Thoughts plan (DAG of reasoning steps).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Complex query to decompose"},
                "tenant_id": {"type": "string", "default": "default"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hbllm_feedback",
        "description": "Submit RLHF feedback (thumbs up/down) for a previous response. Feeds into DPO training.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "ID of the message to rate"},
                "rating": {"type": "integer", "enum": [-1, 0, 1], "description": "-1=bad, 0=neutral, 1=good"},
                "tenant_id": {"type": "string", "default": "default"},
                "comment": {"type": "string", "description": "Optional feedback comment", "default": ""},
            },
            "required": ["message_id", "rating"],
        },
    },
    {
        "name": "hbllm_health",
        "description": "Get the health status of the HBLLM cognitive system (node count, capabilities, bus type).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ─── MCP Server ───────────────────────────────────────────────────────────────

class HBLLMMcpServer:
    """
    MCP-compliant server that exposes HBLLM capabilities as tools.

    Implements the Model Context Protocol over JSON-RPC (stdio transport).
    Tools map to HBLLM MessageBus topics.
    """

    def __init__(self, bus: MessageBus | None = None):
        self.bus = bus
        self._initialized = False
        self._server_info = {
            "name": "hbllm-mcp-server",
            "version": "1.0.0",
        }

    async def ensure_bus(self) -> MessageBus:
        """Lazily create and start the bus if not provided."""
        if self.bus is None:
            self.bus = InProcessBus()
            await self.bus.start()
        return self.bus

    async def handle_request(self, request: dict) -> dict | None:
        """
        Handle an incoming JSON-RPC request and return a response.
        Returns None for notifications (no response expected).
        """
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        # Notifications (no id) don't get responses
        is_notification = req_id is None

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "initialized":
                self._initialized = True
                return None  # notification
            elif method == "tools/list":
                result = await self._handle_tools_list(params)
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            elif method == "resources/list":
                result = await self._handle_resources_list(params)
            elif method == "prompts/list":
                result = await self._handle_prompts_list(params)
            elif method == "ping":
                result = {}
            else:
                if is_notification:
                    return None
                return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")

            if is_notification:
                return None
            return _jsonrpc_response(req_id, result)

        except Exception as e:
            logger.exception("Error handling MCP request: %s", method)
            if is_notification:
                return None
            return _jsonrpc_error(req_id, -32603, str(e))

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle MCP initialize handshake."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": self._server_info,
        }

    async def _handle_tools_list(self, params: dict) -> dict:
        """List all available HBLLM tools."""
        return {"tools": HBLLM_TOOLS}

    async def _handle_tools_call(self, params: dict) -> dict:
        """Execute an HBLLM tool by dispatching to the MessageBus."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        bus = await self.ensure_bus()

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

        try:
            result = await handler(bus, arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    async def _handle_resources_list(self, params: dict) -> dict:
        return {"resources": []}

    async def _handle_prompts_list(self, params: dict) -> dict:
        return {"prompts": []}

    # ─── Tool Implementations ─────────────────────────────────────────────

    async def _tool_hbllm_chat(self, bus: MessageBus, args: dict) -> dict:
        """Full cognitive pipeline inference."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="mcp_server",
            topic="workspace.process",
            tenant_id=args.get("tenant_id", "default"),
            session_id=args.get("session_id", "default"),
            payload={"text": args["text"]},
        )
        try:
            response = await bus.request("workspace.process", msg, timeout=60.0)
            return {"response": response.payload.get("response", ""), "message_id": response.id}
        except TimeoutError:
            return {"response": "Processing timed out", "error": True}

    async def _tool_hbllm_memory_query(self, bus: MessageBus, args: dict) -> dict:
        """Search memory systems."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="mcp_server",
            topic="memory.search",
            tenant_id=args.get("tenant_id", "default"),
            payload={
                "query": args["query"],
                "memory_type": args.get("memory_type", "all"),
                "limit": args.get("limit", 10),
            },
        )
        try:
            response = await bus.request("memory.search", msg, timeout=30.0)
            return response.payload
        except TimeoutError:
            return {"results": [], "error": "Memory query timed out"}

    async def _tool_hbllm_memory_store(self, bus: MessageBus, args: dict) -> dict:
        """Store in memory."""
        memory_type = args.get("memory_type", "semantic")
        topic = f"memory.{memory_type}.store" if memory_type != "skill" else "memory.skill.store"

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="mcp_server",
            topic=topic,
            tenant_id=args.get("tenant_id", "default"),
            payload={
                "content": args["content"],
                "metadata": args.get("metadata", {}),
            },
        )
        await bus.publish(topic, msg)
        return {"stored": True, "memory_type": memory_type}

    async def _tool_hbllm_identity_get(self, bus: MessageBus, args: dict) -> dict:
        """Get identity profile."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="mcp_server",
            topic="identity.query",
            tenant_id=args.get("tenant_id", "default"),
            payload={},
        )
        try:
            response = await bus.request("identity.query", msg, timeout=10.0)
            return response.payload
        except TimeoutError:
            return {"error": "Identity query timed out"}

    async def _tool_hbllm_identity_set(self, bus: MessageBus, args: dict) -> dict:
        """Update identity profile."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="mcp_server",
            topic="identity.update",
            tenant_id=args.get("tenant_id", "default"),
            payload={k: v for k, v in args.items() if k != "tenant_id"},
        )
        await bus.publish("identity.update", msg)
        return {"updated": True}

    async def _tool_hbllm_plan(self, bus: MessageBus, args: dict) -> dict:
        """GoT planning."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="mcp_server",
            topic="planner.plan",
            tenant_id=args.get("tenant_id", "default"),
            payload={"query": args["query"]},
        )
        try:
            response = await bus.request("planner.plan", msg, timeout=60.0)
            return response.payload
        except TimeoutError:
            return {"error": "Planning timed out"}

    async def _tool_hbllm_feedback(self, bus: MessageBus, args: dict) -> dict:
        """Submit RLHF feedback."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="mcp_server",
            topic="system.feedback",
            tenant_id=args.get("tenant_id", "default"),
            payload={
                "message_id": args["message_id"],
                "rating": args["rating"],
                "comment": args.get("comment", ""),
            },
        )
        await bus.publish("system.feedback", msg)
        return {"recorded": True}

    async def _tool_hbllm_health(self, bus: MessageBus, args: dict) -> dict:
        """System health."""
        return {
            "status": "healthy",
            "bus_type": type(bus).__name__,
            "server": self._server_info,
        }


# ─── Stdio Transport ──────────────────────────────────────────────────────────

async def run_stdio(server: HBLLMMcpServer) -> None:
    """Run the MCP server over stdio (stdin/stdout JSON-RPC)."""
    logger.info("HBLLM MCP server starting on stdio")

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

    while True:
        try:
            # Read Content-Length header
            header = await reader.readline()
            if not header:
                break

            header_str = header.decode().strip()
            if header_str.startswith("Content-Length:"):
                content_length = int(header_str.split(":")[1].strip())
                # Read empty line
                await reader.readline()
                # Read body
                body = await reader.readexactly(content_length)
                request = json.loads(body.decode())
            else:
                # Try plain JSON (one per line)
                if header_str:
                    request = json.loads(header_str)
                else:
                    continue

            response = await server.handle_request(request)

            if response is not None:
                response_bytes = json.dumps(response).encode()
                header = f"Content-Length: {len(response_bytes)}\r\n\r\n".encode()
                writer.write(header + response_bytes)
                await writer.drain()

        except (json.JSONDecodeError, asyncio.IncompleteReadError):
            continue
        except Exception as e:
            logger.exception("Error in stdio transport")
            break


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HBLLM MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Keep stdout clean for MCP
    )

    server = HBLLMMcpServer()

    if args.transport == "stdio":
        asyncio.run(run_stdio(server))
    else:
        logger.error("SSE transport not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
