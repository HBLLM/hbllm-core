"""Tests for MCP Integration — server tool listing, tool execution, client node."""

import asyncio
import json
import pytest

from hbllm.network.bus import InProcessBus
from hbllm.serving.mcp_server import HBLLMMcpServer, HBLLM_TOOLS
from hbllm.actions.mcp_client_node import McpClientNode, McpTool


# ─── MCP Server Tests ─────────────────────────────────────────────────────────

class TestMcpServer:
    """Tests for the HBLLM MCP Server."""

    @pytest.fixture
    async def server(self):
        bus = InProcessBus()
        await bus.start()
        s = HBLLMMcpServer(bus=bus)
        yield s
        await bus.stop()

    async def test_initialize(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        })
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in response["result"]["capabilities"]
        assert response["result"]["serverInfo"]["name"] == "hbllm-mcp-server"

    async def test_initialized_notification(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {},
        })
        # Notifications return None
        assert response is None
        assert server._initialized is True

    async def test_tools_list(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        tools = response["result"]["tools"]
        assert len(tools) == len(HBLLM_TOOLS)

        tool_names = {t["name"] for t in tools}
        assert "hbllm_chat" in tool_names
        assert "hbllm_memory_query" in tool_names
        assert "hbllm_identity_get" in tool_names
        assert "hbllm_plan" in tool_names
        assert "hbllm_health" in tool_names
        assert "hbllm_feedback" in tool_names

    async def test_tools_call_health(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "hbllm_health", "arguments": {}},
        })
        result = response["result"]
        assert result["isError"] is False

        content = json.loads(result["content"][0]["text"])
        assert content["status"] == "healthy"
        assert content["bus_type"] == "InProcessBus"

    async def test_tools_call_unknown(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert response["result"]["isError"] is True

    async def test_method_not_found(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method",
            "params": {},
        })
        assert "error" in response
        assert response["error"]["code"] == -32601

    async def test_ping(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "ping",
            "params": {},
        })
        assert response["result"] == {}

    async def test_resources_list(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "resources/list",
            "params": {},
        })
        assert response["result"]["resources"] == []

    async def test_prompts_list(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "prompts/list",
            "params": {},
        })
        assert response["result"]["prompts"] == []

    async def test_tools_call_memory_store(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "hbllm_memory_store",
                "arguments": {"content": "test fact", "memory_type": "semantic"},
            },
        })
        result = response["result"]
        assert result["isError"] is False
        content = json.loads(result["content"][0]["text"])
        assert content["stored"] is True

    async def test_tools_call_identity_set(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "hbllm_identity_set",
                "arguments": {
                    "persona_name": "Test Bot",
                    "system_prompt": "You are helpful",
                },
            },
        })
        result = response["result"]
        assert result["isError"] is False

    async def test_tools_call_feedback(self, server):
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "hbllm_feedback",
                "arguments": {"message_id": "msg-123", "rating": 1},
            },
        })
        result = response["result"]
        assert result["isError"] is False
        content = json.loads(result["content"][0]["text"])
        assert content["recorded"] is True


# ─── Tool Definitions Tests ──────────────────────────────────────────────────

class TestToolDefinitions:
    """Verify tool definition schema validity."""

    def test_all_tools_have_required_fields(self):
        for tool in HBLLM_TOOLS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing description"
            assert "inputSchema" in tool, f"Tool {tool['name']} missing inputSchema"
            assert tool["inputSchema"]["type"] == "object"

    def test_tool_names_unique(self):
        names = [t["name"] for t in HBLLM_TOOLS]
        assert len(names) == len(set(names))


# ─── McpTool Tests ───────────────────────────────────────────────────────────

class TestMcpTool:
    def test_to_dict(self):
        tool = McpTool(name="test", description="Test tool", input_schema={"type": "object"})
        d = tool.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test tool"
        assert d["input_schema"] == {"type": "object"}


# ─── McpClientNode Tests ─────────────────────────────────────────────────────

class TestMcpClientNode:
    def test_node_init(self):
        node = McpClientNode(
            node_id="test_mcp",
            server_command="echo",
            server_args=["hello"],
        )
        assert node.node_id == "test_mcp"
        assert node.server_command == "echo"
        assert "mcp_bridge" in node.capabilities
        assert len(node.tools) == 0

    def test_tools_empty_initially(self):
        node = McpClientNode(
            node_id="test",
            server_command="true",
        )
        assert node.tools == {}
        assert node.server_args == []
