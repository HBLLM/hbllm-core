"""Unit tests for action modules — logic_node, fuzzy_node, mcp_client_node."""

import pytest

from hbllm.actions.fuzzy_node import FuzzyNode
from hbllm.actions.logic_node import LogicNode
from hbllm.actions.mcp_client_node import McpClientNode, McpTool


class TestMcpTool:
    def test_creation(self):
        tool = McpTool(
            name="test-tool",
            description="A test tool",
            input_schema={"type": "object"},
        )
        assert tool.name == "test-tool"

    def test_to_dict(self):
        tool = McpTool(
            name="test-tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        d = tool.to_dict()
        assert d["name"] == "test-tool"
        assert "description" in d


class TestMcpClientNode:
    def test_init(self):
        node = McpClientNode(node_id="mcp-test", server_command="echo")
        assert node.node_id == "mcp-test"

    def test_tools_empty(self):
        node = McpClientNode(node_id="mcp-test", server_command="echo")
        assert node.tools == {}


class TestLogicNode:
    def test_init(self):
        node = LogicNode(node_id="logic-test")
        assert node.node_id == "logic-test"


    @pytest.mark.asyncio
    async def test_on_stop(self):
        node = LogicNode(node_id="logic-test")
        await node.on_stop()


class TestFuzzyNode:
    def test_init(self):
        node = FuzzyNode(node_id="fuzzy-test")
        assert node.node_id == "fuzzy-test"


    @pytest.mark.asyncio
    async def test_on_stop(self):
        node = FuzzyNode(node_id="fuzzy-test")
        await node.on_stop()
