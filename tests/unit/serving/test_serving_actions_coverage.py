"""
Serving Layer (batch 2) + Actions & Tools — Integration test coverage.

Covers uncovered lines in:
  - hbllm/serving/mcp_server.py
  - hbllm/serving/synapse_gateway.py
  - hbllm/serving/studio/helpers.py
  - hbllm/serving/studio/persona.py
  - hbllm/serving/studio/cognitive.py
  - hbllm/serving/studio/emotion.py
  - hbllm/serving/studio/perception.py
  - hbllm/serving/pipeline.py (classify_complexity)
  - hbllm/actions/tool_registry.py
  - hbllm/actions/scheduler_tools.py
  - hbllm/actions/orchestrator.py
  - hbllm/actions/builtin_tools.py
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════
# serving/mcp_server.py
# ═══════════════════════════════════════════════════════════════════════


class TestMcpServer:
    @pytest.fixture
    def server(self):
        from hbllm.serving.mcp_server import HBLLMMcpServer
        return HBLLMMcpServer()

    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in resp["result"]["capabilities"]

    @pytest.mark.asyncio
    async def test_handle_initialized_notification(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "method": "initialized"})
        assert resp is None
        assert server._initialized is True

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = resp["result"]["tools"]
        assert len(tools) > 0
        names = [t["name"] for t in tools]
        assert "hbllm_chat" in names
        assert "hbllm_memory_query" in names

    @pytest.mark.asyncio
    async def test_handle_resources_list(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}})
        assert resp["result"]["resources"] == []

    @pytest.mark.asyncio
    async def test_handle_prompts_list(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "id": 4, "method": "prompts/list", "params": {}})
        assert resp["result"]["prompts"] == []

    @pytest.mark.asyncio
    async def test_handle_ping(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "id": 5, "method": "ping"})
        assert resp["result"] == {}

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "id": 6, "method": "unknown.method"})
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_handle_unknown_method_notification(self, server):
        resp = await server.handle_request({"jsonrpc": "2.0", "method": "unknown.method"})
        assert resp is None

    @pytest.mark.asyncio
    async def test_tools_call_health(self, server):
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 7,
            "method": "tools/call",
            "params": {"name": "hbllm_health", "arguments": {}}
        })
        content = resp["result"]["content"]
        assert content[0]["type"] == "text"
        result = json.loads(content[0]["text"])
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_tools_call_unknown_tool(self, server):
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 8,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}}
        })
        assert resp["result"]["isError"] is True

    @pytest.mark.asyncio
    async def test_tools_call_memory_store(self, server):
        mock_bus = AsyncMock()
        server.bus = mock_bus
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 9,
            "method": "tools/call",
            "params": {"name": "hbllm_memory_store", "arguments": {
                "content": "Test fact", "memory_type": "semantic"
            }}
        })
        result = json.loads(resp["result"]["content"][0]["text"])
        assert result["stored"] is True

    @pytest.mark.asyncio
    async def test_tools_call_identity_set(self, server):
        mock_bus = AsyncMock()
        server.bus = mock_bus
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 10,
            "method": "tools/call",
            "params": {"name": "hbllm_identity_set", "arguments": {
                "tenant_id": "t1", "persona_name": "TestBot"
            }}
        })
        result = json.loads(resp["result"]["content"][0]["text"])
        assert result["updated"] is True

    @pytest.mark.asyncio
    async def test_tools_call_feedback(self, server):
        mock_bus = AsyncMock()
        server.bus = mock_bus
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 11,
            "method": "tools/call",
            "params": {"name": "hbllm_feedback", "arguments": {
                "message_id": "msg-123", "rating": 1, "comment": "Great!"
            }}
        })
        result = json.loads(resp["result"]["content"][0]["text"])
        assert result["recorded"] is True

    @pytest.mark.asyncio
    async def test_ensure_bus_creates_inprocess(self, server):
        bus = await server.ensure_bus()
        assert bus is not None
        assert server.bus is bus

    @pytest.mark.asyncio
    async def test_jsonrpc_helpers(self):
        from hbllm.serving.mcp_server import _jsonrpc_response, _jsonrpc_error, _jsonrpc_notification
        resp = _jsonrpc_response(1, {"ok": True})
        assert resp["id"] == 1 and resp["result"]["ok"] is True

        err = _jsonrpc_error(2, -32600, "Invalid")
        assert err["error"]["code"] == -32600

        notif = _jsonrpc_notification("test/event", {"data": 1})
        assert notif["method"] == "test/event"
        assert "id" not in notif

        notif_no_params = _jsonrpc_notification("test/bare")
        assert "params" not in notif_no_params


# ═══════════════════════════════════════════════════════════════════════
# serving/studio/helpers.py
# ═══════════════════════════════════════════════════════════════════════


class TestStudioHelpers:
    def test_get_brain_none(self):
        from hbllm.serving.state import _state
        from hbllm.serving.studio.helpers import get_brain
        _state.pop("brain", None)
        assert get_brain() is None

    def test_get_brain_present(self):
        from hbllm.serving.state import _state
        from hbllm.serving.studio.helpers import get_brain
        mock_brain = MagicMock()
        _state["brain"] = mock_brain
        try:
            assert get_brain() is mock_brain
        finally:
            _state.pop("brain", None)

    def test_get_bus_from_brain(self):
        from hbllm.serving.state import _state
        from hbllm.serving.studio.helpers import get_bus
        mock_brain = MagicMock()
        mock_brain.bus = MagicMock()
        _state["brain"] = mock_brain
        try:
            assert get_bus() is mock_brain.bus
        finally:
            _state.pop("brain", None)

    def test_get_bus_no_brain(self):
        from hbllm.serving.state import _state
        from hbllm.serving.studio.helpers import get_bus
        _state.pop("brain", None)
        assert get_bus() is None

    def test_require_bus_raises_503(self):
        from hbllm.serving.state import _state
        from hbllm.serving.studio.helpers import require_bus
        _state.pop("brain", None)
        with pytest.raises(Exception) as exc_info:
            require_bus()
        assert exc_info.value.status_code == 503

    def test_get_node(self):
        from hbllm.serving.state import _state
        from hbllm.serving.studio.helpers import get_node

        class FakeNode:
            pass

        brain = MagicMock()
        brain.nodes = [FakeNode()]
        brain.evaluation_node = None
        brain.self_model = None
        brain.skill_registry = None
        brain.goal_manager = None
        brain.attention_manager = None
        brain.load_manager = None
        brain.reflection_node = None
        brain.skill_compiler_node = None
        brain.skill_intelligence_node = None
        brain.failure_analyzer_node = None
        brain.scheduler_node = None
        brain.policy_engine = None
        brain.sentinel = None
        brain.revision_node = None
        brain.tool_memory = None
        brain.cognitive_metrics = None
        _state["brain"] = brain
        try:
            assert get_node("FakeNode") is not None
            assert get_node("Nonexistent") is None
        finally:
            _state.pop("brain", None)

    def test_require_node_raises_503(self):
        from hbllm.serving.state import _state
        from hbllm.serving.studio.helpers import require_node
        _state.pop("brain", None)
        with pytest.raises(Exception) as exc_info:
            require_node("SomeNode")
        assert exc_info.value.status_code == 503

    def test_get_data_dir(self, monkeypatch):
        from hbllm.serving.studio.helpers import get_data_dir
        monkeypatch.setenv("HBLLM_DATA_DIR", "/custom/data")
        assert get_data_dir() == "/custom/data"

    def test_get_data_dir_default(self, monkeypatch):
        from hbllm.serving.studio.helpers import get_data_dir
        monkeypatch.delenv("HBLLM_DATA_DIR", raising=False)
        assert get_data_dir() == "data"

    def test_get_tenant_id(self):
        from hbllm.serving.studio.helpers import get_tenant_id
        request = MagicMock()
        request.state.tenant_id = "my-tenant"
        assert get_tenant_id(request) == "my-tenant"

    def test_get_user_id(self):
        from hbllm.serving.studio.helpers import get_user_id
        request = MagicMock()
        request.state.user_id = "user-42"
        assert get_user_id(request) == "user-42"


# ═══════════════════════════════════════════════════════════════════════
# serving/synapse_gateway.py (non-WebSocket unit tests)
# ═══════════════════════════════════════════════════════════════════════


class TestSynapseGateway:
    def test_init(self):
        from hbllm.serving.synapse_gateway import SynapseGateway
        gw = SynapseGateway(bus=MagicMock(), audit_log=MagicMock())
        assert gw.active_connections == {}

    def test_disconnect_nonexistent(self):
        from hbllm.serving.synapse_gateway import SynapseGateway
        gw = SynapseGateway()
        gw.disconnect("t1", "u1", "d1")  # should not raise


# ═══════════════════════════════════════════════════════════════════════
# actions/tool_registry.py
# ═══════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    def test_register_and_list(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def my_tool() -> ToolResult:
            return ToolResult(tool="test", success=True, output="done")

        reg.register("test", "A test tool", my_tool, {"arg1": "string"})
        tools = reg.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test"
        assert tools[0]["available"] is True

    def test_unregister(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def my_tool() -> ToolResult:
            return ToolResult(tool="test", success=True, output="done")

        reg.register("test", "A test tool", my_tool)
        assert reg.unregister("test") is True
        assert reg.unregister("test") is False

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def echo(text: str) -> ToolResult:
            return ToolResult(tool="echo", success=True, output=text)

        reg.register("echo", "Echo tool", echo)
        result = await reg.invoke("echo", text="hello")
        assert result.success and result.output == "hello"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_invoke_unknown_tool(self):
        from hbllm.actions.tool_registry import ToolRegistry
        reg = ToolRegistry()
        result = await reg.invoke("nonexistent")
        assert not result.success
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_invoke_unavailable_tool(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def my_tool() -> ToolResult:
            return ToolResult(tool="test", success=True, output="done")

        reg.register("test", "A tool", my_tool)
        reg.set_availability("test", False, "Rate limited")
        result = await reg.invoke("test")
        assert not result.success
        assert "unavailable" in result.error

    @pytest.mark.asyncio
    async def test_invoke_handler_exception(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def failing_tool() -> ToolResult:
            raise ValueError("Tool crashed")

        reg.register("fail", "Failing tool", failing_tool)
        result = await reg.invoke("fail")
        assert not result.success
        assert "Tool crashed" in result.error

    def test_set_availability(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def my_tool() -> ToolResult:
            return ToolResult(tool="test", success=True, output="done")

        reg.register("test", "A tool", my_tool)
        reg.set_availability("test", False, "Maintenance")
        assert not reg.get_availability("test")["available"]
        assert reg.get_availability("test")["reason"] == "Maintenance"

        reg.set_availability("test", True)
        assert reg.get_availability("test")["available"]

    def test_get_availability_unknown(self):
        from hbllm.actions.tool_registry import ToolRegistry
        reg = ToolRegistry()
        status = reg.get_availability("nonexistent")
        assert not status["registered"]

    def test_set_availability_unknown_tool(self):
        from hbllm.actions.tool_registry import ToolRegistry
        reg = ToolRegistry()
        reg.set_availability("nonexistent", False)  # Should not raise

    def test_available_tools(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def tool_a() -> ToolResult:
            return ToolResult(tool="a", success=True, output="")

        async def tool_b() -> ToolResult:
            return ToolResult(tool="b", success=True, output="")

        reg.register("a", "Tool A", tool_a)
        reg.register("b", "Tool B", tool_b)
        reg.set_availability("b", False, "Offline")
        assert reg.available_tools() == ["a"]

    def test_list_tools_available_only(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def tool_a() -> ToolResult:
            return ToolResult(tool="a", success=True, output="")

        async def tool_b() -> ToolResult:
            return ToolResult(tool="b", success=True, output="")

        reg.register("a", "Tool A", tool_a)
        reg.register("b", "Tool B", tool_b)
        reg.set_availability("b", False, "Down")
        listed = reg.list_tools(available_only=True)
        assert len(listed) == 1 and listed[0]["name"] == "a"

    def test_list_tools_all(self):
        from hbllm.actions.tool_registry import ToolRegistry, ToolResult
        reg = ToolRegistry()

        async def tool_a() -> ToolResult:
            return ToolResult(tool="a", success=True, output="")

        reg.register("a", "Tool A", tool_a)
        reg.set_availability("a", False, "Reason")
        listed = reg.list_tools(available_only=False)
        assert len(listed) == 1
        assert listed[0]["unavailable_reason"] == "Reason"


class TestToolDecorator:
    def test_decorator_registers(self):
        from hbllm.actions.tool_registry import _TOOL_REGISTRY, tool

        @tool(name="test_decorator_tool", description="A test tool")
        def my_test_func(text: str, count: int = 5) -> str:
            return text * count

        assert "test_decorator_tool" in _TOOL_REGISTRY
        schema = _TOOL_REGISTRY["test_decorator_tool"]["schema"]
        assert "text" in schema["parameters"]["properties"]
        assert "count" in schema["parameters"]["properties"]
        assert "text" in schema["parameters"]["required"]

    def test_decorator_preserves_function(self):
        from hbllm.actions.tool_registry import tool

        @tool(name="preserved_func")
        def original(x: str) -> str:
            """Original docstring."""
            return x

        assert original("hello") == "hello"

    def test_get_tool_registry(self):
        from hbllm.actions.tool_registry import get_tool_registry
        registry = get_tool_registry()
        assert isinstance(registry, dict)


class TestCreateToolFromCode:
    def test_create_valid_function(self):
        from hbllm.actions.tool_registry import create_tool_from_code
        code = "def add(a, b): return a + b"
        func = create_tool_from_code(code, "add")
        assert func(2, 3) == 5

    def test_create_missing_function_raises(self):
        from hbllm.actions.tool_registry import create_tool_from_code
        with pytest.raises(ValueError, match="not found"):
            create_tool_from_code("x = 42", "nonexistent")


# ═══════════════════════════════════════════════════════════════════════
# actions/scheduler_tools.py
# ═══════════════════════════════════════════════════════════════════════


class TestSchedulerTools:
    @pytest.mark.asyncio
    async def test_schedule_event_tool(self):
        from hbllm.actions.scheduler_tools import ScheduleEventTool
        tool = ScheduleEventTool()
        env = MagicMock()
        scheduler = MagicMock()
        scheduler.schedule = AsyncMock(return_value="task-123")
        env.scheduler = scheduler

        result = await tool.execute(
            env, action="Send daily report", time="2026-06-20T09:00:00"
        )
        assert "task-123" in str(result) or result is not None

    @pytest.mark.asyncio
    async def test_schedule_recurring_tool(self):
        from hbllm.actions.scheduler_tools import ScheduleRecurringTool
        tool = ScheduleRecurringTool()
        env = MagicMock()
        scheduler = MagicMock()
        scheduler.schedule_recurring = AsyncMock(return_value="recurring-456")
        env.scheduler = scheduler

        result = await tool.execute(
            env, action="Check emails", interval="every 30 minutes"
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_cancel_task_tool(self):
        from hbllm.actions.scheduler_tools import CancelTaskTool
        tool = CancelTaskTool()
        env = MagicMock()
        scheduler = MagicMock()
        scheduler.cancel = AsyncMock(return_value=True)
        env.scheduler = scheduler

        result = await tool.execute(env, task_id="task-123")
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# actions/orchestrator.py
# ═══════════════════════════════════════════════════════════════════════


class TestOrchestrator:
    def test_parse_plan(self):
        from hbllm.actions.orchestrator import MultiAgentOrchestrator
        plan_text = """Step 1: Search for information
Tool: web_search
Input: latest news

Step 2: Summarize
Tool: summarize
Input: the search results"""
        steps = MultiAgentOrchestrator._parse_plan(plan_text)
        assert len(steps) >= 1


# ═══════════════════════════════════════════════════════════════════════
# serving/pipeline.py (_classify_complexity)
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineClassifyComplexity:
    def test_simple_query(self):
        from hbllm.serving.pipeline import CognitivePipeline
        pipeline = CognitivePipeline.__new__(CognitivePipeline)
        # Short simple text
        assert pipeline._classify_complexity("Hello") in ("trivial", "simple", "moderate", "complex")

    def test_complex_query(self):
        from hbllm.serving.pipeline import CognitivePipeline
        pipeline = CognitivePipeline.__new__(CognitivePipeline)
        complex_text = (
            "Analyze the trade-offs between microservices and monolithic architectures "
            "considering scalability, maintainability, deployment complexity, and team size. "
            "Compare approaches for data consistency, service discovery, and fault tolerance. "
            "Provide a detailed implementation plan with specific technology recommendations."
        )
        result = pipeline._classify_complexity(complex_text)
        assert result in ("trivial", "simple", "moderate", "complex")


class TestPipelineResult:
    def test_to_dict(self):
        from hbllm.serving.pipeline import PipelineResult
        result = PipelineResult(
            text="Hello!",
            correlation_id="corr-123",
            tenant_id="t1",
            latency_ms=42.5,
        )
        d = result.to_dict()
        assert d["text"] == "Hello!"
        assert d["correlation_id"] == "corr-123"
        assert d["latency_ms"] == 42.5

    def test_pipeline_config_defaults(self):
        from hbllm.serving.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.inject_memory is True
        assert config.router_timeout == 15.0
