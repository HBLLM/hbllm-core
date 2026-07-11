"""
Integration and unit tests for global tool and plugin execution in the HBLLM Chat API.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hbllm.brain.control.decision_node import DecisionNode
from hbllm.brain.core.factory import BrainConfig, BrainFactory
from hbllm.brain.core.prompt_helper import get_dynamic_system_prompt
from hbllm.brain.planning.action_planner import ActionPlanner
from hbllm.brain.planning.action_schema import ActionType
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeInfo, NodeType
from hbllm.serving.provider import LLMProvider, LLMResponse


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Bypass safety checks and signature verification in tests."""
    # Patch RouterNode.__init__ to disable vector routing
    import hbllm.brain.control.router_node

    original_init = hbllm.brain.control.router_node.RouterNode.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["use_vectors"] = False
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(hbllm.brain.control.router_node.RouterNode, "__init__", patched_init)

    # Patch ServiceRegistry.verify_message to bypass cryptographic validation in tests
    import hbllm.network.registry

    async def mock_verify_message(*args, **kwargs):
        return True

    monkeypatch.setattr(
        hbllm.network.registry.ServiceRegistry, "verify_message", mock_verify_message
    )

    # Patch WorkspaceNode.__init__ to set a short thinking deadline for fast testing
    import hbllm.brain.planning.workspace_node

    original_ws_init = hbllm.brain.planning.workspace_node.WorkspaceNode.__init__

    def patched_ws_init(self, *args, **kwargs):
        kwargs["thinking_deadline"] = 1.0
        original_ws_init(self, *args, **kwargs)

    monkeypatch.setattr(
        hbllm.brain.planning.workspace_node.WorkspaceNode, "__init__", patched_ws_init
    )


# ── ActionPlanner Tool Parsing Unit Tests ─────────────────────────────────────


class TestActionPlannerToolParsing:
    """Verify that ActionPlanner extracts XML and JSON tool calls to ActionType.MCP_TOOL."""

    def setup_method(self):
        self.planner = ActionPlanner()

    def test_parse_xml_tool_call(self):
        content = (
            "I need to query the database to get information.\n"
            '<tool_call name="db_query">\n'
            "{\n"
            '  "sql": "SELECT * FROM users;"\n'
            "}\n"
            "</tool_call>"
        )
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content=content,
            confidence=0.9,
            original_query={"text": "get all users"},
        )
        assert plan.action_type == ActionType.MCP_TOOL
        assert plan.metadata.get("tool_name") == "db_query"
        assert plan.metadata.get("arguments") == {"sql": "SELECT * FROM users;"}
        assert plan.metadata.get("is_direct_tool_call") is True

    def test_parse_json_markdown_tool_call(self):
        content = (
            "Here is the tool execution block:\n"
            "```json\n"
            "{\n"
            '  "tool_call": "weather_lookup",\n'
            '  "arguments": {"location": "Tokyo"}\n'
            "}\n"
            "```"
        )
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content=content,
            confidence=0.9,
            original_query={"text": "weather in Tokyo"},
        )
        assert plan.action_type == ActionType.MCP_TOOL
        assert plan.metadata.get("tool_name") == "weather_lookup"
        assert plan.metadata.get("arguments") == {"location": "Tokyo"}
        assert plan.metadata.get("is_direct_tool_call") is True


# ── Dynamic System Prompt Injection Unit Tests ──────────────────────────────


@pytest.mark.asyncio
async def test_dynamic_system_prompt_tool_listing():
    """Verify that get_dynamic_system_prompt discovers and lists custom toolbox nodes."""
    bus = InProcessBus()
    await bus.start()

    # We register a mock ServiceRegistry discover handler
    async def mock_discover_handler(message: Message) -> Message | None:
        nodes = [
            # System node (should be ignored)
            NodeInfo(
                node_id="router",
                node_type=NodeType.ROUTER,
                capabilities=["routing"],
                description="Core router",
            ).model_dump(),
            # Plugin/Tool Node
            NodeInfo(
                node_id="weather-plugin",
                node_type=NodeType.ACTION,
                capabilities=["weather_lookup", "geocoding"],
                description="Real-time weather data API",
            ).model_dump(),
            # MCP server Node
            NodeInfo(
                node_id="mcp_db_tools",
                node_type=NodeType.META,
                capabilities=["mcp.query", "mcp.list_tables"],
                description="Database access tools via MCP",
            ).model_dump(),
        ]
        return message.create_response({"nodes": nodes})

    await bus.subscribe("registry.discover", mock_discover_handler)

    try:
        prompt = await get_dynamic_system_prompt(bus, "tenant_test", "test_node")

        # Verify weather-plugin and mcp_db_tools are listed
        assert "weather-plugin" in prompt
        assert "Real-time weather data API" in prompt
        assert "weather_lookup, geocoding" in prompt

        assert "mcp_db_tools" in prompt
        assert "Database access tools via MCP" in prompt
        assert "mcp.query, mcp.list_tables" in prompt

        # Verify router is ignored/skipped
        assert "Core router" not in prompt

        # Verify instructions are added
        assert '<tool_call name="tool_name">' in prompt

    finally:
        await bus.stop()


# ── DecisionNode Tool Execution Unit Test ─────────────────────────────────────


@pytest.mark.asyncio
async def test_decision_node_executes_mcp_tool():
    """Verify that DecisionNode dispatches tool execution to the unified topic and synthesizes results."""
    bus = InProcessBus()
    await bus.start()

    # Subscribe to unified tool topic
    mock_tool_called = asyncio.Event()

    async def handle_weather_lookup(message: Message) -> Message | None:
        mock_tool_called.set()
        return message.create_response({"status": "SUCCESS", "output": "It is sunny in Tokyo, 25C"})

    await bus.subscribe("action.tool.weather_lookup", handle_weather_lookup)

    # Capture sensory output
    outputs = []

    async def capture_output(msg: Message):
        outputs.append(msg)

    await bus.subscribe("sensory.output", capture_output)

    # Setup mock LLM for synthesis and safety classification
    class MockSynthesisLLM:
        async def generate(self, prompt, **kwargs):
            return "Synthesized: The weather in Tokyo is sunny and 25C."

        async def generate_json(self, prompt, **kwargs):
            return {"safe": True}

    node = DecisionNode(node_id="decision_test", llm=MockSynthesisLLM())
    node.utility_engine.weight_risk = 0.0
    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    await node.start(bus)

    try:
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="workspace_01",
            topic="decision.evaluate",
            payload={
                "original_query": {"intent": "answer", "text": "what is the weather in Tokyo?"},
                "selected_thought": {
                    "type": "intuition_general",
                    "confidence": 0.9,
                    "content": (
                        "Checking weather:\n"
                        '<tool_call name="weather_lookup">\n'
                        "{\n"
                        '  "location": "Tokyo"\n'
                        "}\n"
                        "</tool_call>"
                    ),
                },
            },
        )

        await node.evaluate_workspace_decision(msg)
        await asyncio.sleep(0.5)

        assert mock_tool_called.is_set()
        assert len(outputs) == 1
        assert "Synthesized" in outputs[0].payload["text"]
        assert outputs[0].payload["source"] == "tool_execution"
    finally:
        await node.stop()
        await bus.stop()


# ── End-to-End Brain Loop Test for Global Tool/Plugin Calls ──────────────────


class ToolCallMockProvider(LLMProvider):
    """Mock LLM Provider that outputs router classification and synthesizes final output."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        user_msg = messages[-1]["content"] if messages else ""
        user_msg_lower = user_msg.lower()
        print(f"\n--- [ToolCallMockProvider.generate] ---\nUser msg: {user_msg}\n")

        # Synthesis path
        if "present the results" in user_msg_lower or "tool output" in user_msg_lower:
            content = "The current weather in Tokyo is sunny and 25C."
        # Router intent classification
        elif "intent classifier" in user_msg_lower or "classify" in user_msg_lower:
            content = '{"domain": "general", "intent": "general_knowledge", "confidence": 0.9}'
        # Critic evaluation
        elif "evaluator" in user_msg_lower or "verdict" in user_msg_lower:
            content = '{"verdict": "PASS", "reason": "Response is safe"}'
        # Decision safety
        elif "safety classifier" in user_msg_lower:
            content = '{"safe": true, "reason": "Content is safe"}'
        else:
            content = "Unhandled mock path"

        return LLMResponse(
            content=content,
            model="mock-tool-brain",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )

    async def stream(self, messages, max_tokens=1024, temperature=0.7, **kwargs):
        response = await self.generate(messages, max_tokens, temperature, **kwargs)
        yield response.content

    @property
    def name(self) -> str:
        return "mock-tool-brain"


class MockDomainNode(Node):
    """Mock domain node that reacts to workspace updates and posts a tool call thought."""

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.DETECTOR)

    async def on_start(self):
        await self.bus.subscribe("workspace.update", self.handle_update)

    async def on_stop(self):
        pass

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_update(self, message: Message) -> None:
        print(
            f"\n--- [MockDomainNode.handle_update] received message on {message.topic}: {message.payload} ---\n"
        )
        thought_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.thought",
            payload={
                "type": "intuition_general",
                "confidence": 0.95,
                "content": (
                    "I will check the weather using weather_lookup:\n"
                    '<tool_call name="weather_lookup">\n'
                    "{\n"
                    '  "location": "Tokyo"\n'
                    "}\n"
                    "</tool_call>"
                ),
            },
            correlation_id=message.correlation_id or message.id,
        )
        await self.bus.publish("workspace.thought", thought_msg)


def _test_config(tmp_path) -> BrainConfig:
    """Create a minimal test-safe BrainConfig."""
    return BrainConfig(
        data_dir=str(tmp_path),
        watch_plugins=False,
        inject_plugins=False,
        inject_awareness=False,
        inject_load_manager=False,
        inject_scheduler=False,
        inject_knowledge=False,
        inject_persistence=False,
        inject_embodiment=False,
        inject_human_control=False,
        inject_causal_graph=False,
        inject_compaction=False,
        inject_task_graph=False,
        inject_mesh=False,
        inject_shell=False,
        total_timeout=5.0,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_e2e_global_tool_execution_flow(tmp_path):
    """Verify that E2E flow triggers mock domain thought generation, unified tool execution, and synthesis."""
    provider = ToolCallMockProvider()
    brain = await BrainFactory.create(
        provider=provider,
        config=_test_config(tmp_path),
    )

    # Disable utility penalties to bypass policy routing constraint in E2E tool run
    for node in brain.nodes:
        if isinstance(node, DecisionNode):
            node.utility_engine.weight_risk = 0.0
            node.utility_engine.weight_token = 0.0
            node.utility_engine.weight_latency = 0.0
        elif hasattr(node, "decision") and node.decision is not None:
            node.decision.utility_engine.weight_risk = 0.0
            node.decision.utility_engine.weight_token = 0.0
            node.decision.utility_engine.weight_latency = 0.0

    # Register custom mock domain expert
    mock_domain = MockDomainNode(node_id="domain_weather_mock")
    await brain.registry.register(mock_domain.get_info())
    await mock_domain.start(brain.bus)

    # Register the mock tool `weather_lookup` on the bus
    mock_tool_called = asyncio.Event()

    async def handle_weather_lookup(message: Message) -> Message | None:
        print("\n--- [handle_weather_lookup] tool triggered! ---\n")
        mock_tool_called.set()
        return message.create_response({"status": "SUCCESS", "output": "It is sunny in Tokyo, 25C"})

    # Subscribe to the unified tool topic
    await brain.bus.subscribe("action.tool.weather_lookup", handle_weather_lookup)

    outputs = []
    correlation_id = "e2e_tool_test_corr"

    async def capture_output(msg: Message):
        if msg.correlation_id == correlation_id:
            outputs.append(msg)

    await brain.bus.subscribe("sensory.output", capture_output)

    try:
        # Publish query
        query = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id="default",
            session_id="session_tool",
            topic="router.query",
            payload={"text": "what is the weather in Tokyo?"},
            correlation_id=correlation_id,
        )

        await brain.bus.publish("router.query", query)

        # Allow time for routing, planning, execution, and synthesis
        await asyncio.sleep(4.0)

        assert mock_tool_called.is_set(), "The mock tool weather_lookup was never executed"
        assert len(outputs) >= 1, "No output was published to sensory.output"

        payload = outputs[0].payload
        assert "Tokyo" in payload["text"]
        assert "sunny and 25C" in payload["text"]
        assert payload["source"] == "tool_execution"

    finally:
        await mock_domain.stop()
        await brain.shutdown()
