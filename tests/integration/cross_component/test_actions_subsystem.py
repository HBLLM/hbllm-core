"""Integration tests for Actions subsystem — MultiAgentOrchestrator, ToolRegistry."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hbllm.actions.agent_executor import AgentResponse, AgentStep, ConfidenceScorer
from hbllm.actions.orchestrator import MultiAgentOrchestrator
from hbllm.actions.tool_registry import ToolRegistry, ToolResult


# ── Mock LLM ─────────────────────────────────────────────────────────────────


class MockLLM:
    """Mock LLM that returns structured responses for orchestrator phases."""

    def __init__(self):
        self.call_count = 0
        self.messages_received = []

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        self.messages_received.append(messages)

        user_msg = messages[-1]["content"].lower() if messages else ""

        # Planning phase — return a structured plan
        if "task planner" in messages[0].get("content", "").lower():
            return (
                "STEP 1: Search for information\n"
                "TOOL: search\n"
                "INPUT: Python data structures\n"
                "\n"
                "STEP 2: Summarize findings\n"
                "TOOL: NONE\n"
                "INPUT: N/A\n"
            )

        # Review phase
        if "code reviewer" in messages[0].get("content", "").lower():
            return "Task completed successfully. The information was found and summarized."

        # Final synthesis
        return "Here is your answer: Python supports lists, dicts, sets, and tuples."


class SimpleMockLLM:
    """Minimal mock with generate method instead of chat."""

    async def generate(self, prompt: str) -> str:
        return "Generated response"


# ── ToolRegistry Integration Tests ───────────────────────────────────────────


class TestToolRegistryIntegration:
    """Test tool registration, invocation, and lifecycle."""

    def test_register_and_list(self):
        registry = ToolRegistry()

        async def search_tool(query: str) -> ToolResult:
            return ToolResult(tool="search", success=True, output=f"Results for: {query}")

        registry.register("search", "Search for information", search_tool)

        tools = registry.list_tools()
        assert len(tools) >= 1
        assert any(t["name"] == "search" for t in tools)

    @pytest.mark.asyncio
    async def test_invoke_tool(self):
        registry = ToolRegistry()

        async def calculator(expression: str) -> ToolResult:
            return ToolResult(tool="calculator", success=True, output=str(eval(expression)))

        registry.register("calculator", "Calculate", calculator)

        result = await registry.invoke("calculator", expression="2 + 3")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "5"

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_tool(self):
        registry = ToolRegistry()
        result = await registry.invoke("nonexistent", query="test")
        assert result.success is False
        assert "unknown" in result.error.lower() or "not found" in result.error.lower()

    def test_unregister(self):
        registry = ToolRegistry()

        async def dummy(x: str) -> ToolResult:
            return ToolResult(tool="temp", success=True, output=x)

        registry.register("temp", "Temp tool", dummy)
        assert any(t["name"] == "temp" for t in registry.list_tools())

        registry.unregister("temp")
        assert not any(t["name"] == "temp" for t in registry.list_tools())

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        registry = ToolRegistry()

        async def failing_tool(**kwargs) -> ToolResult:
            raise ValueError("Tool failed!")

        registry.register("fail", "Fail", failing_tool)
        result = await registry.invoke("fail", input="test")
        assert result.success is False
        assert "Tool failed" in (result.error or "")

    def test_get_availability(self):
        registry = ToolRegistry()

        async def tool(**kwargs) -> ToolResult:
            return ToolResult(tool="avail", success=True, output="ok")

        registry.register("avail", "Available tool", tool)
        status = registry.get_availability("avail")
        assert status.get("available", False) is True or status.get("registered", False) is True

    def test_available_only_filter(self):
        registry = ToolRegistry()

        async def tool(**kwargs) -> ToolResult:
            return ToolResult(tool="active", success=True, output="ok")

        registry.register("active", "Active", tool)
        tools = registry.list_tools(available_only=True)
        assert len(tools) >= 1


# ── MultiAgentOrchestrator Integration Tests ─────────────────────────────────


class TestMultiAgentOrchestratorIntegration:
    """Test the Planner → Executor → Reviewer pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_full_pipeline_execution(self):
        llm = MockLLM()
        registry = ToolRegistry()

        async def search(query: str = "", **kwargs) -> ToolResult:
            return ToolResult(tool="search", success=True, output=f"Search results for: {query}")

        registry.register("search", "Search", search)


        orchestrator = MultiAgentOrchestrator(llm=llm, tool_registry=registry)
        response = await orchestrator.execute("Find information about Python data structures")

        assert isinstance(response, AgentResponse)
        assert response.content != ""
        assert response.multi_agent is True
        assert len(response.steps) >= 3  # Plan + execution + review + final
        assert response.confidence >= 0.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_pipeline_without_tools(self):
        llm = MockLLM()
        registry = ToolRegistry()  # Empty registry

        orchestrator = MultiAgentOrchestrator(llm=llm, tool_registry=registry)
        response = await orchestrator.execute("Explain quantum computing")

        assert isinstance(response, AgentResponse)
        assert response.content != ""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_pipeline_with_kb_context(self):
        llm = MockLLM()
        registry = ToolRegistry()

        orchestrator = MultiAgentOrchestrator(llm=llm, tool_registry=registry)
        response = await orchestrator.execute(
            "What is HBLLM?",
            kb_context="HBLLM is a Human Brain-Like Language Model.",
        )

        assert isinstance(response, AgentResponse)
        assert response.confidence >= 0.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_pipeline_with_generate_llm(self):
        """Test orchestrator works with generate-based LLM instead of chat."""
        llm = SimpleMockLLM()
        registry = ToolRegistry()

        orchestrator = MultiAgentOrchestrator(llm=llm, tool_registry=registry)
        response = await orchestrator.execute("Simple query")

        assert isinstance(response, AgentResponse)


# ── Plan Parsing Tests ───────────────────────────────────────────────────────


class TestPlanParsing:
    """Test the static plan parsing method."""

    def test_parse_well_formed_plan(self):
        plan = (
            "STEP 1: Search for data\n"
            "TOOL: search\n"
            "INPUT: Python lists\n\n"
            "STEP 2: Format output\n"
            "TOOL: NONE\n"
            "INPUT: N/A\n"
        )
        steps = MultiAgentOrchestrator._parse_plan(plan)
        assert len(steps) == 2
        assert steps[0]["description"] == "Search for data"
        assert steps[0]["tool"] == "search"
        assert steps[0]["input"] == "Python lists"
        assert steps[1]["tool"] == "NONE"

    def test_parse_max_5_steps(self):
        plan = "\n".join(f"STEP {i}: Task {i}\nTOOL: NONE\nINPUT: n/a" for i in range(1, 10))
        steps = MultiAgentOrchestrator._parse_plan(plan)
        assert len(steps) <= 5

    def test_parse_empty_plan(self):
        steps = MultiAgentOrchestrator._parse_plan("")
        assert len(steps) == 0

    def test_parse_malformed_plan(self):
        steps = MultiAgentOrchestrator._parse_plan("Just some random text without structure")
        assert len(steps) == 0


# ── Confidence Scorer Tests ──────────────────────────────────────────────────


class TestConfidenceScorer:
    """Test confidence scoring logic."""

    def test_score_with_context(self):
        score = ConfidenceScorer.score(
            query="What is Python?",
            response="Python is a programming language created by Guido van Rossum.",
            had_context=True,
        )
        assert "overall" in score
        assert "flags" in score
        assert 0.0 <= score["overall"] <= 1.0

    def test_score_without_context(self):
        score = ConfidenceScorer.score(
            query="Obscure question",
            response="I'm not sure about this topic.",
            had_context=False,
        )
        assert score["overall"] >= 0.0

    def test_score_empty_response(self):
        score = ConfidenceScorer.score(
            query="Test",
            response="",
            had_context=False,
        )
        assert score["overall"] >= 0.0
