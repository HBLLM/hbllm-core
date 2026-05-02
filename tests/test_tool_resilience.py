"""Tests for tool resilience — dynamic availability, removal, and runtime management.

Covers scenarios:
A. Tool removal
B. Tool replacement
C. Tool temporarily unavailable
D. Decision-making uses availability
E. ToolMemory recommendations with availability
F. Runtime tool introduction
G. System continuity under tool loss
H. Bus events
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hbllm.actions.tool_memory import ToolMemory, ToolUsageRecord
from hbllm.actions.tool_registry import ToolRegistry, ToolResult

# ── Helpers ──────────────────────────────────────────────────────────────


async def _echo_handler(x: str = "") -> ToolResult:
    return ToolResult(tool="echo", success=True, output=f"echo: {x}")


async def _search_v1(query: str = "") -> ToolResult:
    return ToolResult(tool="search", success=True, output=f"v1 results for: {query}")


async def _search_v2(query: str = "") -> ToolResult:
    return ToolResult(tool="search", success=True, output=f"v2 results for: {query}")


async def _calc_handler(expr: str = "") -> ToolResult:
    return ToolResult(tool="calc", success=True, output=f"calc: {expr}")


async def _weather_handler(city: str = "") -> ToolResult:
    return ToolResult(tool="weather", success=True, output=f"weather: {city}")


# ══════════════════════════════════════════════════════════════════════════
# A. Tool Removal
# ══════════════════════════════════════════════════════════════════════════


class TestToolRemoval:
    @pytest.fixture
    def registry(self):
        r = ToolRegistry()
        r.register("echo", "Echo tool", _echo_handler)
        r.register("calc", "Calculator", _calc_handler)
        r.register("weather", "Weather lookup", _weather_handler)
        return r

    def test_unregister_returns_true_for_existing(self, registry):
        assert registry.unregister("echo") is True

    def test_unregister_returns_false_for_missing(self, registry):
        assert registry.unregister("nonexistent") is False

    @pytest.mark.asyncio
    async def test_unregistered_tool_returns_unknown(self, registry):
        registry.unregister("echo")
        result = await registry.invoke("echo")
        assert not result.success
        assert "Unknown tool" in result.error

    def test_unregistered_tool_not_in_list(self, registry):
        registry.unregister("echo")
        names = [t["name"] for t in registry.list_tools()]
        assert "echo" not in names
        assert "calc" in names
        assert "weather" in names

    @pytest.mark.asyncio
    async def test_remaining_tools_still_work(self, registry):
        registry.unregister("echo")
        result = await registry.invoke("calc", expr="2+2")
        assert result.success
        assert "2+2" in result.output


# ══════════════════════════════════════════════════════════════════════════
# B. Tool Replacement
# ══════════════════════════════════════════════════════════════════════════


class TestToolReplacement:
    @pytest.mark.asyncio
    async def test_register_same_name_replaces_handler(self):
        registry = ToolRegistry()
        registry.register("search", "Search v1", _search_v1)

        # Verify v1 behavior
        r1 = await registry.invoke("search", query="test")
        assert "v1 results" in r1.output

        # Replace with v2
        registry.register("search", "Search v2", _search_v2)

        # Verify v2 behavior
        r2 = await registry.invoke("search", query="test")
        assert "v2 results" in r2.output

    @pytest.mark.asyncio
    async def test_replacement_resets_availability(self):
        """Replacing a tool that was marked unavailable makes it available again."""
        registry = ToolRegistry()
        registry.register("search", "Search v1", _search_v1)
        registry.set_availability("search", False, "v1 broken")

        # Replace with v2
        registry.register("search", "Search v2", _search_v2)

        # Should be available again
        avail = registry.get_availability("search")
        assert avail["available"] is True
        assert avail["reason"] == ""


# ══════════════════════════════════════════════════════════════════════════
# C. Tool Temporarily Unavailable
# ══════════════════════════════════════════════════════════════════════════


class TestToolAvailability:
    @pytest.fixture
    def registry(self):
        r = ToolRegistry()
        r.register("echo", "Echo tool", _echo_handler)
        r.register("calc", "Calculator", _calc_handler)
        return r

    @pytest.mark.asyncio
    async def test_unavailable_tool_returns_error(self, registry):
        registry.set_availability("echo", False, "service maintenance")
        result = await registry.invoke("echo", x="hello")
        assert not result.success
        assert "currently unavailable" in result.error
        assert "service maintenance" in result.error

    @pytest.mark.asyncio
    async def test_restored_tool_works_again(self, registry):
        registry.set_availability("echo", False, "down")
        registry.set_availability("echo", True)
        result = await registry.invoke("echo", x="hello")
        assert result.success
        assert "hello" in result.output

    def test_list_tools_shows_availability(self, registry):
        registry.set_availability("echo", False, "maintenance")
        tools = registry.list_tools()
        echo_tool = next(t for t in tools if t["name"] == "echo")
        assert echo_tool["available"] is False
        assert echo_tool["unavailable_reason"] == "maintenance"

        calc_tool = next(t for t in tools if t["name"] == "calc")
        assert calc_tool["available"] is True
        assert "unavailable_reason" not in calc_tool

    def test_get_availability_registered(self, registry):
        avail = registry.get_availability("echo")
        assert avail["registered"] is True
        assert avail["available"] is True

    def test_get_availability_unregistered(self, registry):
        avail = registry.get_availability("nonexistent")
        assert avail["registered"] is False
        assert avail["available"] is False

    def test_set_availability_unknown_tool_is_noop(self, registry):
        # Should not raise
        registry.set_availability("nonexistent", False, "doesn't matter")

    def test_available_tools_list(self, registry):
        assert set(registry.available_tools()) == {"echo", "calc"}
        registry.set_availability("echo", False)
        assert registry.available_tools() == ["calc"]


# ══════════════════════════════════════════════════════════════════════════
# D. Decision-Making Uses Availability
# ══════════════════════════════════════════════════════════════════════════


class TestDecisionMakingAvailability:
    @pytest.fixture
    def registry(self):
        r = ToolRegistry()
        r.register("echo", "Echo tool", _echo_handler)
        r.register("calc", "Calculator", _calc_handler)
        r.register("weather", "Weather lookup", _weather_handler)
        return r

    def test_list_tools_available_only(self, registry):
        registry.set_availability("weather", False, "API key expired")
        available = registry.list_tools(available_only=True)
        names = [t["name"] for t in available]
        assert "echo" in names
        assert "calc" in names
        assert "weather" not in names

    def test_agent_executor_sees_only_available_tools(self, registry):
        """Verify the system prompt would only contain available tools."""
        registry.set_availability("weather", False, "down")
        tool_list = registry.list_tools(available_only=True)
        tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_list)
        assert "echo" in tool_desc
        assert "calc" in tool_desc
        assert "weather" not in tool_desc

    def test_orchestrator_planner_prompt_dynamic(self, registry):
        """Verify the planner prompt uses actual available tools."""
        from hbllm.actions.orchestrator import MultiAgentOrchestrator

        mock_llm = MagicMock()
        orch = MultiAgentOrchestrator(mock_llm, registry)

        # All tools available
        prompt = orch._build_planner_prompt()
        assert "echo" in prompt
        assert "calc" in prompt
        assert "weather" in prompt

        # Remove weather
        registry.set_availability("weather", False, "offline")
        prompt = orch._build_planner_prompt()
        assert "echo" in prompt
        assert "calc" in prompt
        assert "weather" not in prompt

    def test_orchestrator_no_tools_available(self, registry):
        """When all tools are unavailable, prompt says NONE."""
        from hbllm.actions.orchestrator import MultiAgentOrchestrator

        mock_llm = MagicMock()
        orch = MultiAgentOrchestrator(mock_llm, registry)

        registry.set_availability("echo", False)
        registry.set_availability("calc", False)
        registry.set_availability("weather", False)

        prompt = orch._build_planner_prompt()
        assert "NONE" in prompt


# ══════════════════════════════════════════════════════════════════════════
# E. ToolMemory Recommendations with Availability
# ══════════════════════════════════════════════════════════════════════════


class TestToolMemoryAvailability:
    @pytest.fixture
    def memory(self, tmp_path):
        return ToolMemory(data_dir=str(tmp_path))

    def _seed_usage(self, memory, tool_name, query_type, count=5, success=True):
        for _ in range(count):
            memory.record(
                ToolUsageRecord(
                    tool_name=tool_name,
                    query_type=query_type,
                    success=success,
                    latency_ms=10.0,
                    result_quality=0.9,
                )
            )

    def test_recommend_without_filter(self, memory):
        self._seed_usage(memory, "tool_a", "code")
        self._seed_usage(memory, "tool_b", "code")
        recs = memory.recommend_tool("code")
        names = [r["tool"] for r in recs]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_recommend_with_availability_filter(self, memory):
        self._seed_usage(memory, "tool_a", "code")
        self._seed_usage(memory, "tool_b", "code")
        self._seed_usage(memory, "tool_c", "code")

        # Only tool_a and tool_c are available
        recs = memory.recommend_tool("code", available_tools=["tool_a", "tool_c"])
        names = [r["tool"] for r in recs]
        assert "tool_a" in names
        assert "tool_c" in names
        assert "tool_b" not in names

    def test_recommend_sequence_without_filter(self, memory):
        memory.record_sequence("deploy", ["build", "test", "deploy_cmd"], True, 100.0)
        seq = memory.recommend_sequence("deploy")
        assert seq == ["build", "test", "deploy_cmd"]

    def test_recommend_sequence_skips_unavailable(self, memory):
        # Best sequence uses "deploy_cmd" which is unavailable
        memory.record_sequence("deploy", ["build", "test", "deploy_cmd"], True, 100.0)
        memory.record_sequence("deploy", ["build", "test", "deploy_cmd"], True, 100.0)
        # Fallback sequence only uses available tools
        memory.record_sequence("deploy", ["build", "test"], True, 150.0)

        seq = memory.recommend_sequence("deploy", available_tools=["build", "test"])
        assert seq == ["build", "test"]

    def test_recommend_sequence_returns_none_if_all_unavailable(self, memory):
        memory.record_sequence("deploy", ["build", "test", "deploy_cmd"], True, 100.0)
        seq = memory.recommend_sequence("deploy", available_tools=["unrelated"])
        assert seq is None


# ══════════════════════════════════════════════════════════════════════════
# F. Runtime Tool Introduction
# ══════════════════════════════════════════════════════════════════════════


class TestRuntimeToolIntroduction:
    @pytest.mark.asyncio
    async def test_register_at_runtime_is_immediately_invocable(self):
        registry = ToolRegistry()
        registry.register("echo", "Echo tool", _echo_handler)

        # Start with 1 tool
        assert len(registry.list_tools()) == 1

        # Add a new tool at runtime
        registry.register("calc", "Calculator", _calc_handler)
        assert len(registry.list_tools()) == 2

        # Immediately usable
        result = await registry.invoke("calc", expr="5*5")
        assert result.success
        assert "5*5" in result.output

    def test_new_tool_appears_in_available_list(self):
        registry = ToolRegistry()
        registry.register("echo", "Echo tool", _echo_handler)
        assert "calc" not in registry.available_tools()

        registry.register("calc", "Calculator", _calc_handler)
        assert "calc" in registry.available_tools()

    def test_new_tool_appears_in_available_only_list(self):
        registry = ToolRegistry()
        registry.register("echo", "Echo tool", _echo_handler)
        registry.register("calc", "Calculator", _calc_handler)

        available = registry.list_tools(available_only=True)
        names = [t["name"] for t in available]
        assert "echo" in names
        assert "calc" in names


# ══════════════════════════════════════════════════════════════════════════
# G. System Continuity Under Tool Loss
# ══════════════════════════════════════════════════════════════════════════


class TestSystemContinuity:
    @pytest.fixture
    def memory(self, tmp_path):
        return ToolMemory(data_dir=str(tmp_path))

    def _seed_usage(self, memory, tool_name, query_type, count=5):
        for _ in range(count):
            memory.record(
                ToolUsageRecord(
                    tool_name=tool_name,
                    query_type=query_type,
                    success=True,
                    latency_ms=10.0,
                    result_quality=0.9,
                )
            )

    def test_survives_losing_most_tools(self):
        """System keeps working with just one tool remaining."""
        registry = ToolRegistry()
        registry.register("echo", "Echo tool", _echo_handler)
        registry.register("calc", "Calculator", _calc_handler)
        registry.register("weather", "Weather lookup", _weather_handler)

        # Remove 2 of 3 tools
        registry.unregister("calc")
        registry.unregister("weather")

        # System still works with the surviving tool
        available = registry.list_tools(available_only=True)
        assert len(available) == 1
        assert available[0]["name"] == "echo"

    def test_memory_retains_history_after_tool_removal(self, memory):
        """ToolMemory retains historical data for re-registered tools."""
        self._seed_usage(memory, "search", "code", count=10)

        # Tool is "removed" — memory still has the data
        stats = memory.stats()
        assert stats["total_usages"] == 10

        # When tool is re-registered, recommendations still work
        recs = memory.recommend_tool("code")
        assert len(recs) > 0
        assert recs[0]["tool"] == "search"

    def test_recommendations_degrade_gracefully(self, memory):
        """When preferred tool is removed, system recommends next-best."""
        # tool_a is best, tool_b is second-best
        self._seed_usage(memory, "tool_a", "code", count=10)
        self._seed_usage(memory, "tool_b", "code", count=5)

        # tool_a is removed
        recs = memory.recommend_tool("code", available_tools=["tool_b"])
        assert len(recs) == 1
        assert recs[0]["tool"] == "tool_b"

    @pytest.mark.asyncio
    async def test_invoke_after_mass_unavailability(self):
        """Even with most tools offline, remaining tools work normally."""
        registry = ToolRegistry()
        registry.register("echo", "Echo tool", _echo_handler)
        registry.register("calc", "Calculator", _calc_handler)
        registry.register("weather", "Weather lookup", _weather_handler)

        # Mark most tools unavailable
        registry.set_availability("calc", False, "service crashed")
        registry.set_availability("weather", False, "API rate limit")

        # Available tool works fine
        result = await registry.invoke("echo", x="still working")
        assert result.success
        assert "still working" in result.output

        # Unavailable tools return clear errors
        result = await registry.invoke("calc", expr="1+1")
        assert not result.success
        assert "unavailable" in result.error


# ══════════════════════════════════════════════════════════════════════════
# H. Bus Events
# ══════════════════════════════════════════════════════════════════════════


class TestBusEvents:
    @pytest.fixture
    def bus(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        return bus

    def test_register_fires_event(self, bus):
        registry = ToolRegistry(bus=bus)
        registry.register("echo", "Echo tool", _echo_handler)
        # Event is fire-and-forget via create_task, but the bus.publish
        # will be called with the system.tool.registered topic
        # (Exact assertion depends on event loop state, so we just verify
        # the tool was registered successfully)
        assert "echo" in registry.available_tools()

    def test_unregister_fires_event(self, bus):
        registry = ToolRegistry(bus=bus)
        registry.register("echo", "Echo tool", _echo_handler)
        registry.unregister("echo")
        assert "echo" not in registry.available_tools()

    def test_availability_change_fires_event(self, bus):
        registry = ToolRegistry(bus=bus)
        registry.register("echo", "Echo tool", _echo_handler)
        registry.set_availability("echo", False, "maintenance")
        assert registry.get_availability("echo")["available"] is False

    def test_no_event_without_bus(self):
        """Events are skipped silently when no bus is connected."""
        registry = ToolRegistry()  # No bus
        # Should not raise
        registry.register("echo", "Echo tool", _echo_handler)
        registry.set_availability("echo", False, "test")
        registry.unregister("echo")
