"""Tests for core ToolRegistry, @tool decorator, and ToolNode."""

import asyncio

import pytest

from hbllm.actions.complexity import ComplexityDetector
from hbllm.actions.tool_registry import (
    _TOOL_REGISTRY,
    ToolNode,
    ToolRegistry,
    ToolResult,
    create_tool_from_code,
    get_tool_registry,
    tool,
)


class TestToolResult:
    def test_success_result(self):
        r = ToolResult(tool="test", success=True, output="done")
        assert r.success
        assert r.output == "done"
        assert r.error == ""

    def test_failure_result(self):
        r = ToolResult(tool="test", success=False, output="", error="boom")
        assert not r.success
        assert r.error == "boom"


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.mark.asyncio
    async def test_register_and_invoke(self, registry):
        async def my_handler(x: str = "") -> ToolResult:
            return ToolResult(tool="echo", success=True, output=f"echo: {x}")

        registry.register("echo", "Echo tool", my_handler)
        result = await registry.invoke("echo", x="hello")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_invoke_unknown_tool(self, registry):
        result = await registry.invoke("nonexistent")
        assert not result.success
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_invoke_error_handling(self, registry):
        async def bad_handler() -> ToolResult:
            raise ValueError("intentional error")

        registry.register("bad", "Bad tool", bad_handler)
        result = await registry.invoke("bad")
        assert not result.success
        assert "intentional error" in result.error

    def test_list_tools(self, registry):
        async def noop() -> ToolResult:
            return ToolResult(tool="noop", success=True, output="")

        registry.register("noop", "No-op tool", noop, parameters={"x": "string"})
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "noop"


class TestToolDecorator:
    def setup_method(self):
        """Clear the global registry before each test."""
        _TOOL_REGISTRY.clear()

    def test_decorator_registers(self):
        @tool(name="greet", description="Greet someone")
        def greet(name: str):
            return f"Hello, {name}!"

        registry = get_tool_registry()
        assert "greet" in registry
        assert registry["greet"]["schema"]["name"] == "greet"

    def test_decorator_extracts_params(self):
        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        schema = get_tool_registry()["add"]["schema"]
        assert "a" in schema["parameters"]["properties"]
        assert schema["parameters"]["properties"]["a"]["type"] == "integer"

    def test_decorator_uses_function_name(self):
        @tool()
        def my_func():
            pass

        assert "my_func" in get_tool_registry()

    def test_decorated_function_still_callable(self):
        @tool(name="square")
        def square(x: int) -> int:
            return x * x

        assert square(5) == 25


class TestCreateToolFromCode:
    def test_creates_function(self):
        code = """
def multiply(a, b):
    return a * b
"""
        func = create_tool_from_code(code, "multiply")
        assert func(3, 4) == 12

    def test_missing_function_raises(self):
        code = "x = 42"
        with pytest.raises(ValueError, match="not found"):
            create_tool_from_code(code, "missing_func")


class TestComplexityDetector:
    def test_simple_question(self):
        assert not ComplexityDetector.needs_multi_agent("What is Python?")

    def test_complex_task(self):
        msg = "Analyze the codebase, refactor the authentication module, and then write comprehensive tests"
        assert ComplexityDetector.needs_multi_agent(msg)

    def test_short_question_negative(self):
        assert not ComplexityDetector.needs_multi_agent("How are you?")
