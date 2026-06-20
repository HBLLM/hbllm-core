"""Unit tests for ReAct-style tool reasoning loop."""

import pytest

from hbllm.actions.tool_chain import (
    Observation,
    ReActConfig,
    ReActLoop,
    ReActResult,
    ThoughtStep,
)
from hbllm.actions.tool_registry import ToolRegistry, ToolResult

# ── Test Data Structures ─────────────────────────────────────────────────────


class TestReActConfig:
    def test_defaults(self):
        cfg = ReActConfig()
        assert cfg.max_iterations == 8
        assert cfg.max_wall_time_seconds == 60.0
        assert cfg.max_parallel_tools == 3
        assert cfg.allow_parallel is True

    def test_custom(self):
        cfg = ReActConfig(max_iterations=3, allow_parallel=False)
        assert cfg.max_iterations == 3
        assert cfg.allow_parallel is False


class TestObservation:
    def test_creation(self):
        obs = Observation(source="web_search", content="Result text", success=True)
        assert obs.source == "web_search"
        assert obs.success is True
        assert obs.timestamp > 0

    def test_failure(self):
        obs = Observation(source="api", content="Error: timeout", success=False)
        assert obs.success is False


class TestThoughtStep:
    def test_creation(self):
        step = ThoughtStep(step_num=1, thought="I need to search")
        assert step.step_num == 1
        assert step.action is None
        assert step.observation is None

    def test_with_action(self):
        step = ThoughtStep(
            step_num=2,
            thought="Let me search",
            action="web_search",
            action_input="python asyncio",
        )
        assert step.action == "web_search"


class TestReActResult:
    def test_to_dict(self):
        result = ReActResult(
            answer="The answer is 42",
            steps=[
                ThoughtStep(step_num=1, thought="Thinking..."),
            ],
            total_iterations=1,
            total_tool_calls=0,
            total_duration_ms=100.0,
            finished_reason="answer",
        )
        d = result.to_dict()
        assert d["answer"] == "The answer is 42"
        assert d["finished_reason"] == "answer"
        assert len(d["steps"]) == 1

    def test_with_observation(self):
        obs = Observation(source="calc", content="42")
        step = ThoughtStep(step_num=1, thought="Calc", action="calc", observation=obs)
        result = ReActResult(
            answer="42",
            steps=[step],
            total_iterations=1,
            total_tool_calls=1,
            total_duration_ms=50.0,
            finished_reason="answer",
        )
        d = result.to_dict()
        assert d["steps"][0]["observation"]["source"] == "calc"


# ── Test Parsing ─────────────────────────────────────────────────────────────


class TestReActParsing:
    def test_parse_basic_output(self):
        text = (
            "Thought: I need to search for information.\n"
            "Action: web_search\n"
            "Action Input: python asyncio tutorial"
        )
        thought, actions = ReActLoop._parse_react_output(text)
        assert "search for information" in thought
        assert len(actions) == 1
        assert actions[0]["tool"] == "web_search"
        assert actions[0]["input"] == "python asyncio tutorial"

    def test_parse_finish(self):
        text = "Thought: I have the answer.\nAction: FINISH\nAction Input: The answer is 42."
        thought, actions = ReActLoop._parse_react_output(text)
        assert len(actions) == 1
        assert actions[0]["tool"] == "FINISH"
        assert "42" in actions[0]["input"]

    def test_parse_no_structure(self):
        text = "Just a plain answer without any structure."
        thought, actions = ReActLoop._parse_react_output(text)
        assert thought == text
        # Should auto-detect as FINISH
        assert len(actions) == 1
        assert actions[0]["tool"] == "FINISH"

    def test_parse_multiple_actions(self):
        text = (
            "Thought: I need two pieces of info.\n"
            "Action: web_search\n"
            "Action Input: topic A\n"
            "Action: calculator\n"
            "Action Input: 2 + 2"
        )
        thought, actions = ReActLoop._parse_react_output(text)
        assert len(actions) == 2

    def test_extract_partial_answer_with_observations(self):
        steps = [
            ThoughtStep(
                step_num=1,
                thought="Found it",
                observation=Observation(source="search", content="The answer is here"),
            )
        ]
        answer = ReActLoop._extract_partial_answer(steps)
        assert "answer is here" in answer

    def test_extract_partial_answer_empty(self):
        answer = ReActLoop._extract_partial_answer([])
        assert "Unable to complete" in answer


# ── Test ReAct Loop Execution ────────────────────────────────────────────────


class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0

    async def generate(self, messages):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class TestReActLoop:
    @pytest.mark.asyncio
    async def test_direct_answer(self):
        """LLM answers immediately without tools."""
        llm = MockLLM(
            [
                "Thought: This is a simple question.\n"
                "Action: FINISH\n"
                "Action Input: The capital of France is Paris."
            ]
        )
        tools = ToolRegistry()
        loop = ReActLoop(llm=llm, tools=tools)
        result = await loop.run("What is the capital of France?")

        assert "Paris" in result.answer
        assert result.finished_reason == "answer"
        assert result.total_tool_calls == 0

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        """LLM uses one tool then answers."""
        llm = MockLLM(
            [
                "Thought: I need to calculate this.\nAction: calculator\nAction Input: 2 + 2",
                "Thought: I got the result.\nAction: FINISH\nAction Input: 2 + 2 = 4",
            ]
        )
        tools = ToolRegistry()

        async def calc(expression: str) -> ToolResult:
            return ToolResult(tool="calculator", success=True, output="4")

        tools.register("calculator", "Calculate math", calc, {"expression": "Math expression"})

        loop = ReActLoop(llm=llm, tools=tools)
        result = await loop.run("What is 2 + 2?")

        assert result.total_tool_calls == 1
        assert result.finished_reason == "answer"
        assert "4" in result.answer

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        """Loop stops after max iterations."""
        # LLM never says FINISH
        llm = MockLLM(["Thought: Let me think more.\nAction: calculator\nAction Input: 1 + 1"] * 5)

        tools = ToolRegistry()

        async def calc(expression: str) -> ToolResult:
            return ToolResult(tool="calculator", success=True, output="2")

        tools.register("calculator", "Calculate", calc, {"expression": "expr"})

        config = ReActConfig(max_iterations=3)
        loop = ReActLoop(llm=llm, tools=tools, config=config)
        result = await loop.run("Keep calculating")

        assert result.total_iterations == 3
        assert result.finished_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_tool_failure_handling(self):
        """Loop handles tool failures gracefully."""
        llm = MockLLM(
            [
                "Thought: Let me try.\nAction: broken_tool\nAction Input: test",
                "Thought: The tool failed, let me answer directly.\n"
                "Action: FINISH\n"
                "Action Input: Could not retrieve the information.",
            ]
        )
        tools = ToolRegistry()

        async def broken(x: str) -> ToolResult:
            return ToolResult(
                tool="broken_tool", success=False, output="", error="Connection refused"
            )

        tools.register("broken_tool", "A broken tool", broken, {"x": "input"})

        loop = ReActLoop(llm=llm, tools=tools)
        result = await loop.run("Use the broken tool")

        assert result.finished_reason == "answer"
        # Should have observation with failure
        tool_steps = [s for s in result.steps if s.observation and not s.observation.success]
        assert len(tool_steps) >= 1

    @pytest.mark.asyncio
    async def test_scratchpad(self):
        """Scratchpad captures reasoning chain."""
        llm = MockLLM(["Thought: Simple answer.\nAction: FINISH\nAction Input: Done."])
        tools = ToolRegistry()
        config = ReActConfig(include_scratchpad=True)
        loop = ReActLoop(llm=llm, tools=tools, config=config)
        result = await loop.run("Quick task")

        assert "FINISH" in result.scratchpad

    @pytest.mark.asyncio
    async def test_no_llm(self):
        """Handles missing LLM gracefully."""

        class NoLLM:
            pass

        tools = ToolRegistry()
        loop = ReActLoop(llm=NoLLM(), tools=tools)
        result = await loop.run("Test without LLM")

        assert result.finished_reason == "answer"
        assert "no LLM" in result.answer.lower() or "unable" in result.answer.lower()
