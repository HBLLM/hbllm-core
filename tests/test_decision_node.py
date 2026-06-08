"""Tests for DecisionNode — the safety gatekeeper and output dispatcher."""

import asyncio

import pytest

from hbllm.brain.action_planner import ActionPlanner
from hbllm.brain.action_schema import ActionPlan, ActionType, RiskLevel
from hbllm.brain.decision_node import DecisionNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ── Mock LLMs ────────────────────────────────────────────────────────────────


class MockSafeLLM:
    """Mock LLM that classifies content as safe."""

    async def generate_json(self, prompt):
        return {"safe": True, "reason": "Content is safe"}

    async def generate(self, prompt, **kwargs):
        return "Synthesized output"


class MockUnsafeLLM:
    """Mock LLM that classifies content as unsafe."""

    async def generate_json(self, prompt):
        return {"safe": False, "reason": "Contains harmful instructions"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_decision_message(content="Hello, world!", intent="answer", **kwargs):
    return Message(
        type=MessageType.EVENT,
        source_node_id="workspace_01",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": intent, "text": "test query", **kwargs},
            "selected_thought": {
                "type": "intuition_general",
                "confidence": 0.9,
                "content": content,
            },
        },
    )


class OutputCollector:
    """Collects messages from the bus for assertions."""

    def __init__(self):
        self.messages = []

    async def collect(self, msg):
        self.messages.append(msg)


# ── Safety Classification Tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_safe_content_passes_through():
    """Safe content should be published to sensory.output."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_safe", llm=MockSafeLLM())
    await node.start(bus)

    msg = _make_decision_message("This is a helpful answer")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "This is a helpful answer"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_unsafe_content_blocked():
    """Unsafe content should be rejected with safety warning.

    Note: LLM safety is only invoked for HIGH risk actions (code, IoT, MCP).
    A plain text response (LOW risk) skips the LLM classifier entirely.
    So we test with code content to trigger the HIGH risk path.
    """
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_unsafe", llm=MockUnsafeLLM())
    await node.start(bus)

    # Use code content to trigger HIGH risk → LLM safety check
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace_01",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "answer", "text": "run this code"},
            "selected_thought": {
                "type": "code_execution",
                "confidence": 0.9,
                "content": "```python\nimport os; os.system('rm -rf /')\n```",
            },
        },
    )
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert "safety constraints" in collector.messages[0].payload["text"]

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_no_llm_skips_safety_check():
    """Without an LLM, safety check is skipped and content passes through."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_nollm", llm=None)
    await node.start(bus)

    msg = _make_decision_message("Unfiltered content")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "Unfiltered content"

    await node.stop()
    await bus.stop()


# ── Output Routing Tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_routes_to_audio_output():
    """Intent=speak should route to audio output node."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.audio.out", collector.collect)

    node = DecisionNode(node_id="decision_audio", llm=MockSafeLLM())
    await node.start(bus)

    msg = _make_decision_message("Say this out loud", intent="speak")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "Say this out loud"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_routes_code_to_execution():
    """Content with python code blocks should route to execution node."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()

    async def mock_execute_handler(msg):
        await collector.collect(msg)
        return msg.create_response({"output": "hello output", "error": ""})

    await bus.subscribe("task.execute.python", mock_execute_handler)

    node = DecisionNode(node_id="decision_code", llm=MockSafeLLM())
    await node.start(bus)

    code_content = "Here's the solution:\n```python\nprint('hello')\n```"
    msg = _make_decision_message(code_content)
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["code"] == "print('hello')"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_routes_plain_text_to_ui():
    """Plain text should route to sensory.output (UI)."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()
    await bus.subscribe("sensory.output", collector.collect)

    node = DecisionNode(node_id="decision_ui", llm=MockSafeLLM())
    await node.start(bus)

    msg = _make_decision_message("Just a regular answer")
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    assert len(collector.messages) == 1
    assert collector.messages[0].payload["text"] == "Just a regular answer"
    assert collector.messages[0].payload["source"] == "intuition_general"

    await node.stop()
    await bus.stop()


# ── Node Lifecycle Tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decision_node_starts_and_stops():
    """Node should subscribe on start and clean up on stop."""
    bus = InProcessBus()
    await bus.start()

    node = DecisionNode(node_id="decision_lifecycle")
    await node.start(bus)
    assert node.node_id == "decision_lifecycle"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_handle_message_returns_none():
    """The generic handle_message should return None (unused)."""
    bus = InProcessBus()
    await bus.start()

    node = DecisionNode(node_id="decision_handle")
    await node.start(bus)

    msg = _make_decision_message("test")
    result = await node.handle_message(msg)
    assert result is None

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_routes_web_search_with_vague_query_resolution():
    """Web search with a vague query (e.g. 'search it') should resolve context using memory and route."""
    bus = InProcessBus()
    await bus.start()

    collector = OutputCollector()

    # 1. Mock memory node retrieve_recent response
    async def mock_memory_handler(msg):
        return msg.create_response(
            {
                "session_id": "session_123",
                "turns": [
                    {"role": "user", "content": "who is the current president of sri lanka?"},
                    {
                        "role": "assistant",
                        "content": "I don't have the latest information. I can search for it.",
                    },
                ],
            }
        )

    await bus.subscribe("memory.retrieve_recent", mock_memory_handler)

    # 2. Mock browser node search response
    async def mock_search_handler(msg):
        await collector.collect(msg)
        return msg.create_response(
            {"text": "The current president of Sri Lanka is Anura Kumara Dissanayake."}
        )

    await bus.subscribe("task.execute.search", mock_search_handler)

    # 3. Mock LLM with generate method that resolves 'search it' and synthesizes response
    class MockResolverLLM:
        async def generate_json(self, prompt):
            return {"safe": True, "reason": "Content is safe"}

        async def generate(self, prompt, **kwargs):
            if "Resolved Search Query:" in prompt:
                return "current president of Sri Lanka"
            return "Synthesized: Anura Kumara Dissanayake is the president."

    node = DecisionNode(node_id="decision_search", llm=MockResolverLLM())
    await node.start(bus)

    # Create evaluate message with vague query
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace_01",
        topic="decision.evaluate",
        payload={
            "original_query": {
                "intent": "web_search",
                "text": "search it",
                "session_id": "session_123",
                "tenant_id": "default",
            },
            "selected_thought": {
                "type": "web_search",
                "confidence": 0.9,
                "content": "I will search it.",
            },
        },
    )
    await node.evaluate_workspace_decision(msg)
    await asyncio.sleep(0.2)

    # Assert that the search query was resolved to "current president of Sri Lanka"
    assert len(collector.messages) == 1
    assert collector.messages[0].payload["query"] == "current president of Sri Lanka"

    await node.stop()
    await bus.stop()


# ── ActionPlanner Unit Tests ─────────────────────────────────────────────────


class TestActionPlannerMapsIntents:
    """Verify that the ActionPlanner maps intents to the correct ActionType."""

    def setup_method(self):
        self.planner = ActionPlanner()

    def _plan(
        self, intent="answer", thought_type="intuition", content="test", confidence=0.9, **kwargs
    ):
        return self.planner.plan(
            intent=intent,
            thought_type=thought_type,
            content=content,
            confidence=confidence,
            original_query={"text": "test query", "intent": intent, **kwargs},
        )

    def test_speak_intent(self):
        plan = self._plan(intent="speak")
        assert plan.action_type == ActionType.AUDIO_OUTPUT

    def test_force_audio(self):
        plan = self._plan(force_audio=True)
        assert plan.action_type == ActionType.AUDIO_OUTPUT

    def test_web_search_intent(self):
        plan = self._plan(intent="web_search")
        assert plan.action_type == ActionType.WEB_SEARCH

    def test_force_search(self):
        plan = self._plan(force_search=True)
        assert plan.action_type == ActionType.WEB_SEARCH

    def test_api_synthesis_thought(self):
        plan = self._plan(thought_type="api_synthesis")
        assert plan.action_type == ActionType.API_CALL

    def test_tool_synthesis_intent(self):
        plan = self._plan(intent="tool_synthesis")
        assert plan.action_type == ActionType.API_CALL

    def test_iot_command_intent(self):
        plan = self._plan(intent="iot_command")
        assert plan.action_type == ActionType.IOT_COMMAND

    def test_iot_topic_present(self):
        plan = self._plan(iot_topic="home/lights")
        assert plan.action_type == ActionType.IOT_COMMAND

    def test_mcp_tool_intent(self):
        plan = self._plan(intent="mcp_tool")
        assert plan.action_type == ActionType.MCP_TOOL

    def test_mcp_tool_name_present(self):
        plan = self._plan(mcp_tool_name="weather_tool")
        assert plan.action_type == ActionType.MCP_TOOL

    def test_code_execution_thought_type(self):
        plan = self._plan(thought_type="code_execution")
        assert plan.action_type == ActionType.CODE_EXECUTION

    def test_code_block_in_content(self):
        plan = self._plan(content="Here:\n```python\nprint(1)\n```")
        assert plan.action_type == ActionType.CODE_EXECUTION

    def test_default_text_response(self):
        plan = self._plan()
        assert plan.action_type == ActionType.TEXT_RESPONSE


class TestConfidenceThreshold:
    """Verify that low confidence triggers CLARIFY."""

    def setup_method(self):
        self.planner = ActionPlanner()

    def test_low_confidence_triggers_clarify(self):
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content="Maybe something?",
            confidence=0.1,
            original_query={"text": "test", "intent": "answer"},
        )
        assert plan.action_type == ActionType.CLARIFY
        assert plan.metadata["confidence"] == 0.1

    def test_threshold_boundary(self):
        """Confidence exactly at 0.3 should NOT trigger clarify."""
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content="Some answer",
            confidence=0.3,
            original_query={"text": "test", "intent": "answer"},
        )
        assert plan.action_type == ActionType.TEXT_RESPONSE

    def test_above_threshold_proceeds(self):
        plan = self.planner.plan(
            intent="web_search",
            thought_type="web_search",
            content="search results",
            confidence=0.8,
            original_query={"text": "who is the president", "intent": "web_search"},
        )
        assert plan.action_type == ActionType.WEB_SEARCH


class TestVagueDetection:
    """Verify that vague detection uses pronouns, not verbs."""

    def setup_method(self):
        self.planner = ActionPlanner()

    def test_pronoun_reference_triggers_vague(self):
        plan = self.planner.plan(
            intent="web_search",
            thought_type="web_search",
            content="search results",
            confidence=0.9,
            original_query={"text": "search it", "intent": "web_search"},
        )
        assert plan.metadata.get("needs_context_resolution") is True

    def test_short_query_is_vague(self):
        plan = self.planner.plan(
            intent="web_search",
            thought_type="web_search",
            content="search results",
            confidence=0.9,
            original_query={"text": "yes", "intent": "web_search"},
        )
        assert plan.metadata.get("needs_context_resolution") is True

    def test_specific_query_is_not_vague(self):
        plan = self.planner.plan(
            intent="web_search",
            thought_type="web_search",
            content="search results",
            confidence=0.9,
            original_query={
                "text": "search for quantum computing breakthroughs 2026",
                "intent": "web_search",
            },
        )
        assert plan.metadata.get("needs_context_resolution") is False

    def test_that_with_enough_words_not_vague(self):
        """'that' in a long enough query should NOT trigger vague."""
        plan = self.planner.plan(
            intent="web_search",
            thought_type="web_search",
            content="search results",
            confidence=0.9,
            original_query={
                "text": "can you search for that specific research paper about transformers in NLP",
                "intent": "web_search",
            },
        )
        assert plan.metadata.get("needs_context_resolution") is False


class TestTieredSafety:
    """Verify the LLM safety classifier is only called for HIGH risk actions."""

    def setup_method(self):
        self.planner = ActionPlanner()

    def test_text_response_is_low_risk(self):
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content="Hello",
            confidence=0.9,
            original_query={"text": "hi", "intent": "answer"},
        )
        assert plan.risk_level == RiskLevel.LOW
        assert plan.requires_safety_llm is False

    def test_web_search_is_medium_risk(self):
        plan = self.planner.plan(
            intent="web_search",
            thought_type="web_search",
            content="query",
            confidence=0.9,
            original_query={"text": "latest news", "intent": "web_search"},
        )
        assert plan.risk_level == RiskLevel.MEDIUM
        assert plan.requires_safety_llm is False

    def test_code_execution_is_high_risk(self):
        plan = self.planner.plan(
            intent="answer",
            thought_type="code_execution",
            content="```python\nprint(1)\n```",
            confidence=0.9,
            original_query={"text": "run code", "intent": "answer"},
        )
        assert plan.risk_level == RiskLevel.HIGH
        assert plan.requires_safety_llm is True

    def test_iot_command_is_high_risk(self):
        plan = self.planner.plan(
            intent="iot_command",
            thought_type="intuition",
            content="turn on lights",
            confidence=0.9,
            original_query={"text": "lights on", "intent": "iot_command"},
        )
        assert plan.risk_level == RiskLevel.HIGH
        assert plan.requires_safety_llm is True

    def test_mcp_tool_is_high_risk(self):
        plan = self.planner.plan(
            intent="mcp_tool",
            thought_type="intuition",
            content="call tool",
            confidence=0.9,
            original_query={
                "text": "use weather",
                "intent": "mcp_tool",
                "mcp_tool_name": "weather",
            },
        )
        assert plan.risk_level == RiskLevel.HIGH
        assert plan.requires_safety_llm is True

    @pytest.mark.asyncio
    async def test_llm_not_called_for_text_response(self):
        """Verify the LLM safety classifier is NOT called for a text response (LOW risk)."""
        bus = InProcessBus()
        await bus.start()

        call_log = []

        class SpyLLM:
            async def generate_json(self, prompt):
                call_log.append("generate_json")
                return {"safe": True, "reason": "ok"}

            async def generate(self, prompt, **kwargs):
                return "output"

        collector = OutputCollector()
        await bus.subscribe("sensory.output", collector.collect)

        node = DecisionNode(node_id="decision_spy", llm=SpyLLM())
        await node.start(bus)

        msg = _make_decision_message("Hello world")
        await node.evaluate_workspace_decision(msg)
        await asyncio.sleep(0.2)

        # The LLM safety classifier (generate_json) should NOT have been called
        assert "generate_json" not in call_log
        assert len(collector.messages) == 1

        await node.stop()
        await bus.stop()
