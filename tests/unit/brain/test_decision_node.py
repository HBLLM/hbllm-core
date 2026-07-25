"""Tests for DecisionNode — the safety gatekeeper and output dispatcher."""

import asyncio

import pytest

from hbllm.brain.control.decision_node import DecisionNode
from hbllm.brain.evaluation.utility_calibrator import UtilityCalibrator
from hbllm.brain.planning.action_planner import ActionPlanner
from hbllm.brain.planning.action_schema import ActionPlan, ActionType, RiskLevel
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


# ── Level 2 Policy Router & Level 3 Budget Controller Tests ──────────────────


def test_utility_calibrator_readiness(tmp_path):
    calibrator = UtilityCalibrator(data_dir=str(tmp_path))
    # Sample count < 10 -> bootstrap should be active
    readiness = calibrator.get_calibration_readiness()
    assert readiness["bootstrap_active"] is True
    assert "Insufficient samples" in readiness["reason"]

    # Record 12 traces (still < 15, so bootstrap should be active)
    for i in range(12):
        calibrator.record_trace(
            trace_id=f"t_{i}",
            decision_point="domain_a",
            predicted_utility=0.5,
            actual_outcome=0.5,
        )
    readiness = calibrator.get_calibration_readiness()
    assert readiness["bootstrap_active"] is True

    # Record 4 more traces (now sample_count = 16, but unique_domains = 1 < 3)
    for i in range(12, 16):
        calibrator.record_trace(
            trace_id=f"t_{i}",
            decision_point="domain_a",
            predicted_utility=0.5,
            actual_outcome=0.5,
        )
    readiness = calibrator.get_calibration_readiness()
    assert readiness["bootstrap_active"] is True

    # Record traces in domain_b and domain_c to satisfy unique_domains >= 3
    # Also keep variance low (predicted_utility ≈ actual_outcome)
    for i in range(16, 18):
        calibrator.record_trace(
            trace_id=f"t_{i}",
            decision_point="domain_b",
            predicted_utility=0.5,
            actual_outcome=0.5,
        )
    for i in range(18, 20):
        calibrator.record_trace(
            trace_id=f"t_{i}",
            decision_point="domain_c",
            predicted_utility=0.5,
            actual_outcome=0.5,
        )
    readiness = calibrator.get_calibration_readiness()
    assert readiness["bootstrap_active"] is False


def test_utility_calibrator_percentiles_anchor_mix(tmp_path):
    calibrator = UtilityCalibrator(data_dir=str(tmp_path))
    # Default without enough samples (lt_rows < 10) returns anchor percentiles (0.7, 0.3, 0.0)
    high, med, low = calibrator.get_utility_percentiles()
    assert high == 0.7
    assert med == 0.3
    assert low == 0.0

    # Populate 20 traces with predicted_utility = 1.0
    for i in range(20):
        calibrator.record_trace(
            trace_id=f"t_{i}",
            decision_point=f"domain_{i % 3}",
            predicted_utility=1.0,
            actual_outcome=1.0,
        )
    high, med, low = calibrator.get_utility_percentiles()
    assert abs(high - 0.91) < 1e-5
    assert abs(med - 0.79) < 1e-5
    assert abs(low - 0.70) < 1e-5


def test_utility_calibrator_zscore_drift_detection(tmp_path):
    calibrator = UtilityCalibrator(data_dir=str(tmp_path))
    assert calibrator.detect_drift() is False

    # Seed 100 traces with constant 0.5 (std dev = 0)
    for i in range(100):
        val = 0.5
        calibrator.record_trace(
            trace_id=f"t_{i}",
            decision_point="domain_a",
            predicted_utility=val,
            actual_outcome=val,
        )
    assert calibrator.detect_drift() is False

    # Introduce sudden drift upwards: record 20 samples of 0.9
    for i in range(100, 120):
        calibrator.record_trace(
            trace_id=f"t_{i}",
            decision_point="domain_a",
            predicted_utility=0.9,
            actual_outcome=0.9,
        )
    assert calibrator.detect_drift() is True


@pytest.mark.asyncio
async def test_router_sigmoid_load_penalty(tmp_path):
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}
    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 1.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    msg_low = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "answer", "cpu_percent": 10.0},
            "selected_thought": {
                "type": "intuition",
                "confidence": 1.0,
                "predicted_latency": 10.0,
            },
        },
    )
    msg_high = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "answer", "cpu_percent": 95.0},
            "selected_thought": {
                "type": "intuition",
                "confidence": 1.0,
                "predicted_latency": 10.0,
            },
        },
    )

    recorded_utilities = []
    original_record = node.calibrator.record_trace
    node.calibrator.record_trace = lambda *args, **kwargs: (
        recorded_utilities.append(kwargs.get("predicted_utility")),
        original_record(*args, **kwargs),
    )[1]

    await node._arbitrate_utility(plan, msg_low, msg_low.payload["original_query"])
    await node._arbitrate_utility(plan, msg_high, msg_high.payload["original_query"])

    assert recorded_utilities[0] > recorded_utilities[1]


@pytest.mark.asyncio
async def test_router_execution_hysteresis(tmp_path):
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_utility_percentiles = lambda: (0.8, 0.5, 0.2)
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}

    node.smoothed_high = 0.8
    node.smoothed_med = 0.5
    node.smoothed_low = 0.2
    node.last_mode = "high"

    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "answer", "cpu_percent": 20.0},
            "selected_thought": {
                "type": "intuition",
                "confidence": 0.76,
                "predicted_latency": 0.0,
            },
        },
    )
    await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    assert node.last_mode == "high"

    msg.payload["selected_thought"]["confidence"] = 0.74
    await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    assert node.last_mode == "medium"

    msg.payload["selected_thought"]["confidence"] = 0.82
    await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    assert node.last_mode == "medium"

    msg.payload["selected_thought"]["confidence"] = 0.86
    await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    assert node.last_mode == "high"


@pytest.mark.asyncio
async def test_router_context_aware_replanning_recursion(tmp_path):
    bus = InProcessBus()
    await bus.start()

    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    await node.start(bus)

    decomposes = []
    deferreds = []
    await bus.subscribe("planner.decompose", decomposes.append)
    await bus.subscribe("tasks.deferred", deferreds.append)

    node.calibrator.get_utility_percentiles = lambda: (0.9, 0.8, 0.7)
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}
    node.smoothed_high = 0.9
    node.smoothed_med = 0.8
    node.smoothed_low = 0.7
    node.last_mode = "high"

    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    msg_code = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "code_execution", "replan_depth": 3},
            "selected_thought": {
                "type": "code_execution",
                "confidence": 0.1,
                "predicted_latency": 0.0,
            },
        },
    )
    plan_code = ActionPlan(action_type=ActionType.CODE_EXECUTION, content="print(1)")

    await node._arbitrate_utility(plan_code, msg_code, msg_code.payload["original_query"])
    await asyncio.sleep(0.1)
    assert len(decomposes) == 1
    assert decomposes[-1].payload["original_query"]["replan_depth"] == 4

    decomposes.clear()

    msg_code.payload["original_query"]["replan_depth"] = 4
    await node._arbitrate_utility(plan_code, msg_code, msg_code.payload["original_query"])
    await asyncio.sleep(0.1)
    assert len(decomposes) == 0
    assert len(deferreds) == 1
    assert deferreds[-1].payload["reason"] == "replan_depth_exceeded"

    deferreds.clear()

    msg_search = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "web_search", "replan_depth": 1},
            "selected_thought": {
                "type": "web_search",
                "confidence": 0.1,
                "predicted_latency": 0.0,
            },
        },
    )
    plan_search = ActionPlan(action_type=ActionType.WEB_SEARCH, content="search")

    await node._arbitrate_utility(plan_search, msg_search, msg_search.payload["original_query"])
    await asyncio.sleep(0.1)
    assert len(decomposes) == 1
    assert decomposes[-1].payload["original_query"]["replan_depth"] == 2

    decomposes.clear()

    msg_search.payload["original_query"]["replan_depth"] = 2
    await node._arbitrate_utility(plan_search, msg_search, msg_search.payload["original_query"])
    await asyncio.sleep(0.1)
    assert len(decomposes) == 0
    assert len(deferreds) == 1

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_router_exploration_ttl_decay(tmp_path):
    bus = InProcessBus()
    await bus.start()

    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    await node.start(bus)

    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0
    node.smoothed_high = 0.9
    node.smoothed_med = 0.8
    node.smoothed_low = 0.7
    node.last_mode = "high"
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    from datetime import datetime, timedelta, timezone

    past_time = datetime.now(timezone.utc) - timedelta(seconds=400)

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {
                "intent": "answer",
                "cpu_percent": 20.0,
                "exploration_override": True,
                "exploration_ttl": 300.0,
            },
            "selected_thought": {
                "type": "intuition",
                "confidence": 0.5,
                "predicted_latency": 0.0,
            },
        },
        timestamp=past_time,
    )

    res = await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    assert res is False
    assert node.last_mode == "low"

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_router_exploration_restricted_tools(tmp_path):
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}
    node.smoothed_high = 0.9
    node.smoothed_med = 0.8
    node.smoothed_low = 0.7
    node.last_mode = "low"

    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan_restricted = ActionPlan(action_type=ActionType.MCP_TOOL, content="use mcp")
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {
                "intent": "mcp_tool",
                "cpu_percent": 20.0,
                "exploration_override": True,
            },
            "selected_thought": {
                "type": "intuition",
                "confidence": 0.5,
                "predicted_latency": 0.0,
            },
        },
    )

    res = await node._arbitrate_utility(plan_restricted, msg, msg.payload["original_query"])
    assert res is False

    plan_allowed = ActionPlan(action_type=ActionType.WEB_SEARCH, content="search")
    res_allowed = await node._arbitrate_utility(plan_allowed, msg, msg.payload["original_query"])
    assert res_allowed is True


@pytest.mark.asyncio
async def test_router_bootstrap_readiness_offset(tmp_path):
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": True}
    node.smoothed_high = 0.8
    node.smoothed_med = 0.5
    node.smoothed_low = 0.2
    node.last_mode = "medium"

    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "answer", "cpu_percent": 20.0},
            "selected_thought": {
                "type": "intuition",
                "confidence": 0.6,
                "predicted_latency": 0.0,
            },
        },
    )
    await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    assert node.last_mode == "high"


def test_budget_controller_token_limits(tmp_path):
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    node.last_mode = "high"
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {
                "thought_budget": {"max_tokens": 4000},
            },
            "selected_thought": {"tokens_used": 3000},
        },
    )
    assert node._check_hard_limits(plan, msg) is True

    node.last_mode = "medium"
    assert node._check_hard_limits(plan, msg) is False

    node.last_mode = "low"
    msg_low = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {
                "thought_budget": {"max_tokens": 4000},
            },
            "selected_thought": {"tokens_used": 1020},
        },
    )
    assert node._check_hard_limits(plan, msg_low) is True

    msg_low_fail = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {
                "thought_budget": {"max_tokens": 4000},
            },
            "selected_thought": {"tokens_used": 1050},
        },
    )
    assert node._check_hard_limits(plan, msg_low_fail) is False


@pytest.mark.asyncio
async def test_router_timescale_separation(tmp_path):
    """Verify that percentiles are only queried at smooth ticks (jittered ~7) and
    Lyapunov checks fire at observe ticks (jittered ~13)."""
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))

    percentile_calls = []
    original_percentiles = node.calibrator.get_utility_percentiles

    def spy_percentiles():
        percentile_calls.append(node.decision_count)
        return original_percentiles()

    node.calibrator.get_utility_percentiles = spy_percentiles
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}

    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    import random

    original_randint = random.randint
    random.randint = lambda a, b: 0
    try:
        for _ in range(15):
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="workspace",
                topic="decision.evaluate",
                payload={
                    "original_query": {"intent": "answer", "cpu_percent": 20.0},
                    "selected_thought": {
                        "type": "intuition",
                        "confidence": 0.9,
                        "predicted_latency": 0.0,
                    },
                },
            )
            await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    finally:
        random.randint = original_randint

    assert node.decision_count == 15
    assert percentile_calls == [7, 14]


@pytest.mark.asyncio
async def test_router_replanning_quadratic_penalty(tmp_path):
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}

    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    recorded_utilities = []
    original_record = node.calibrator.record_trace
    node.calibrator.record_trace = lambda *args, **kwargs: (
        recorded_utilities.append(kwargs.get("predicted_utility")),
        original_record(*args, **kwargs),
    )[1]

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="decision.evaluate",
        payload={
            "original_query": {"intent": "answer", "replan_depth": 3},
            "selected_thought": {
                "type": "intuition",
                "confidence": 0.9,
            },
        },
    )
    await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
    assert abs(recorded_utilities[0] - 0.45) < 1e-5


@pytest.mark.asyncio
async def test_router_lyapunov_cooling_hysteresis(tmp_path):
    """Verify cooling enters at S_diag > 0.08 and exits only after 3
    consecutive checks with S_diag < 0.04 (Schmitt trigger)."""
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": True}
    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    def make_msg(confidence=0.9):
        return Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="decision.evaluate",
            payload={
                "original_query": {"intent": "answer", "cpu_percent": 20.0},
                "selected_thought": {
                    "type": "intuition",
                    "confidence": confidence,
                    "predicted_latency": 0.0,
                },
            },
        )

    from hbllm.brain.evaluation.utility_calibrator import CalibrationTrace

    high_var_traces = [
        CalibrationTrace(
            trace_id=f"t_{i}",
            decision_point="decision_node:text_response",
            predicted_utility=0.9 if i < 7 else 0.5,
            actual_outcome=0.1 if i < 7 else 0.5,
            prediction_error=0.8 if i < 7 else 0.0,
            timestamp=0.0,
            metadata={},
        )
        for i in range(15)
    ]
    node.calibrator.get_traces = lambda: high_var_traces

    import random

    original_randint = random.randint
    random.randint = lambda a, b: 0
    try:
        for _ in range(13):
            msg = make_msg(confidence=0.5)
            await node._arbitrate_utility(plan, msg, msg.payload["original_query"])

        assert node.decision_count == 13
        assert node.in_cooling is True

        stable_traces = [
            CalibrationTrace(
                trace_id=f"t_{i}",
                decision_point="decision_node:text_response",
                predicted_utility=0.5,
                actual_outcome=0.5,
                prediction_error=0.0,
                timestamp=0.0,
                metadata={},
            )
            for i in range(15)
        ]
        node.calibrator.get_traces = lambda: stable_traces

        for _ in range(13):
            msg = make_msg(confidence=0.5)
            await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
        assert node.decision_count == 26
        assert node.in_cooling is True

        for _ in range(13):
            msg = make_msg(confidence=0.5)
            await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
        assert node.decision_count == 39
        assert node.in_cooling is True

        for _ in range(13):
            msg = make_msg(confidence=0.5)
            await node._arbitrate_utility(plan, msg, msg.payload["original_query"])
        assert node.decision_count == 52
        assert node.in_cooling is False

        node.in_cooling = True
        node.smoothed_high = 0.9
        node.smoothed_med = 0.8
        node.smoothed_low = 0.7
        msg_exp = Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="decision.evaluate",
            payload={
                "original_query": {
                    "intent": "answer",
                    "cpu_percent": 20.0,
                    "exploration_override": True,
                },
                "selected_thought": {
                    "type": "intuition",
                    "confidence": 0.1,
                    "predicted_latency": 0.0,
                },
            },
        )
        await node._arbitrate_utility(plan, msg_exp, msg_exp.payload["original_query"])
        assert node.last_mode == "negative"
        assert plan.metadata.get("optimize_resources") is True

    finally:
        random.randint = original_randint


@pytest.mark.asyncio
async def test_router_high_confidence_anchor_gating(tmp_path):
    """Verify that anchor is only updated if cooling exit is high confidence (S_diag < 0.03)."""
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": True}
    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    node._anchor_percentiles = (0.7, 0.3, 0.0)
    node.smoothed_high = 0.9
    node.smoothed_med = 0.8
    node.smoothed_low = 0.7

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    def make_msg(confidence=0.5):
        return Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="decision.evaluate",
            payload={
                "original_query": {"intent": "answer", "cpu_percent": 20.0},
                "selected_thought": {
                    "type": "intuition",
                    "confidence": confidence,
                    "predicted_latency": 0.0,
                },
            },
        )

    from hbllm.brain.evaluation.utility_calibrator import CalibrationTrace

    high_var_traces = [
        CalibrationTrace(
            trace_id=f"t_{i}",
            decision_point="decision_node:text_response",
            predicted_utility=0.9 if i < 7 else 0.5,
            actual_outcome=0.1 if i < 7 else 0.5,
            prediction_error=0.8 if i < 7 else 0.0,
            timestamp=0.0,
            metadata={},
        )
        for i in range(15)
    ]
    node.calibrator.get_traces = lambda: high_var_traces

    import random

    original_randint = random.randint
    random.randint = lambda a, b: 0
    try:
        await node._arbitrate_utility(plan, make_msg(), {})
        node.decision_count = 12
        node._next_observe_tick = 13
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True

        marginal_traces = [
            CalibrationTrace(
                trace_id=f"t_{i}",
                decision_point="decision_node:text_response",
                predicted_utility=0.632,
                actual_outcome=0.5,
                prediction_error=0.132,
                timestamp=0.0,
                metadata={},
            )
            for i in range(15)
        ]
        node.calibrator.get_traces = lambda: marginal_traces

        for target in (25, 38, 51):
            node.decision_count = target
            node._next_observe_tick = target + 1
            await node._arbitrate_utility(plan, make_msg(), {})

        assert node.in_cooling is False
        assert node._anchor_percentiles == (0.7, 0.3, 0.0)

        node.calibrator.get_traces = lambda: high_var_traces
        node.decision_count = 12
        node._next_observe_tick = 13
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True

        zero_traces = [
            CalibrationTrace(
                trace_id=f"t_{i}",
                decision_point="decision_node:text_response",
                predicted_utility=0.5,
                actual_outcome=0.5,
                prediction_error=0.0,
                timestamp=0.0,
                metadata={},
            )
            for i in range(15)
        ]
        node.calibrator.get_traces = lambda: zero_traces

        for target in (25, 38, 51):
            node.decision_count = target
            node._next_observe_tick = target + 1
            await node._arbitrate_utility(plan, make_msg(), {})

        assert node.in_cooling is False
        assert node._anchor_percentiles == (
            node.smoothed_high,
            node.smoothed_med,
            node.smoothed_low,
        )

    finally:
        random.randint = original_randint


@pytest.mark.asyncio
async def test_router_stable_lock_invariance(tmp_path):
    """Verify that 3 cooling entries within 60 decisions engage stable lock."""
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": True}
    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    def make_msg(confidence=0.5):
        return Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="decision.evaluate",
            payload={
                "original_query": {"intent": "answer", "cpu_percent": 20.0},
                "selected_thought": {
                    "type": "intuition",
                    "confidence": confidence,
                    "predicted_latency": 0.0,
                },
            },
        )

    from hbllm.brain.evaluation.utility_calibrator import CalibrationTrace

    high_var_traces = [
        CalibrationTrace(
            trace_id=f"t_{i}",
            decision_point="decision_node:text_response",
            predicted_utility=0.9 if i < 7 else 0.5,
            actual_outcome=0.1 if i < 7 else 0.5,
            prediction_error=0.8 if i < 7 else 0.0,
            timestamp=0.0,
            metadata={},
        )
        for i in range(15)
    ]
    stable_traces = [
        CalibrationTrace(
            trace_id=f"t_{i}",
            decision_point="decision_node:text_response",
            predicted_utility=0.5,
            actual_outcome=0.5,
            prediction_error=0.0,
            timestamp=0.0,
            metadata={},
        )
        for i in range(15)
    ]

    import random

    original_randint = random.randint
    random.randint = lambda a, b: 0
    try:
        node.calibrator.get_traces = lambda: high_var_traces
        node.decision_count = 12
        node._next_observe_tick = 13
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True
        assert len(node._cooling_cycle_history) == 1

        node.calibrator.get_traces = lambda: stable_traces
        for t in (25, 38, 51):
            node.decision_count = t
            node._next_observe_tick = t + 1
            await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is False

        node.calibrator.get_traces = lambda: high_var_traces
        node.decision_count = 63
        node._next_observe_tick = 64
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True
        assert len(node._cooling_cycle_history) == 2

        node.calibrator.get_traces = lambda: stable_traces
        for t in (76, 89, 102):
            node.decision_count = t
            node._next_observe_tick = t + 1
            await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is False

        node.calibrator.get_traces = lambda: high_var_traces
        node.decision_count = 114
        node._next_observe_tick = 115
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True
        assert node._stable_lock is False

        node.calibrator.get_traces = lambda: stable_traces
        for t in (127, 140, 153):
            node.decision_count = t
            node._next_observe_tick = t + 1
            await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is False

        node._cooling_cycle_history.clear()

        node.calibrator.get_traces = lambda: high_var_traces
        node.decision_count = 159
        node._next_observe_tick = 160
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True

        node.calibrator.get_traces = lambda: stable_traces
        for t in (172, 185, 198):
            node.decision_count = t
            node._next_observe_tick = t + 1
            await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is False

        node.calibrator.get_traces = lambda: high_var_traces
        node.decision_count = 209
        node._next_observe_tick = 210
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True

        node.calibrator.get_traces = lambda: stable_traces
        for t in (222, 235, 248):
            node.decision_count = t
            node._next_observe_tick = t + 1
            await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is False

        node.calibrator.get_traces = lambda: high_var_traces
        node.decision_count = 219
        node._next_observe_tick = 220
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is True
        assert node._stable_lock is True

        node.calibrator.get_traces = lambda: stable_traces
        for t in (231, 244, 257, 270, 283):
            node.decision_count = t
            node._next_observe_tick = t + 1
            await node._arbitrate_utility(plan, make_msg(), {})
            assert node.in_cooling is True

        node.decision_count = 296
        node._next_observe_tick = 297
        await node._arbitrate_utility(plan, make_msg(), {})
        assert node.in_cooling is False
        assert node._stable_lock is False

    finally:
        random.randint = original_randint


@pytest.mark.asyncio
async def test_router_s_ctrl_smoothing(tmp_path):
    """Verify S_ctrl smoothing calculations and adaptive anchor blending weight."""
    node = DecisionNode(node_id="test_node", data_dir=str(tmp_path))
    node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}
    node.utility_engine.weight_token = 0.0
    node.utility_engine.weight_latency = 0.0
    node.utility_engine.weight_risk = 0.0

    plan = ActionPlan(action_type=ActionType.TEXT_RESPONSE, content="test")

    def make_msg():
        return Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="decision.evaluate",
            payload={
                "original_query": {"intent": "answer", "cpu_percent": 20.0},
                "selected_thought": {
                    "type": "intuition",
                    "confidence": 0.5,
                    "predicted_latency": 0.0,
                },
            },
        )

    from hbllm.brain.evaluation.utility_calibrator import CalibrationTrace

    mock_traces = [
        CalibrationTrace(
            trace_id=f"t_{i}",
            decision_point="decision_node:text_response",
            predicted_utility=1.0,
            actual_outcome=0.5,
            prediction_error=0.5,
            timestamp=0.0,
            metadata={},
        )
        for i in range(15)
    ]
    node.calibrator.get_traces = lambda: mock_traces
    node.calibrator.get_utility_percentiles = lambda: (0.9, 0.8, 0.7)

    import random

    original_randint = random.randint
    random.randint = lambda a, b: 0
    try:
        assert node.S_ctrl == 0.0

        node.decision_count = 12
        node._next_observe_tick = 13
        await node._arbitrate_utility(plan, make_msg(), {})
        assert abs(node.S_ctrl - 0.15) < 1e-5

        node.smoothed_high = 0.81
        node.decision_count = 13
        node._next_observe_tick = 14
        await node._arbitrate_utility(plan, make_msg(), {})
        assert abs(node.smoothed_high - 0.81) < 1e-5

    finally:
        random.randint = original_randint
