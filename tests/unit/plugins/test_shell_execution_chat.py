"""
Integration and unit tests for safe shell execution in the HBLLM Chat API.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hbllm.brain.control.decision_node import DecisionNode
from hbllm.brain.core.factory import BrainConfig, BrainFactory
from hbllm.brain.planning.action_planner import ActionPlanner
from hbllm.brain.planning.action_schema import ActionType, RiskLevel
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.serving.provider import LLMProvider, LLMResponse


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Bypass manual approval, disable vector routing (ONNX), and disable signature checks in tests."""
    monkeypatch.setenv("HBLLM_REQUIRE_SHELL_APPROVAL", "false")

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


# ── ActionPlanner Unit Tests ──────────────────────────────────────────────────


class TestActionPlannerShellExecution:
    """Verify that ActionPlanner maps shell command structures to SHELL_EXECUTION."""

    def setup_method(self):
        self.planner = ActionPlanner()

    def test_plan_shell_execution_by_thought_type(self):
        plan = self.planner.plan(
            intent="answer",
            thought_type="shell_execution",
            content="ls -la",
            confidence=0.9,
            original_query={"text": "list files"},
        )
        assert plan.action_type == ActionType.SHELL_EXECUTION
        assert plan.content == "ls -la"
        assert plan.risk_level == RiskLevel.HIGH
        assert plan.requires_safety_llm is True

    def test_plan_shell_execution_by_markdown_bash(self):
        content = "Here is the command:\n```bash\nls -lh\n```"
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content=content,
            confidence=0.9,
            original_query={"text": "list files"},
        )
        assert plan.action_type == ActionType.SHELL_EXECUTION
        assert plan.content == "ls -lh"

    def test_plan_shell_execution_by_markdown_sh(self):
        content = "```sh\npwd\n```"
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content=content,
            confidence=0.9,
            original_query={"text": "where am I"},
        )
        assert plan.action_type == ActionType.SHELL_EXECUTION
        assert plan.content == "pwd"

    def test_plan_shell_execution_unclosed_markdown(self):
        content = "```bash\nwhoami"
        plan = self.planner.plan(
            intent="answer",
            thought_type="intuition",
            content=content,
            confidence=0.9,
            original_query={"text": "who am I"},
        )
        assert plan.action_type == ActionType.SHELL_EXECUTION
        assert plan.content == "whoami"


# ── DecisionNode Unit Tests ───────────────────────────────────────────────────


class TestDecisionNodeShellExecution:
    """Verify that DecisionNode dispatches and synthesizes SHELL_EXECUTION."""

    @pytest.mark.asyncio
    async def test_exec_shell_execution_success(self, tmp_path):
        bus = InProcessBus()
        await bus.start()

        # Mock HostShellNode receiver/handler
        async def mock_shell_handler(msg: Message) -> Message:
            assert msg.payload["command"] == "echo 'hello'"
            return msg.create_response(
                {"status": "SUCCESS", "output": "hello\n", "error": "", "exit_code": 0}
            )

        await bus.subscribe("action.execute_shell", mock_shell_handler)

        outputs = []

        async def capture_output(msg: Message):
            outputs.append(msg)

        await bus.subscribe("sensory.output", capture_output)

        # Mock LLM for synthesis
        class MockSynthesisLLM:
            async def generate_json(self, prompt):
                return {"safe": True, "reason": "Content is safe"}

            async def generate(self, prompt, **kwargs):
                return f"Synthesized: {prompt}"

        node = DecisionNode(
            node_id="decision_shell", llm=MockSynthesisLLM(), data_dir=str(tmp_path)
        )
        node.utility_engine.weight_risk = 0.0
        node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}
        await node.start(bus)

        try:
            # Setup decision evaluate message with SHELL_EXECUTION
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="workspace_01",
                topic="decision.evaluate",
                payload={
                    "original_query": {"intent": "answer", "text": "run command"},
                    "selected_thought": {
                        "type": "shell_execution",
                        "confidence": 0.9,
                        "content": "echo 'hello'",
                    },
                },
            )

            await node.evaluate_workspace_decision(msg)
            await asyncio.sleep(0.2)

            assert len(outputs) == 1
            assert "Synthesized:" in outputs[0].payload["text"]
            assert "hello\n" in outputs[0].payload["text"]
        finally:
            await node.stop()
            await bus.stop()

    @pytest.mark.asyncio
    async def test_exec_shell_execution_failure(self, tmp_path):
        bus = InProcessBus()
        await bus.start()

        # Mock HostShellNode failure responder
        async def mock_shell_handler(msg: Message) -> Message:
            return msg.create_response(
                {"status": "FAILURE", "output": "", "error": "some_error", "exit_code": 1}
            )

        await bus.subscribe("action.execute_shell", mock_shell_handler)

        outputs = []

        async def capture_output(msg: Message):
            outputs.append(msg)

        await bus.subscribe("sensory.output", capture_output)

        # No LLM -> falls back to text response
        node = DecisionNode(node_id="decision_shell", llm=None, data_dir=str(tmp_path))
        node.utility_engine.weight_risk = 0.0
        node.calibrator.get_calibration_readiness = lambda: {"bootstrap_active": False}
        await node.start(bus)

        try:
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="workspace_01",
                topic="decision.evaluate",
                payload={
                    "original_query": {"intent": "answer", "text": "run bad command"},
                    "selected_thought": {
                        "type": "shell_execution",
                        "confidence": 0.9,
                        "content": "false",
                    },
                },
            )

            await node.evaluate_workspace_decision(msg)
            await asyncio.sleep(0.2)

            assert len(outputs) == 1
            assert "Command exited with code 1" in outputs[0].payload["text"]
            assert "some_error" in outputs[0].payload["text"]
        finally:
            await node.stop()
            await bus.stop()


# ── End-to-End Brain Loop Test ────────────────────────────────────────────────


class ShellExecutionMockProvider(LLMProvider):
    """Mock LLM Provider that maps queries to a shell execution plan and synthesizes output."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        user_msg = messages[-1]["content"] if messages else ""
        user_msg_lower = user_msg.lower()

        # Synthesis
        if "present the results" in user_msg_lower or "executed" in user_msg_lower:
            content = "Shell execution complete. Output: test_file.txt"
        # Router intent classification
        elif "intent classifier" in user_msg_lower or "classify" in user_msg_lower:
            content = '{"domain": "coding", "intent": "code_generation", "confidence": 0.9}'
        # Critic evaluation
        elif "evaluator" in user_msg_lower or "verdict" in user_msg_lower:
            content = '{"verdict": "PASS", "reason": "Response is relevant and safe"}'
        # Decision safety
        elif "safety classifier" in user_msg_lower:
            content = '{"safe": true, "reason": "Content is safe"}'
        # Planner / thought generation
        elif "generate" in user_msg_lower and "thought" in user_msg_lower:
            content = '{"thought": "I will list the active directory using list files:\\n```bash\\necho \'test_file.txt\'\\n```", "score": 0.95}'
        # Score evaluation
        elif "score" in user_msg_lower or "evaluate" in user_msg_lower:
            content = '{"score": 0.9, "explanation": "Good command"}'
        else:
            # For general domain / intuition thoughts, return the bash code thought JSON so it gets picked up
            content = '{"thought": "I will list the active directory using list files:\\n```bash\\necho \'test_file.txt\'\\n```", "score": 0.95}'

        return LLMResponse(
            content=content,
            model="mock-shell-brain",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )

    async def stream(self, messages, max_tokens=1024, temperature=0.7, **kwargs):
        response = await self.generate(messages, max_tokens, temperature, **kwargs)
        yield response.content

    @property
    def name(self) -> str:
        return "mock-shell-brain"


def _test_config(tmp_path, **overrides) -> BrainConfig:
    """Create a test-safe BrainConfig with minimal subsystems and shell injected."""
    defaults = dict(
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
        inject_shell=True,  # Ensure HostShellNode is wired
        total_timeout=5.0,
    )
    defaults.update(overrides)
    return BrainConfig(**defaults)


@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_e2e_shell_execution_flow(tmp_path):
    """Verify full E2E query flow triggers HostShellNode execution and synthesis."""
    provider = ShellExecutionMockProvider()
    brain = await BrainFactory.create(
        provider=provider,
        config=_test_config(tmp_path),
    )

    # Disable risk constraints and bootstrap mode to allow shell execution in tests
    brain.reasoning_core._decision.utility_engine.weight_risk = 0.0
    brain.reasoning_core._decision.calibrator.get_calibration_readiness = lambda: {
        "bootstrap_active": False
    }

    outputs = []
    correlation_id = "e2e_shell_test_corr"

    async def capture_output(msg: Message):
        if msg.correlation_id == correlation_id:
            outputs.append(msg)

    await brain.bus.subscribe("sensory.output", capture_output)

    try:
        # Publish query matching our mock setup
        query = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id="default",
            session_id="session_shell",
            topic="router.query",
            payload={"text": "list the files in the directory"},
            correlation_id=correlation_id,
        )

        await brain.bus.publish("router.query", query)

        # Allow time for routing, planning, and execution
        await asyncio.sleep(4.0)

        assert len(outputs) >= 1, "No output was published to sensory.output"
        payload = outputs[0].payload
        assert "Shell execution complete" in payload["text"]
        assert "test_file.txt" in payload["text"]
        assert payload["source"] == "shell_execution"

    finally:
        await brain.shutdown()
