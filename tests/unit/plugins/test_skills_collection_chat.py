"""
Integration test for skills collection and execution via the Chat API.
"""

from __future__ import annotations

from typing import Any

import httpx
import jwt
import pytest

from hbllm.brain.factory import BrainConfig, BrainFactory
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message
from hbllm.serving.api import _state, app
from hbllm.serving.provider import LLMProvider, LLMResponse


class SkillCallMockProvider(LLMProvider):
    """Mock LLM Provider that outputs skill_calls and synthesizes output."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        user_msg = messages[-1]["content"] if messages else ""
        user_msg_lower = user_msg.lower()
        print(f"\n--- [SkillCallMockProvider.generate] ---\nUser msg: {user_msg}\n")

        # Critic evaluation (check this first to avoid collisions with safety/evaluator prompts)
        if (
            "evaluator" in user_msg_lower
            or "violations" in user_msg_lower
            or "verdict" in user_msg_lower
        ):
            content = '{"violations": [], "rationale": "All clear"}'
        # Synthesis path
        elif (
            "present the results" in user_msg_lower
            or "executed" in user_msg_lower
            or "tool output" in user_msg_lower
        ):
            content = "Successfully completed Deploy Website skill. Output: Deploy successful!"
        # Router intent classification
        elif "intent classifier" in user_msg_lower or "classify" in user_msg_lower:
            content = '{"domain": "coding", "intent": "code_generation", "confidence": 0.9}'
        # Decision safety
        elif "safety classifier" in user_msg_lower:
            content = '{"safe": true, "reason": "Content is safe"}'
        # Conclusion after observation
        elif "observation" in user_msg_lower or "sil perfectly executed" in user_msg_lower:
            content = '{"thought": "Successfully completed Deploy Website skill. Output: Deploy successful!", "score": 0.95}'
        # ExpressionStream prompts (deep path: "generating one section",
        # shallow: "RENDER", broca: "TYPE:") — must check BEFORE planner
        # because ExpressionStream prompts also contain "generate" + "thought"
        elif (
            "generating one section" in user_msg_lower
            or "render" in user_msg_lower
            or user_msg_lower.startswith("type:")
        ):
            content = "Successfully completed Deploy Website skill. Output: Deploy successful!"
        # Planner / thought generation / general evaluate
        elif "generate" in user_msg_lower and "thought" in user_msg_lower:
            content = '{"thought": "I will execute the Deploy Website skill:\\n<skill_call task=\\"Deploy Website\\">deploy args</skill_call>", "score": 0.95}'
        else:
            # Default to skill call thought
            content = '{"thought": "I will execute the Deploy Website skill:\\n<skill_call task=\\"Deploy Website\\">deploy args</skill_call>", "score": 0.95}'

        return LLMResponse(
            content=content,
            model="mock-skill-brain",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )

    async def stream(self, messages, max_tokens=1024, temperature=0.7, **kwargs):
        response = await self.generate(messages, max_tokens, temperature, **kwargs)
        yield response.content

    @property
    def name(self) -> str:
        return "mock-skill-brain"


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Bypass safety checks and signature verification in tests."""
    monkeypatch.setenv("HBLLM_REQUIRE_SHELL_APPROVAL", "false")

    # Patch RouterNode.__init__ to disable vector routing
    import hbllm.brain.router_node

    original_init = hbllm.brain.router_node.RouterNode.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["use_vectors"] = False
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(hbllm.brain.router_node.RouterNode, "__init__", patched_init)

    # Patch ServiceRegistry.verify_message to bypass cryptographic validation
    import hbllm.network.registry

    async def mock_verify_message(*args, **kwargs):
        return True

    monkeypatch.setattr(
        hbllm.network.registry.ServiceRegistry, "verify_message", mock_verify_message
    )

    # Patch WorkspaceNode.__init__ to set a short thinking deadline for fast testing
    import hbllm.brain.workspace_node

    original_ws_init = hbllm.brain.workspace_node.WorkspaceNode.__init__

    def patched_ws_init(self, *args, **kwargs):
        kwargs["thinking_deadline"] = 1.0
        original_ws_init(self, *args, **kwargs)

    monkeypatch.setattr(hbllm.brain.workspace_node.WorkspaceNode, "__init__", patched_ws_init)

    # Patch WorkspaceNode._finalize_board to prioritize GoT thoughts
    original_finalize = hbllm.brain.workspace_node.WorkspaceNode._finalize_board

    async def patched_finalize(self, corr_id):
        board = self.blackboards.get(corr_id)
        if board:
            for t in board["thoughts"]:
                if t.get("type") == "graph_of_thoughts":
                    t["confidence"] = 1.0
        await original_finalize(self, corr_id)

    monkeypatch.setattr(
        hbllm.brain.workspace_node.WorkspaceNode, "_finalize_board", patched_finalize
    )

    # Patch PlannerNode.handle_message to strip GoT stats prefix for JSON parsing
    import hbllm.brain.planner_node

    original_handle_message = hbllm.brain.planner_node.PlannerNode.handle_message

    async def patched_handle_message(self, message):
        resp = await original_handle_message(self, message)
        if resp and "text" in resp.payload:
            text = resp.payload["text"]
            if "[MCTS Planner]" in text:
                parts = text.split("\n\n", 1)
                if len(parts) > 1:
                    resp.payload["text"] = parts[1]
        return resp

    monkeypatch.setattr(
        hbllm.brain.planner_node.PlannerNode, "handle_message", patched_handle_message
    )


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
        inject_persistence=True,  # Ensures skill registry and SIL are wired
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
async def test_skills_collection_via_chat_api(tmp_path, monkeypatch):
    """Verify that Chat API correctly resolves and executes skills from procedural memory."""
    jwt_secret = "test_secret_key_for_jwt_testing_32ch"
    monkeypatch.setenv("HBLLM_JWT_SECRET", jwt_secret)
    monkeypatch.setenv("HBLLM_ENV", "production")
    monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")

    # Override JWT secret in API middleware
    for middleware in app.user_middleware:
        if middleware.cls.__name__ == "JWTAuthMiddleware":
            middleware.kwargs["secret_key"] = jwt_secret

    # Setup the mock provider
    mock_provider = SkillCallMockProvider()

    # Completely patch the app lifespan to run a minimal test brain
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_lifespan(app):
        bus = InProcessBus()
        await bus.start()
        cfg = _test_config(tmp_path)
        brain = await BrainFactory.create(provider=mock_provider, config=cfg, bus=bus)
        _state["brain"] = brain
        _state["config"] = cfg
        _state["bus"] = bus
        _state["mode"] = "full"
        try:
            yield
        finally:
            if brain:
                await brain.shutdown()
            await bus.stop()

    monkeypatch.setattr(app.router, "lifespan_context", mock_lifespan)

    # Use AsyncClient to boot the app lifespan and make async HTTP requests
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # The brain has booted! Let's get the active bus and register a mock execute_code handler
            bus = _state["bus"]
            brain = _state["brain"]

            # Add the skill in the database
            skill = brain.skill_registry.extract_and_store(
                "Deploy Website",
                [{"action": "print('Deploy successful!')"}],
                [],
                True,
                "general",
                tenant_id="tenant_test",
            )
            assert skill is not None

            # Boost its confidence so it's selected by SIL
            for _ in range(5):
                brain.skill_registry.record_execution(skill.skill_id, True, 10.0)

            # Register mock responders for execution and simulation on the test bus
            async def mock_execute(msg: Message) -> Message:
                return msg.create_response({"status": "SUCCESS", "output": "Deploy successful!"})

            await bus.subscribe("action.execute_code", mock_execute)

            async def mock_simulate(msg: Message) -> Message:
                return msg.create_response({"status": "SUCCESS", "prediction": "SUCCESS"})

            await bus.subscribe("workspace.simulate", mock_simulate)

            # Generate Bearer token for authentication
            import time

            token = jwt.encode(
                {"tenant_id": "tenant_test", "user_id": "user_1", "exp": int(time.time()) + 3600},
                jwt_secret,
                algorithm="HS256",
            )
            headers = {"Authorization": f"Bearer {token}"}

            # Send request to `/v1/chat`
            chat_request = {
                "text": "run the Deploy Website task",
                "session_id": "test_skills_session",
                "tenant_id": "tenant_test",
            }

            response = await client.post("/v1/chat", json=chat_request, headers=headers)

            assert response.status_code == 200
            response_json = response.json()
            assert "response_text" in response_json
            assert "Successfully completed Deploy Website skill" in response_json["response_text"]
