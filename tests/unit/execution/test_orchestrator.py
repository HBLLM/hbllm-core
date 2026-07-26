"""Tests for ExecutionOrchestrator — end-to-end orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.execution.bus import ExecutionBus
from hbllm.execution.capability import CapabilityResolver, RuntimeCapabilities
from hbllm.execution.orchestrator import ExecutionOrchestrator
from hbllm.execution.payload import ExecutionPayload
from hbllm.execution.plan import ExecutionPlan, ExecutionRequest, TaskType
from hbllm.execution.policy import GenerationPolicy
from hbllm.execution.registry import ProviderRegistry, RuntimeRegistry
from hbllm.execution.result import ExecutionResult


class MockTextRuntime:
    """Mock runtime for testing."""

    def __init__(self, response: str = "orchestrated response") -> None:
        self._response = response
        self.executed_plans: list[ExecutionPlan] = []

    @property
    def runtime_type(self) -> str:
        return "text"

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(streaming=True, json_mode=True)

    def supported_task_types(self) -> list[TaskType]:
        return [TaskType.TEXT_GENERATION, TaskType.TEXT_COMPLETION]

    async def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        self.executed_plans.append(plan)
        return ExecutionResult(content=self._response, plan_id=plan.plan_id)

    async def is_available(self) -> bool:
        return True


class MockProvider:
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        return {"content": "mock"}

    async def is_available(self) -> bool:
        return True


@pytest.fixture()
def orchestrator_setup() -> tuple[ExecutionOrchestrator, MockTextRuntime, ExecutionBus]:
    runtime_registry = RuntimeRegistry()
    mock_runtime = MockTextRuntime()
    runtime_registry.register(mock_runtime)

    provider_registry = ProviderRegistry()
    provider_registry.register(MockProvider())

    policy = GenerationPolicy.default()
    resolver = CapabilityResolver()
    bus = ExecutionBus()

    orchestrator = ExecutionOrchestrator(
        policy=policy,
        capability_resolver=resolver,
        runtime_registry=runtime_registry,
        provider_registry=provider_registry,
        execution_bus=bus,
    )

    return orchestrator, mock_runtime, bus


class TestExecutionOrchestrator:
    @pytest.mark.asyncio()
    async def test_end_to_end(
        self,
        orchestrator_setup: tuple[ExecutionOrchestrator, MockTextRuntime, ExecutionBus],
    ) -> None:
        """Full orchestration: request → plan → bus → runtime → result."""
        orchestrator, mock_runtime, bus = orchestrator_setup

        request = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=ExecutionPayload.from_prompt("Hello world"),
        )

        result = await orchestrator.execute(request)

        assert result.content == "orchestrated response"
        assert len(mock_runtime.executed_plans) == 1

    @pytest.mark.asyncio()
    async def test_plan_has_identity(
        self,
        orchestrator_setup: tuple[ExecutionOrchestrator, MockTextRuntime, ExecutionBus],
    ) -> None:
        """Plans should have unique IDs."""
        orchestrator, mock_runtime, _ = orchestrator_setup

        request = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=ExecutionPayload.from_prompt("Test"),
        )

        plan = await orchestrator.plan(request)
        assert plan.plan_id is not None
        assert len(plan.plan_id) > 0
        assert plan.created_at is not None

    @pytest.mark.asyncio()
    async def test_bus_journals_execution(
        self,
        orchestrator_setup: tuple[ExecutionOrchestrator, MockTextRuntime, ExecutionBus],
    ) -> None:
        """The bus should journal all execution events."""
        orchestrator, _, bus = orchestrator_setup

        request = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=ExecutionPayload.from_prompt("Test"),
        )

        await orchestrator.execute(request)

        # Should have at least submitted + started + completed
        assert bus.journal_size >= 3

    @pytest.mark.asyncio()
    async def test_async_execution(
        self,
        orchestrator_setup: tuple[ExecutionOrchestrator, MockTextRuntime, ExecutionBus],
    ) -> None:
        """Async execution returns a handle for later awaiting."""
        orchestrator, _, bus = orchestrator_setup

        request = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=ExecutionPayload.from_prompt("Async test"),
        )

        handle = await orchestrator.execute_async(request)
        assert handle.plan_id is not None

        result = await bus.wait(handle)
        assert result.content == "orchestrated response"

    @pytest.mark.asyncio()
    async def test_no_cognitive_metadata_in_plan(
        self,
        orchestrator_setup: tuple[ExecutionOrchestrator, MockTextRuntime, ExecutionBus],
    ) -> None:
        """Plans should contain zero cognitive metadata."""
        orchestrator, _, _ = orchestrator_setup

        request = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=ExecutionPayload.from_prompt("Test"),
        )

        plan = await orchestrator.plan(request)

        # Plan should not have cognitive fields
        assert not hasattr(plan, "domain")
        assert not hasattr(plan, "style")
        assert not hasattr(plan, "persona")
        assert not hasattr(plan, "cognitive_metadata")
