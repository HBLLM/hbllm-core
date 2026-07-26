"""Tests for TextRuntime — full generation pipeline execution."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.execution.plan import ExecutionPlan, TaskType
from hbllm.execution.text.modifiers.modifier import ModifierPipeline
from hbllm.execution.text.modifiers.no_modifier import NoModifier
from hbllm.execution.text.modifiers.prompt_modifier import PromptModifier
from hbllm.execution.text.serializer import ExecutionSerializer
from hbllm.execution.text.text_runtime import TextRuntime


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "Hello from mock") -> None:
        self._response = response
        self.last_call: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "mock"

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.last_call = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return {
            "content": self._response,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "finish_reason": "stop",
        }

    async def is_available(self) -> bool:
        return True


@pytest.fixture()
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture()
def runtime(mock_provider: MockProvider) -> TextRuntime:
    pipeline = ModifierPipeline()
    pipeline.add(NoModifier())
    return TextRuntime(
        providers={"mock": mock_provider},
        pipeline=pipeline,
        serializer=ExecutionSerializer(),
    )


class TestTextRuntime:
    def test_runtime_type(self, runtime: TextRuntime) -> None:
        assert runtime.runtime_type == "text"

    def test_supported_task_types(self, runtime: TextRuntime) -> None:
        types = runtime.supported_task_types()
        assert TaskType.TEXT_GENERATION in types
        assert TaskType.TEXT_COMPLETION in types
        assert TaskType.JSON_GENERATION in types

    @pytest.mark.asyncio()
    async def test_is_available(self, runtime: TextRuntime) -> None:
        assert await runtime.is_available() is True

    @pytest.mark.asyncio()
    async def test_basic_execution(self, runtime: TextRuntime, mock_provider: MockProvider) -> None:
        plan = ExecutionPlan(
            provider="mock",
            payload_messages=(("user", "Hello world"),),
            max_tokens=100,
        )

        result = await runtime.execute(plan)

        assert result.status == "completed"
        assert result.content == "Hello from mock"
        assert result.plan_id == plan.plan_id
        assert result.usage.total_tokens == 15
        assert result.metrics.latency_ms > 0

    @pytest.mark.asyncio()
    async def test_execution_with_system_message(
        self, runtime: TextRuntime, mock_provider: MockProvider
    ) -> None:
        plan = ExecutionPlan(
            provider="mock",
            payload_messages=(
                ("system", "Be helpful"),
                ("user", "Hi"),
            ),
        )

        result = await runtime.execute(plan)
        assert result.status == "completed"

        # Provider should have received the messages
        assert mock_provider.last_call is not None

    @pytest.mark.asyncio()
    async def test_modifier_pipeline_runs(self) -> None:
        """Modifiers should transform the prompt during execution."""
        provider = MockProvider()
        pipeline = ModifierPipeline()
        pipeline.add(PromptModifier(style="formal"))

        runtime = TextRuntime(
            providers={"mock": provider},
            pipeline=pipeline,
        )

        plan = ExecutionPlan(
            provider="mock",
            payload_messages=(("user", "Hello"),),
        )

        result = await runtime.execute(plan)
        assert result.status == "completed"
        assert result.metrics.modifiers_applied == ["prompt-formal"]

    @pytest.mark.asyncio()
    async def test_missing_provider_fails(self, runtime: TextRuntime) -> None:
        plan = ExecutionPlan(
            provider="nonexistent",
            payload_messages=(("user", "Hello"),),
        )

        result = await runtime.execute(plan)
        assert result.status == "failed"

    @pytest.mark.asyncio()
    async def test_provider_metadata_separated(self, runtime: TextRuntime) -> None:
        """Provider metadata should be in ProviderMetadata, not polluting result."""
        plan = ExecutionPlan(
            provider="mock",
            payload_messages=(("user", "Hello"),),
        )

        result = await runtime.execute(plan)

        assert result.provider_metadata is not None
        assert result.provider_metadata.provider == "mock"
        assert result.provider_metadata.finish_reason == "stop"
        # Raw provider response is preserved
        assert result.provider_metadata.raw is not None

    @pytest.mark.asyncio()
    async def test_empty_runtime_not_available(self) -> None:
        runtime = TextRuntime(providers={})
        assert await runtime.is_available() is False


class TestExecutionSerializer:
    @pytest.mark.asyncio()
    async def test_serialize_local(self) -> None:
        serializer = ExecutionSerializer()
        result = await serializer.serialize_prompt(
            (("system", "Be helpful"), ("user", "Hi")),
            "local",
        )
        assert isinstance(result, str)
        assert "Be helpful" in result
        assert "Hi" in result

    @pytest.mark.asyncio()
    async def test_serialize_openai(self) -> None:
        serializer = ExecutionSerializer()
        result = await serializer.serialize_prompt(
            (("system", "Be helpful"), ("user", "Hi")),
            "openai",
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Hi"
