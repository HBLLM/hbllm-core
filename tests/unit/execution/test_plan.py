"""Tests for ExecutionPlan, ExecutionRequest, and related types."""

from __future__ import annotations

import pytest

from hbllm.execution.payload import ExecutionPayload
from hbllm.execution.plan import (
    ExecutionConstraints,
    ExecutionPlan,
    ExecutionRequest,
    TaskType,
)


class TestTaskType:
    def test_inference_types(self) -> None:
        assert TaskType.TEXT_GENERATION == "text_generation"
        assert TaskType.JSON_GENERATION == "json_generation"

    def test_training_types(self) -> None:
        assert TaskType.LORA_TRAINING == "lora_training"
        assert TaskType.DPO_TRAINING == "dpo_training"

    def test_future_types(self) -> None:
        assert TaskType.VISION == "vision"
        assert TaskType.CODE_EXECUTION == "code_execution"


class TestExecutionConstraints:
    def test_defaults(self) -> None:
        c = ExecutionConstraints()
        assert c.max_tokens == 4096
        assert c.max_latency_ms is None
        assert c.require_streaming is False
        assert c.require_json is False
        assert c.required_capabilities == ()

    def test_frozen(self) -> None:
        c = ExecutionConstraints(max_tokens=1024)
        with pytest.raises(AttributeError):
            c.max_tokens = 2048  # type: ignore[misc]

    def test_custom_constraints(self) -> None:
        c = ExecutionConstraints(
            max_tokens=512,
            max_latency_ms=100,
            require_streaming=True,
            required_capabilities=("json_mode", "streaming"),
        )
        assert c.max_tokens == 512
        assert c.max_latency_ms == 100
        assert "json_mode" in c.required_capabilities


class TestExecutionRequest:
    def test_minimal_request(self) -> None:
        payload = ExecutionPayload.from_prompt("Hello")
        req = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=payload,
        )
        assert req.task_type == TaskType.TEXT_GENERATION
        assert req.tenant_id is None
        assert req.session_id is None

    def test_frozen(self) -> None:
        payload = ExecutionPayload.from_prompt("Hello")
        req = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=payload,
        )
        with pytest.raises(AttributeError):
            req.tenant_id = "test"  # type: ignore[misc]

    def test_no_cognitive_metadata(self) -> None:
        """ExecutionRequest has NO cognitive metadata fields."""
        payload = ExecutionPayload.from_prompt("Hello")
        req = ExecutionRequest(
            task_type=TaskType.TEXT_GENERATION,
            payload=payload,
        )
        # These should NOT exist as attributes
        assert not hasattr(req, "domain")
        assert not hasattr(req, "style")
        assert not hasattr(req, "persona")
        assert not hasattr(req, "cognitive_metadata")


class TestExecutionPlan:
    def test_has_identity(self) -> None:
        plan = ExecutionPlan()
        assert plan.plan_id is not None
        assert len(plan.plan_id) > 0
        assert plan.created_at is not None
        assert plan.version == 1
        assert plan.parent_plan is None

    def test_unique_ids(self) -> None:
        plan1 = ExecutionPlan()
        plan2 = ExecutionPlan()
        assert plan1.plan_id != plan2.plan_id

    def test_frozen(self) -> None:
        plan = ExecutionPlan()
        with pytest.raises(AttributeError):
            plan.provider = "openai"  # type: ignore[misc]

    def test_defaults(self) -> None:
        plan = ExecutionPlan()
        assert plan.task_type == TaskType.TEXT_GENERATION
        assert plan.runtime == "text"
        assert plan.provider == "local"
        assert plan.temperature == 0.7
        assert plan.streaming is False
        assert plan.modifiers == ()

    def test_retry(self) -> None:
        plan = ExecutionPlan(provider="openai")
        retry = plan.with_retry()
        assert retry.plan_id == plan.plan_id  # Same plan
        assert retry.version == plan.version + 1  # Incremented
        assert retry.provider == "openai"  # Preserved
        assert retry.created_at != plan.created_at  # New timestamp

    def test_fork(self) -> None:
        plan = ExecutionPlan(provider="local")
        fork = plan.with_fork(provider="openai")
        assert fork.plan_id != plan.plan_id  # New identity
        assert fork.parent_plan == plan.plan_id  # Parent link
        assert fork.provider == "openai"  # Overridden
        assert fork.version == 1  # Reset
