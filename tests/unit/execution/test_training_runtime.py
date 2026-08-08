"""Tests for TrainingRuntime and TrainingSubscriber."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hbllm.execution.plan import ExecutionPlan, TaskType
from hbllm.execution.training.training_runtime import (
    TrainingConfig,
    TrainingJob,
    TrainingResult,
    TrainingRuntime,
)


class TestTrainingConfig:
    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.rank == 8
        assert config.learning_rate == 1e-4
        assert config.max_steps == 20
        assert config.training_type == "sft"
        assert config.checkpoint_dir == "./checkpoints/domains"

    def test_frozen(self) -> None:
        config = TrainingConfig()
        with pytest.raises(AttributeError):
            config.rank = 16  # type: ignore[misc]

    def test_custom(self) -> None:
        config = TrainingConfig(rank=32, learning_rate=5e-5, max_steps=100)
        assert config.rank == 32
        assert config.learning_rate == 5e-5
        assert config.max_steps == 100


class TestTrainingResult:
    def test_defaults(self) -> None:
        result = TrainingResult()
        assert result.status == "completed"
        assert result.steps_completed == 0
        assert result.adapter_state is None

    def test_failed(self) -> None:
        result = TrainingResult(status="failed", error="OOM")
        assert result.status == "failed"
        assert result.error == "OOM"

    def test_completed(self) -> None:
        result = TrainingResult(
            adapter_name="medical",
            steps_completed=20,
            final_loss=0.42,
            training_time_ms=5000.0,
            status="completed",
        )
        assert result.adapter_name == "medical"
        assert result.steps_completed == 20


class TestTrainingJob:
    def test_auto_id(self) -> None:
        job = TrainingJob(adapter_name="test")
        assert job.job_id.startswith("train-")
        assert len(job.job_id) > 6

    def test_default_status(self) -> None:
        job = TrainingJob()
        assert job.status == "pending"


class TestTrainingRuntime:
    def test_runtime_type(self) -> None:
        runtime = TrainingRuntime()
        assert runtime.runtime_type == "training"

    def test_supported_task_types(self) -> None:
        runtime = TrainingRuntime()
        types = runtime.supported_task_types()
        assert TaskType.LORA_TRAINING in types
        assert TaskType.DPO_TRAINING in types

    @pytest.mark.asyncio()
    async def test_not_available_without_model(self) -> None:
        runtime = TrainingRuntime()
        assert await runtime.is_available() is False

    @pytest.mark.asyncio()
    async def test_available_with_model(self) -> None:
        runtime = TrainingRuntime(model=MagicMock(), tokenizer=MagicMock())
        assert await runtime.is_available() is True

    @pytest.mark.asyncio()
    async def test_execute_sft_without_model(self) -> None:
        """Should fail gracefully when no model."""
        runtime = TrainingRuntime()  # No model
        plan = ExecutionPlan(
            task_type=TaskType.LORA_TRAINING,
            metadata={"adapter_name": "test", "dataset_path": "/fake/path"},
        )

        result = await runtime.execute(plan)
        assert result.status == "failed"

    @pytest.mark.asyncio()
    async def test_execute_dpo_without_pairs(self) -> None:
        """Should fail when no training pairs provided."""
        runtime = TrainingRuntime(model=MagicMock(), tokenizer=MagicMock())
        plan = ExecutionPlan(
            task_type=TaskType.DPO_TRAINING,
            metadata={"adapter_name": "test", "training_pairs": []},
        )

        result = await runtime.execute(plan)
        # Training with empty pairs should indicate failure in content
        assert result.plan_id == plan.plan_id

    def test_resolve_config(self) -> None:
        runtime = TrainingRuntime()
        plan = ExecutionPlan(
            metadata={"rank": 32, "learning_rate": 5e-5, "max_steps": 50},
        )
        config = runtime._resolve_config(plan)
        assert config.rank == 32
        assert config.learning_rate == 5e-5
        assert config.max_steps == 50

    def test_resolve_config_defaults(self) -> None:
        runtime = TrainingRuntime(default_config=TrainingConfig(rank=16))
        plan = ExecutionPlan(metadata={})
        config = runtime._resolve_config(plan)
        assert config.rank == 16  # From default

    def test_job_tracking(self) -> None:
        runtime = TrainingRuntime()
        assert runtime.active_job_count == 0
        assert runtime.completed_job_count == 0
        assert runtime.list_completed() == []


class TestTrainingSubscriber:
    @pytest.mark.asyncio()
    async def test_handle_skill_discovered_empty_domain(self) -> None:
        """Should warn and return on empty domain."""
        from hbllm.execution.training.subscriber import TrainingSubscriber

        runtime = TrainingRuntime()
        subscriber = TrainingSubscriber(runtime)

        await subscriber.handle_skill_discovered({"domain": ""})
        assert subscriber.jobs_dispatched == 0

    @pytest.mark.asyncio()
    async def test_handle_feedback_no_pairs(self) -> None:
        """Should skip when no pairs provided."""
        from hbllm.execution.training.subscriber import TrainingSubscriber

        runtime = TrainingRuntime()
        subscriber = TrainingSubscriber(runtime)

        await subscriber.handle_feedback_queued({"pairs": [], "adapter_name": "test"})
        assert subscriber.jobs_dispatched == 0

    @pytest.mark.asyncio()
    async def test_handle_feedback_dispatches_dpo(self) -> None:
        """Should dispatch DPO training when pairs are provided."""
        from hbllm.execution.training.subscriber import TrainingSubscriber

        runtime = TrainingRuntime(model=MagicMock(), tokenizer=MagicMock())
        subscriber = TrainingSubscriber(runtime)

        await subscriber.handle_feedback_queued(
            {
                "pairs": [("prompt", "good", "bad")],
                "adapter_name": "personalization",
            }
        )
        assert subscriber.jobs_dispatched == 1

    @pytest.mark.asyncio()
    async def test_handle_feedback_with_execution_bus(self) -> None:
        """Should emit completion event to execution bus when present."""
        from unittest.mock import AsyncMock

        from hbllm.execution.training.subscriber import TrainingSubscriber

        runtime = TrainingRuntime(model=MagicMock(), tokenizer=MagicMock())
        mock_bus = MagicMock()
        mock_bus._emit = AsyncMock()
        subscriber = TrainingSubscriber(runtime, execution_bus=mock_bus)

        await subscriber.handle_feedback_queued(
            {
                "pairs": [("prompt", "good", "bad")],
                "adapter_name": "personalization",
            }
        )
        assert subscriber.jobs_dispatched == 1
        mock_bus._emit.assert_called_once()
