"""
Training Subscriber — bridges cognitive bus events to training execution.

Subscribes to:
    - ``skill.discovered`` → dispatches SFT training jobs
    - ``learning.feedback_queued`` → dispatches DPO training jobs

This component sits at the boundary between the cognitive event bus
(hbllm.network) and the Execution OS. It translates cognitive events
into ExecutionRequests and dispatches them to the TrainingRuntime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.execution.plan import ExecutionPlan, TaskType
from hbllm.execution.training.training_runtime import TrainingConfig, TrainingRuntime

if TYPE_CHECKING:
    from hbllm.execution.bus import ExecutionBus

logger = logging.getLogger(__name__)


class TrainingSubscriber:
    """
    Bridges cognitive bus events to training execution.

    This is NOT a cognitive node — it's an execution adapter.
    It translates events into training plans.
    """

    def __init__(
        self,
        training_runtime: TrainingRuntime,
        execution_bus: ExecutionBus | None = None,
        default_config: TrainingConfig | None = None,
    ) -> None:
        self._runtime = training_runtime
        self._bus = execution_bus
        self._config = default_config or TrainingConfig()
        self._jobs_dispatched = 0

    async def handle_skill_discovered(self, event_payload: dict[str, Any]) -> None:
        """
        Handle a skill.discovered event from the SpawnerNode.

        Translates the cognitive event into an SFT training plan.
        """
        domain = event_payload.get("domain", "")
        topic = event_payload.get("topic", domain)
        suggested_rank = event_payload.get("suggested_rank", self._config.rank)

        if not domain:
            logger.warning("[TrainingSubscriber] skill.discovered with empty domain")
            return

        logger.info(
            "[TrainingSubscriber] Dispatching SFT training for '%s' (rank=%d)",
            domain,
            suggested_rank,
        )

        # First, generate synthetic data
        dataset_path = await self._generate_dataset(topic)
        if not dataset_path:
            logger.error("[TrainingSubscriber] Failed to generate dataset for '%s'", domain)
            return

        # Build execution plan
        plan = ExecutionPlan(
            task_type=TaskType.LORA_TRAINING,
            metadata={
                "adapter_name": domain,
                "dataset_path": dataset_path,
                "rank": suggested_rank,
                "training_type": "sft",
            },
        )

        # Execute via runtime
        result = await self._runtime.execute(plan)
        self._jobs_dispatched += 1

        logger.info(
            "[TrainingSubscriber] Training result for '%s': %s",
            domain,
            result.status,
        )

        # Emit completion event on ExecutionBus
        if self._bus:
            from hbllm.execution.events import ExecutionEvent

            event = ExecutionEvent(
                event_type="training.completed",
                plan_id=plan.plan_id,
                data={
                    "domain": domain,
                    "status": result.status,
                    "adapter_name": domain,
                    "metadata": result.metadata,
                },
            )
            self._bus._emit(event)

    async def handle_feedback_queued(self, event_payload: dict[str, Any]) -> None:
        """
        Handle a learning.feedback_queued event from the LearnerNode.

        Translates the cognitive event into a DPO training plan.
        """
        pairs = event_payload.get("pairs", [])
        adapter_name = event_payload.get("adapter_name", "personalization")

        if not pairs:
            logger.debug("[TrainingSubscriber] feedback_queued with no pairs")
            return

        logger.info(
            "[TrainingSubscriber] Dispatching DPO training '%s' (%d pairs)",
            adapter_name,
            len(pairs),
        )

        plan = ExecutionPlan(
            task_type=TaskType.DPO_TRAINING,
            metadata={
                "adapter_name": adapter_name,
                "training_pairs": pairs,
                "training_type": "dpo",
            },
        )

        result = await self._runtime.execute(plan)
        self._jobs_dispatched += 1

        logger.info(
            "[TrainingSubscriber] DPO result for '%s': %s",
            adapter_name,
            result.status,
        )

    async def _generate_dataset(self, topic: str) -> str:
        """Generate synthetic training data for a topic.

        Returns the path to the generated dataset, or empty string on failure.
        """
        import asyncio

        try:
            from hbllm.data.synthesizer import DataSynthesizer

            if self._runtime._model is None or self._runtime._tokenizer is None:
                logger.warning("[TrainingSubscriber] No model/tokenizer for data generation")
                return ""

            synthesizer = DataSynthesizer(
                model=self._runtime._model,
                tokenizer=self._runtime._tokenizer,
            )
            dataset_path = await asyncio.to_thread(
                synthesizer.generate_dataset,
                topic=topic,
                num_samples=10,
            )
            return str(dataset_path)

        except Exception as e:
            logger.error("[TrainingSubscriber] Dataset generation failed: %s", e)
            return ""

    @property
    def jobs_dispatched(self) -> int:
        return self._jobs_dispatched
