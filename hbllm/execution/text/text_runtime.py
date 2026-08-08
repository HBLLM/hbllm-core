"""
Text Runtime — executes text generation plans.

Does NOT know about domains, styles, personas, or cognitive state.
Receives a frozen ExecutionPlan, creates mutable ExecutionState,
runs the modifier pipeline, calls the provider, returns result.

Full pipeline:
    1. pipeline.run_before_context(plan)
    2. pipeline.run_before_prompt(prompt, plan)
    3. pipeline.run_before_generation(plan)
    4. provider.generate(...)
    5. pipeline.run_after_generation(text, plan)
    6. pipeline.run_after_validation(text, plan)
    7. pipeline.cleanup()
    8. Build ExecutionResult
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.execution.capability import RuntimeCapabilities
from hbllm.execution.plan import ExecutionPlan, TaskType
from hbllm.execution.result import (
    ExecutionMetrics,
    ExecutionResult,
    ProviderMetadata,
    TokenUsage,
)
from hbllm.execution.state import ExecutionState
from hbllm.execution.text.modifiers.modifier import ModifierPipeline
from hbllm.execution.text.serializer import ExecutionSerializer

logger = logging.getLogger(__name__)


class TextRuntime:
    """
    Executes text generation plans.

    Zero cognitive knowledge. Receives a frozen ExecutionPlan,
    creates mutable ExecutionState, runs the modifier pipeline,
    calls the provider, returns a minimal ExecutionResult.
    """

    def __init__(
        self,
        providers: dict[str, Any] | None = None,
        pipeline: ModifierPipeline | None = None,
        serializer: ExecutionSerializer | None = None,
    ) -> None:
        self._providers: dict[str, Any] = providers or {}
        self._pipeline = pipeline or ModifierPipeline()
        self._serializer = serializer or ExecutionSerializer()

    @property
    def runtime_type(self) -> str:
        return "text"

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            streaming=True,
            json_mode=True,
            tool_calls=True,
            max_context=128_000,
            max_output=8192,
            supports_lora=any(m.modifier_type == "lora" for m in self._pipeline._modifiers)
            if self._pipeline._modifiers
            else False,
            supported_modifiers=tuple(m.modifier_type for m in self._pipeline._modifiers),
        )

    def supported_task_types(self) -> list[TaskType]:
        return [
            TaskType.TEXT_GENERATION,
            TaskType.TEXT_COMPLETION,
            TaskType.JSON_GENERATION,
        ]

    async def is_available(self) -> bool:
        return len(self._providers) > 0

    def register_provider(self, name: str, provider: Any) -> None:
        """Register a provider for use by this runtime."""
        self._providers[name] = provider

    async def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        Full text generation pipeline.

        Steps:
            1. Create mutable state
            2. Run modifier pipeline (before_context, before_prompt, before_generation)
            3. Call provider
            4. Run modifier pipeline (after_generation, after_validation)
            5. Cleanup
            6. Build result
        """
        state = ExecutionState(plan=plan)
        state.mark_started()

        try:
            # 1. Run before_context
            modified_plan = await self._pipeline.run_before_context(plan)

            # 2. Serialize payload
            prompt = await self._serializer.serialize_prompt(
                modified_plan.payload_messages,
                modified_plan.provider,
            )

            # 3. Run before_prompt
            if isinstance(prompt, str):
                prompt = await self._pipeline.run_before_prompt(prompt, modified_plan)

            # 4. Run before_generation
            await self._pipeline.run_before_generation(modified_plan)
            state.active_modifiers = list(self._pipeline.modifier_names)

            # 5. Call provider
            provider_start = time.monotonic()
            raw_result = await self._call_provider(modified_plan, prompt)
            state.provider_latency_ms = (time.monotonic() - provider_start) * 1000

            # 6. Extract text from provider result
            text = self._extract_text(raw_result)

            # 7. Run after_generation
            text = await self._pipeline.run_after_generation(text, modified_plan)

            # 8. Run after_validation
            text = await self._pipeline.run_after_validation(text, modified_plan)

            # 9. Mark completed
            state.mark_completed()

            # 10. Build result
            return ExecutionResult(
                content=text,
                plan_id=plan.plan_id,
                metrics=ExecutionMetrics(
                    latency_ms=state.elapsed_ms,
                    provider_latency_ms=state.provider_latency_ms,
                    modifier_latency_ms=self._pipeline.total_latency_ms,
                    modifiers_applied=list(self._pipeline.modifier_names),
                ),
                usage=self._extract_usage(raw_result),
                status="completed",
                provider_metadata=ProviderMetadata(
                    provider=plan.provider,
                    model=plan.model_id or "",
                    finish_reason=raw_result.get("finish_reason", "stop")
                    if isinstance(raw_result, dict)
                    else "stop",
                    raw=raw_result,
                ),
            )

        except Exception as exc:
            state.mark_failed(str(exc))
            logger.exception("Text generation failed: plan_id=%s", plan.plan_id)
            return ExecutionResult(
                content="",
                plan_id=plan.plan_id,
                metrics=ExecutionMetrics(
                    latency_ms=state.elapsed_ms,
                    modifiers_applied=list(self._pipeline.modifier_names),
                ),
                status="failed",
            )

        finally:
            await self._pipeline.cleanup()

    async def _call_provider(
        self, plan: ExecutionPlan, prompt: str | list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Call the appropriate provider."""
        provider = self._providers.get(plan.provider)

        if provider is None:
            raise RuntimeError(
                f"Provider '{plan.provider}' not registered with TextRuntime. "
                f"Available: {list(self._providers.keys())}"
            )

        # If provider implements the LLMProvider protocol
        if hasattr(provider, "generate"):
            if isinstance(prompt, list):
                return await provider.generate(
                    messages=prompt,
                    model=plan.model_id,
                    temperature=plan.temperature,
                    max_tokens=plan.max_tokens,
                )
            # String prompt → wrap in message
            return await provider.generate(
                messages=[{"role": "user", "content": prompt}],
                model=plan.model_id,
                temperature=plan.temperature,
                max_tokens=plan.max_tokens,
            )

        raise TypeError(f"Provider '{plan.provider}' does not implement generate()")

    def _extract_text(self, result: dict[str, Any] | Any) -> str:
        """Extract generated text from provider result."""
        if isinstance(result, dict):
            # OpenAI-style response
            if "content" in result:
                return str(result["content"])
            if "choices" in result:
                choices = result["choices"]
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    return str(msg.get("content", ""))
            if "text" in result:
                return str(result["text"])
        if isinstance(result, str):
            return result
        return str(result)

    def _extract_usage(self, result: dict[str, Any] | Any) -> TokenUsage:
        """Extract token usage from provider result."""
        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
            return TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
        return TokenUsage()
