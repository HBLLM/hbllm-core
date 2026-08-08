"""
Generation Modifier Protocol & Modifier Pipeline.

Modifiers change how the LLM generates text — style, not substance.

Full lifecycle (7 hooks):
    before_context → before_prompt → before_generation →
    [provider.generate()] →
    after_generation → after_validation → cleanup

ModifierPipeline composes multiple modifiers (LoRA + Grammar +
Safety + Watermark) simultaneously, executed in priority order.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

from hbllm.execution.plan import ExecutionPlan

logger = logging.getLogger(__name__)


@runtime_checkable
class GenerationModifier(Protocol):
    """
    Protocol for generation modifiers.

    Modifiers are composable via ModifierPipeline and execute
    in priority order. Each modifier can hook into multiple
    lifecycle points.

    A modifier changes how text is generated — never what.
    """

    @property
    def name(self) -> str:
        """Unique modifier name."""
        ...

    @property
    def modifier_type(self) -> str:
        """Type identifier: 'lora', 'prompt', 'grammar', 'safety', 'none'."""
        ...

    def priority(self) -> int:
        """Higher priority → earlier in pipeline. Default 0."""
        ...

    def supports(self, model_id: str) -> bool:
        """Check if this modifier is compatible with the given model."""
        ...

    async def is_available(self) -> bool:
        """Check if this modifier can currently be activated."""
        ...

    # ── Full Lifecycle (7 hooks) ──────────────────────────────

    async def before_context(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Modify the execution plan before any generation begins."""
        ...

    async def before_prompt(self, prompt: str, plan: ExecutionPlan) -> str:
        """Transform the prompt before sending to the model."""
        ...

    async def before_generation(self, plan: ExecutionPlan) -> None:
        """Activate state before model inference (e.g. set LoRA weights)."""
        ...

    async def after_generation(self, text: str, plan: ExecutionPlan) -> str:
        """Transform generated text after model output."""
        ...

    async def after_validation(self, text: str, plan: ExecutionPlan) -> str:
        """Post-validation hook (safety checks, watermarking)."""
        ...

    async def cleanup(self) -> None:
        """Release resources after generation completes."""
        ...

    def metrics(self) -> dict[str, Any]:
        """Report modifier-specific metrics for telemetry."""
        ...


class ModifierPipeline:
    """
    Composable pipeline of modifiers.

    Supports LoRA + Grammar + Safety + Watermark simultaneously.
    Modifiers execute in priority order (highest first).

    Usage:
        pipeline = ModifierPipeline()
        pipeline.add(lora_modifier)
        pipeline.add(grammar_modifier)
        pipeline.add(safety_modifier)

        plan = await pipeline.run_before_context(plan)
        prompt = await pipeline.run_before_prompt(prompt, plan)
        await pipeline.run_before_generation(plan)
        # ... provider.generate() ...
        text = await pipeline.run_after_generation(text, plan)
        text = await pipeline.run_after_validation(text, plan)
        await pipeline.cleanup()
    """

    def __init__(self) -> None:
        self._modifiers: list[GenerationModifier] = []
        self._total_latency_ms: float = 0.0

    def add(self, modifier: GenerationModifier) -> None:
        """Add a modifier to the pipeline, maintaining priority order."""
        self._modifiers.append(modifier)
        self._modifiers.sort(key=lambda m: m.priority(), reverse=True)

    def remove(self, name: str) -> bool:
        """Remove a modifier by name. Returns True if found and removed."""
        before = len(self._modifiers)
        self._modifiers = [m for m in self._modifiers if m.name != name]
        return len(self._modifiers) < before

    @property
    def modifier_names(self) -> list[str]:
        """List of modifier names in pipeline order."""
        return [m.name for m in self._modifiers]

    @property
    def size(self) -> int:
        return len(self._modifiers)

    @property
    def total_latency_ms(self) -> float:
        return self._total_latency_ms

    async def run_before_context(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Run before_context on all modifiers in order."""
        for modifier in self._modifiers:
            start = time.monotonic()
            plan = await modifier.before_context(plan)
            self._total_latency_ms += (time.monotonic() - start) * 1000
        return plan

    async def run_before_prompt(self, prompt: str, plan: ExecutionPlan) -> str:
        """Run before_prompt on all modifiers in order."""
        for modifier in self._modifiers:
            start = time.monotonic()
            prompt = await modifier.before_prompt(prompt, plan)
            self._total_latency_ms += (time.monotonic() - start) * 1000
        return prompt

    async def run_before_generation(self, plan: ExecutionPlan) -> None:
        """Run before_generation on all modifiers in order."""
        for modifier in self._modifiers:
            start = time.monotonic()
            await modifier.before_generation(plan)
            self._total_latency_ms += (time.monotonic() - start) * 1000

    async def run_after_generation(self, text: str, plan: ExecutionPlan) -> str:
        """Run after_generation on all modifiers in order."""
        for modifier in self._modifiers:
            start = time.monotonic()
            text = await modifier.after_generation(text, plan)
            self._total_latency_ms += (time.monotonic() - start) * 1000
        return text

    async def run_after_validation(self, text: str, plan: ExecutionPlan) -> str:
        """Run after_validation on all modifiers in order."""
        for modifier in self._modifiers:
            start = time.monotonic()
            text = await modifier.after_validation(text, plan)
            self._total_latency_ms += (time.monotonic() - start) * 1000
        return text

    async def cleanup(self) -> None:
        """Run cleanup on all modifiers."""
        for modifier in self._modifiers:
            try:
                await modifier.cleanup()
            except Exception:
                logger.exception("Error during cleanup of modifier '%s'", modifier.name)

    def all_metrics(self) -> dict[str, dict[str, Any]]:
        """Collect metrics from all modifiers."""
        return {m.name: m.metrics() for m in self._modifiers}
