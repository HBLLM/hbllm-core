"""
No Modifier — default pass-through.

Applied when no modification is needed. All lifecycle hooks
are identity operations.
"""

from __future__ import annotations

from typing import Any

from hbllm.execution.plan import ExecutionPlan


class NoModifier:
    """Default pass-through modifier. No modification applied."""

    @property
    def name(self) -> str:
        return "none"

    @property
    def modifier_type(self) -> str:
        return "none"

    def priority(self) -> int:
        return 0

    def supports(self, model_id: str) -> bool:
        return True

    async def is_available(self) -> bool:
        return True

    async def before_context(self, plan: ExecutionPlan) -> ExecutionPlan:
        return plan

    async def before_prompt(self, prompt: str, plan: ExecutionPlan) -> str:
        return prompt

    async def before_generation(self, plan: ExecutionPlan) -> None:
        pass

    async def after_generation(self, text: str, plan: ExecutionPlan) -> str:
        return text

    async def after_validation(self, text: str, plan: ExecutionPlan) -> str:
        return text

    async def cleanup(self) -> None:
        pass

    def metrics(self) -> dict[str, Any]:
        return {"type": "none", "active": False}
