"""
Prompt Modifier — style adaptation via system prompt injection.

No model weight changes. Works with any provider (local, API).
Modifies the prompt by injecting style/persona instructions.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.execution.plan import ExecutionPlan

logger = logging.getLogger(__name__)

# ── Built-in style templates ─────────────────────────────────────────────────

_STYLE_TEMPLATES: dict[str, str] = {
    "concise": "Respond concisely. Use short sentences. Avoid unnecessary elaboration.",
    "verbose": "Provide detailed, thorough responses with examples and explanations.",
    "formal": "Use formal, professional language. Avoid colloquialisms and contractions.",
    "casual": "Use a friendly, conversational tone. Be approachable and natural.",
    "technical": "Use precise technical terminology. Include code examples where relevant.",
    "creative": "Be creative and expressive. Use vivid language and original metaphors.",
    "academic": "Write in academic style with citations and structured arguments.",
}


class PromptModifier:
    """
    Style adaptation via system prompt injection.

    Works with any provider — no weight changes needed.
    Injects a style instruction into the prompt during
    the ``before_prompt`` lifecycle hook.
    """

    def __init__(
        self,
        style: str = "concise",
        custom_instruction: str | None = None,
    ) -> None:
        self._style = style
        self._custom_instruction = custom_instruction
        self._instruction = self._resolve_instruction()

    @property
    def name(self) -> str:
        return f"prompt-{self._style}"

    @property
    def modifier_type(self) -> str:
        return "prompt"

    def priority(self) -> int:
        return 50  # Lower than LoRA, higher than grammar

    def supports(self, model_id: str) -> bool:
        return True  # Works with any model

    async def is_available(self) -> bool:
        return True  # Always available

    async def before_context(self, plan: ExecutionPlan) -> ExecutionPlan:
        return plan

    async def before_prompt(self, prompt: str, plan: ExecutionPlan) -> str:
        """Inject style instruction at the beginning of the prompt."""
        if self._instruction:
            return f"[Style: {self._instruction}]\n\n{prompt}"
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
        return {
            "type": "prompt",
            "style": self._style,
            "has_custom_instruction": self._custom_instruction is not None,
        }

    def _resolve_instruction(self) -> str:
        """Resolve the style instruction from template or custom."""
        if self._custom_instruction:
            return self._custom_instruction
        return _STYLE_TEMPLATES.get(self._style, "")
