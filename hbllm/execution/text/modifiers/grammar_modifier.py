"""
Grammar Modifier — constrains generation to a grammar/schema.

Enforces structured output formats (JSON, YAML, etc.) by:
1. Injecting format instructions in ``before_prompt``
2. Validating output structure in ``after_validation``

Works with both provider-native JSON modes and manual validation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hbllm.execution.plan import ExecutionPlan

logger = logging.getLogger(__name__)


class GrammarModifier:
    """
    Constrains generation to structured output formats.

    Currently supports JSON schema validation. Future: YAML, XML,
    custom grammars (GBNF), regex constraints.
    """

    def __init__(
        self,
        schema: dict[str, Any] | None = None,
        format_type: str = "json",
        strict: bool = False,
    ) -> None:
        self._schema = schema
        self._format_type = format_type
        self._strict = strict
        self._validation_errors: list[str] = []

    @property
    def name(self) -> str:
        return f"grammar-{self._format_type}"

    @property
    def modifier_type(self) -> str:
        return "grammar"

    def priority(self) -> int:
        return 75  # Between LoRA (100) and prompt (50)

    def supports(self, model_id: str) -> bool:
        return True

    async def is_available(self) -> bool:
        return True

    async def before_context(self, plan: ExecutionPlan) -> ExecutionPlan:
        return plan

    async def before_prompt(self, prompt: str, plan: ExecutionPlan) -> str:
        """Inject format instructions into the prompt."""
        if self._format_type == "json":
            instruction = "\n\nRespond with ONLY a valid JSON object, no other text."
            if self._schema:
                schema_str = json.dumps(self._schema, indent=2)
                instruction += f"\n\nFollow this JSON schema:\n```json\n{schema_str}\n```"
            return prompt + instruction
        return prompt

    async def before_generation(self, plan: ExecutionPlan) -> None:
        self._validation_errors.clear()

    async def after_generation(self, text: str, plan: ExecutionPlan) -> str:
        return text

    async def after_validation(self, text: str, plan: ExecutionPlan) -> str:
        """Validate that the output matches the expected format."""
        if self._format_type == "json":
            return self._validate_json(text)
        return text

    async def cleanup(self) -> None:
        pass

    def metrics(self) -> dict[str, Any]:
        return {
            "type": "grammar",
            "format": self._format_type,
            "strict": self._strict,
            "validation_errors": list(self._validation_errors),
        }

    def _validate_json(self, text: str) -> str:
        """Validate and optionally extract JSON from text."""
        # Try to parse the raw text as JSON
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        import re

        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the text
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Validation failed
        error = f"Output is not valid {self._format_type}"
        self._validation_errors.append(error)
        if self._strict:
            logger.warning("Grammar validation failed (strict): %s", error)
        return text
