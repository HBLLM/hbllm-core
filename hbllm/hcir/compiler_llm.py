"""
Structured LLM Compiler Frontend — parses structured LLM output into SemanticAST & HCIR Bytecode.

Demotes the LLM from runtime decision-maker to structured compiler frontend:

    User Prompt / NL
           ↓
    LLM Structured Output (JSON / Pydantic Schema)
           ↓
    LLMCompilerFrontend
           ↓
    SemanticAST
           ↓
    HCIRCompiler / Optimizer
           ↓
    HCIR Bytecode Stream
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from hbllm.hcir.bytecode import InstructionStream
from hbllm.hcir.compiler import HCIRCompiler, IntentType, SemanticAST, SemanticSlot

logger = logging.getLogger(__name__)


class StructuredIntentPayload(BaseModel):
    """Pydantic schema enforced on LLM structured JSON output.

    The LLM outputs strictly this schema rather than unstructured text.
    """

    intent: str = Field(
        ...,
        description="Intent type: investigate, create, modify, delete, query, plan, learn, simulate, monitor",
    )
    subject: str = Field(..., description="Primary subject of the request")
    action: str = Field(default="", description="Specific action or goal target")
    slots: dict[str, Any] = Field(
        default_factory=dict, description="Extracted key-value slots"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Resource or policy constraints"
    )
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class LLMCompilerFrontend:
    """Translates structured LLM JSON/Dict responses into verified HCIR InstructionStreams.

    Usage::

        frontend = LLMCompilerFrontend()
        # LLM generated json_response
        payload_json = '{"intent": "plan", "subject": "solar dehydrator", "action": "design"}'
        stream = frontend.compile_json(payload_json, author="qwen_llm")
    """

    def __init__(self, tenant_id: str = "default") -> None:
        self._compiler = HCIRCompiler(tenant_id=tenant_id)

    def parse_payload(self, data: dict[str, Any] | str) -> StructuredIntentPayload:
        """Parse dict or JSON string into a validated StructuredIntentPayload."""
        if isinstance(data, str):
            parsed_dict = json.loads(data)
        else:
            parsed_dict = data
        return StructuredIntentPayload.model_validate(parsed_dict)

    def payload_to_ast(self, payload: StructuredIntentPayload) -> SemanticAST:
        """Convert a validated StructuredIntentPayload into a SemanticAST."""
        intent_str = payload.intent.lower().strip()
        try:
            intent_enum = IntentType(intent_str)
        except ValueError:
            logger.warning("Unknown intent '%s', falling back to QUERY", intent_str)
            intent_enum = IntentType.QUERY

        slot_objs = [
            SemanticSlot(name=k, value=v) for k, v in payload.slots.items()
        ]

        return SemanticAST(
            intent=intent_enum,
            subject=payload.subject,
            action=payload.action,
            slots=slot_objs,
            constraints=payload.constraints,
            priority=payload.priority,
        )

    def compile_json(
        self, data: dict[str, Any] | str, author: str = "llm_compiler"
    ) -> InstructionStream:
        """Parse structured LLM input and compile to HCIR InstructionStream."""
        payload = self.parse_payload(data)
        ast = self.payload_to_ast(payload)
        return self._compiler.compile(ast, author=author)
