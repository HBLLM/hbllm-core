"""
HCIR Compiler Frontend — Natural Language → HCIR Bytecode.

Pipeline::

    Natural Language
          ↓
    Semantic AST          (intent extraction)
          ↓
    Intent Graph          (goal/constraint decomposition)
          ↓
    Constraint Injection  (policy/scope/resource bounds)
          ↓
    HCIR IR               (graph operations)
          ↓
    Bytecode              (instruction stream)
          ↓
    Scheduler             (dispatch)

The compiler is itself a cognitive node — it receives user
intent and produces executable HCIR bytecode.  LLM-backed
compilation is one strategy; rule-based compilation is another.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import (
    GoalNode,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Semantic AST — intermediate representation of user intent
# ═══════════════════════════════════════════════════════════════════════════


class IntentType(StrEnum):
    """High-level intent categories extracted from natural language."""

    INVESTIGATE = "investigate"  # "Find out why...", "Analyze..."
    CREATE = "create"  # "Build...", "Generate...", "Write..."
    MODIFY = "modify"  # "Change...", "Update...", "Fix..."
    DELETE = "delete"  # "Remove...", "Delete...", "Clean up..."
    QUERY = "query"  # "What is...", "Show me...", "List..."
    PLAN = "plan"  # "Plan...", "Design...", "How should I..."
    LEARN = "learn"  # "Remember...", "Learn...", "Note that..."
    SIMULATE = "simulate"  # "What if...", "Imagine...", "Try..."
    MONITOR = "monitor"  # "Watch...", "Alert me...", "Track..."


@dataclass
class SemanticSlot:
    """A named semantic slot extracted from user intent."""

    name: str
    value: Any
    confidence: float = 0.8


@dataclass
class SemanticAST:
    """Abstract syntax tree of user intent.

    Represents the structured meaning of a natural language input,
    ready for compilation into HCIR bytecode.

    Example::

        "Find why battery temperature increased"

        SemanticAST(
            intent=IntentType.INVESTIGATE,
            subject="battery_temperature",
            action="find_cause",
            slots=[
                SemanticSlot("metric", "temperature"),
                SemanticSlot("direction", "increased"),
            ],
        )
    """

    intent: IntentType
    subject: str = ""
    action: str = ""
    slots: list[SemanticSlot] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priority: float = 0.5
    raw_text: str = ""

    def get_slot(self, name: str) -> Any | None:
        for slot in self.slots:
            if slot.name == name:
                return slot.value
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Compiler — AST → HCIR Bytecode
# ═══════════════════════════════════════════════════════════════════════════


class HCIRCompiler:
    """Compiles a SemanticAST into an HCIR InstructionStream.

    The compiler maps high-level intents to sequences of
    primitive bytecode instructions.  Each intent type has
    a compilation strategy.

    Usage::

        compiler = HCIRCompiler()
        ast = SemanticAST(
            intent=IntentType.INVESTIGATE,
            subject="battery_temperature",
            action="find_cause",
        )
        stream = compiler.compile(ast, author="user_session")
    """

    def __init__(self, tenant_id: str = "default") -> None:
        self._tenant_id = tenant_id
        # Intent → compilation strategy mapping
        self._strategies: dict[IntentType, Any] = {
            IntentType.INVESTIGATE: self._compile_investigate,
            IntentType.CREATE: self._compile_create,
            IntentType.MODIFY: self._compile_modify,
            IntentType.DELETE: self._compile_delete,
            IntentType.QUERY: self._compile_query,
            IntentType.PLAN: self._compile_plan,
            IntentType.LEARN: self._compile_learn,
            IntentType.SIMULATE: self._compile_simulate,
            IntentType.MONITOR: self._compile_monitor,
        }

    def compile(self, ast: SemanticAST, author: str = "compiler") -> InstructionStream:
        """Compile a SemanticAST into bytecode."""
        strategy = self._strategies.get(ast.intent)
        if strategy is None:
            logger.warning("No compilation strategy for intent: %s", ast.intent)
            return InstructionStream(author=author, description=f"Unknown intent: {ast.intent}")

        stream = strategy(ast, author)
        logger.debug(
            "Compiled '%s' intent → %d instructions",
            ast.intent,
            stream.length,
        )
        return stream

    def compile_text(self, text: str, author: str = "user") -> InstructionStream:
        """Parse natural language prompt into a SemanticAST and compile to InstructionStream."""
        text_lower = text.lower().strip()

        # Heuristic intent parser
        if any(w in text_lower for w in ["find why", "investigate", "analyze", "why did", "diagnose"]):
            intent = IntentType.INVESTIGATE
        elif any(w in text_lower for w in ["plan", "how to", "design", "steps for"]):
            intent = IntentType.PLAN
        elif any(w in text_lower for w in ["create", "build", "generate", "write", "make"]):
            intent = IntentType.CREATE
        elif any(w in text_lower for w in ["modify", "change", "update", "fix"]):
            intent = IntentType.MODIFY
        elif any(w in text_lower for w in ["simulate", "what if", "imagine", "try"]):
            intent = IntentType.SIMULATE
        elif any(w in text_lower for w in ["remember", "learn", "note"]):
            intent = IntentType.LEARN
        else:
            intent = IntentType.QUERY

        ast = SemanticAST(
            intent=intent,
            subject=text[:100],
            action="evaluate_prompt",
            raw_text=text,
        )
        return self.compile(ast, author=author)

    # ── Compilation Strategies ───────────────────────────────────────

    def _compile_investigate(self, ast: SemanticAST, author: str) -> InstructionStream:
        """INVESTIGATE: ASSERT goal → QUERY observations → QUERY procedures → EXECUTE analysis."""
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        stream = InstructionStream(
            author=author,
            description=f"Investigate: {ast.subject}",
        )
        # 1. Assert investigation goal
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=goal_id,
                        description=f"Investigate: {ast.subject} ({ast.action})",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        # 2. Query relevant observations
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={
                    "node_type": "observation",
                    "text_contains": ast.subject,
                },
                cost_estimate=10,
            )
        )
        # 3. Query relevant procedures
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={
                    "node_type": "procedure",
                    "text_contains": ast.action or ast.subject,
                },
                cost_estimate=10,
            )
        )
        # 4. Execute analysis capability
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "causal_analysis",
                    "params": {
                        "subject": ast.subject,
                        "action": ast.action,
                        "goal_id": goal_id,
                    },
                },
                cost_estimate=100,
            )
        )
        return stream

    def _compile_create(self, ast: SemanticAST, author: str) -> InstructionStream:
        """CREATE: ASSERT goal → EXECUTE creation → ASSERT result."""
        stream = InstructionStream(author=author, description=f"Create: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=f"goal_{uuid.uuid4().hex[:8]}",
                        description=f"Create: {ast.subject}",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "content_generation",
                    "params": {"subject": ast.subject, "action": ast.action},
                },
                cost_estimate=200,
            )
        )
        return stream

    def _compile_modify(self, ast: SemanticAST, author: str) -> InstructionStream:
        """MODIFY: QUERY target → EXECUTE modification."""
        stream = InstructionStream(author=author, description=f"Modify: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"text_contains": ast.subject},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "content_modification",
                    "params": {"subject": ast.subject, "action": ast.action},
                },
                cost_estimate=100,
            )
        )
        return stream

    def _compile_delete(self, ast: SemanticAST, author: str) -> InstructionStream:
        """DELETE: QUERY target → RETRACT."""
        stream = InstructionStream(author=author, description=f"Delete: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"text_contains": ast.subject},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.RETRACT,
                params={"node_id": ast.subject},
                cost_estimate=5,
            )
        )
        return stream

    def _compile_query(self, ast: SemanticAST, author: str) -> InstructionStream:
        """QUERY: direct graph query."""
        stream = InstructionStream(author=author, description=f"Query: {ast.subject}")
        params: dict[str, Any] = {}
        if ast.subject:
            params["text_contains"] = ast.subject
        node_type = ast.get_slot("node_type")
        if node_type:
            params["node_type"] = node_type
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params=params,
                cost_estimate=10,
            )
        )
        return stream

    def _compile_plan(self, ast: SemanticAST, author: str) -> InstructionStream:
        """PLAN: ASSERT goal → QUERY constraints → EXECUTE planner."""
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        stream = InstructionStream(author=author, description=f"Plan: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=goal_id,
                        description=f"Plan: {ast.subject}",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"node_type": "constraint"},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "planning",
                    "params": {"goal_id": goal_id, "subject": ast.subject},
                },
                cost_estimate=150,
            )
        )
        return stream

    def _compile_learn(self, ast: SemanticAST, author: str) -> InstructionStream:
        """LEARN: QUERY existing → ASSERT new knowledge → EXECUTE skill extraction."""
        stream = InstructionStream(author=author, description=f"Learn: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"text_contains": ast.subject},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": {
                        "id": f"belief_{uuid.uuid4().hex[:8]}",
                        "node_type": "belief",
                        "claim": ast.subject,
                        "belief_type": "learned",
                    },
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "skill_extraction",
                    "params": {"subject": ast.subject},
                },
                cost_estimate=50,
            )
        )
        return stream

    def _compile_simulate(self, ast: SemanticAST, author: str) -> InstructionStream:
        """SIMULATE: FORK → EXECUTE scenario → MERGE or ROLLBACK."""
        branch_name = f"sim_{uuid.uuid4().hex[:6]}"
        stream = InstructionStream(author=author, description=f"Simulate: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.FORK,
                params={"branch_name": branch_name},
                cost_estimate=20,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "simulation",
                    "params": {"scenario": ast.subject, "branch": branch_name},
                },
                cost_estimate=200,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.MERGE,
                params={"branch_name": branch_name},
                cost_estimate=20,
            )
        )
        return stream

    def _compile_monitor(self, ast: SemanticAST, author: str) -> InstructionStream:
        """MONITOR: ASSERT goal → WAIT for condition."""
        stream = InstructionStream(author=author, description=f"Monitor: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=f"goal_{uuid.uuid4().hex[:8]}",
                        description=f"Monitor: {ast.subject}",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.WAIT,
                params={"condition": ast.subject, "timeout_ms": 60000},
                cost_estimate=5,
            )
        )
        return stream
