"""
HCIR Bytecode — minimal register-unification cognitive instruction set (ISA v2).

12 canonical opcodes.  State operations (ASSERT, RETRACT, QUERY, EXECUTE,
WAIT, FORK, MERGE, ROLLBACK) compose with unification and subroutine control
flow (UNIFY, BRANCH, CALL, RETURN) over working-memory registers (`$r0`..`$r7`).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from hbllm.hcir.types import CostMetric

# ═══════════════════════════════════════════════════════════════════════════
# Opcode Enumeration
# ═══════════════════════════════════════════════════════════════════════════


class Opcode(StrEnum):
    """The 12 canonical HCIR ISA v2 opcodes."""

    ASSERT = "ASSERT"  # Inject state, goal, or belief into the graph
    RETRACT = "RETRACT"  # Deprecate or remove an active node
    QUERY = "QUERY"  # Retrieve elements matching a structural constraint
    EXECUTE = "EXECUTE"  # Dispatch a declarative Layer-0 syscall via CapabilityResolver
    WAIT = "WAIT"  # Block execution thread until condition resolves
    FORK = "FORK"  # Clone current execution context for simulation
    MERGE = "MERGE"  # Commit simulated modifications back to parent
    ROLLBACK = "ROLLBACK"  # Reset state to a prior event log index
    UNIFY = "UNIFY"  # Match GraphPattern against workspace or $rN and bind ?vars
    BRANCH = "BRANCH"  # Conditional/unconditional jump on registers, labels, or surprise
    CALL = "CALL"  # Invoke homoiconic SkillNode/ProcedureNode in child CallFrame
    RETURN = "RETURN"  # Pop CallFrame and yield register outputs to caller


# ═══════════════════════════════════════════════════════════════════════════
# Instruction
# ═══════════════════════════════════════════════════════════════════════════


class Instruction(BaseModel):
    """A single bytecode instruction."""

    id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    opcode: Opcode
    params: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None
    cost_estimate: CostMetric = 0  # Estimated cost in tokens

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        prefix = f"{self.label}: " if self.label else ""
        return f"{prefix}{self.opcode}({param_str})"


# ═══════════════════════════════════════════════════════════════════════════
# Instruction Stream
# ═══════════════════════════════════════════════════════════════════════════


def _canonical_instructions(instructions: list[Instruction]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ins in instructions:
        entry: dict[str, Any] = {"opcode": ins.opcode.value, "params": ins.params}
        if ins.label is not None:
            entry["label"] = ins.label
        items.append(entry)
    return items


class InstructionStream(BaseModel):
    """An ordered sequence of ISA v2 bytecode instructions."""

    id: str = Field(default_factory=lambda: f"stream_{uuid.uuid4().hex[:8]}")
    instructions: list[Instruction] = Field(default_factory=list)
    author: str = ""  # Node that compiled this stream
    description: str = ""

    @property
    def total_cost_estimate(self) -> int:
        return sum(i.cost_estimate for i in self.instructions)

    @property
    def length(self) -> int:
        return len(self.instructions)

    def append(self, instruction: Instruction) -> None:
        self.instructions.append(instruction)

    def compute_hash(self) -> str:
        """Compute canonical deterministic SHA256 checksum (independent of random UUIDs)."""
        payload = json.dumps(
            {
                "author": self.author,
                "description": self.description,
                "instructions": _canonical_instructions(self.instructions),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def semantic_hash(self) -> str:
        """Compute SHA256 checksum of instructions only (independent of id/author)."""
        payload = json.dumps(
            _canonical_instructions(self.instructions),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def semantic_digest(self) -> str:
        """Compute canonical BLAKE3 semantic identity digest (independent of id/author)."""
        from hbllm.hcir.sexpr import semantic_digest as _semantic_digest

        return _semantic_digest(self)

    def exact_digest(self) -> str:
        """Compute exact BLAKE3 digest preserving stream and instruction IDs."""
        from hbllm.hcir.sexpr import exact_digest as _exact_digest

        return _exact_digest(self)
