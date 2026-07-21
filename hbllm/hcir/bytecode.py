"""
HCIR Bytecode — minimal cognitive instruction set.

8 stable primitive opcodes.  All high-level behaviors (learning,
resolving, reflecting) are compiled down to combinations of
these primitives — like LLVM has surprisingly few instructions.

Stable Core:
    ASSERT, RETRACT, QUERY, EXECUTE, WAIT, FORK, MERGE, ROLLBACK

Higher-level operations are macros::

    LEARN = QUERY + ASSERT + EXECUTE + ASSERT
    REFLECT = QUERY + ASSERT(annotations)
    PLAN = QUERY + ASSERT(goals) + ASSERT(actions)
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from hbllm.hcir.types import CostMetric

# ═══════════════════════════════════════════════════════════════════════════
# Opcode Enumeration
# ═══════════════════════════════════════════════════════════════════════════


class Opcode(StrEnum):
    """The 8 stable HCIR primitive opcodes.

    Every cognitive operation compiles down to sequences of these.
    """

    ASSERT = "ASSERT"  # Inject state, goal, or belief into the graph
    RETRACT = "RETRACT"  # Deprecate or remove an active node
    QUERY = "QUERY"  # Retrieve elements matching a structural constraint
    EXECUTE = "EXECUTE"  # Dispatch a declarative action via CapabilityResolver
    WAIT = "WAIT"  # Block execution thread until condition resolves
    FORK = "FORK"  # Clone current execution context for simulation
    MERGE = "MERGE"  # Commit simulated modifications back to parent
    ROLLBACK = "ROLLBACK"  # Reset state to a prior event log index


# ═══════════════════════════════════════════════════════════════════════════
# Instruction
# ═══════════════════════════════════════════════════════════════════════════


class Instruction(BaseModel):
    """A single bytecode instruction.

    Each instruction carries its opcode, typed parameters, and
    an estimated resource cost for scheduler decision-making.
    """

    id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    opcode: Opcode
    params: dict[str, Any] = Field(default_factory=dict)
    cost_estimate: CostMetric = 0  # Estimated cost in tokens

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.opcode}({param_str})"


# ═══════════════════════════════════════════════════════════════════════════
# Instruction Stream
# ═══════════════════════════════════════════════════════════════════════════


class InstructionStream(BaseModel):
    """An ordered sequence of bytecode instructions.

    Represents a compiled plan, procedure, or macro expansion.
    The interpreter executes instructions sequentially, with
    FORK/MERGE enabling parallel simulation branches.
    """

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
        """Compute SHA256 checksum of the instruction stream for audit verification."""
        import hashlib
        import json

        payload = json.dumps(
            {
                "id": self.id,
                "author": self.author,
                "instructions": [
                    {"opcode": ins.opcode.value, "params": ins.params} for ins in self.instructions
                ],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
