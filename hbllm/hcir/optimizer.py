"""
HCIR Optimizer — Intermediate Representation (CIR) & Bytecode Optimization.

Pipeline::

    SemanticAST / Frontends
              ↓
    Cognitive IR (CIR)
              ↓
    HCIROptimizer Passes
      ├── DeadInstructionEliminationPass
      ├── QueryMergingPass
      ├── TransactionBatchingPass
      └── CostPruningPass
              ↓
    Optimized Bytecode Stream
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive IR (CIR)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CIRBlock:
    """A basic block of Cognitive IR instructions."""

    block_id: str
    instructions: list[Instruction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveIRProgram:
    """Intermediate Representation program before bytecode emission."""

    author: str = ""
    description: str = ""
    blocks: list[CIRBlock] = field(default_factory=list)

    def flatten(self) -> InstructionStream:
        """Convert CIR program to an InstructionStream."""
        all_ins: list[Instruction] = []
        for block in self.blocks:
            all_ins.extend(block.instructions)
        return InstructionStream(
            author=self.author,
            description=self.description,
            instructions=all_ins,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer Pass Base Interface
# ═══════════════════════════════════════════════════════════════════════════


class IOptimizerPass(ABC):
    """Abstract interface for an HCIR optimization pass."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, stream: InstructionStream) -> InstructionStream:
        """Transform an InstructionStream into an optimized InstructionStream."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Concrete Optimization Passes
# ═══════════════════════════════════════════════════════════════════════════


class DeadInstructionEliminationPass(IOptimizerPass):
    """Eliminates redundant or no-op instructions."""

    @property
    def name(self) -> str:
        return "DeadInstructionElimination"

    def run(self, stream: InstructionStream) -> InstructionStream:
        optimized: list[Instruction] = []
        for ins in stream.instructions:
            # Eliminate ASSERT with empty node_data and edge_data
            if (
                ins.opcode == Opcode.ASSERT
                and not ins.params.get("node_data")
                and not ins.params.get("edge_data")
            ):
                logger.debug("Pruned empty ASSERT instruction")
                continue
            # Eliminate RETRACT with missing node_id and edge_id
            if (
                ins.opcode == Opcode.RETRACT
                and not ins.params.get("node_id")
                and not ins.params.get("edge_id")
            ):
                logger.debug("Pruned empty RETRACT instruction")
                continue
            optimized.append(ins)

        return InstructionStream(
            author=stream.author,
            description=f"{stream.description} [opt:dead_elim]",
            instructions=optimized,
        )


class QueryMergingPass(IOptimizerPass):
    """Combines consecutive identical or compatible QUERY instructions."""

    @property
    def name(self) -> str:
        return "QueryMerging"

    def run(self, stream: InstructionStream) -> InstructionStream:
        optimized: list[Instruction] = []
        prev_query: Instruction | None = None

        for ins in stream.instructions:
            if ins.opcode == Opcode.QUERY:
                if prev_query is not None and prev_query.params == ins.params:
                    # Duplicate consecutive query — skip
                    logger.debug("Merged duplicate QUERY instruction: %s", ins.params)
                    continue
                prev_query = ins
            else:
                prev_query = None

            optimized.append(ins)

        return InstructionStream(
            author=stream.author,
            description=f"{stream.description} [opt:query_merge]",
            instructions=optimized,
        )


class CostPruningPass(IOptimizerPass):
    """Prunes instructions if total estimated cost exceeds a budget limit."""

    def __init__(self, max_cost_budget: int = 1000) -> None:
        self._max_cost_budget = max_cost_budget

    @property
    def name(self) -> str:
        return "CostPruning"

    def run(self, stream: InstructionStream) -> InstructionStream:
        optimized: list[Instruction] = []
        accumulated_cost = 0

        for ins in stream.instructions:
            if accumulated_cost + ins.cost_estimate > self._max_cost_budget:
                logger.warning(
                    "CostPruningPass: Pruned instruction %s due to budget (%d + %d > %d)",
                    ins.opcode,
                    accumulated_cost,
                    ins.cost_estimate,
                    self._max_cost_budget,
                )
                break
            accumulated_cost += ins.cost_estimate
            optimized.append(ins)

        return InstructionStream(
            author=stream.author,
            description=f"{stream.description} [opt:cost_prune]",
            instructions=optimized,
        )


# ═══════════════════════════════════════════════════════════════════════════
# HCIROptimizer Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class HCIROptimizer:
    """Optimization pipeline for HCIR bytecode streams.

    Usage::

        optimizer = HCIROptimizer([
            DeadInstructionEliminationPass(),
            QueryMergingPass(),
        ])
        optimized_stream = optimizer.optimize(raw_stream)
    """

    def __init__(self, passes: list[IOptimizerPass] | None = None) -> None:
        self._passes = passes or [
            DeadInstructionEliminationPass(),
            QueryMergingPass(),
        ]

    def optimize(self, stream: InstructionStream) -> InstructionStream:
        """Run all optimization passes sequentially."""
        current = stream
        for opt_pass in self._passes:
            current = opt_pass.run(current)
        return current
