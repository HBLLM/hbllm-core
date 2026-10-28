"""HCIR — HBLLM Cognitive Intermediate Representation.

A typed, versioned, transactional cognitive execution model composed of
graph state, runtime context, executable cognitive instructions, and
immutable transformation history.

HCIR is the Instruction Set Architecture (ISA) of the HBLLM Cognitive OS.
LLMs act as compilers/synthesizers; the kernel executes HCIR bytecodes
over a typed cognitive hypergraph.
"""

from __future__ import annotations

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.compiler import HCIRCompiler, IntentType, SemanticAST
from hbllm.hcir.compiler_llm import LLMCompilerFrontend, StructuredIntentPayload
from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIRNode, HCIRNodeType
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.optimizer import HCIROptimizer
from hbllm.hcir.replay_debugger import CognitiveBreakpoint, ReplayDebugger
from hbllm.hcir.skills import GF2LinearSolver
from hbllm.hcir.spatial_planner import (
    AVATAR,
    BARRIER,
    DOORWAY,
    EXIT,
    MOVABLE_ITEM,
    REFILL,
    SWITCH,
    EntityGraph,
    EntityRole,
    HCIRSpatialEntityPlanner,
    SpatialEntity,
)
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.types import (
    Confidence,
    CostMetric,
    Priority,
    TimeDuration,
    Timestamp,
    UncertaintyVector,
)
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world_kernel import WorldKernel

__hcir_version__ = "1.2.0"

__all__ = [
    # --- Cognitive Type System ---
    "Priority",
    "Confidence",
    "CostMetric",
    "TimeDuration",
    "Timestamp",
    "UncertaintyVector",
    # --- Graph & State ---
    "CognitiveGraph",
    "HCIRNode",
    "HCIREdge",
    "HCIRNodeType",
    "HCIRWorkspaceState",
    "TieredWorkspace",
    # --- Transactions ---
    "HCIRTransaction",
    "TransactionOperation",
    "TransactionOp",
    # --- Bytecode & Virtual Machine ---
    "Instruction",
    "InstructionStream",
    "Opcode",
    "HCIRInterpreter",
    # --- Compiler & Optimizer ---
    "HCIRCompiler",
    "HCIROptimizer",
    "LLMCompilerFrontend",
    "StructuredIntentPayload",
    "SemanticAST",
    "IntentType",
    # --- Cognitive Planners ---
    "CounterfactualPlanner",
    "HierarchicalGoalDecomposer",
    "HCIRSpatialEntityPlanner",
    "EntityGraph",
    "SpatialEntity",
    "EntityRole",
    # --- World Kernel & Debugging ---
    "WorldKernel",
    "ReplayDebugger",
    "CognitiveBreakpoint",
    # --- Backward Compat Aliases (deprecated) ---
    "AVATAR",
    "BARRIER",
    "MOVABLE_ITEM",
    "DOORWAY",
    "SWITCH",
    "REFILL",
    "EXIT",
    "GF2LinearSolver",
]
