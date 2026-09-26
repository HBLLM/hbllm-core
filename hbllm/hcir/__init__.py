"""
HCIR — HBLLM Cognitive Intermediate Representation.

A typed, versioned, transactional cognitive execution model composed of
graph state, runtime context, executable cognitive instructions, and
immutable transformation history.

HCIR is the Instruction Set Architecture (ISA) of the HBLLM Cognitive OS.
LLMs act as compilers/decompilers; the kernel executes HCIR bytecodes
over a typed cognitive hypergraph.

Architecture layers (strict dependency order)::

    Application Layer         (User interfaces, NL translation)
            ▲
    Reasoning Nodes           (Planners, LLMs, logic solvers)
            ▲
    Kernel Services           (Scheduler, TransactionManager)
            ▲
    HCIR Runtime              (Execution graphs, instruction streams)
            ▲
    HCIR Data Model           (Typed graph, types, validation)
"""

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
from hbllm.hcir.types import (
    Confidence,
    CostMetric,
    Priority,
    TimeDuration,
    Timestamp,
    UncertaintyVector,
)

__all__ = [
    # --- Cognitive Type System ---
    "Priority",
    "Confidence",
    "CostMetric",
    "TimeDuration",
    "Timestamp",
    "UncertaintyVector",
    # --- Spatial Entity Planning ---
    "HCIRSpatialEntityPlanner",
    "EntityGraph",
    "SpatialEntity",
    "EntityRole",
    # --- Backward Compat Aliases (deprecated) ---
    "AVATAR",
    "BARRIER",
    "MOVABLE_ITEM",
    "DOORWAY",
    "SWITCH",
    "REFILL",
    "EXIT",
    # --- Skill Acquisition ---
    "AutonomousTrajectoryModel",
    "CoupledControllableSkillAcquisition",
    "GF2LinearSolver",
    "KinematicMomentumSkillAcquisition",
    "MorphologicalProgramSynthesis",
    "PermutationAlgebraSkillAcquisition",
    "RelationalAffordanceSkillAcquisition",
    "SpatiotemporalSkillAcquisition",
]

from hbllm.hcir.skills import (
    AutonomousTrajectoryModel,
    CoupledControllableSkillAcquisition,
    GF2LinearSolver,
    KinematicMomentumSkillAcquisition,
    MorphologicalProgramSynthesis,
    PermutationAlgebraSkillAcquisition,
    RelationalAffordanceSkillAcquisition,
    SpatiotemporalSkillAcquisition,
)

__hcir_version__ = "1.2.0"
