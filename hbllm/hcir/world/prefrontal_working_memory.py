"""Prefrontal Working Memory & Hierarchical Subgoal Schema System (Facade).

Modeled on the primate dorsolateral prefrontal cortex (dlPFC).
This facade re-exports all symbols from `hbllm.hcir.world.working_memory` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.working_memory import (
    AttractorMemorySlot,
    DomainCompetenceProfile,
    HeldItem,
    HomeostaticState,
    PBWMWorkingMemoryBuffer,
    PrefrontalWorkingMemory,
    SubgoalSchema,
    SubgoalStage,
    WorkingMemorySalienceState,
)

__all__ = [
    "AttractorMemorySlot",
    "DomainCompetenceProfile",
    "HeldItem",
    "HomeostaticState",
    "PBWMWorkingMemoryBuffer",
    "PrefrontalWorkingMemory",
    "SubgoalSchema",
    "SubgoalStage",
    "WorkingMemorySalienceState",
]
