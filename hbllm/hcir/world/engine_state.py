"""Cognitive Engine State, Frozen Contexts, and Faculty Views (Facade).

Authoritative Cognitive Engine State representation owned by HCIR.
This facade re-exports all symbols from `hbllm.hcir.world.state` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.state import (
    CognitiveContext,
    EngineState,
    FacultyWriteJournal,
    FrozenBodySchema,
    FrozenSubgoalStack,
    FrozenSymbolicTheory,
    FrozenWorkingMemory,
    JournaledEngineView,
    ReadOnlyEngineView,
    StateDelta,
    _JournaledStateProxy,
    apply_wm_field_delta,
    get_engine_state_attr,
)

__all__ = [
    "CognitiveContext",
    "EngineState",
    "FacultyWriteJournal",
    "FrozenBodySchema",
    "FrozenSubgoalStack",
    "FrozenSymbolicTheory",
    "FrozenWorkingMemory",
    "JournaledEngineView",
    "ReadOnlyEngineView",
    "StateDelta",
    "_JournaledStateProxy",
    "apply_wm_field_delta",
    "get_engine_state_attr",
]
