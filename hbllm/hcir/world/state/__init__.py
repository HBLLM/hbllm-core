"""Cognitive Engine State Subsystem — Authoritative State, Snapshots, and Faculty Views."""

from __future__ import annotations

from hbllm.hcir.world.state.context import CognitiveContext
from hbllm.hcir.world.state.models import (
    FrozenBodySchema,
    FrozenSubgoalStack,
    FrozenSymbolicTheory,
    FrozenWorkingMemory,
    apply_wm_field_delta,
    get_engine_state_attr,
)
from hbllm.hcir.world.state.state import EngineState, StateDelta
from hbllm.hcir.world.state.views import (
    FacultyWriteJournal,
    JournaledEngineView,
    ReadOnlyEngineView,
    _JournaledStateProxy,
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
