"""Macro-Action Operators and Hierarchical Option Scheduling (Facade).

Decomposes monolithic procedural cognitive loops into discrete, composable macro-actions.
This facade re-exports all symbols from `hbllm.hcir.world.macro` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.macro import (
    CustomMacroAction,
    DepositCargo,
    ExploreFrontier,
    ExploreObject,
    InteractWith,
    MacroAction,
    MacroActionDriverManager,
    MacroActionPlan,
    MacroActionStage,
    MacroActionStatus,
    MacroPlanSynthesizer,
    MicroActionSpec,
    NavigateTo,
    ObjectSpec,
    PickUpCargo,
    ProceduralReplayAction,
    PushObject,
    SwitchFocus,
    TranslateObject,
    UseToolOnBarrier,
    _spatial_effector_payload,
    attribute_macro_action_outcome,
    get_sub_attr,
)

__all__ = [
    "CustomMacroAction",
    "DepositCargo",
    "ExploreFrontier",
    "ExploreObject",
    "InteractWith",
    "MacroAction",
    "MacroActionDriverManager",
    "MacroActionPlan",
    "MacroActionStage",
    "MacroActionStatus",
    "MacroPlanSynthesizer",
    "MicroActionSpec",
    "NavigateTo",
    "ObjectSpec",
    "PickUpCargo",
    "ProceduralReplayAction",
    "PushObject",
    "SwitchFocus",
    "TranslateObject",
    "UseToolOnBarrier",
    "_spatial_effector_payload",
    "attribute_macro_action_outcome",
    "get_sub_attr",
]
