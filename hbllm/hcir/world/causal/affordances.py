"""Embodied motor and conditional affordance predicates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.causal.predicates import CausalPredicate


@dataclass
class ActionAffordance:
    """Declared or empirically discovered motor affordance for an embodied action.

    Decouples cognitive reasoning from specific game environments, hardware drivers,
    cameras, depth sensors, or LiDAR peripherals.
    """

    action_id: Any
    name: str = ""
    requires_spatial_target: bool = False
    target_param_keys: tuple[str, ...] = ("x", "y")
    is_displacement: bool = False
    delta: tuple[int, ...] | None = None
    is_state_transform: bool = False
    is_focus_switch: bool = False
    is_undo: bool = False
    is_value_mutator: bool = False
    is_index_selector: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionalAffordance:
    """Expresses that feature is traversable / operable iff condition predicate holds.

    General form of key->door, switch->gate, cargo->receptacle causal dependencies.
    """

    feature: Any
    condition: CausalPredicate
    evidence: str = ""
    target_action: str = "traverse"
    prerequisite_feature: Any | None = None


__all__ = [
    "ActionAffordance",
    "ConditionalAffordance",
]
