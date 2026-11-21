"""Embodied motor and conditional affordance predicates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.hcir.world.causal.predicates import CausalPredicate


class AffordanceType(str, Enum):
    """Core ontological affordance types grounded in physical interaction dynamics.

    Decouples cognitive reasoning and planning from domain-specific or game-specific
    terminologies (e.g. 'barrier', 'gem', 'key', 'door') and expresses universal physical
    capabilities and interaction constraints.
    """

    IMPASSABLE = "barrier"  # Non-traversable solid matter / topological obstacle
    PASSABLE = "walkable"  # Free navigable space / medium
    TELIC = "goal"  # Terminal attractor / desired state
    CANDIDATE_TELIC = "candidate_goal"  # Hypothesized goal pending confirmation
    PUSHABLE = "cargo"  # Manipulable entity under contact displacement
    RECEPTACLE = "receptacle"  # Container locus, drop-off, or target boundary
    ACTUATABLE = "actuator"  # Distal state switch / trigger / button
    HAZARD = "hazard"  # Lethal hazard / penalty zone
    HOMEOTROPIC = "vital_resource"  # Sustenance / replenishment resource (e.g. time, energy, fuel)


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
    requires_contact: bool = False
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
    required_property: Any | None = None


__all__ = [
    "ActionAffordance",
    "AffordanceType",
    "ConditionalAffordance",
]
