"""
Stochastic Robustness Adapter Types.

Defines strongly-typed representations of noisy actions, sensory perturbations,
epistemic belief states, and multi-tier noise benchmark specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class StochasticAction(IntEnum):
    """Primitive 4-way discrete navigation actions."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NOOP = 4


class StochasticTier(StrEnum):
    """Canonical noise and perturbation evaluation tiers."""

    TIER_1_DETERMINISTIC_BASELINE = "tier_1_deterministic_baseline"
    TIER_2_ACTUATOR_SLIP = "tier_2_actuator_slip"
    TIER_3_SENSORY_DROPOUT = "tier_3_sensory_dropout"
    TIER_4_COMPOUND_PERTURBATION = "tier_4_compound_perturbation"
    TIER_5_DYNAMIC_DRIFT = "tier_5_dynamic_drift"


@dataclass
class EpistemicEntityBelief:
    """Belief state for an entity under perceptual noise."""

    entity_id: str
    estimated_pos: tuple[int, int]
    confidence: float  # 0.0 to 1.0
    last_seen_step: int
    occluded: bool = False


@dataclass
class StochasticObservation:
    """Observation returned by stochastic environment wrapper."""

    grid: list[list[int]]  # May contain dropped/occluded cells (-1 for unseen)
    player_pos: tuple[int, int]
    target_pos: tuple[int, int] | None  # None if dropped/occluded this step
    obstacles: list[tuple[int, int]]
    step_count: int
    max_steps: int
    done: bool = False
    won: bool = False
    was_slipped: bool = False
    was_occluded: bool = False
    info: dict[str, Any] = field(default_factory=dict)
