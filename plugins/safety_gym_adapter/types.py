"""
Safety-Gymnasium Adapter Types.

Defines strongly-typed representations of safety navigation actions,
hazards, dynamic gremlins, pillars, goals, cost signals, and observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class SafetyGymAction(IntEnum):
    """Discrete navigation actions."""

    NOOP = 0
    FORWARD = 1
    BACKWARD = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4


class SafetyEntityType(StrEnum):
    """Types of entities in safety navigation arenas."""

    AGENT = "agent"
    GOAL = "goal"
    HAZARD = "hazard"
    GREMLIN = "gremlin"
    PILLAR = "pillar"
    WALL = "wall"


@dataclass
class SafetyEntity:
    """An obstacle, hazard, or goal in the 2D arena."""

    id: str
    entity_type: SafetyEntityType
    x: float
    y: float
    radius: float = 0.3
    vx: float = 0.0
    vy: float = 0.0
    cost_weight: float = 1.0


@dataclass
class SafetyGoal:
    """Navigation target with strict safety constraints."""

    target_pos: tuple[float, float]
    target_radius: float = 0.3
    max_cost_allowed: float = 0.0


@dataclass
class SafetyObservation:
    """Observation returned by the safety environment."""

    agent_pos: tuple[float, float]
    agent_heading: float  # heading angle in radians
    agent_vel: tuple[float, float]
    goal_pos: tuple[float, float]
    hazards: list[SafetyEntity] = field(default_factory=list)
    gremlins: list[SafetyEntity] = field(default_factory=list)
    pillars: list[SafetyEntity] = field(default_factory=list)
    lidar_distances: list[float] = field(default_factory=list)
    current_cost: float = 0.0
    cumulative_cost: float = 0.0
    step_count: int = 0
    raw_obs: Any = None
