"""
Sokoban Adapter Types.

Defines strongly-typed representations of Sokoban actions, tiles, observations,
deadlock categories, and multi-tier benchmark configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class SokobanAction(IntEnum):
    """Sokoban 4 primitive directional movements / pushes."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SokobanTile(IntEnum):
    """Grid tile types in Sokoban."""

    EMPTY = 0
    WALL = 1
    TARGET = 2
    BOX = 3
    BOX_ON_TARGET = 4
    PLAYER = 5
    PLAYER_ON_TARGET = 6


class SokobanDeadlockType(StrEnum):
    """Irreversible deadlock failure modes."""

    NONE = "none"
    CORNER = "corner_deadlock"
    WALL_FREEZE = "wall_freeze_deadlock"
    SQUARE_2X2 = "square_2x2_deadlock"


class SokobanTier(StrEnum):
    """Canonical Sokoban benchmark evaluation tiers."""

    TIER_1_DIRECT_PUSH = "tier_1_direct_push"
    TIER_2_OBSTACLE_NAVIGATION = "tier_2_obstacle_navigation"
    TIER_3_CORNER_DEADLOCK_AVOIDANCE = "tier_3_corner_deadlock_avoidance"
    TIER_4_MULTI_BOX_ASSIGNMENT = "tier_4_multi_box_assignment"
    TIER_5_COMBINATORIAL_MAZE = "tier_5_combinatorial_maze"


@dataclass(frozen=True)
class Coordinate:
    """2D grid coordinate (row, col)."""

    r: int
    c: int

    def add(self, dr: int, dc: int) -> Coordinate:
        return Coordinate(self.r + dr, self.c + dc)

    def manhattan(self, other: Coordinate) -> int:
        return abs(self.r - other.r) + abs(self.c - other.c)


@dataclass
class SokobanObservation:
    """Typed observation returned by Sokoban environments."""

    grid: list[list[int]]
    player_pos: tuple[int, int]
    boxes: list[tuple[int, int]]
    targets: list[tuple[int, int]]
    step_count: int
    max_steps: int
    done: bool = False
    won: bool = False
    deadlock_detected: bool = False
    info: dict[str, Any] = field(default_factory=dict)
