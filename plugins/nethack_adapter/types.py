"""
NetHack Adapter Types.

Defines strongly-typed representations of NetHack/MiniHack actions,
glyphs, dungeon features, player stats, inventory, and observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class NetHackAction(IntEnum):
    """Primitive NetHack discrete movement and interaction commands."""

    NOOP = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4
    NORTHEAST = 5
    SOUTHEAST = 6
    SOUTHWEST = 7
    NORTHWEST = 8
    PICKUP = 9
    OPEN_DOOR = 10
    KICK = 11
    DESCEND_STAIRS = 12
    WAIT = 13
    EAT = 14


ACTION_VECTORS: dict[NetHackAction, tuple[int, int]] = {
    NetHackAction.NORTH: (0, -1),
    NetHackAction.EAST: (1, 0),
    NetHackAction.SOUTH: (0, 1),
    NetHackAction.WEST: (-1, 0),
    NetHackAction.NORTHEAST: (1, -1),
    NetHackAction.SOUTHEAST: (1, 1),
    NetHackAction.SOUTHWEST: (-1, 1),
    NetHackAction.NORTHWEST: (-1, -1),
}


class NetHackGlyph(IntEnum):
    """Canonical dungeon glyph categories."""

    UNEXPLORED = 0
    FLOOR = 1
    WALL = 2
    CORRIDOR = 3
    DOOR_CLOSED = 4
    DOOR_OPEN = 5
    STAIRS_DOWN = 6
    STAIRS_UP = 7
    PLAYER = 8
    MONSTER = 9
    FOOD = 10
    KEY = 11
    GOLD = 12


GLYPH_CHARS: dict[NetHackGlyph, str] = {
    NetHackGlyph.UNEXPLORED: " ",
    NetHackGlyph.FLOOR: ".",
    NetHackGlyph.WALL: "#",
    NetHackGlyph.CORRIDOR: "#",
    NetHackGlyph.DOOR_CLOSED: "+",
    NetHackGlyph.DOOR_OPEN: "-",
    NetHackGlyph.STAIRS_DOWN: ">",
    NetHackGlyph.STAIRS_UP: "<",
    NetHackGlyph.PLAYER: "@",
    NetHackGlyph.MONSTER: "D",
    NetHackGlyph.FOOD: "%",
    NetHackGlyph.KEY: "(",
    NetHackGlyph.GOLD: "$",
}


@dataclass
class NetHackStats:
    """Bottom-line character status in NetHack."""

    hp: int = 15
    max_hp: int = 15
    dungeon_level: int = 1
    hunger_state: str = "Normal"  # Normal, Hungry, Weak, Fainting
    gold: int = 0
    armor_class: int = 10
    exp_level: int = 1


@dataclass
class NetHackObservation:
    """Observation returned by NetHack / MiniHack."""

    glyphs: list[list[int]]  # 2D grid of NetHackGlyph
    chars: list[list[str]]  # 2D ASCII character array
    player_pos: tuple[int, int]
    stats: NetHackStats
    inventory: list[str] = field(default_factory=list)
    message: str = ""
    step_count: int = 0
    raw_obs: Any = None


@dataclass
class NetHackGoal:
    """Target objective (e.g. reach staircase and descend)."""

    target_action: str = "descend_stairs"
    target_dungeon_level: int = 2
    target_pos: tuple[int, int] | None = None
