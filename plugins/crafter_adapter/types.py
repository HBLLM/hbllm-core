"""
Crafter Adapter Types.

Defines strongly-typed representations of Crafter actions, semantic objects,
achievements, inventory, vitals, and goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class CrafterAction(IntEnum):
    """Crafter 17 discrete primitive actions."""

    NOOP = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    DO = 5
    SLEEP = 6
    PLACE_STONE = 7
    PLACE_TABLE = 8
    PLACE_FURNACE = 9
    PLACE_PLANT = 10
    MAKE_WOOD_PICKAXE = 11
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16


class CrafterObject(IntEnum):
    """Semantic object and terrain IDs in Crafter."""

    EMPTY = 0
    GRASS = 1
    PATH = 2
    SAND = 3
    TREE = 4
    WATER = 5
    STONE = 6
    COAL = 7
    IRON = 8
    DIAMOND = 9
    LAVA = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    PLANT = 13
    PLAYER = 14
    COW = 15
    ZOMBIE = 16
    SKELETON = 17
    ARROW = 18


class CrafterAchievement(StrEnum):
    """All 22 canonical Crafter benchmark achievements."""

    COLLECT_WOOD = "collect_wood"
    PLACE_TABLE = "place_table"
    EAT_COW = "eat_cow"
    COLLECT_SAPLING = "collect_sapling"
    COLLECT_DRINK = "collect_drink"
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    MAKE_WOOD_SWORD = "make_wood_sword"
    PLACE_PLANT = "place_plant"
    DEFEAT_ZOMBIE = "defeat_zombie"
    COLLECT_STONE = "collect_stone"
    PLACE_STONE = "place_stone"
    EAT_PLANT = "eat_plant"
    DEFEAT_SKELETON = "defeat_skeleton"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    MAKE_STONE_SWORD = "make_stone_sword"
    PLACE_FURNACE = "place_furnace"
    COLLECT_COAL = "collect_coal"
    COLLECT_IRON = "collect_iron"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    MAKE_IRON_SWORD = "make_iron_sword"
    COLLECT_DIAMOND = "collect_diamond"
    SURVIVE = "survive"


ACTION_NAMES = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

NAME_TO_ACTION = {name: CrafterAction(i) for i, name in enumerate(ACTION_NAMES)}

MOVE_VECTORS: dict[CrafterAction, tuple[int, int]] = {
    CrafterAction.MOVE_LEFT: (-1, 0),
    CrafterAction.MOVE_RIGHT: (1, 0),
    CrafterAction.MOVE_UP: (0, -1),
    CrafterAction.MOVE_DOWN: (0, 1),
}


@dataclass
class CrafterInventory:
    """Agent inventory counts."""

    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "wood": self.wood,
            "stone": self.stone,
            "coal": self.coal,
            "iron": self.iron,
            "diamond": self.diamond,
            "sapling": self.sapling,
            "wood_pickaxe": self.wood_pickaxe,
            "stone_pickaxe": self.stone_pickaxe,
            "iron_pickaxe": self.iron_pickaxe,
            "wood_sword": self.wood_sword,
            "stone_sword": self.stone_sword,
            "iron_sword": self.iron_sword,
        }


@dataclass
class CrafterVitals:
    """Agent life metrics (range 0 to 9)."""

    health: int = 9
    food: int = 9
    drink: int = 9
    energy: int = 9

    @property
    def is_critical(self) -> bool:
        return self.health <= 2 or self.food <= 2 or self.drink <= 2 or self.energy <= 1


@dataclass
class CrafterObservation:
    """Strongly-typed observation for Crafter."""

    semantic_grid: list[list[int]]  # 2D grid of CrafterObject IDs
    player_pos: tuple[int, int]
    player_facing: tuple[int, int]
    inventory: CrafterInventory
    vitals: CrafterVitals
    achievements: set[CrafterAchievement] = field(default_factory=set)
    step_count: int = 0
    day_time: float = 0.0  # 0.0 to 1.0 (day/night)
    raw_obs: Any = None


@dataclass
class CrafterGoal:
    """Target objective for causal planning."""

    target_achievement: CrafterAchievement | None = None
    target_item: str | None = None
    target_object: CrafterObject | None = None
    priority: int = 0
