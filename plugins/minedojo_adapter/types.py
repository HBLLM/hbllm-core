"""
MineDojo Adapter Types.

Defines strongly-typed representations of MineDojo 3D voxels, actions,
inventory counts, crafting recipes, and observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class MineDojoAction(IntEnum):
    """Discrete Minecraft locomotion, mining, and crafting primitives."""

    NOOP = 0
    MOVE_FORWARD = 1
    MOVE_BACK = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    JUMP = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    MINE_BLOCK = 8
    PLACE_BLOCK = 9
    CRAFT_PLANKS = 10
    CRAFT_STICKS = 11
    CRAFT_TABLE = 12
    CRAFT_WOOD_PICKAXE = 13
    CRAFT_STONE_PICKAXE = 14
    CRAFT_FURNACE = 15
    CRAFT_IRON_PICKAXE = 16


class MineDojoVoxel(IntEnum):
    """3D block voxel types."""

    AIR = 0
    GRASS_BLOCK = 1
    DIRT = 2
    STONE = 3
    WOOD_LOG = 4
    WOOD_PLANKS = 5
    LEAVES = 6
    COAL_ORE = 7
    IRON_ORE = 8
    CRAFTING_TABLE = 9
    FURNACE = 10


@dataclass
class MineDojoInventory:
    """Player inventory storage."""

    log: int = 0
    planks: int = 0
    stick: int = 0
    crafting_table: int = 0
    wooden_pickaxe: int = 0
    cobblestone: int = 0
    stone_pickaxe: int = 0
    iron_ore: int = 0
    coal: int = 0
    iron_ingot: int = 0
    iron_pickaxe: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "log": self.log,
            "planks": self.planks,
            "stick": self.stick,
            "crafting_table": self.crafting_table,
            "wooden_pickaxe": self.wooden_pickaxe,
            "cobblestone": self.cobblestone,
            "stone_pickaxe": self.stone_pickaxe,
            "iron_ore": self.iron_ore,
            "coal": self.coal,
            "iron_ingot": self.iron_ingot,
            "iron_pickaxe": self.iron_pickaxe,
        }


@dataclass
class MineDojoObservation:
    """Local 3D voxel and player state observation."""

    voxels: list[list[list[int]]]  # [z][y][x] voxel grid
    player_pos: tuple[int, int, int]  # (x, y, z)
    player_yaw: float
    player_pitch: float
    inventory: MineDojoInventory
    equipped: str | None = None
    step_count: int = 0
    raw_obs: Any = None


@dataclass
class MineDojoGoal:
    """Target item or block synthesis milestone."""

    target_item: str
    target_count: int = 1
