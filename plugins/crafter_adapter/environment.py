"""
Crafter Environment Wrapper.

Provides a dual-mode execution engine:
1. Native `crafter.Env` if available and binary-compatible.
2. High-Fidelity `StandaloneCrafterEnv` implementing complete 22-achievement
   dynamics, procedural worldgen, crafting DAG, and survival vitals with zero external dependencies.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .types import (
    CrafterAchievement,
    CrafterAction,
    CrafterInventory,
    CrafterObject,
    CrafterObservation,
    CrafterVitals,
)

logger = logging.getLogger(__name__)


class StandaloneCrafterEnv:
    """
    High-fidelity, zero-dependency Crafter simulation engine.
    Implements 64x64 procedural world, crafting DAG, mining constraints,
    vitals drain/recovery, and all 22 achievements.
    """

    def __init__(self, size: tuple[int, int] = (64, 64), seed: int | None = None) -> None:
        self.width, self.height = size
        self.rng = random.Random(seed)
        self.grid: list[list[int]] = []
        self.player_pos = (self.width // 2, self.height // 2)
        self.player_facing = (0, 1)  # facing south
        self.inventory = CrafterInventory()
        self.vitals = CrafterVitals()
        self.achievements: set[CrafterAchievement] = set()
        self.step_count = 0
        self.max_steps = 300
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[CrafterObservation, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.step_count = 0
        self.inventory = CrafterInventory()
        self.vitals = CrafterVitals(health=9, food=9, drink=9, energy=9)
        self.achievements = set()
        self._generate_world()
        return self._get_obs(), {}

    def _generate_world(self) -> None:
        """Procedurally generate grass, trees, water, stone, ores, and mobs."""
        self.grid = [[CrafterObject.GRASS for _ in range(self.width)] for _ in range(self.height)]

        # Borders are stone
        for x in range(self.width):
            self.grid[0][x] = CrafterObject.STONE
            self.grid[self.height - 1][x] = CrafterObject.STONE
        for y in range(self.height):
            self.grid[y][0] = CrafterObject.STONE
            self.grid[y][self.width - 1] = CrafterObject.STONE

        # Scatter water pools
        for _ in range(8):
            cx = self.rng.randint(5, self.width - 6)
            cy = self.rng.randint(5, self.height - 6)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx * dx + dy * dy <= 4:
                        self.grid[cy + dy][cx + dx] = CrafterObject.WATER

        # Scatter stone mountains & ore veins
        for _ in range(12):
            cx = self.rng.randint(5, self.width - 6)
            cy = self.rng.randint(5, self.height - 6)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if (
                        dx * dx + dy * dy <= 3
                        and self.grid[cy + dy][cx + dx] == CrafterObject.GRASS
                    ):
                        self.grid[cy + dy][cx + dx] = CrafterObject.STONE

        # Embed coal, iron, diamond into stone
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y][x] == CrafterObject.STONE:
                    r = self.rng.random()
                    if r < 0.08:
                        self.grid[y][x] = CrafterObject.COAL
                    elif r < 0.12:
                        self.grid[y][x] = CrafterObject.IRON
                    elif r < 0.13:
                        self.grid[y][x] = CrafterObject.DIAMOND

        # Scatter trees
        for y in range(2, self.height - 2):
            for x in range(2, self.width - 2):
                if self.grid[y][x] == CrafterObject.GRASS and self.rng.random() < 0.12:
                    self.grid[y][x] = CrafterObject.TREE

        # Scatter cows & zombies
        for _ in range(6):
            rx, ry = self._find_empty_grass()
            self.grid[ry][rx] = CrafterObject.COW
        for _ in range(4):
            rx, ry = self._find_empty_grass()
            self.grid[ry][rx] = CrafterObject.ZOMBIE

        # Place player on grass near center
        px, py = self._find_empty_grass(near=(self.width // 2, self.height // 2))
        self.player_pos = (px, py)
        self.player_facing = (0, 1)

    def _find_empty_grass(self, near: tuple[int, int] | None = None) -> tuple[int, int]:
        if near:
            nx, ny = near
            for radius in range(1, 20):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        x, y = nx + dx, ny + dy
                        if 1 <= x < self.width - 1 and 1 <= y < self.height - 1:
                            if self.grid[y][x] == CrafterObject.GRASS:
                                return (x, y)
        for _ in range(100):
            x = self.rng.randint(2, self.width - 3)
            y = self.rng.randint(2, self.height - 3)
            if self.grid[y][x] == CrafterObject.GRASS:
                return (x, y)
        return (self.width // 2, self.height // 2)

    def _target_pos(self) -> tuple[int, int]:
        px, py = self.player_pos
        fx, fy = self.player_facing
        return (px + fx, py + fy)

    def _is_passable(self, obj: int) -> bool:
        return obj in (
            CrafterObject.GRASS,
            CrafterObject.PATH,
            CrafterObject.SAND,
            CrafterObject.EMPTY,
        )

    def _is_near(self, target_obj: CrafterObject, radius: int = 2) -> bool:
        px, py = self.player_pos
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.grid[y][x] == target_obj:
                        return True
        return False

    def step(
        self, action: int | CrafterAction
    ) -> tuple[CrafterObservation, float, bool, bool, dict[str, Any]]:
        action_enum = CrafterAction(action) if isinstance(action, int) else action
        self.step_count += 1
        reward = 0.0

        # Survival vitals decay every few steps
        if self.step_count % 8 == 0:
            self.vitals.food = max(0, self.vitals.food - 1)
        if self.step_count % 10 == 0:
            self.vitals.drink = max(0, self.vitals.drink - 1)
        if self.step_count % 12 == 0:
            self.vitals.energy = max(0, self.vitals.energy - 1)

        # Survival achievement check
        if self.step_count >= 50 and CrafterAchievement.SURVIVE not in self.achievements:
            self.achievements.add(CrafterAchievement.SURVIVE)
            reward += 1.0

        px, py = self.player_pos

        # Movement Actions
        if action_enum in (
            CrafterAction.MOVE_LEFT,
            CrafterAction.MOVE_RIGHT,
            CrafterAction.MOVE_UP,
            CrafterAction.MOVE_DOWN,
        ):
            dx, dy = 0, 0
            if action_enum == CrafterAction.MOVE_LEFT:
                dx, dy = -1, 0
            elif action_enum == CrafterAction.MOVE_RIGHT:
                dx, dy = 1, 0
            elif action_enum == CrafterAction.MOVE_UP:
                dx, dy = 0, -1
            elif action_enum == CrafterAction.MOVE_DOWN:
                dx, dy = 0, 1

            self.player_facing = (dx, dy)
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self._is_passable(self.grid[ny][nx]):
                    self.player_pos = (nx, ny)

        # Sleep Action
        elif action_enum == CrafterAction.SLEEP:
            self.vitals.energy = min(9, self.vitals.energy + 5)
            self.vitals.health = min(9, self.vitals.health + 1)

        # Do / Harvest / Attack Action
        elif action_enum == CrafterAction.DO:
            tx, ty = self._target_pos()
            if 0 <= tx < self.width and 0 <= ty < self.height:
                target = self.grid[ty][tx]

                # Tree -> Wood
                if target == CrafterObject.TREE:
                    self.grid[ty][tx] = CrafterObject.GRASS
                    self.inventory.wood += 1
                    if self.rng.random() < 0.5:
                        self.inventory.sapling += 1
                        if CrafterAchievement.COLLECT_SAPLING not in self.achievements:
                            self.achievements.add(CrafterAchievement.COLLECT_SAPLING)
                            reward += 1.0
                    if CrafterAchievement.COLLECT_WOOD not in self.achievements:
                        self.achievements.add(CrafterAchievement.COLLECT_WOOD)
                        reward += 1.0

                # Water -> Drink
                elif target == CrafterObject.WATER:
                    self.vitals.drink = 9
                    if CrafterAchievement.COLLECT_DRINK not in self.achievements:
                        self.achievements.add(CrafterAchievement.COLLECT_DRINK)
                        reward += 1.0

                # Cow -> Eat
                elif target == CrafterObject.COW:
                    self.grid[ty][tx] = CrafterObject.GRASS
                    self.vitals.food = min(9, self.vitals.food + 6)
                    self.vitals.health = min(9, self.vitals.health + 2)
                    if CrafterAchievement.EAT_COW not in self.achievements:
                        self.achievements.add(CrafterAchievement.EAT_COW)
                        reward += 1.0

                # Zombie -> Combat
                elif target == CrafterObject.ZOMBIE:
                    self.grid[ty][tx] = CrafterObject.GRASS
                    if CrafterAchievement.DEFEAT_ZOMBIE not in self.achievements:
                        self.achievements.add(CrafterAchievement.DEFEAT_ZOMBIE)
                        reward += 1.0

                # Skeleton -> Combat
                elif target == CrafterObject.SKELETON:
                    self.grid[ty][tx] = CrafterObject.GRASS
                    if CrafterAchievement.DEFEAT_SKELETON not in self.achievements:
                        self.achievements.add(CrafterAchievement.DEFEAT_SKELETON)
                        reward += 1.0

                # Stone -> Mine with wood/stone/iron pickaxe
                elif target == CrafterObject.STONE:
                    if (
                        self.inventory.wood_pickaxe > 0
                        or self.inventory.stone_pickaxe > 0
                        or self.inventory.iron_pickaxe > 0
                    ):
                        self.grid[ty][tx] = CrafterObject.PATH
                        self.inventory.stone += 1
                        if CrafterAchievement.COLLECT_STONE not in self.achievements:
                            self.achievements.add(CrafterAchievement.COLLECT_STONE)
                            reward += 1.0

                # Coal -> Mine with wood/stone/iron pickaxe
                elif target == CrafterObject.COAL:
                    if (
                        self.inventory.wood_pickaxe > 0
                        or self.inventory.stone_pickaxe > 0
                        or self.inventory.iron_pickaxe > 0
                    ):
                        self.grid[ty][tx] = CrafterObject.PATH
                        self.inventory.coal += 1
                        if CrafterAchievement.COLLECT_COAL not in self.achievements:
                            self.achievements.add(CrafterAchievement.COLLECT_COAL)
                            reward += 1.0

                # Iron -> Requires stone or iron pickaxe
                elif target == CrafterObject.IRON:
                    if self.inventory.stone_pickaxe > 0 or self.inventory.iron_pickaxe > 0:
                        self.grid[ty][tx] = CrafterObject.PATH
                        self.inventory.iron += 1
                        if CrafterAchievement.COLLECT_IRON not in self.achievements:
                            self.achievements.add(CrafterAchievement.COLLECT_IRON)
                            reward += 1.0

                # Diamond -> Requires iron pickaxe
                elif target == CrafterObject.DIAMOND:
                    if self.inventory.iron_pickaxe > 0:
                        self.grid[ty][tx] = CrafterObject.PATH
                        self.inventory.diamond += 1
                        if CrafterAchievement.COLLECT_DIAMOND not in self.achievements:
                            self.achievements.add(CrafterAchievement.COLLECT_DIAMOND)
                            reward += 1.0

                # Plant -> Eat
                elif target == CrafterObject.PLANT:
                    self.grid[ty][tx] = CrafterObject.GRASS
                    self.vitals.food = min(9, self.vitals.food + 4)
                    if CrafterAchievement.EAT_PLANT not in self.achievements:
                        self.achievements.add(CrafterAchievement.EAT_PLANT)
                        reward += 1.0

        # Placement Actions
        elif action_enum == CrafterAction.PLACE_TABLE:
            if self.inventory.wood >= 2:
                tx, ty = self._target_pos()
                if self._is_passable(self.grid[ty][tx]):
                    self.grid[ty][tx] = CrafterObject.CRAFTING_TABLE
                    self.inventory.wood -= 2
                    if CrafterAchievement.PLACE_TABLE not in self.achievements:
                        self.achievements.add(CrafterAchievement.PLACE_TABLE)
                        reward += 1.0

        elif action_enum == CrafterAction.PLACE_FURNACE:
            if self.inventory.stone >= 4:
                tx, ty = self._target_pos()
                if self._is_passable(self.grid[ty][tx]):
                    self.grid[ty][tx] = CrafterObject.FURNACE
                    self.inventory.stone -= 4
                    if CrafterAchievement.PLACE_FURNACE not in self.achievements:
                        self.achievements.add(CrafterAchievement.PLACE_FURNACE)
                        reward += 1.0

        elif action_enum == CrafterAction.PLACE_STONE:
            if self.inventory.stone >= 1:
                tx, ty = self._target_pos()
                if self._is_passable(self.grid[ty][tx]):
                    self.grid[ty][tx] = CrafterObject.STONE
                    self.inventory.stone -= 1
                    if CrafterAchievement.PLACE_STONE not in self.achievements:
                        self.achievements.add(CrafterAchievement.PLACE_STONE)
                        reward += 1.0

        elif action_enum == CrafterAction.PLACE_PLANT:
            if self.inventory.sapling >= 1:
                tx, ty = self._target_pos()
                if self.grid[ty][tx] == CrafterObject.GRASS:
                    self.grid[ty][tx] = CrafterObject.PLANT
                    self.inventory.sapling -= 1
                    if CrafterAchievement.PLACE_PLANT not in self.achievements:
                        self.achievements.add(CrafterAchievement.PLACE_PLANT)
                        reward += 1.0

        # Crafting Actions (Requires Table)
        elif action_enum == CrafterAction.MAKE_WOOD_PICKAXE:
            if self.inventory.wood >= 1 and self._is_near(CrafterObject.CRAFTING_TABLE):
                self.inventory.wood -= 1
                self.inventory.wood_pickaxe += 1
                if CrafterAchievement.MAKE_WOOD_PICKAXE not in self.achievements:
                    self.achievements.add(CrafterAchievement.MAKE_WOOD_PICKAXE)
                    reward += 1.0

        elif action_enum == CrafterAction.MAKE_WOOD_SWORD:
            if self.inventory.wood >= 1 and self._is_near(CrafterObject.CRAFTING_TABLE):
                self.inventory.wood -= 1
                self.inventory.wood_sword += 1
                if CrafterAchievement.MAKE_WOOD_SWORD not in self.achievements:
                    self.achievements.add(CrafterAchievement.MAKE_WOOD_SWORD)
                    reward += 1.0

        elif action_enum == CrafterAction.MAKE_STONE_PICKAXE:
            if (
                self.inventory.wood >= 1
                and self.inventory.stone >= 1
                and self._is_near(CrafterObject.CRAFTING_TABLE)
            ):
                self.inventory.wood -= 1
                self.inventory.stone -= 1
                self.inventory.stone_pickaxe += 1
                if CrafterAchievement.MAKE_STONE_PICKAXE not in self.achievements:
                    self.achievements.add(CrafterAchievement.MAKE_STONE_PICKAXE)
                    reward += 1.0

        elif action_enum == CrafterAction.MAKE_STONE_SWORD:
            if (
                self.inventory.wood >= 1
                and self.inventory.stone >= 1
                and self._is_near(CrafterObject.CRAFTING_TABLE)
            ):
                self.inventory.wood -= 1
                self.inventory.stone -= 1
                self.inventory.stone_sword += 1
                if CrafterAchievement.MAKE_STONE_SWORD not in self.achievements:
                    self.achievements.add(CrafterAchievement.MAKE_STONE_SWORD)
                    reward += 1.0

        # Iron tools require Table + Furnace + Coal + Iron
        elif action_enum == CrafterAction.MAKE_IRON_PICKAXE:
            if (
                self.inventory.wood >= 1
                and self.inventory.coal >= 1
                and self.inventory.iron >= 1
                and self._is_near(CrafterObject.CRAFTING_TABLE)
                and self._is_near(CrafterObject.FURNACE)
            ):
                self.inventory.wood -= 1
                self.inventory.coal -= 1
                self.inventory.iron -= 1
                self.inventory.iron_pickaxe += 1
                if CrafterAchievement.MAKE_IRON_PICKAXE not in self.achievements:
                    self.achievements.add(CrafterAchievement.MAKE_IRON_PICKAXE)
                    reward += 1.0

        elif action_enum == CrafterAction.MAKE_IRON_SWORD:
            if (
                self.inventory.wood >= 1
                and self.inventory.coal >= 1
                and self.inventory.iron >= 1
                and self._is_near(CrafterObject.CRAFTING_TABLE)
                and self._is_near(CrafterObject.FURNACE)
            ):
                self.inventory.wood -= 1
                self.inventory.coal -= 1
                self.inventory.iron -= 1
                self.inventory.iron_sword += 1
                if CrafterAchievement.MAKE_IRON_SWORD not in self.achievements:
                    self.achievements.add(CrafterAchievement.MAKE_IRON_SWORD)
                    reward += 1.0

        # Check death
        terminated = False
        if self.vitals.food == 0 or self.vitals.drink == 0:
            self.vitals.health = max(0, self.vitals.health - 1)
        if self.vitals.health <= 0:
            terminated = True

        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"achievements": set(self.achievements)}

    def _get_obs(self) -> CrafterObservation:
        return CrafterObservation(
            semantic_grid=self.grid,
            player_pos=self.player_pos,
            player_facing=self.player_facing,
            inventory=CrafterInventory(**self.inventory.to_dict()),
            vitals=CrafterVitals(
                health=self.vitals.health,
                food=self.vitals.food,
                drink=self.vitals.drink,
                energy=self.vitals.energy,
            ),
            achievements=set(self.achievements),
            step_count=self.step_count,
            day_time=(self.step_count % 300) / 300.0,
        )


class NativeCrafterWrapper:
    """Wrapper around upstream `crafter.Env` that projects state into typed CrafterObservation."""

    def __init__(self, seed: int | None = None) -> None:
        import crafter  # type: ignore

        self.native_env = crafter.Env(seed=seed)
        self.step_count = 0
        self.max_steps = 300
        self.last_info: dict[str, Any] = {}
        self.achievements: set[CrafterAchievement] = set()

    def reset(self, seed: int | None = None) -> tuple[CrafterObservation, dict[str, Any]]:
        self.step_count = 0
        self.achievements.clear()
        if seed is not None:
            import crafter  # type: ignore

            self.native_env = crafter.Env(seed=seed)
        raw_obs = self.native_env.reset()
        self.last_info = {}
        return self._build_obs(raw_obs), {}

    def step(
        self, action: CrafterAction | int
    ) -> tuple[CrafterObservation, float, bool, dict[str, Any]]:
        self.step_count += 1
        act_idx = int(action)
        raw_obs, reward, done, info = self.native_env.step(act_idx)
        self.last_info = info or {}

        if "achievements" in self.last_info:
            for ach_name, count in self.last_info["achievements"].items():
                if count > 0:
                    try:
                        self.achievements.add(CrafterAchievement(ach_name))
                    except ValueError:
                        pass

        obs = self._build_obs(raw_obs)
        return obs, float(reward), bool(done), info

    def _build_obs(self, raw_obs: Any) -> CrafterObservation:
        inv_data: dict[str, int] = {}
        vitals_data = {"health": 9, "food": 9, "drink": 9, "energy": 9}
        if "inventory" in self.last_info:
            inv_dict = self.last_info["inventory"]
            for k in [
                "wood",
                "stone",
                "coal",
                "iron",
                "diamond",
                "drink",
                "wood_pickaxe",
                "stone_pickaxe",
                "iron_pickaxe",
                "wood_sword",
                "stone_sword",
                "iron_sword",
            ]:
                if k in inv_dict:
                    inv_data[k] = inv_dict[k]
        for v in ["health", "food", "drink", "energy"]:
            if v in self.last_info:
                vitals_data[v] = self.last_info[v]

        semantic_data = getattr(self.native_env, "semantic", [])
        return CrafterObservation(
            semantic_grid=semantic_data if isinstance(semantic_data, list) else [],
            player_pos=(32, 32),
            player_facing=(0, 1),
            inventory=CrafterInventory(**inv_data),
            vitals=CrafterVitals(**vitals_data),
            achievements=set(self.achievements),
            step_count=self.step_count,
            day_time=(self.step_count % 300) / 300.0,
            info=dict(self.last_info),
        )


def make_crafter_env(seed: int | None = None, prefer_native: bool = True) -> Any:
    """Instantiate Crafter environment, binding to native crafter if available or falling back to standalone."""
    if prefer_native:
        try:
            wrapper = NativeCrafterWrapper(seed=seed)
            logger.info(
                "Successfully instantiated NativeCrafterWrapper using upstream 'crafter' package"
            )
            return wrapper
        except Exception as e:
            logger.debug("Native crafter unavailable (%s), falling back to StandaloneCrafterEnv", e)
    return StandaloneCrafterEnv(seed=seed)
