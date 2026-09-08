"""
Crafter Action and Causal Planning Adapter.

Implements CrafterCausalPlanner which combines:
1. Urgent survival vital interrupts (energy, drink, food).
2. Recursive tech-tree causal recipe DAG (wood -> table -> pickaxe -> stone -> furnace -> iron -> diamond).
3. 2D grid BFS pathfinding and target interaction.
"""

from __future__ import annotations

import logging
from collections import deque

from .types import (
    CrafterAchievement,
    CrafterAction,
    CrafterGoal,
    CrafterObject,
    CrafterObservation,
)

logger = logging.getLogger(__name__)


class CrafterActionAdapter:
    """
    Translates high-level causal decisions into executable Crafter discrete actions.
    """

    def __init__(self) -> None:
        self.current_plan: list[CrafterAction] = []

    def plan_next_action(
        self, obs: CrafterObservation, goal: CrafterGoal | None = None
    ) -> CrafterAction:
        """Select next optimal action respecting survival and tech-tree DAG."""
        # 1. Vital Survival Interrupts
        if obs.vitals.energy <= 2:
            return CrafterAction.SLEEP

        if obs.vitals.drink <= 2:
            water_act = self._navigate_and_interact(obs, CrafterObject.WATER)
            if water_act is not None:
                return water_act

        if obs.vitals.food <= 2:
            cow_act = self._navigate_and_interact(obs, CrafterObject.COW)
            if cow_act is not None:
                return cow_act

        # 2. Target Goal Planning
        target = goal.target_achievement if goal else None
        if target is None:
            # Default progressive tech-tree roadmap
            target = self._select_next_achievement(obs)

        return self._plan_achievement(obs, target)

    def _select_next_achievement(self, obs: CrafterObservation) -> CrafterAchievement:
        """Progressive technology tree roadmap."""
        achs = obs.achievements
        inv = obs.inventory

        if CrafterAchievement.COLLECT_WOOD not in achs or inv.wood < 2:
            return CrafterAchievement.COLLECT_WOOD
        if CrafterAchievement.PLACE_TABLE not in achs:
            return CrafterAchievement.PLACE_TABLE
        if CrafterAchievement.MAKE_WOOD_PICKAXE not in achs and inv.wood_pickaxe == 0:
            return CrafterAchievement.MAKE_WOOD_PICKAXE
        if CrafterAchievement.COLLECT_STONE not in achs or inv.stone < 1:
            return CrafterAchievement.COLLECT_STONE
        if CrafterAchievement.MAKE_STONE_PICKAXE not in achs and inv.stone_pickaxe == 0:
            return CrafterAchievement.MAKE_STONE_PICKAXE
        if CrafterAchievement.COLLECT_COAL not in achs or inv.coal < 1:
            return CrafterAchievement.COLLECT_COAL
        if CrafterAchievement.COLLECT_IRON not in achs or inv.iron < 1:
            return CrafterAchievement.COLLECT_IRON
        if CrafterAchievement.PLACE_FURNACE not in achs and inv.stone >= 4:
            return CrafterAchievement.PLACE_FURNACE
        if CrafterAchievement.MAKE_IRON_PICKAXE not in achs and inv.iron_pickaxe == 0:
            return CrafterAchievement.MAKE_IRON_PICKAXE
        if CrafterAchievement.COLLECT_DIAMOND not in achs:
            return CrafterAchievement.COLLECT_DIAMOND

        return CrafterAchievement.SURVIVE

    def _plan_achievement(self, obs: CrafterObservation, ach: CrafterAchievement) -> CrafterAction:
        """Causal dispatch for target achievement."""
        inv = obs.inventory

        if ach == CrafterAchievement.COLLECT_WOOD:
            act = self._navigate_and_interact(obs, CrafterObject.TREE)
            return act or CrafterAction.NOOP

        if ach == CrafterAchievement.PLACE_TABLE:
            if inv.wood < 2:
                act = self._navigate_and_interact(obs, CrafterObject.TREE)
                return act or CrafterAction.NOOP
            tx = obs.player_pos[0] + obs.player_facing[0]
            ty = obs.player_pos[1] + obs.player_facing[1]
            if 0 <= tx < len(obs.semantic_grid[0]) and 0 <= ty < len(obs.semantic_grid):
                if obs.semantic_grid[ty][tx] in (
                    CrafterObject.GRASS,
                    CrafterObject.PATH,
                    CrafterObject.SAND,
                ):
                    return CrafterAction.PLACE_TABLE
            return CrafterAction.MOVE_LEFT

        if ach == CrafterAchievement.MAKE_WOOD_PICKAXE:
            if CrafterAchievement.PLACE_TABLE not in obs.achievements and not self._is_near(
                obs, CrafterObject.CRAFTING_TABLE, radius=30
            ):
                return self._plan_achievement(obs, CrafterAchievement.PLACE_TABLE)
            if inv.wood < 1:
                act = self._navigate_and_interact(obs, CrafterObject.TREE)
                return act or CrafterAction.NOOP
            if not self._is_near(obs, CrafterObject.CRAFTING_TABLE):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act
            return CrafterAction.MAKE_WOOD_PICKAXE

        if ach == CrafterAchievement.COLLECT_STONE:
            if inv.wood_pickaxe == 0 and inv.stone_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.MAKE_WOOD_PICKAXE)
            act = self._navigate_and_interact(obs, CrafterObject.STONE)
            return act or CrafterAction.NOOP

        if ach == CrafterAchievement.MAKE_STONE_PICKAXE:
            if CrafterAchievement.PLACE_TABLE not in obs.achievements and not self._is_near(
                obs, CrafterObject.CRAFTING_TABLE, radius=30
            ):
                return self._plan_achievement(obs, CrafterAchievement.PLACE_TABLE)
            if inv.stone < 1:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_STONE)
            if inv.wood < 1:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_WOOD)
            if not self._is_near(obs, CrafterObject.CRAFTING_TABLE):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act
            return CrafterAction.MAKE_STONE_PICKAXE

        if ach == CrafterAchievement.COLLECT_COAL:
            if inv.wood_pickaxe == 0 and inv.stone_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.MAKE_WOOD_PICKAXE)
            act = self._navigate_and_interact(obs, CrafterObject.COAL)
            return act or CrafterAction.NOOP

        if ach == CrafterAchievement.COLLECT_IRON:
            if inv.stone_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.MAKE_STONE_PICKAXE)
            act = self._navigate_and_interact(obs, CrafterObject.IRON)
            return act or CrafterAction.NOOP

        if ach == CrafterAchievement.PLACE_FURNACE:
            if inv.stone < 4:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_STONE)
            return CrafterAction.PLACE_FURNACE

        if ach == CrafterAchievement.COLLECT_DIAMOND:
            if inv.iron_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_IRON)
            act = self._navigate_and_interact(obs, CrafterObject.DIAMOND)
            return act or CrafterAction.NOOP

        if ach == CrafterAchievement.EAT_COW:
            act = self._navigate_and_interact(obs, CrafterObject.COW)
            return act or CrafterAction.NOOP

        if ach == CrafterAchievement.COLLECT_DRINK:
            act = self._navigate_and_interact(obs, CrafterObject.WATER)
            return act or CrafterAction.NOOP

        return CrafterAction.NOOP

    def _is_near(self, obs: CrafterObservation, obj_type: CrafterObject, radius: int = 2) -> bool:
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    if obs.semantic_grid[y][x] == obj_type:
                        return True
        return False

    def _navigate_and_interact(
        self,
        obs: CrafterObservation,
        target_type: CrafterObject,
        face_only: bool = False,
    ) -> CrafterAction | None:
        """Find nearest target, navigate adjacent via BFS, and interact."""
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])

        # 1. Find nearest instance of target_type
        target_pos = None
        best_dist = float("inf")
        radius = 24
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    if obs.semantic_grid[y][x] == target_type:
                        dist = abs(px - x) + abs(py - y)
                        if dist < best_dist:
                            best_dist = dist
                            target_pos = (x, y)

        if target_pos is None:
            return None

        tx, ty = target_pos
        # Check if already adjacent
        if abs(px - tx) + abs(py - ty) == 1:
            dx, dy = tx - px, ty - py
            if obs.player_facing == (dx, dy):
                return CrafterAction.NOOP if face_only else CrafterAction.DO
            # Turn to face target
            if dx == -1:
                return CrafterAction.MOVE_LEFT
            if dx == 1:
                return CrafterAction.MOVE_RIGHT
            if dy == -1:
                return CrafterAction.MOVE_UP
            if dy == 1:
                return CrafterAction.MOVE_DOWN

        # 2. BFS to adjacent passable tile
        return self._bfs_path_step(obs, target_pos)

    def _bfs_path_step(
        self, obs: CrafterObservation, target_pos: tuple[int, int]
    ) -> CrafterAction | None:
        """Compute one-step BFS movement toward an adjacent tile of target_pos."""
        start = obs.player_pos
        tx, ty = target_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0])

        passable_ids = (
            CrafterObject.GRASS,
            CrafterObject.PATH,
            CrafterObject.SAND,
            CrafterObject.EMPTY,
        )

        queue = deque([(start[0], start[1], [])])
        visited = {start}

        while queue:
            cx, cy, path = queue.popleft()

            # Target reached if adjacent to target
            if abs(cx - tx) + abs(cy - ty) == 1:
                if path:
                    return path[0]
                return None

            if len(path) >= 30:  # Search depth cap
                continue

            for act, (dx, dy) in (
                (CrafterAction.MOVE_LEFT, (-1, 0)),
                (CrafterAction.MOVE_RIGHT, (1, 0)),
                (CrafterAction.MOVE_UP, (0, -1)),
                (CrafterAction.MOVE_DOWN, (0, 1)),
            ):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    if obs.semantic_grid[ny][nx] in passable_ids:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [act]))

        return None
