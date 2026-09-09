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
        self.table_pos: tuple[int, int] | None = None
        self.furnace_pos: tuple[int, int] | None = None

    def reset(self) -> None:
        """Reset internal plan and spatial landmarks."""
        self.current_plan.clear()
        self.table_pos = None
        self.furnace_pos = None

    def plan_next_action(
        self, obs: CrafterObservation, goal: CrafterGoal | None = None
    ) -> CrafterAction:
        """Select next optimal action respecting survival, mob combat, and tech-tree DAG."""
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0

        # 1. Tactical Monster Combat & Self-Defense
        if height > 0 and width > 0:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if obs.semantic_grid[ny][nx] in (CrafterObject.ZOMBIE, CrafterObject.SKELETON):
                        if obs.player_facing == (dx, dy):
                            return CrafterAction.DO
                        if (dx, dy) == (-1, 0):
                            return CrafterAction.MOVE_LEFT
                        if (dx, dy) == (1, 0):
                            return CrafterAction.MOVE_RIGHT
                        if (dx, dy) == (0, -1):
                            return CrafterAction.MOVE_UP
                        if (dx, dy) == (0, 1):
                            return CrafterAction.MOVE_DOWN

        # 2. Vital Survival Interrupts (raised proactive thresholds)
        if obs.vitals.energy <= 2:
            if not self._is_near(obs, CrafterObject.ZOMBIE, radius=3) and not self._is_near(
                obs, CrafterObject.SKELETON, radius=3
            ):
                return CrafterAction.SLEEP

        if obs.vitals.drink <= 4:
            water_act = self._navigate_and_interact(obs, CrafterObject.WATER)
            if water_act is not None:
                return water_act

        if obs.vitals.food <= 4:
            cow_act = self._navigate_and_interact(obs, CrafterObject.COW)
            if cow_act is not None:
                return cow_act

        # 3. Target Goal Planning
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
            return act or self._explore_passable(obs)

        if ach == CrafterAchievement.PLACE_TABLE:
            if inv.wood < 2:
                act = self._navigate_and_interact(obs, CrafterObject.TREE)
                return act or self._explore_passable(obs)
            tx = obs.player_pos[0] + obs.player_facing[0]
            ty = obs.player_pos[1] + obs.player_facing[1]
            if 0 <= tx < len(obs.semantic_grid[0]) and 0 <= ty < len(obs.semantic_grid):
                if obs.semantic_grid[ty][tx] in (
                    CrafterObject.GRASS,
                    CrafterObject.PATH,
                    CrafterObject.SAND,
                ):
                    self.table_pos = (tx, ty)
                    return CrafterAction.PLACE_TABLE
            return CrafterAction.MOVE_LEFT

        if ach == CrafterAchievement.MAKE_WOOD_PICKAXE:
            if CrafterAchievement.PLACE_TABLE not in obs.achievements and not self._is_near(
                obs, CrafterObject.CRAFTING_TABLE, radius=30
            ):
                return self._plan_achievement(obs, CrafterAchievement.PLACE_TABLE)
            if inv.wood < 1:
                act = self._navigate_and_interact(obs, CrafterObject.TREE)
                return act or self._explore_passable(obs)
            if not self._is_near(obs, CrafterObject.CRAFTING_TABLE, radius=1):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act
            return CrafterAction.MAKE_WOOD_PICKAXE

        if ach == CrafterAchievement.COLLECT_STONE:
            if inv.wood_pickaxe == 0 and inv.stone_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.MAKE_WOOD_PICKAXE)
            act = self._navigate_and_interact(obs, CrafterObject.STONE)
            return act or self._explore_passable(obs)

        if ach == CrafterAchievement.MAKE_STONE_PICKAXE:
            if CrafterAchievement.PLACE_TABLE not in obs.achievements and not self._is_near(
                obs, CrafterObject.CRAFTING_TABLE, radius=30
            ):
                return self._plan_achievement(obs, CrafterAchievement.PLACE_TABLE)
            if inv.stone < 1:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_STONE)
            if inv.wood < 1:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_WOOD)
            if not self._is_near(obs, CrafterObject.CRAFTING_TABLE, radius=1):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act
            return CrafterAction.MAKE_STONE_PICKAXE

        if ach == CrafterAchievement.COLLECT_COAL:
            if inv.wood_pickaxe == 0 and inv.stone_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.MAKE_WOOD_PICKAXE)
            act = self._navigate_and_interact(obs, CrafterObject.COAL)
            return act or self._explore_passable(obs)

        if ach == CrafterAchievement.COLLECT_IRON:
            if inv.stone_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.MAKE_STONE_PICKAXE)
            act = self._navigate_and_interact(obs, CrafterObject.IRON)
            return act or self._explore_passable(obs)

        if ach == CrafterAchievement.PLACE_FURNACE:
            if inv.stone < 4:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_STONE)
            # If crafting table exists, ensure we place furnace right next to table
            if self.table_pos is not None and not self._is_near(
                obs, CrafterObject.CRAFTING_TABLE, radius=1
            ):
                act = self._navigate_and_interact(obs, CrafterObject.CRAFTING_TABLE, face_only=True)
                if act:
                    return act

            px, py = obs.player_pos
            passable = (CrafterObject.GRASS, CrafterObject.PATH, CrafterObject.SAND)
            tx = px + obs.player_facing[0]
            ty = py + obs.player_facing[1]
            if 0 <= tx < len(obs.semantic_grid[0]) and 0 <= ty < len(obs.semantic_grid):
                if obs.semantic_grid[ty][tx] in passable:
                    self.furnace_pos = (tx, ty)
                    return CrafterAction.PLACE_FURNACE

            # Rotate to a passable neighbor tile
            for (dx, dy), action in (
                ((-1, 0), CrafterAction.MOVE_LEFT),
                ((1, 0), CrafterAction.MOVE_RIGHT),
                ((0, -1), CrafterAction.MOVE_UP),
                ((0, 1), CrafterAction.MOVE_DOWN),
            ):
                nx, ny = px + dx, py + dy
                if 0 <= nx < len(obs.semantic_grid[0]) and 0 <= ny < len(obs.semantic_grid):
                    if obs.semantic_grid[ny][nx] in passable:
                        return action

            return CrafterAction.MOVE_LEFT

        if ach == CrafterAchievement.MAKE_IRON_PICKAXE:
            if CrafterAchievement.PLACE_TABLE not in obs.achievements and not self._is_near(
                obs, CrafterObject.CRAFTING_TABLE, radius=30
            ):
                return self._plan_achievement(obs, CrafterAchievement.PLACE_TABLE)
            if CrafterAchievement.PLACE_FURNACE not in obs.achievements and not self._is_near(
                obs, CrafterObject.FURNACE, radius=30
            ):
                return self._plan_achievement(obs, CrafterAchievement.PLACE_FURNACE)
            if inv.wood < 1:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_WOOD)
            if inv.coal < 1:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_COAL)
            if inv.iron < 1:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_IRON)

            near_table = self._is_near(obs, CrafterObject.CRAFTING_TABLE, radius=1)
            near_furnace = self._is_near(obs, CrafterObject.FURNACE, radius=1)
            if not (near_table and near_furnace):
                overlap_step = self._navigate_to_overlap(
                    obs, CrafterObject.CRAFTING_TABLE, CrafterObject.FURNACE
                )
                if overlap_step:
                    return overlap_step
                if not near_table:
                    act = self._navigate_and_interact(
                        obs, CrafterObject.CRAFTING_TABLE, face_only=True
                    )
                    if act:
                        return act
                elif not near_furnace:
                    act = self._navigate_and_interact(obs, CrafterObject.FURNACE, face_only=True)
                    if act:
                        return act
            return CrafterAction.MAKE_IRON_PICKAXE

        if ach == CrafterAchievement.COLLECT_DIAMOND:
            if inv.iron_pickaxe == 0:
                return self._plan_achievement(obs, CrafterAchievement.COLLECT_IRON)
            act = self._navigate_and_interact(obs, CrafterObject.DIAMOND)
            return act or self._explore_passable(obs)

        if ach == CrafterAchievement.EAT_COW:
            act = self._navigate_and_interact(obs, CrafterObject.COW)
            return act or self._explore_passable(obs)

        if ach == CrafterAchievement.COLLECT_DRINK:
            act = self._navigate_and_interact(obs, CrafterObject.WATER)
            return act or self._explore_passable(obs)

        if ach == CrafterAchievement.SURVIVE:
            if obs.vitals.drink <= 6:
                act = self._navigate_and_interact(obs, CrafterObject.WATER)
                if act:
                    return act
            if obs.vitals.food <= 6:
                act = self._navigate_and_interact(obs, CrafterObject.COW)
                if act:
                    return act
            return self._explore_passable(obs)

        return self._explore_passable(obs)

    def _is_near(self, obs: CrafterObservation, obj_type: CrafterObject, radius: int = 2) -> bool:
        px, py = obs.player_pos
        if obj_type == CrafterObject.CRAFTING_TABLE and self.table_pos is not None:
            tx, ty = self.table_pos
            if abs(px - tx) <= radius and abs(py - ty) <= radius:
                return True
        if obj_type == CrafterObject.FURNACE and self.furnace_pos is not None:
            fx, fy = self.furnace_pos
            if abs(px - fx) <= radius and abs(py - fy) <= radius:
                return True
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0
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
        for y in range(height):
            for x in range(width):
                if obs.semantic_grid[y][x] == target_type:
                    dist = abs(px - x) + abs(py - y)
                    if dist < best_dist:
                        best_dist = dist
                        target_pos = (x, y)

        if (
            target_pos is None
            and target_type == CrafterObject.CRAFTING_TABLE
            and self.table_pos is not None
        ):
            target_pos = self.table_pos

        if (
            target_pos is None
            and target_type == CrafterObject.FURNACE
            and self.furnace_pos is not None
        ):
            target_pos = self.furnace_pos

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

            if len(path) >= 60:  # Search depth cap
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

    def _explore_passable(self, obs: CrafterObservation) -> CrafterAction:
        """Move in an available passable direction to discover new terrain."""
        px, py = obs.player_pos
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0
        passable_ids = (
            CrafterObject.GRASS,
            CrafterObject.PATH,
            CrafterObject.SAND,
            CrafterObject.EMPTY,
        )
        fx, fy = obs.player_facing
        nx, ny = px + fx, py + fy
        if 0 <= nx < width and 0 <= ny < height and obs.semantic_grid[ny][nx] in passable_ids:
            if fx == -1:
                return CrafterAction.MOVE_LEFT
            if fx == 1:
                return CrafterAction.MOVE_RIGHT
            if fy == -1:
                return CrafterAction.MOVE_UP
            if fy == 1:
                return CrafterAction.MOVE_DOWN

        for act, (dx, dy) in (
            (CrafterAction.MOVE_UP, (0, -1)),
            (CrafterAction.MOVE_RIGHT, (1, 0)),
            (CrafterAction.MOVE_DOWN, (0, 1)),
            (CrafterAction.MOVE_LEFT, (-1, 0)),
        ):
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height and obs.semantic_grid[ny][nx] in passable_ids:
                return act

        return CrafterAction.DO

    def _navigate_to_overlap(
        self,
        obs: CrafterObservation,
        obj_a: CrafterObject,
        obj_b: CrafterObject,
    ) -> CrafterAction | None:
        """Find a passable tile within Chebyshev radius 1 of both objects and step toward it."""
        pos_a = self.table_pos if obj_a == CrafterObject.CRAFTING_TABLE else self.furnace_pos
        pos_b = self.furnace_pos if obj_b == CrafterObject.FURNACE else self.table_pos
        if pos_a is None or pos_b is None:
            return None

        px, py = obs.player_pos
        ax, ay = pos_a
        bx, by = pos_b
        height = len(obs.semantic_grid)
        width = len(obs.semantic_grid[0]) if height > 0 else 0

        passable_ids = (
            CrafterObject.GRASS,
            CrafterObject.PATH,
            CrafterObject.SAND,
            CrafterObject.EMPTY,
        )

        candidates = set()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx, cy = ax + dx, ay + dy
                if 0 <= cx < width and 0 <= cy < height:
                    if abs(cx - bx) <= 1 and abs(cy - by) <= 1:
                        if obs.semantic_grid[cy][cx] in passable_ids:
                            candidates.add((cx, cy))

        if not candidates or (px, py) in candidates:
            return None

        queue = deque([(px, py, [])])
        visited = {(px, py)}
        while queue:
            cx, cy, path = queue.popleft()
            if (cx, cy) in candidates:
                if path:
                    return path[0]
                return None

            if len(path) >= 60:
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
