"""
Sokoban Standalone and Gym Environment Wrapper.

Provides a deterministic, zero-dependency Sokoban engine supporting all 5 tiers
with exact push physics, win verification, and deadlock detection.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import (
    SokobanAction,
    SokobanDeadlockType,
    SokobanObservation,
    SokobanTier,
    SokobanTile,
)

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = {
    SokobanAction.UP: (-1, 0),
    SokobanAction.DOWN: (1, 0),
    SokobanAction.LEFT: (0, -1),
    SokobanAction.RIGHT: (0, 1),
}


class StandaloneSokobanEnv:
    """Zero-dependency deterministic Sokoban environment engine."""

    def __init__(
        self,
        tier: SokobanTier | str = SokobanTier.TIER_1_DIRECT_PUSH,
        seed: int = 42,
        max_steps: int = 120,
    ) -> None:
        self.tier = SokobanTier(tier)
        self.seed = seed
        self.max_steps = max_steps
        self.step_count = 0
        self.player_pos = (1, 1)
        self.boxes: set[tuple[int, int]] = set()
        self.targets: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        self.width = 7
        self.height = 7
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> SokobanObservation:
        """Reset environment to initial level layout according to tier."""
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        self._build_tier_layout()
        return self._get_obs()

    def _build_tier_layout(self) -> None:
        """Construct canonical level layout according to evaluation tier."""
        self.walls.clear()
        self.boxes.clear()
        self.targets.clear()

        if self.tier == SokobanTier.TIER_1_DIRECT_PUSH:
            # Simple corridor / open room: Player pushes box directly to target
            self.height, self.width = 6, 7
            for r in range(self.height):
                for c in range(self.width):
                    if r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1:
                        self.walls.add((r, c))
            self.player_pos = (2, 2)
            self.boxes = {(2, 3)}
            self.targets = {(2, 5)}

        elif self.tier == SokobanTier.TIER_2_OBSTACLE_NAVIGATION:
            # Agent must maneuver around a wall pillar to get behind box
            self.height, self.width = 7, 8
            for r in range(self.height):
                for c in range(self.width):
                    if r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1:
                        self.walls.add((r, c))
            # Center pillar
            self.walls.add((2, 3))
            self.walls.add((3, 3))
            self.player_pos = (4, 1)
            self.boxes = {(4, 4)}
            self.targets = {(1, 4)}

        elif self.tier == SokobanTier.TIER_3_CORNER_DEADLOCK_AVOIDANCE:
            # Trap layout: Corners (1, 1), (1, 5), (5, 1) are fatal dead-ends.
            # Only (5, 5) is a valid target. Pushing carelessly into corners causes deadlock.
            self.height, self.width = 7, 7
            for r in range(self.height):
                for c in range(self.width):
                    if r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1:
                        self.walls.add((r, c))
            # Wall obstacle that prevents trivial diagonal pushes
            self.walls.add((3, 2))
            self.player_pos = (2, 2)
            self.boxes = {(3, 3)}
            self.targets = {(5, 5)}

        elif self.tier == SokobanTier.TIER_4_MULTI_BOX_ASSIGNMENT:
            # 2 boxes, 2 targets. Placing Box 1 into target (2, 5) must happen before
            # or without blocking path to target (4, 5).
            self.height, self.width = 7, 8
            for r in range(self.height):
                for c in range(self.width):
                    if r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1:
                        self.walls.add((r, c))
            self.walls.add((3, 4))
            self.player_pos = (1, 1)
            self.boxes = {(2, 2), (4, 2)}
            self.targets = {(2, 5), (4, 5)}

        elif self.tier == SokobanTier.TIER_5_COMBINATORIAL_MAZE:
            # Classic Boxoban mini-puzzle requiring unblocking and keyhole pushing
            self.height, self.width = 8, 8
            for r in range(self.height):
                for c in range(self.width):
                    if r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1:
                        self.walls.add((r, c))
            # Maze partition
            self.walls.add((2, 2))
            self.walls.add((2, 4))
            self.walls.add((4, 2))
            self.walls.add((4, 4))
            self.player_pos = (1, 3)
            self.boxes = {(2, 3), (3, 3)}
            self.targets = {(5, 3), (6, 3)}

    def check_deadlock(self, box_pos: tuple[int, int]) -> SokobanDeadlockType:
        """Check if a box is in an irreversible deadlock."""
        if box_pos in self.targets:
            return SokobanDeadlockType.NONE

        r, c = box_pos
        up_wall = (r - 1, c) in self.walls
        down_wall = (r + 1, c) in self.walls
        left_wall = (r, c - 1) in self.walls
        right_wall = (r, c + 1) in self.walls

        # Corner deadlock: blocked in both vertical and horizontal directions
        if (up_wall or down_wall) and (left_wall or right_wall):
            return SokobanDeadlockType.CORNER

        # 2x2 square deadlock of boxes and walls
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
            quad = [
                (r + dr, c + dc),
                (r + dr + 1, c + dc),
                (r + dr, c + dc + 1),
                (r + dr + 1, c + dc + 1),
            ]
            all_blocked = all(pos in self.walls or pos in self.boxes for pos in quad)
            if all_blocked:
                # If any box in this quad is not on a target, it is an irreversible 2x2 deadlock
                if any(pos in self.boxes and pos not in self.targets for pos in quad):
                    return SokobanDeadlockType.SQUARE_2X2

        return SokobanDeadlockType.NONE

    def step(
        self, action: SokobanAction | int
    ) -> tuple[SokobanObservation, float, bool, dict[str, Any]]:
        """Apply directional action, executing push physics."""
        self.step_count += 1
        act = SokobanAction(action)
        dr, dc = DIRECTION_DELTAS[act]

        pr, pc = self.player_pos
        nr, nc = pr + dr, pc + dc

        deadlock_occurred = False

        if (nr, nc) in self.walls:
            # Blocked by wall - no movement
            pass
        elif (nr, nc) in self.boxes:
            # Attempt to push box
            nnr, nnc = nr + dr, nc + dc
            if (nnr, nnc) not in self.walls and (nnr, nnc) not in self.boxes:
                # Valid push!
                self.boxes.remove((nr, nc))
                self.boxes.add((nnr, nnc))
                self.player_pos = (nr, nc)
                # Check for deadlocks
                dl = self.check_deadlock((nnr, nnc))
                if dl != SokobanDeadlockType.NONE:
                    deadlock_occurred = True
        else:
            # Free walk
            self.player_pos = (nr, nc)

        won = self.boxes == self.targets
        done = won or self.step_count >= self.max_steps
        reward = 10.0 if won else (-1.0 if deadlock_occurred else -0.01)

        info = {
            "won": won,
            "deadlock": deadlock_occurred,
            "steps": self.step_count,
        }

        obs = self._get_obs(done=done, won=won, deadlock=deadlock_occurred, info=info)
        return obs, reward, done, info

    def _get_obs(
        self,
        done: bool = False,
        won: bool = False,
        deadlock: bool = False,
        info: dict[str, Any] | None = None,
    ) -> SokobanObservation:
        """Construct full typed observation grid."""
        grid = [[int(SokobanTile.EMPTY) for _ in range(self.width)] for _ in range(self.height)]

        for r, c in self.walls:
            grid[r][c] = int(SokobanTile.WALL)
        for r, c in self.targets:
            grid[r][c] = int(SokobanTile.TARGET)
        for r, c in self.boxes:
            if (r, c) in self.targets:
                grid[r][c] = int(SokobanTile.BOX_ON_TARGET)
            else:
                grid[r][c] = int(SokobanTile.BOX)

        pr, pc = self.player_pos
        if (pr, pc) in self.targets:
            grid[pr][pc] = int(SokobanTile.PLAYER_ON_TARGET)
        else:
            grid[pr][pc] = int(SokobanTile.PLAYER)

        return SokobanObservation(
            grid=grid,
            player_pos=self.player_pos,
            boxes=sorted(list(self.boxes)),
            targets=sorted(list(self.targets)),
            step_count=self.step_count,
            max_steps=self.max_steps,
            done=done,
            won=won,
            deadlock_detected=deadlock,
            info=info or {},
        )


class NativeSokobanWrapper:
    """Wrapper around upstream `gym-sokoban` environment (e.g. Sokoban-v0).

    Translates raw gym observations and room_state into typed SokobanObservation.
    """

    def __init__(
        self,
        env_name: str = "Sokoban-v0",
        seed: int = 42,
        max_steps: int = 120,
    ) -> None:
        import gym  # type: ignore
        import gym_sokoban  # type: ignore # noqa: F401

        self.native_env = gym.make(env_name)
        self.seed = seed
        self.max_steps = max_steps
        self.step_count = 0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> SokobanObservation:
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        try:
            self.native_env.seed(self.seed)
        except Exception:
            pass
        _raw_obs = self.native_env.reset()
        return self._extract_obs(done=False, won=False)

    def step(
        self, action: SokobanAction | int
    ) -> tuple[SokobanObservation, float, bool, dict[str, Any]]:
        self.step_count += 1
        # In gym-sokoban: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        gym_act = int(action) + 1
        _raw_obs, reward, done, info = self.native_env.step(gym_act)

        room = getattr(self.native_env, "room_state", None)
        won = False
        if room is not None:
            has_unsolved_boxes = 4 in room
            won = not has_unsolved_boxes and (3 in room)

        obs = self._extract_obs(
            done=done or won or self.step_count >= self.max_steps,
            won=won,
            info=info or {},
        )
        return obs, float(reward), done or won, info or {}

    def _extract_obs(
        self,
        done: bool = False,
        won: bool = False,
        info: dict[str, Any] | None = None,
    ) -> SokobanObservation:
        room = getattr(self.native_env, "room_state", None)
        player_pos = tuple(getattr(self.native_env, "player_position", (1, 1)))

        boxes: list[tuple[int, int]] = []
        targets: list[tuple[int, int]] = []
        grid: list[list[int]] = []

        if room is not None:
            h, w = room.shape
            grid = [[int(room[r][c]) for c in range(w)] for r in range(h)]
            for r in range(h):
                for c in range(w):
                    tile = int(room[r][c])
                    if tile in (3, 4):
                        boxes.append((r, c))
                    if tile in (2, 3):
                        targets.append((r, c))
        else:
            grid = [[0 for _ in range(7)] for _ in range(7)]

        return SokobanObservation(
            grid=grid,
            player_pos=(int(player_pos[0]), int(player_pos[1])),
            boxes=sorted(boxes),
            targets=sorted(targets),
            step_count=self.step_count,
            max_steps=self.max_steps,
            done=done,
            won=won,
            deadlock_detected=False,
            info=info or {},
        )


def make_sokoban_env(
    tier: SokobanTier | str = SokobanTier.TIER_1_DIRECT_PUSH,
    seed: int = 42,
    max_steps: int = 120,
    prefer_native: bool = False,
) -> Any:
    """Factory creating configured Sokoban environments.

    Supports native upstream `gym-sokoban` or deterministic 5-tier `StandaloneSokobanEnv`.
    """
    if prefer_native:
        try:
            wrapper = NativeSokobanWrapper(seed=seed, max_steps=max_steps)
            logger.info("Successfully bound to native gym-sokoban environment")
            return wrapper
        except Exception as e:
            logger.debug(
                "Native gym-sokoban unavailable (%s), falling back to StandaloneSokobanEnv", e
            )
    return StandaloneSokobanEnv(tier=tier, seed=seed, max_steps=max_steps)
