"""
Sokoban Native Gym Environment Wrapper.

Provides native integration with upstream gym-sokoban for all 5 Boxoban tiers
with exact push physics, observation translation, and win verification.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import (
    SokobanAction,
    SokobanObservation,
    SokobanTier,
    SokobanTile,
)

logger = logging.getLogger(__name__)


class NativeSokobanWrapper:
    """Wrapper around upstream `gym-sokoban` environment (e.g. Sokoban-v0).

    Translates raw gym observations and room_state into typed SokobanObservation.
    """

    is_native: bool = True

    TIER_ENV_MAP = {
        SokobanTier.TIER_1_DIRECT_PUSH: "Sokoban-small-v0",
        SokobanTier.TIER_2_OBSTACLE_NAVIGATION: "Sokoban-small-v1",
        SokobanTier.TIER_3_CORNER_DEADLOCK_AVOIDANCE: "Sokoban-v0",
        SokobanTier.TIER_4_MULTI_BOX_ASSIGNMENT: "Sokoban-v1",
        SokobanTier.TIER_5_COMBINATORIAL_MAZE: "Sokoban-large-v0",
    }

    def __init__(
        self,
        tier: SokobanTier | str = SokobanTier.TIER_1_DIRECT_PUSH,
        env_name: str | None = None,
        seed: int = 42,
        max_steps: int = 120,
    ) -> None:
        import numpy as np

        if not hasattr(np, "bool8"):
            np.bool8 = np.bool_  # type: ignore

        import gym  # type: ignore
        import gym_sokoban  # type: ignore # noqa: F401

        self.tier = SokobanTier(tier)
        if env_name is None:
            env_name = self.TIER_ENV_MAP.get(self.tier, "Sokoban-v0")
        self.native_env = gym.make(env_name)
        self.seed = seed
        self.max_steps = max_steps
        self.step_count = 0

    @property
    def player_pos(self) -> tuple[int, int]:
        pos = getattr(self.native_env, "player_position", (1, 1))
        return (int(pos[0]), int(pos[1]))

    @property
    def boxes(self) -> list[tuple[int, int]]:
        room = getattr(self.native_env, "room_state", None)
        if room is None:
            return []
        h, w = room.shape
        return [(r, c) for r in range(h) for c in range(w) if room[r][c] in (3, 4)]

    @property
    def targets(self) -> list[tuple[int, int]]:
        room = getattr(self.native_env, "room_state", None)
        if room is None:
            return []
        h, w = room.shape
        return [(r, c) for r in range(h) for c in range(w) if room[r][c] in (2, 3)]

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

        is_done = bool(done or won or self.step_count >= self.max_steps)
        obs = self._extract_obs(
            done=is_done,
            won=won,
            info=info or {},
        )
        return obs, float(reward), is_done, info or {}

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

        GYM_TO_SOKOBAN_TILE = {
            0: int(SokobanTile.WALL),
            1: int(SokobanTile.EMPTY),
            2: int(SokobanTile.TARGET),
            3: int(SokobanTile.BOX_ON_TARGET),
            4: int(SokobanTile.BOX),
            5: int(SokobanTile.PLAYER),
        }

        if room is not None:
            h, w = room.shape
            grid = [
                [GYM_TO_SOKOBAN_TILE.get(int(room[r][c]), int(SokobanTile.EMPTY)) for c in range(w)]
                for r in range(h)
            ]
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
) -> NativeSokobanWrapper:
    """Factory creating configured native Sokoban environments."""
    try:
        wrapper = NativeSokobanWrapper(tier=tier, seed=seed, max_steps=max_steps)
        logger.info(
            "Successfully bound to native gym-sokoban environment (%s)",
            wrapper.native_env.spec.id if hasattr(wrapper.native_env, "spec") else "native",
        )
        return wrapper
    except Exception as e:
        raise RuntimeError(
            f"Native 'gym-sokoban' upstream package is required but failed: {e}"
        ) from e
