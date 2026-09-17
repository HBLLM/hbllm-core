"""
NetHack Native Environment Wrapper.

Provides native integration with upstream minihack / nle packages for
procedural multi-room dungeons, glyph grids, and tactical action execution.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import (
    NetHackAction,
    NetHackGlyph,
    NetHackObservation,
    NetHackStats,
)

logger = logging.getLogger(__name__)


class NativeNetHackWrapper:
    """Wrapper around authentic upstream minihack / nle package."""

    is_native: bool = True

    ENV_MAP = {
        1: "MiniHack-Room-5x5-v0",
        2: "MiniHack-Room-15x15-v0",
        3: "MiniHack-Corridor-R3-v0",
        4: "MiniHack-Room-Monster-5x5-v0",
        5: "MiniHack-MultiRoom-N4-v0",
    }

    ACTION_MAP = {
        NetHackAction.NORTH: 0,
        NetHackAction.EAST: 1,
        NetHackAction.SOUTH: 2,
        NetHackAction.WEST: 3,
        NetHackAction.NORTHEAST: 4,
        NetHackAction.SOUTHEAST: 5,
        NetHackAction.SOUTHWEST: 6,
        NetHackAction.NORTHWEST: 7,
        NetHackAction.WAIT: 8,
        NetHackAction.OPEN_DOOR: 8,
        NetHackAction.DESCEND_STAIRS: 11,
    }

    def __init__(self, seed: int | None = None, tier: int = 5) -> None:
        try:
            import gymnasium as gym  # type: ignore
            import minihack  # type: ignore # noqa: F401

            self._is_gymnasium = True
        except ImportError:
            try:
                import gym  # type: ignore
                import minihack  # type: ignore # noqa: F401

                self._is_gymnasium = False
            except ImportError as err:
                raise ImportError(
                    "minihack and gymnasium/gym are required for NativeNetHackWrapper. "
                    "Install via 'pip install minihack'."
                ) from err

        self._gym = gym
        self.tier = tier
        self.seed = seed
        self.step_count = 0
        self.max_steps = 200
        self.dungeon_level = 1
        self.player_pos = (0, 0)
        self.known_glyphs: list[list[int]] = []
        self.known_chars: list[list[str]] = []

        env_id = self.ENV_MAP.get(tier, "MiniHack-MultiRoom-N4-v0")
        self.env = self._gym.make(env_id)

        # Introspect authentic action space definitions
        self._action_name_to_idx: dict[str, int] = {}
        unwrapped_actions = getattr(self.env.unwrapped, "actions", None)
        if unwrapped_actions:
            for idx, act in enumerate(unwrapped_actions):
                name = getattr(act, "name", str(act))
                self._action_name_to_idx[name] = idx

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[NetHackObservation, dict[str, Any]]:
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        self.dungeon_level = 1
        self.known_glyphs = []
        self.known_chars = []
        if self._is_gymnasium:
            raw_obs, info = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        else:
            raw_obs = self.env.reset()
            info = {}
        obs = self._build_obs(raw_obs)
        return obs, info

    def step(
        self, action: NetHackAction | int
    ) -> tuple[NetHackObservation, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        act_enum = NetHackAction(action)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        # Direction vector mapping for modal actions
        dir_to_name = {
            (0, -1): "N",
            (1, 0): "E",
            (0, 1): "S",
            (-1, 0): "W",
            (1, -1): "NE",
            (1, 1): "SE",
            (-1, 1): "SW",
            (-1, -1): "NW",
        }

        # Handle door interactions (modal OPEN/KICK if supported, or step-into-door)
        if act_enum in (NetHackAction.OPEN_DOOR, NetHackAction.KICK):
            px, py = self.player_pos
            dir_name = "E"
            for (dx, dy), name in dir_to_name.items():
                nx, ny = px + dx, py + dy
                if 0 <= ny < len(self.known_glyphs) and 0 <= nx < len(self.known_glyphs[0]):
                    if self.known_glyphs[ny][nx] == NetHackGlyph.DOOR_CLOSED:
                        dir_name = name
                        break

            if act_enum == NetHackAction.OPEN_DOOR and "OPEN" in self._action_name_to_idx:
                open_idx = self._action_name_to_idx["OPEN"]
                dir_idx = self._action_name_to_idx.get(dir_name, 0)
                if self._is_gymnasium:
                    _, r1, term1, trunc1, i1 = self.env.step(open_idx)
                    raw_obs, r2, term2, trunc2, i2 = self.env.step(dir_idx)
                    reward = float(r1 + r2)
                    terminated = term1 or term2
                    truncated = trunc1 or trunc2 or (self.step_count >= self.max_steps)
                    info = {**i1, **i2}
                else:
                    _, r1, d1, i1 = self.env.step(open_idx)
                    raw_obs, r2, d2, i2 = self.env.step(dir_idx)
                    reward = float(r1 + r2)
                    terminated = d1 or d2
                    truncated = self.step_count >= self.max_steps
                    info = {**i1, **i2}
            elif act_enum == NetHackAction.KICK and "KICK" in self._action_name_to_idx:
                kick_idx = self._action_name_to_idx["KICK"]
                dir_idx = self._action_name_to_idx.get(dir_name, 0)
                if self._is_gymnasium:
                    _, r1, term1, trunc1, i1 = self.env.step(kick_idx)
                    raw_obs, r2, term2, trunc2, i2 = self.env.step(dir_idx)
                    reward = float(r1 + r2)
                    terminated = term1 or term2
                    truncated = trunc1 or trunc2 or (self.step_count >= self.max_steps)
                    info = {**i1, **i2}
                else:
                    _, r1, d1, i1 = self.env.step(kick_idx)
                    raw_obs, r2, d2, i2 = self.env.step(dir_idx)
                    reward = float(r1 + r2)
                    terminated = d1 or d2
                    truncated = self.step_count >= self.max_steps
                    info = {**i1, **i2}
            else:
                # In standard MiniHack environments, stepping into the closed door opens it!
                act_idx = self._action_name_to_idx.get(dir_name, 0)
                if self._is_gymnasium:
                    raw_obs, reward, terminated, truncated, info = self.env.step(act_idx)
                else:
                    raw_obs, reward, terminated, info = self.env.step(act_idx)
                reward = float(reward)
                truncated = truncated or (self.step_count >= self.max_steps)

        else:
            name_map = {
                "NORTH": "N",
                "EAST": "E",
                "SOUTH": "S",
                "WEST": "W",
                "NORTHEAST": "NE",
                "SOUTHEAST": "SE",
                "SOUTHWEST": "SW",
                "NORTHWEST": "NW",
                "PICKUP": "PICKUP",
                "WAIT": "WAIT",
            }
            target_name = name_map.get(act_enum.name, act_enum.name)
            act_idx = self._action_name_to_idx.get(target_name, self.ACTION_MAP.get(act_enum, 0))
            action_space_size = getattr(self.env.action_space, "n", 8)
            if act_idx >= action_space_size:
                act_idx = act_idx % action_space_size

            if self._is_gymnasium:
                raw_obs, reward_env, term_env, trunc_env, info = self.env.step(act_idx)
                reward = float(reward_env)
                terminated = term_env
                truncated = trunc_env or (self.step_count >= self.max_steps)
            else:
                raw_obs, reward_env, done_env, info = self.env.step(act_idx)
                reward = float(reward_env)
                terminated = done_env
                truncated = self.step_count >= self.max_steps

        if reward >= 1.0:
            self.dungeon_level = 2
        obs = self._build_obs(raw_obs)
        return obs, reward, terminated, truncated, info

    def _build_obs(self, raw_obs: Any) -> NetHackObservation:
        message = ""
        px, py = 0, 0
        stats = NetHackStats(dungeon_level=self.dungeon_level)

        if isinstance(raw_obs, dict):
            if "blstats" in raw_obs:
                bl = raw_obs["blstats"]
                if len(bl) >= 25:
                    px, py = int(bl[0]), int(bl[1])
                    stats = NetHackStats(
                        hp=int(bl[10]),
                        max_hp=int(bl[11]),
                        energy=int(bl[12]),
                        max_energy=int(bl[13]),
                        level=int(bl[18]),
                        gold=int(bl[19]),
                        dungeon_level=self.dungeon_level,
                    )
            self.player_pos = (px, py)

            if "chars" in raw_obs:
                raw_chars = raw_obs["chars"]
                CHAR_TO_GLYPH = {
                    ord("."): int(NetHackGlyph.FLOOR),
                    ord("#"): int(NetHackGlyph.CORRIDOR),
                    ord("+"): int(NetHackGlyph.DOOR_CLOSED),
                    ord("'"): int(NetHackGlyph.DOOR_OPEN),
                    ord(">"): int(NetHackGlyph.STAIRS_DOWN),
                    ord("<"): int(NetHackGlyph.STAIRS_UP),
                    ord("@"): int(NetHackGlyph.PLAYER),
                    ord("$"): int(NetHackGlyph.GOLD),
                    ord("0"): int(NetHackGlyph.KEY),
                    ord("("): int(NetHackGlyph.KEY),
                    ord("%"): int(NetHackGlyph.FOOD),
                }
                h = len(raw_chars)
                w = len(raw_chars[0]) if h > 0 else 0

                if (
                    not self.known_glyphs
                    or len(self.known_glyphs) != h
                    or len(self.known_glyphs[0]) != w
                ):
                    self.known_glyphs = [
                        [int(NetHackGlyph.UNEXPLORED) for _ in range(w)] for _ in range(h)
                    ]
                    self.known_chars = [[" " for _ in range(w)] for _ in range(h)]

                for r in range(h):
                    for c in range(w):
                        ch = raw_chars[r][c]
                        if ch in CHAR_TO_GLYPH:
                            self.known_glyphs[r][c] = CHAR_TO_GLYPH[ch]
                            self.known_chars[r][c] = chr(ch)
                        elif (65 <= ch <= 90) or (97 <= ch <= 122):
                            self.known_glyphs[r][c] = int(NetHackGlyph.MONSTER)
                            self.known_chars[r][c] = chr(ch)
                        elif ch in (ord("|"), ord("-")):
                            self.known_glyphs[r][c] = int(NetHackGlyph.WALL)
                            self.known_chars[r][c] = chr(ch)

            if "message" in raw_obs:
                message = "".join([chr(c) for c in raw_obs["message"] if c != 0])

        glyphs_copy = (
            [list(row) for row in self.known_glyphs]
            if self.known_glyphs
            else [[int(NetHackGlyph.FLOOR)]]
        )
        chars_copy = [list(row) for row in self.known_chars] if self.known_chars else [["."]]

        return NetHackObservation(
            glyphs=glyphs_copy,
            chars=chars_copy,
            player_pos=(px, py),
            stats=stats,
            inventory=[],
            message=message,
            step_count=self.step_count,
            raw_obs=raw_obs,
        )


def make_nethack_env(
    seed: int | None = None,
    tier: int = 5,
) -> NativeNetHackWrapper:
    """Instantiate NetHack environment binding strictly to native minihack."""
    try:
        wrapper = NativeNetHackWrapper(seed=seed, tier=tier)
        logger.info("Successfully bound to native minihack environment (tier=%d)", tier)
        return wrapper
    except Exception as e:
        raise RuntimeError(
            f"Native 'minihack' / 'nle' upstream package is required but failed: {e}"
        ) from e
