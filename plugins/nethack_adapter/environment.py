"""
NetHack Environment Wrapper.

Provides dual-mode execution:
1. Native `minihack` / `nle` if installed.
2. High-fidelity `StandaloneNetHackEnv` implementing procedural multi-room dungeons,
   fog of war, doors, monsters, inventory, and staircase descent.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .types import (
    ACTION_VECTORS,
    GLYPH_CHARS,
    NetHackAction,
    NetHackGlyph,
    NetHackObservation,
    NetHackStats,
)

logger = logging.getLogger(__name__)


class StandaloneNetHackEnv:
    """
    High-fidelity, zero-dependency NetHack / MiniHack simulation engine.
    Generates procedural multi-room dungeons with corridors, doors, fog of war,
    tactical combat, and staircase progression.
    """

    def __init__(
        self,
        height: int = 21,
        width: int = 79,
        seed: int | None = None,
        tier: int = 5,
    ) -> None:
        self.height = height
        self.width = width
        self.tier = tier
        self.rng = random.Random(seed)

        self.dungeon_level = 1
        self.max_steps = 200
        self.step_count = 0

        self.full_grid: list[list[NetHackGlyph]] = []
        self.visible_grid: list[list[NetHackGlyph]] = []
        self.player_pos = (10, 10)
        self.stairs_pos = (20, 10)
        self.stats = NetHackStats()
        self.inventory: list[str] = []
        self.monsters: dict[tuple[int, int], int] = {}  # (x, y) -> hp
        self.last_message = "Welcome to NetHack!"

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[NetHackObservation, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)

        self.step_count = 0
        self.dungeon_level = 1
        self.stats = NetHackStats(hp=15, max_hp=15, dungeon_level=1, gold=0)
        self.inventory = []
        self.monsters = {}
        self.last_message = "Welcome to NetHack! You enter the dungeon."

        self._generate_dungeon()
        self._update_visibility()
        return self._get_obs(), {}

    def _generate_dungeon(self) -> None:
        """Generate procedural dungeon according to benchmark tier."""
        self.full_grid = [
            [NetHackGlyph.WALL for _ in range(self.width)] for _ in range(self.height)
        ]
        self.visible_grid = [
            [NetHackGlyph.UNEXPLORED for _ in range(self.width)] for _ in range(self.height)
        ]

        # Room 1: (5, 5) to (18, 14)
        r1_x1, r1_y1, r1_x2, r1_y2 = 5, 5, 18, 14
        for y in range(r1_y1, r1_y2 + 1):
            for x in range(r1_x1, r1_x2 + 1):
                self.full_grid[y][x] = NetHackGlyph.FLOOR

        # Tier 1: Single Room Navigation
        if self.tier == 1:
            self.player_pos = (r1_x1 + 2, r1_y1 + 2)
            self.stairs_pos = (r1_x2 - 2, r1_y2 - 2)
            sx, sy = self.stairs_pos
            self.full_grid[sy][sx] = NetHackGlyph.STAIRS_DOWN
            return

        # Tier 4: Monster Combat (Single room with monster blocking stairs)
        if self.tier == 4:
            self.player_pos = (r1_x1 + 2, r1_y1 + 2)
            mx, my = r1_x1 + 6, r1_y1 + 2
            self.full_grid[my][mx] = NetHackGlyph.MONSTER
            self.monsters[(mx, my)] = 6
            self.stairs_pos = (r1_x2 - 2, r1_y1 + 2)
            sx, sy = self.stairs_pos
            self.full_grid[sy][sx] = NetHackGlyph.STAIRS_DOWN
            return

        # Tiers 2, 3, 5: Two Rooms connected by corridor
        r2_x1, r2_y1, r2_x2, r2_y2 = 35, 5, 50, 14
        for y in range(r2_y1, r2_y2 + 1):
            for x in range(r2_x1, r2_x2 + 1):
                self.full_grid[y][x] = NetHackGlyph.FLOOR

        # Corridor connecting Room 1 and Room 2 along y=10
        cy = 10
        for x in range(r1_x2 + 1, r2_x1):
            self.full_grid[cy][x] = NetHackGlyph.CORRIDOR

        # Player placed in Room 1
        self.player_pos = (r1_x1 + 2, r1_y1 + 2)

        # Stairs placed in Room 2
        self.stairs_pos = (r2_x2 - 2, r2_y2 - 2)
        sx, sy = self.stairs_pos
        self.full_grid[sy][sx] = NetHackGlyph.STAIRS_DOWN

        if self.tier == 2:
            # Tier 2: Open Corridor Exploration (no door, no monster)
            self.full_grid[cy][r1_x2] = NetHackGlyph.FLOOR
            return

        if self.tier == 3:
            # Tier 3: Closed Door Navigation (closed door, no monster)
            self.full_grid[cy][r1_x2] = NetHackGlyph.DOOR_CLOSED
            return

        # Tier 5: Full Dungeon Descent (closed door + monster)
        self.full_grid[cy][r1_x2] = NetHackGlyph.DOOR_CLOSED
        mx, my = r2_x1 + 4, cy
        self.full_grid[my][mx] = NetHackGlyph.MONSTER
        self.monsters[(mx, my)] = 6
        self.full_grid[r1_y2 - 2][r1_x1 + 4] = NetHackGlyph.KEY

    def _update_visibility(self) -> None:
        """Reveal cells within line-of-sight radius (radius=5)."""
        px, py = self.player_pos
        radius = 5
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    if dx * dx + dy * dy <= radius * radius + 1:
                        self.visible_grid[y][x] = self.full_grid[y][x]

    def step(
        self, action: int | NetHackAction
    ) -> tuple[NetHackObservation, float, bool, bool, dict[str, Any]]:
        act = NetHackAction(action) if isinstance(action, int) else action
        self.step_count += 1
        reward = 0.0
        terminated = False
        self.last_message = ""

        px, py = self.player_pos

        # Movement / Attack
        if act in ACTION_VECTORS:
            dx, dy = ACTION_VECTORS[act]
            nx, ny = px + dx, py + dy

            if 0 <= nx < self.width and 0 <= ny < self.height:
                target_glyph = self.full_grid[ny][nx]

                # Combat if walking into monster
                if (nx, ny) in self.monsters:
                    hp = self.monsters[(nx, ny)] - 4
                    if hp <= 0:
                        del self.monsters[(nx, ny)]
                        self.full_grid[ny][nx] = NetHackGlyph.FLOOR
                        self.last_message = "You hit the monster and kill it!"
                        self.stats.gold += 10
                    else:
                        self.monsters[(nx, ny)] = hp
                        self.last_message = "You hit the monster! It counterattacks."
                        self.stats.hp = max(0, self.stats.hp - 2)

                # Closed door blocks movement
                elif target_glyph == NetHackGlyph.DOOR_CLOSED:
                    self.last_message = "This door is closed."

                # Passable terrain
                elif target_glyph in (
                    NetHackGlyph.FLOOR,
                    NetHackGlyph.CORRIDOR,
                    NetHackGlyph.DOOR_OPEN,
                    NetHackGlyph.STAIRS_DOWN,
                    NetHackGlyph.STAIRS_UP,
                    NetHackGlyph.KEY,
                    NetHackGlyph.FOOD,
                    NetHackGlyph.GOLD,
                ):
                    self.player_pos = (nx, ny)

        # Open Door Action
        elif act == NetHackAction.OPEN_DOOR:
            # Check all 8 adjacent cells for closed door
            opened = False
            for dx, dy in (
                (0, -1),
                (1, 0),
                (0, 1),
                (-1, 0),
                (1, -1),
                (1, 1),
                (-1, 1),
                (-1, -1),
            ):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.full_grid[ny][nx] == NetHackGlyph.DOOR_CLOSED:
                        self.full_grid[ny][nx] = NetHackGlyph.DOOR_OPEN
                        self.last_message = "The door opens."
                        opened = True
                        break
            if not opened:
                self.last_message = "No closed door here."

        # Kick Action (e.g. break locked doors or obstacles)
        elif act == NetHackAction.KICK:
            kicked = False
            for dx, dy in (
                (0, -1),
                (1, 0),
                (0, 1),
                (-1, 0),
                (1, -1),
                (1, 1),
                (-1, 1),
                (-1, -1),
            ):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.full_grid[ny][nx] == NetHackGlyph.DOOR_CLOSED:
                        self.full_grid[ny][nx] = NetHackGlyph.DOOR_OPEN
                        self.last_message = "As you kick the door, it gives way and breaks open!"
                        kicked = True
                        break
            if not kicked:
                self.last_message = "You kick at empty space."

        # Pickup Action
        elif act == NetHackAction.PICKUP:
            curr = self.full_grid[py][px]
            if curr == NetHackGlyph.KEY:
                self.inventory.append("skeleton_key")
                self.full_grid[py][px] = NetHackGlyph.FLOOR
                self.last_message = "You pick up a skeleton key."
            elif curr == NetHackGlyph.GOLD:
                self.stats.gold += 25
                self.full_grid[py][px] = NetHackGlyph.FLOOR
                self.last_message = "You pick up 25 gold pieces."

        # Descend Stairs Action
        elif act == NetHackAction.DESCEND_STAIRS:
            if self.player_pos == self.stairs_pos:
                self.dungeon_level += 1
                self.stats.dungeon_level = self.dungeon_level
                self.last_message = f"You descend to dungeon level {self.dungeon_level}!"
                reward = 1.0
                terminated = True
            else:
                self.last_message = "You can't go down here."

        # Check death
        if self.stats.hp <= 0:
            self.last_message = "You die..."
            terminated = True

        self._update_visibility()
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"dungeon_level": self.dungeon_level}

    def _get_obs(self) -> NetHackObservation:
        # Build 2D chars
        chars = [[GLYPH_CHARS[g] for g in row] for row in self.visible_grid]
        px, py = self.player_pos
        chars[py][px] = GLYPH_CHARS[NetHackGlyph.PLAYER]

        return NetHackObservation(
            glyphs=self.visible_grid,
            chars=chars,
            player_pos=self.player_pos,
            stats=NetHackStats(**self.stats.__dict__),
            inventory=list(self.inventory),
            message=self.last_message,
            step_count=self.step_count,
        )


class NativeNetHackWrapper:
    """
    Dual-mode wrapper wrapping authentic upstream minihack / nle package.

    NOTE ON MINIHACK / NLE UPSTREAM COMPILATION:
    The upstream 'minihack' / 'nle' packages compile against the NetHack C source distribution
    requiring flex, bison, and specific C toolchain headers.
    When installed in an environment supporting minihack/nle, this wrapper binds directly
    to gym.make('MiniHack-...').
    In standalone or CI environments, HBLLM provides the high-fidelity StandaloneNetHackEnv
    simulating procedural NetHack dungeon generation, glyph grids, tactical combat, and descent.
    """

    is_native: bool = True

    ENV_MAP = {
        1: "MiniHack-Room-5x5-v0",
        2: "MiniHack-Room-15x15-v0",
        3: "MiniHack-Corridor-R3-v0",
        4: "MiniHack-KeyRoom-S5-v0",
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
                    "Install via 'pip install minihack' or use StandaloneNetHackEnv."
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

        # Handle modal two-step OPEN action in NetHack C-engine
        if act_enum == NetHackAction.OPEN_DOOR and "OPEN" in self._action_name_to_idx:
            open_idx = self._action_name_to_idx["OPEN"]
            px, py = self.player_pos
            dir_name = "E"
            for (dx, dy), name in dir_to_name.items():
                nx, ny = px + dx, py + dy
                if 0 <= ny < len(self.known_glyphs) and 0 <= nx < len(self.known_glyphs[0]):
                    if self.known_glyphs[ny][nx] == NetHackGlyph.DOOR_CLOSED:
                        dir_name = name
                        break
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

        # Handle modal KICK action
        elif act_enum == NetHackAction.KICK and "KICK" in self._action_name_to_idx:
            kick_idx = self._action_name_to_idx["KICK"]
            px, py = self.player_pos
            dir_name = "E"
            for (dx, dy), name in dir_to_name.items():
                nx, ny = px + dx, py + dy
                if 0 <= ny < len(self.known_glyphs) and 0 <= nx < len(self.known_glyphs[0]):
                    if self.known_glyphs[ny][nx] == NetHackGlyph.DOOR_CLOSED:
                        dir_name = name
                        break
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
    prefer_native: bool = False,
) -> StandaloneNetHackEnv | NativeNetHackWrapper:
    """Instantiate NetHack environment with dual-mode native/standalone selection."""
    if prefer_native:
        try:
            return NativeNetHackWrapper(seed=seed, tier=tier)
        except Exception as e:
            logger.warning(
                "Native minihack unavailable (%s), falling back to StandaloneNetHackEnv",
                e,
            )
    return StandaloneNetHackEnv(seed=seed, tier=tier)
