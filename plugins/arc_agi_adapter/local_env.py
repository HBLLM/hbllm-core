"""Deterministic local ARC-like simulator for closed-loop WP0/WP2 measurement.

Mirrors the arcengine FrameData surface consumed by ``kaggle_submission.my_agent``
and ``run_all_25_games``: ``state`` (GameState), ``frame`` (int layers),
``grid``, ``levels_completed``, ``win_levels``, ``available_actions``,
``reset()`` and ``step(action_id, data)``.

The games are generic gridworld physics fixtures (no solvers): each encodes
one failure mode observed in the harness. Everything is seeded and lazily
importable — no arc_agi/arcengine dependency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GameState(Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    GAME_OVER = "GAME_OVER"
    WIN = "WIN"


class GameAction(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7

    def is_complex(self) -> bool:
        return self.value == 6

    def set_data(self, data: Any) -> None:
        self.data = data


# Cell codes
EMPTY, WALL, AVATAR, GOAL, KEY, DOOR, BLOCK, TARGET, TOGGLE, HAZARD, GATE, GATE_KEY = range(12)

# Action semantics (gridworld-generic):
#   1 up, 2 down, 3 left, 4 right, 5 interact/recolour, 6 point-actuate (x,y), 7 wait
_DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


@dataclass
class LocalFrame:
    """FrameData-compatible frame emitted by LocalGame."""

    state: GameState = GameState.NOT_PLAYED
    frame: list[Any] = field(default_factory=list)  # list of 2D layers (ARC-faithful)
    grid: list[list[int]] = field(default_factory=list)
    levels_completed: int = 0
    win_levels: int = 1
    available_actions: list[int] = field(default_factory=list)
    step_count: int = 0


class LocalGame:
    """One game = an ordered list of level rule dicts.

    Supported rule flags (all generic physics):
      walls: set of (r,c)    avatar: start pos     goal: win cell
      stride: 1|2            key/door: cells       block/target: push-to-place
      toggles: {(r,c): opens cells}                action5: recolour required
      hazard_period: (cells, period)               max_steps_per_level
    """

    def __init__(
        self, name: str, levels: list[dict[str, Any]], seed: int = 0, scale: int | None = None
    ) -> None:
        self.name = name
        self._levels = levels
        self._rng = random.Random(seed)
        # Cell->pixel upscaling; default renders a 64x64 frame like ARC.
        max_dim = max(lv.get("shape", (16, 16))[0] for lv in levels)
        self.scale = scale if scale is not None else max(1, 64 // max_dim)
        self.win_levels = len(levels)
        self.state = GameState.NOT_PLAYED
        self.levels_completed = 0
        self.step_count = 0
        self._level_idx = 0
        self._reset_level()

    # ── level setup ──────────────────────────────────────────────────
    def _reset_level(self) -> None:
        lv = self._levels[self._level_idx]
        H, W = lv.get("shape", (16, 16))
        self.grid = [[EMPTY] * W for _ in range(H)]
        for r in range(H):
            self.grid[r][0] = self.grid[r][W - 1] = WALL
        for c in range(W):
            self.grid[0][c] = self.grid[H - 1][c] = WALL
        for r, c in lv.get("walls", []):
            self.grid[r][c] = WALL
        self.avatar = lv["avatar"]
        self.stride = lv.get("stride", 1)
        self.has_key = False
        self.recoloured = False
        self._opens: set[tuple[int, int]] = set()
        self._toggle_pos = set()
        for cell, code in (
            ("goal", GOAL),
            ("key", KEY),
            ("door", DOOR),
            ("block", BLOCK),
            ("target", TARGET),
            ("gate", GATE),
            ("gate_key", GATE_KEY),
        ):
            for rc in lv.get(cell) or []:
                if isinstance(rc, tuple):
                    self.grid[rc[0]][rc[1]] = code
        for (r, c), opens in lv.get("toggles", {}).items():
            self.grid[r][c] = TOGGLE
            self._toggle_pos.add((r, c))
            for oc in opens:
                self.grid[oc[0]][oc[1]] = GATE
        self._toggle_map = lv.get("toggles", {})
        hz = lv.get("hazard")
        self._hazard_cells = hz[0] if hz else []
        self._hazard_period = hz[1] if hz else 0
        self._max_steps = lv.get("max_steps", 500)
        self.grid[self.avatar[0]][self.avatar[1]] = AVATAR

    # ── FrameData surface ────────────────────────────────────────────
    def available_actions(self) -> list[int]:  # noqa: N802 — mirrors ARC
        acts = [1, 2, 3, 4, 7]
        lv = self._levels[self._level_idx]
        if lv.get("action5"):
            acts.append(5)
        if self._toggle_pos:
            acts.append(6)
        return sorted(acts)

    def _frame(self) -> LocalFrame:
        if self.scale > 1:
            layer = []
            for row in self.grid:
                big = [v for v in row for _ in range(self.scale)]
                layer.extend(big[:] for _ in range(self.scale))
        else:
            layer = [row[:] for row in self.grid]
        return LocalFrame(
            state=self.state,
            # ARC fidelity: frame is a LIST of 2D layers (frame[-1] is read).
            frame=[layer],
            grid=[row[:] for row in self.grid],
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            available_actions=self.available_actions()
            if self.state == GameState.NOT_FINISHED
            else [],
            step_count=self.step_count,
        )

    def reset(self) -> LocalFrame:
        self.state = GameState.NOT_FINISHED
        self.step_count = 0
        self._level_idx = 0
        self.levels_completed = 0
        self._reset_level()
        return self._frame()

    def _die(self) -> None:
        self.state = GameState.GAME_OVER

    def _win_level(self) -> None:
        self.levels_completed += 1
        self._level_idx += 1
        if self.levels_completed >= self.win_levels:
            self.state = GameState.WIN
        else:
            self._reset_level()

    def _move(self, dr: int, dc: int) -> None:
        r, c = self.avatar
        nr, nc = r + dr * self.stride, c + dc * self.stride
        cell = self.grid[nr][nc]
        if cell == WALL:
            return  # blocked — zero motion
        if cell == DOOR and not self.has_key:
            return
        if cell == GATE:
            return
        if cell == BLOCK:
            br, bc = nr + dr * self.stride, nc + dc * self.stride
            bcell = self.grid[br][bc]
            if bcell not in (EMPTY, TARGET):
                return
            self.grid[br][bc] = BLOCK if bcell != TARGET else TARGET | 0x10
        else:
            pass
        self.grid[r][c] = EMPTY
        self.grid[nr][nc] = AVATAR
        self.avatar = (nr, nc)
        if cell == KEY:
            self.has_key = True
        if cell == GATE_KEY:
            self.recoloured = True
        if cell == HAZARD:
            self._die()
            return
        # win checks
        if cell == GOAL and not self._levels[self._level_idx].get("needs_key"):
            self._win_level()
        elif cell == GOAL and self._levels[self._level_idx].get("needs_key") and self.has_key:
            self._win_level()
        elif self._levels[self._level_idx].get("block_win"):
            for tr, tc in self._levels[self._level_idx].get("target", []):
                if self.grid[tr][tc] == TARGET | 0x10:
                    self._win_level()
                    break

    def step(self, action_id: int, data: dict[str, Any] | None = None) -> LocalFrame:
        if action_id == 0 and self.state != GameState.WIN:
            # RESET semantics: revive from GAME_OVER / restart current level.
            self._reset_level()
            self.state = GameState.NOT_FINISHED
            return self._frame()
        if self.state == GameState.GAME_OVER or self.state == GameState.WIN:
            return self._frame()
        if self.state == GameState.NOT_PLAYED:
            self.state = GameState.NOT_FINISHED
        self.step_count += 1
        if self.step_count > self._max_steps:
            self._die()
            return self._frame()
        lv = self._levels[self._level_idx]
        if action_id in _DIRS:
            dr, dc = _DIRS[action_id]
            self._move(dr, dc)
        elif action_id == 5 and lv.get("action5"):
            self.recoloured = True  # recolour avatar -> can pass colour gate
        elif action_id == 6 and data is not None:
            x, y = int(data.get("x", -1)) // self.scale, int(data.get("y", -1)) // self.scale
            if (y, x) in self._toggle_pos:
                for oc in self._toggle_map[(y, x)]:
                    self.grid[oc[0]][oc[1]] = EMPTY
        # hazard tick
        if self._hazard_period and self.step_count % self._hazard_period == 0:
            for hr, hc in self._hazard_cells:
                if self.grid[hr][hc] == EMPTY:
                    self.grid[hr][hc] = HAZARD
                elif self.grid[hr][hc] == HAZARD:
                    self.grid[hr][hc] = EMPTY
                if (hr, hc) == self.avatar:
                    self._die()
        return self._frame()


# ── Game registry: ≥8 games, each covering a distinct failure mode ─────
def build_registry(seed: int = 0) -> dict[str, LocalGame]:
    def nav_level(av=(1, 1), goal=(6, 6), walls=(), shape=(8, 8), **kw):
        return {
            "shape": shape,
            "avatar": av,
            "goal": [goal],
            "walls": list(walls),
            "max_steps": 300,
            **kw,
        }

    games: dict[str, LocalGame] = {}
    # 1. reach-goal navigation, two levels
    games["nav_basic"] = LocalGame(
        "nav_basic", [nav_level(), nav_level(goal=(4, 10), shape=(8, 12))], seed
    )
    # 2. first move blocked by a wall (zero motion on first try, free later)
    games["blocked_first"] = LocalGame(
        "blocked_first", [nav_level(av=(2, 2), walls=[(1, 2), (2, 1)])], seed
    )
    # 3. stride-2 movement
    games["stride2"] = LocalGame("stride2", [nav_level(goal=(5, 5), stride=2)], seed)
    # 4. key -> door -> goal
    games["key_door"] = LocalGame(
        "key_door",
        [
            {
                "shape": (8, 8),
                "avatar": (1, 1),
                "goal": [(6, 6)],
                "key": [(1, 5)],
                "door": [(4, 4)],
                "needs_key": True,
                "walls": [(4, 3), (4, 5)],
                "max_steps": 300,
            },
        ],
        seed,
    )
    # 5. push block onto target
    games["push_block"] = LocalGame(
        "push_block",
        [
            {
                "shape": (8, 8),
                "avatar": (2, 2),
                "block": [(2, 3)],
                "target": [(2, 5)],
                "block_win": True,
                "max_steps": 300,
            },
        ],
        seed,
    )
    # 6. ACTION6 toggle opens gated path
    games["click_toggle"] = LocalGame(
        "click_toggle",
        [
            {
                "shape": (8, 8),
                "avatar": (1, 1),
                "goal": [(6, 6)],
                "toggles": {(1, 5): [(4, 4)]},
                "max_steps": 300,
            },
        ],
        seed,
    )
    # 7. ACTION5 recolour needed to pass colour gate
    games["recolour_gate"] = LocalGame(
        "recolour_gate",
        [
            {
                "shape": (8, 8),
                "avatar": (1, 1),
                "goal": [(6, 6)],
                "gate_key": [(2, 2)],
                "gate": [(4, 4)],
                "action5": True,
                "needs_key": False,
                "max_steps": 300,
            },
        ],
        seed,
    )
    # 8. periodic hazard corridor
    games["hazard"] = LocalGame(
        "hazard",
        [
            {
                "shape": (8, 8),
                "avatar": (1, 1),
                "goal": [(6, 6)],
                "hazard": ([(3, 3)], 4),
                "max_steps": 300,
            },
        ],
        seed,
    )
    # 9. death -> GAME_OVER requires RESET (hazard adjacent to spawn)
    games["death_reset"] = LocalGame(
        "death_reset",
        [
            {
                "shape": (8, 8),
                "avatar": (1, 1),
                "goal": [(6, 6)],
                "walls": [(2, 2)],
                "hazard": ([(2, 1)], 1),
                "max_steps": 40,
            },
        ],
        seed,
    )
    return games


class LocalEnv:
    """Registry facade mirroring the ARC game surface used by the runners."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._games = build_registry(seed)

    @property
    def game_ids(self) -> list[str]:
        return list(self._games.keys())

    def make(self, game_id: str, seed: int | None = None, scale: int | None = None) -> LocalGame:
        spec = self._games[game_id]
        return LocalGame(game_id, spec._levels, self.seed if seed is None else seed, scale=scale)


LOCAL_ENV = LocalEnv
