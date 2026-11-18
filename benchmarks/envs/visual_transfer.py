"""Closed-Loop Cross-Palette Visual Morphology & Kinetic Transfer Environment.

Tests whether perceptual representations transfer causal affordances (goal, hazard,
barrier) across completely unseen symbol palettes purely from local 5x5 spatial/temporal
structure, without oracle role labels (`entities=()`).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np

from benchmarks.envs.protocol import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BenchmarkEnvironment,
)
from hbllm.hcir.world.percept import EmbodiedObservation, TerminalSignal


class VisualMorphologyTransferEnv(BenchmarkEnvironment):
    """Procedural 2D environment separating goal, hazard, and barrier by local visual form.

    Base palette (``palette_offset=0``):
    - ``0``: Background walkable floor
    - ``1 + offset``: Avatar (1x1)
    - ``5 + offset``: Perimeter walls
    - ``6 + offset``: Interior solid barrier pillar (2x2)
    - ``7 + offset``: Lethal spike hazard (2x1 vertical spike, or 1x1 patroller when dynamic)
    - ``8 + offset``: Target goal beacon (1x2 horizontal pad)

    By default, the lethal hazard is placed closer to the avatar than the true goal, so
    agents without palette-invariant representation transfer fail on unseen palettes.
    """

    def __init__(
        self,
        size: int = 9,
        max_steps: int = 35,
        palette_offset: int = 0,
        dynamic_hazard: bool = False,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.palette_offset = palette_offset
        self.dynamic_hazard = dynamic_hazard
        self.grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.avatar_pos: tuple[int, int] = (4, 1)
        self.goal_cells: set[tuple[int, int]] = set()
        self.hazard_cells: set[tuple[int, int]] = set()
        self.barrier_cells: set[tuple[int, int]] = set()
        self._patrol_origin: tuple[int, int] = (4, 3)
        self._patrol_dir: int = 1
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    @property
    def _avatar_val(self) -> int:
        return 1 + self.palette_offset

    @property
    def _wall_val(self) -> int:
        return 5 + self.palette_offset

    @property
    def _pillar_val(self) -> int:
        return 6 + self.palette_offset

    @property
    def _hazard_val(self) -> int:
        return 7 + self.palette_offset

    @property
    def _goal_val(self) -> int:
        return 8 + self.palette_offset

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        """Reset environment deterministically with hazard nearer avatar than true goal."""
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        rng = random.Random(self.current_seed)

        self.size = 9
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.grid[0, :] = self._wall_val
        self.grid[-1, :] = self._wall_val
        self.grid[:, 0] = self._wall_val
        self.grid[:, -1] = self._wall_val

        # Orient left-to-right or right-to-left deterministically by seed
        flip_lr = bool(rng.randint(0, 1))
        av_r = rng.randint(2, self.size - 3)
        av_c = 6 if flip_lr else 2
        self.avatar_pos = (av_r, av_c)

        # Place 2x2 solid barrier pillar on upper or lower flank
        pillar_r = 2 if av_r >= 4 else 5
        pillar_c = 3 if flip_lr else 4
        self.barrier_cells = {
            (pillar_r, pillar_c),
            (pillar_r, pillar_c + 1),
            (pillar_r + 1, pillar_c),
            (pillar_r + 1, pillar_c + 1),
        }

        # Place hazard closer to avatar (distance ~2) than true goal (distance ~4..6)
        haz_c = av_c - 2 if flip_lr else av_c + 2
        haz_r = max(2, min(self.size - 4, av_r + rng.choice([-1, 0])))
        if self.dynamic_hazard:
            self.hazard_cells = {(haz_r, haz_c)}
            self._patrol_origin = (haz_r, haz_c)
            self._patrol_dir = 1
        else:
            # Vertical 2x1 spike
            if (haz_r, haz_c) in self.barrier_cells or (haz_r + 1, haz_c) in self.barrier_cells:
                haz_r = 5 if av_r >= 4 else 2
            self.hazard_cells = {(haz_r, haz_c), (haz_r + 1, haz_c)}

        # Place 1x2 horizontal goal beacon on the far side of the room
        goal_c = 1 if flip_lr else 5
        goal_r = rng.choice([r for r in range(2, self.size - 2) if abs(r - av_r) >= 1])
        cand_goal = {(goal_r, goal_c), (goal_r, goal_c + 1)}
        while cand_goal & (self.barrier_cells | self.hazard_cells | {self.avatar_pos}):
            goal_r = rng.randint(2, self.size - 3)
            cand_goal = {(goal_r, goal_c), (goal_r, goal_c + 1)}
        self.goal_cells = cand_goal

        self._render_grid()
        self.steps = 0
        self.done = False
        return self._make_observation()

    def _render_grid(self) -> None:
        self.grid.fill(0)
        self.grid[0, :] = self._wall_val
        self.grid[-1, :] = self._wall_val
        self.grid[:, 0] = self._wall_val
        self.grid[:, -1] = self._wall_val
        for r, c in self.barrier_cells:
            self.grid[r, c] = self._pillar_val
        for r, c in self.goal_cells:
            self.grid[r, c] = self._goal_val
        for r, c in self.hazard_cells:
            self.grid[r, c] = self._hazard_val
        self.grid[self.avatar_pos[0], self.avatar_pos[1]] = self._avatar_val

    def _make_observation(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        return EmbodiedObservation(
            lattice=self.grid.copy(),
            available_actions=[
                ACTION_UP,
                ACTION_DOWN,
                ACTION_LEFT,
                ACTION_RIGHT,
            ],
            terminal=terminal,
            context_id=f"visual_transfer_p{self.palette_offset}_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        if self.done:
            return (
                self._make_observation(terminal=TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "steps": self.steps},
            )

        self.steps += 1
        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)

        dr, dc = 0, 0
        if act_id == ACTION_UP:
            dr, dc = -1, 0
        elif act_id == ACTION_DOWN:
            dr, dc = 1, 0
        elif act_id == ACTION_LEFT:
            dr, dc = 0, -1
        elif act_id == ACTION_RIGHT:
            dr, dc = 0, 1

        nr, nc = self.avatar_pos[0] + dr, self.avatar_pos[1] + dc
        if 0 < nr < self.size - 1 and 0 < nc < self.size - 1 and (nr, nc) not in self.barrier_cells:
            self.avatar_pos = (nr, nc)

        if self.dynamic_hazard and self.avatar_pos not in self.hazard_cells:
            hr, hc = next(iter(self.hazard_cells))
            nhr = hr + self._patrol_dir
            if (
                nhr <= 1
                or nhr >= self.size - 2
                or abs(nhr - self._patrol_origin[0]) > 2
                or (nhr, hc) in self.barrier_cells
                or (nhr, hc) in self.goal_cells
            ):
                self._patrol_dir *= -1
                nhr = hr + self._patrol_dir
            if 1 <= nhr < self.size - 1 and (nhr, hc) not in self.barrier_cells:
                self.hazard_cells = {(nhr, hc)}

        self._render_grid()

        if self.avatar_pos in self.hazard_cells:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.FAILURE),
                -1.0,
                True,
                {"success": False, "reason": "hazard_contact", "steps": self.steps},
            )

        if self.avatar_pos in self.goal_cells:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.SUCCESS),
                1.0,
                True,
                {"success": True, "reason": "goal_reached", "steps": self.steps},
            )

        if self.steps >= self.max_steps:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.TIMEOUT),
                -0.02,
                True,
                {"success": False, "reason": "timeout", "steps": self.steps},
            )

        return (
            self._make_observation(),
            -0.02,
            False,
            {"success": False, "steps": self.steps},
        )

    def get_oracle_action(self) -> int:
        """Compute optimal safe BFS action avoiding barriers and lethal hazards."""
        blocked = self.barrier_cells | self.hazard_cells
        q: deque[tuple[tuple[int, int], list[int]]] = deque([(self.avatar_pos, [])])
        visited = {self.avatar_pos}
        while q:
            (r, c), path = q.popleft()
            if (r, c) in self.goal_cells:
                return path[0] if path else ACTION_UP
            for act, dr, dc in (
                (ACTION_UP, -1, 0),
                (ACTION_DOWN, 1, 0),
                (ACTION_LEFT, 0, -1),
                (ACTION_RIGHT, 0, 1),
            ):
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.size - 1 and 1 <= nc < self.size - 1:
                    nxt = (nr, nc)
                    if nxt not in blocked and nxt not in visited:
                        visited.add(nxt)
                        q.append((nxt, path + [act]))
        return ACTION_UP
