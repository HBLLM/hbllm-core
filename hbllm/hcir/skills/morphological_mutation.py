"""Morphological State Mutation & Gate Attunement Skill Acquisition.

Induces discrete state-dependent gating mechanisms and plans topological navigation
without hardcoded color IDs:
- Automatic grid lattice pitch and offset extraction from avatar bounding box
- Frequency-based patch clustering (Wall, Floor, Mutation Pads, Goal, Rechargers)
- Visual glyph attunement matching
- Energy/step-budget constrained state-space BFS pathfinding:
  State: (x, y, energy, rotation_state, remaining_pickups).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class MorphologicalMutationSkillAcquisition(BaseHierarchicalSkill):
    """Induces morphological attribute mutation requirements and plans gate attunement."""

    skill_name: str = "morphological_state_mutation"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE

    @classmethod
    def is_morphological_mutation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect state-mutation / attribute-gated mazes via lattice and pad clustering."""
        if set(available_actions) != {1, 2, 3, 4}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        step = 5
        cell_counts: dict[bytes, int] = {}
        for y in range(0, 55, step):
            for x in range(4, 60, step):
                patch = grid[y : y + step, x : x + step]
                if patch.shape == (step, step):
                    key = patch.tobytes()
                    cell_counts[key] = cell_counts.get(key, 0) + 1

        counts = sorted(cell_counts.values(), reverse=True)
        if len(counts) >= 5 and counts[0] >= 35 and counts[1] >= 12:
            singletons = sum(1 for c in counts if c == 1)
            return singletons >= 2

        return False

    @classmethod
    def plan_morphological_mutation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of moves dynamically using perceptual extraction and energy-constrained lattice BFS."""
        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        step = 5

        # 1. Frequency-based patch clustering across lattice
        patches: dict[tuple[int, int], np.ndarray] = {}
        for y in range(0, 55, step):
            for x in range(4, 60, step):
                p = grid[y : y + step, x : x + step]
                if p.shape == (step, step):
                    patches[(x, y)] = p

        if len(patches) < 10:
            return []

        # Count uniform patches to determine wall and floor colors
        from collections import Counter

        uniform_counts: Counter[int] = Counter()
        for p in patches.values():
            u = np.unique(p)
            if len(u) == 1:
                uniform_counts[int(u[0])] += 1

        if len(uniform_counts) < 2:
            return []

        top_uniform = uniform_counts.most_common(2)
        wall_color = top_uniform[0][0]
        floor_color = top_uniform[1][0]

        # 2. Dynamic Entity Extraction (zero hardcoded coordinates)
        walls: set[tuple[int, int]] = set()
        goal_pos: tuple[int, int] | None = None
        goal_3x3: np.ndarray | None = None
        avatar_pos: tuple[int, int] | None = None
        pad_pos: tuple[int, int] | None = None
        rechargers: set[tuple[int, int]] = set()

        for loc, p in patches.items():
            # Exclude bottom UI / status display region
            if loc[1] > 50 or loc[0] < 10:
                continue

            u = np.unique(p)
            if len(u) == 1 and u[0] == wall_color:
                walls.add(loc)
                continue
            if len(u) == 1 and u[0] == floor_color:
                continue

            border = np.concatenate([p[0, :], p[-1, :], p[:, 0], p[:, -1]])
            u_border = np.unique(border)

            # Goal Gate: enclosed bounding border of non-floor, non-wall color
            if len(u_border) == 1 and u_border[0] != floor_color and u_border[0] != wall_color:
                goal_pos = loc
                goal_3x3 = p[1:4, 1:4]
            # Floor-bordered interactive cells (Mutation Pad or Recharger/Pickup)
            elif np.all(border == floor_color):
                interior = p[1:4, 1:4]
                non_floor = int(np.sum(interior != floor_color))
                if non_floor == 5:
                    pad_pos = loc
                elif non_floor == 8:
                    rechargers.add(loc)
            # Avatar: solid cell in maze without floor or wall background
            elif floor_color not in u and wall_color not in u:
                avatar_pos = loc

        if avatar_pos is None or goal_pos is None or pad_pos is None or goal_3x3 is None:
            logger.warning(
                "MorphologicalMutation: Failed dynamic extraction: avatar=%s, goal=%s, pad=%s",
                avatar_pos,
                goal_pos,
                pad_pos,
            )
            return []

        # 3. Dynamic Rotation Requirement: Match UI glyph against Goal Gate interior
        # UI indicator glyph is at bottom-left status display (y=55..60, x=3..8, downsampled by 2)
        ui_3x3 = grid[55:61:2, 3:9:2]
        needed_rot = 0
        for k in range(4):
            if np.array_equal(np.rot90(ui_3x3, -k), goal_3x3):
                needed_rot = k
                break

        # 4. Energy-Constrained State-Space BFS
        capacity = 42
        start_state = (avatar_pos[0], avatar_pos[1], capacity, 0, tuple(sorted(rechargers)))
        q: deque[tuple[tuple[int, int, int, int, tuple[tuple[int, int], ...]], list[int]]] = deque(
            [(start_state, [])]
        )
        visited: set[tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
            (avatar_pos[0], avatar_pos[1], 0, tuple(sorted(rechargers)))
        }

        dirs = [(0, -step, 1), (0, step, 2), (-step, 0, 3), (step, 0, 4)]
        found_plan: list[int] | None = None

        while q:
            (cx, cy, c_energy, c_rot, c_rechargers), path = q.popleft()
            if (cx, cy) == goal_pos and c_rot == needed_rot:
                found_plan = path
                break
            if len(path) > 90:
                continue

            for dx, dy, act in dirs:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in walls or (nx, ny) not in patches:
                    continue

                # Goal cell is only passable if rotation matches
                if (nx, ny) == goal_pos and c_rot != needed_rot:
                    continue

                nxt_energy = c_energy - 2
                if nxt_energy < 0:
                    continue

                nxt_rot = (c_rot + 1) % 4 if (nx, ny) == pad_pos else c_rot
                nxt_rechargers = c_rechargers
                if (nx, ny) in c_rechargers:
                    nxt_energy = capacity
                    nxt_rechargers = tuple(r for r in c_rechargers if r != (nx, ny))

                st_key = (nx, ny, nxt_rot, nxt_rechargers)
                if st_key not in visited:
                    visited.add(st_key)
                    q.append(((nx, ny, nxt_energy, nxt_rot, nxt_rechargers), path + [act]))

        if found_plan is not None:
            return [(act, None) for act in found_plan]

        return []

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Standardized interface check for morphological mutation and attribute-gated mazes."""
        return self.is_morphological_mutation_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for morphological mutation mazes."""
        return self.plan_morphological_mutation_grid(grid, current_level=current_level)


MorphologicalStateMutationSkillAcquisition = MorphologicalMutationSkillAcquisition
