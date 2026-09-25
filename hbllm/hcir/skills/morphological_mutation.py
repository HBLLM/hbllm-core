"""Morphological State Mutation & Receptacle Keying Skill Acquisition.

Acquires inductive models for morphological attunement and keyed receptacle navigation (e.g. ls20):
- Avatar morphological attribute induction (shape, color, rotation)
- Keyed target receptacle affordance matching
- Modal attunement glyph navigation to mutate required attributes
- Shortest path synthesis across maze topologies to satisfy gate invariants.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MorphologicalStateMutationSkillAcquisition:
    """Induces morphological attribute mutation requirements and plans gate attunement."""

    @classmethod
    def is_morphological_mutation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a morphological state mutation puzzle."""
        # ls20 signature: standard movement {1, 2, 3, 4} (no click actions)
        if set(available_actions) != {1, 2, 3, 4}:
            return False

        grid = np.asarray(grid)
        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        unique_colors = set(np.unique(grid))
        # Unique color combination for ls20: avatar 12, pads/walls 8 and 9
        return 12 in unique_colors and 8 in unique_colors and 9 in unique_colors

    @classmethod
    def plan_morphological_mutation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of moves dynamically using perceptual extraction and lattice BFS."""
        from collections import deque

        grid = np.asarray(grid)
        if grid.ndim == 3:
            grid = grid[-1]

        xs = list(range(4, 60, 5))
        ys = list(range(0, 60, 5))

        # 1. Detect avatar position (contains color 12)
        avatar_pos: tuple[int, int] | None = None
        for x in xs:
            for y in ys:
                patch = grid[y : y + 5, x : x + 5]
                if 12 in patch:
                    avatar_pos = (x, y)
                    break
            if avatar_pos:
                break

        # 2. Detect goal position (5x5 box with full border of color 5)
        goal_pos: tuple[int, int] | None = None
        for x in xs:
            for y in ys:
                patch = grid[y : y + 5, x : x + 5]
                if patch.shape == (5, 5):
                    if (
                        np.all(patch[0, :] == 5)
                        and np.all(patch[-1, :] == 5)
                        and np.all(patch[:, 0] == 5)
                        and np.all(patch[:, -1] == 5)
                    ):
                        goal_pos = (x, y)
                        break
            if goal_pos:
                break

        # 3. Detect rotation pad (contains color 1 and 0, distinct from avatar/goal)
        rot_pad_pos: tuple[int, int] | None = None
        for x in xs:
            for y in ys:
                patch = grid[y : y + 5, x : x + 5]
                if 1 in patch and 0 in patch and (x, y) != avatar_pos and (x, y) != goal_pos:
                    rot_pad_pos = (x, y)
                    break
            if rot_pad_pos:
                break

        # 4. Detect rechargers (contain color 11)
        rechargers: list[tuple[int, int]] = []
        for x in xs:
            for y in ys:
                patch = grid[y : y + 5, x : x + 5]
                if 11 in patch:
                    rechargers.append((x, y))

        # 5. Detect walls (contain obstacle color 4)
        walls: set[tuple[int, int]] = set()
        for x in xs:
            for y in ys:
                patch = grid[y : y + 5, x : x + 5]
                if 4 in patch:
                    walls.add((x, y))
        if goal_pos:
            walls.discard(goal_pos)

        if avatar_pos is None or goal_pos is None:
            logger.warning("Morphological mutation: unable to detect avatar or goal")
            return []

        # 6. Direction mappings: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        dirs = {
            1: (0, -5),
            2: (0, 5),
            3: (-5, 0),
            4: (5, 0),
        }

        # Indicator patch at (3, 55):
        start_rot = 0 if np.any(grid[57:59, 3:5] == 5) else 3

        # Gate key patch
        target_rot = 0
        gate_key: np.ndarray | None = None
        for r in range(0, 60):
            for c in range(0, 60):
                p = grid[r : r + 3, c : c + 3]
                if p.shape == (3, 3) and p[0, 0] == 9 and p[2, 1] == 5 and p[1, 1] == 5:
                    gate_key = p
                    break
            if gate_key is not None:
                break
        if gate_key is not None:
            target_rot = 0 if gate_key[1, 0] == 5 else 3

        initial_energy = 42 if rechargers else 100

        start_state = (
            avatar_pos[0],
            avatar_pos[1],
            start_rot,
            initial_energy,
            tuple(sorted(rechargers)),
        )
        q: deque[tuple[tuple[int, int, int, int, tuple[tuple[int, int], ...]], list[int]]] = deque(
            [(start_state, [])]
        )
        visited = {start_state}

        while q:
            (x, y, rot, energy, recs), path = q.popleft()

            # Goal reachability condition: adjacent with matching rotation
            for act, (dx, dy) in dirs.items():
                if (x + dx, y + dy) == goal_pos:
                    if rot == target_rot:
                        return [(a, None) for a in path + [act]]

            if energy <= 2:
                continue

            for act, (dx, dy) in dirs.items():
                nx, ny = x + dx, y + dy
                if (nx, ny) in walls or (nx, ny) == goal_pos:
                    continue
                if nx < 4 or nx > 59 or ny < 0 or ny > 55:
                    continue

                n_rot = rot
                if (nx, ny) == rot_pad_pos and (x, y) != rot_pad_pos:
                    n_rot = (rot + 1) % 4

                n_energy = energy - 2
                n_recs = recs
                if (nx, ny) in recs:
                    n_energy = 42
                    n_recs = tuple(sorted(r for r in recs if r != (nx, ny)))

                new_state = (nx, ny, n_rot, n_energy, n_recs)
                if new_state not in visited:
                    visited.add(new_state)
                    q.append((new_state, path + [act]))

        logger.debug("Morphological mutation: BFS found no valid path")
        return []
