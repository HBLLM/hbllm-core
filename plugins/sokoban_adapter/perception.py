"""
Sokoban Perception Adapter.

Projects raw Sokoban grid observations into structured representations, identifying
agent position, active box states, target assignments, and static deadlock taboo cells.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import SokobanObservation, SokobanTile

logger = logging.getLogger(__name__)


class SokobanPerceptionAdapter:
    """Extracts topological and semantic knowledge from Sokoban observations."""

    def __init__(self) -> None:
        self.static_deadlock_cells: set[tuple[int, int]] = set()
        self._analyzed_layout = False

    def reset(self) -> None:
        """Reset internal perceptual state."""
        self.static_deadlock_cells.clear()
        self._analyzed_layout = False

    def process_observation(self, obs: SokobanObservation) -> dict[str, Any]:
        """Convert raw observation into perceptual feature map and taboo dead-end set."""
        if not self._analyzed_layout:
            self._analyze_static_deadlocks(obs)
            self._analyzed_layout = True

        boxes = set(obs.boxes)
        targets = set(obs.targets)

        solved_boxes = boxes.intersection(targets)
        unsolved_boxes = boxes.difference(targets)

        return {
            "player_pos": obs.player_pos,
            "boxes": boxes,
            "targets": targets,
            "solved_boxes": solved_boxes,
            "unsolved_boxes": unsolved_boxes,
            "deadlock_taboo_cells": set(self.static_deadlock_cells),
            "step_count": obs.step_count,
            "grid": obs.grid,
        }

    def _analyze_static_deadlocks(self, obs: SokobanObservation) -> None:
        """
        Pre-compute static non-goal dead-end cells.
        Any empty cell that forms a corner with two adjacent walls and is NOT a target
        is a permanent static taboo cell for boxes.
        """
        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0
        targets = set(obs.targets)

        walls = set()
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(SokobanTile.WALL):
                    walls.add((r, c))

        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if (r, c) in walls or (r, c) in targets:
                    continue

                up_wall = (r - 1, c) in walls
                down_wall = (r + 1, c) in walls
                left_wall = (r, c - 1) in walls
                right_wall = (r, c + 1) in walls

                # Corner condition
                if (up_wall or down_wall) and (left_wall or right_wall):
                    self.static_deadlock_cells.add((r, c))
