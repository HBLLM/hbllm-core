"""
Sokoban Action Adapter.

Implements causal push planning and dead-end avoidance search to determine
optimal collision-free and deadlock-free action sequences.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from .types import SokobanAction, SokobanObservation, SokobanTile

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = [
    (SokobanAction.UP, -1, 0),
    (SokobanAction.DOWN, 1, 0),
    (SokobanAction.LEFT, 0, -1),
    (SokobanAction.RIGHT, 0, 1),
]


class SokobanActionAdapter:
    """Causal planner for Sokoban that prunes deadlocks and solves box arrangements."""

    def __init__(self) -> None:
        self.planned_actions: list[SokobanAction] = []

    def reset(self) -> None:
        """Clear action buffer."""
        self.planned_actions.clear()

    def select_action(
        self,
        obs: SokobanObservation,
        perception_data: dict[str, Any],
    ) -> SokobanAction:
        """Select next primitive action, computing path if queue is empty."""
        if not self.planned_actions:
            self._plan_solution(obs, perception_data)

        if self.planned_actions:
            return self.planned_actions.pop(0)

        # Fallback default action
        return SokobanAction.UP

    def _plan_solution(
        self,
        obs: SokobanObservation,
        perception_data: dict[str, Any],
    ) -> None:
        """Compute deadlock-free push path using state-space BFS search."""
        player_pos = obs.player_pos
        boxes = frozenset(obs.boxes)
        targets = frozenset(obs.targets)
        taboo_cells = perception_data.get("deadlock_taboo_cells", set())

        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0

        walls = set()
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(SokobanTile.WALL):
                    walls.add((r, c))

        start_state = (player_pos, boxes)
        queue: deque[
            tuple[tuple[tuple[int, int], frozenset[tuple[int, int]]], list[SokobanAction]]
        ] = deque([(start_state, [])])
        visited = {start_state}

        max_nodes = 2500
        nodes = 0
        best_path: list[SokobanAction] = []
        best_score = float("inf")

        while queue and nodes < max_nodes:
            nodes += 1
            (curr_player, curr_boxes), path = queue.popleft()

            if curr_boxes == targets:
                self.planned_actions = list(path)
                return

            if targets:
                score = sum(
                    min(abs(br - tr) + abs(bc - tc) for tr, tc in targets) for br, bc in curr_boxes
                )
                if score < best_score and path:
                    best_score = score
                    best_path = path

            pr, pc = curr_player
            for act, dr, dc in DIRECTION_DELTAS:
                nr, nc = pr + dr, pc + dc

                if (nr, nc) in walls:
                    continue

                if (nr, nc) in curr_boxes:
                    # Attempt push
                    nnr, nnc = nr + dr, nc + dc
                    if (nnr, nnc) in walls or (nnr, nnc) in curr_boxes:
                        continue

                    # Deadlock pruning
                    if (nnr, nnc) in taboo_cells and (nnr, nnc) not in targets:
                        continue

                    new_boxes = set(curr_boxes)
                    new_boxes.remove((nr, nc))
                    new_boxes.add((nnr, nnc))

                    if self._is_2x2_deadlock((nnr, nnc), walls, new_boxes, targets):
                        continue

                    next_state = ((nr, nc), frozenset(new_boxes))
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [act]))
                else:
                    # Free walk
                    next_state = ((nr, nc), curr_boxes)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [act]))

        logger.debug("Sokoban planner reached search limit, executing best partial trajectory")
        self.planned_actions = list(best_path) if best_path else [SokobanAction.UP] * 4

    def _is_2x2_deadlock(
        self,
        box: tuple[int, int],
        walls: set[tuple[int, int]],
        boxes: set[tuple[int, int]],
        targets: frozenset[tuple[int, int]],
    ) -> bool:
        """Check whether pushing a box creates an irreversible 2x2 box/wall deadlock."""
        r, c = box
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
            quad = [
                (r + dr, c + dc),
                (r + dr + 1, c + dc),
                (r + dr, c + dc + 1),
                (r + dr + 1, c + dc + 1),
            ]
            if all(pos in walls or pos in boxes for pos in quad):
                if any(pos in boxes and pos not in targets for pos in quad):
                    return True
        return False
