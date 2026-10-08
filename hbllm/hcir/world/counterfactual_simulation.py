"""Orbitofrontal Cortex (OFC) Counterfactual Simulation & Deadlock Detector.

Modeled on primate orbitofrontal cortex and hippocampal preplay (CA3 sharp-wave ripples):
1. Counterfactual Regret Simulation: Mentally simulates the downstream physical state
   of an object before committing motor actions to it.
2. Topological Deadlock Pruning: Identifies irreversible absorbing states in push/Sokoban dynamics:
   - Corner Deadlock: Block with two orthogonal impassable barriers, not on a goal.
   - Interior Wall Deadlock: Block against a continuous barrier wall with no goals on the segment.
   - 2x2 Square Deadlock: Block part of an immovable 2x2 cluster of barriers/blocks.
   - Line Freeze Deadlock: Two contiguous blocks along a wall segment without goal receptors.
3. Negative Valence Assignment: Assigns extreme negative valence (-inf) to prune hopeless
   branches from the mental simulation tree before issuing physical efference copies.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class DeadlockEvaluation(NamedTuple):
    """Result of an OFC counterfactual deadlock evaluation."""

    is_deadlock: bool
    deadlock_type: str
    description: str


class CounterfactualDeadlockDetector:
    """Orbitofrontal counterfactual simulator evaluating topological irreversibility and deadlocks."""

    @staticmethod
    def is_corner_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> bool:
        """Check whether a cargo block is trapped in a corner formed by two orthogonal barriers."""
        if pos in goals:
            return False

        H, W = grid_shape
        r, c = pos
        blocked_up = (r - 1 < 0) or ((r - 1, c) in static_barriers)
        blocked_down = (r + 1 >= H) or ((r + 1, c) in static_barriers)
        blocked_left = (c - 1 < 0) or ((r, c - 1) in static_barriers)
        blocked_right = (c + 1 >= W) or ((r, c + 1) in static_barriers)

        return (blocked_up or blocked_down) and (blocked_left or blocked_right)

    @staticmethod
    def is_square_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        all_blocks: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> bool:
        """Check whether a cargo block is part of an immovable 2x2 block/barrier square cluster."""
        H, W = grid_shape
        r, c = pos
        solid_obstacles = static_barriers | all_blocks

        # A 2x2 window containing (r, c) can have (r, c) as top-left, top-right, bottom-left, or bottom-right
        candidate_top_lefts = [
            (r, c),
            (r, c - 1),
            (r - 1, c),
            (r - 1, c - 1),
        ]

        for tr, tc in candidate_top_lefts:
            if not (0 <= tr < H - 1 and 0 <= tc < W - 1):
                continue

            window = [(tr, tc), (tr, tc + 1), (tr + 1, tc), (tr + 1, tc + 1)]
            # If all 4 cells in the 2x2 window are obstacles (either static barriers or blocks)
            if all(w in solid_obstacles for w in window):
                blocks_in_win = [w for w in window if w in all_blocks]
                # If there are blocks in this 2x2, check if any of them is NOT on a goal
                if blocks_in_win and any(b not in goals for b in blocks_in_win):
                    return True

        return False

    @staticmethod
    def is_wall_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> bool:
        """Check whether a block is against a barrier wall segment that contains no goals."""
        if pos in goals:
            return False

        H, W = grid_shape
        r, c = pos

        # Boundary grid walls:
        if r == 0 and not any(g[0] == 0 for g in goals):
            return True
        if r == H - 1 and not any(g[0] == H - 1 for g in goals):
            return True
        if c == 0 and not any(g[1] == 0 for g in goals):
            return True
        if c == W - 1 and not any(g[1] == W - 1 for g in goals):
            return True

        # Interior continuous barrier walls:
        # Check horizontal wall directly above (r-1, c) or below (r+1, c)
        for wall_dr in (-1, 1):
            wall_r = r + wall_dr
            if 0 <= wall_r < H and (wall_r, c) in static_barriers:
                # Trace left along this wall segment
                left_c = c
                left_has_goal = False
                while left_c >= 0 and (wall_r, left_c) in static_barriers:
                    if (r, left_c) in goals or (wall_r, left_c) in goals:
                        left_has_goal = True
                        break
                    # If pathway is blocked by vertical wall, segment ends
                    if (r, left_c) in static_barriers:
                        break
                    left_c -= 1

                # Trace right along this wall segment
                right_c = c
                right_has_goal = False
                while right_c < W and (wall_r, right_c) in static_barriers:
                    if (r, right_c) in goals or (wall_r, right_c) in goals:
                        right_has_goal = True
                        break
                    if (r, right_c) in static_barriers:
                        break
                    right_c += 1

                # If the continuous wall segment terminates in corners on both sides and has NO goal
                left_terminated = (left_c < 0) or ((r, left_c) in static_barriers)
                right_terminated = (right_c >= W) or ((r, right_c) in static_barriers)
                if (
                    left_terminated
                    and right_terminated
                    and not left_has_goal
                    and not right_has_goal
                ):
                    return True

        # Check vertical wall directly left (r, c-1) or right (r, c+1)
        for wall_dc in (-1, 1):
            wall_c = c + wall_dc
            if 0 <= wall_c < W and (r, wall_c) in static_barriers:
                # Trace up
                up_r = r
                up_has_goal = False
                while up_r >= 0 and (up_r, wall_c) in static_barriers:
                    if (up_r, c) in goals or (up_r, wall_c) in goals:
                        up_has_goal = True
                        break
                    if (up_r, c) in static_barriers:
                        break
                    up_r -= 1

                # Trace down
                down_r = r
                down_has_goal = False
                while down_r < H and (down_r, wall_c) in static_barriers:
                    if (down_r, c) in goals or (down_r, wall_c) in goals:
                        down_has_goal = True
                        break
                    if (down_r, c) in static_barriers:
                        break
                    down_r += 1

                up_terminated = (up_r < 0) or ((up_r, c) in static_barriers)
                down_terminated = (down_r >= H) or ((down_r, c) in static_barriers)
                if up_terminated and down_terminated and not up_has_goal and not down_has_goal:
                    return True

        return False

    @staticmethod
    def is_line_freeze_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        all_blocks: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> bool:
        """Check whether two adjacent blocks are trapped against a wall side-by-side."""
        if pos in goals:
            return False

        H, W = grid_shape
        r, c = pos

        # Check horizontal neighbor block
        for dc in (-1, 1):
            nbr = (r, c + dc)
            if nbr in all_blocks and nbr not in goals:
                # If both are backed by a wall above or below
                for dr in (-1, 1):
                    wr = r + dr
                    if (wr < 0 or wr >= H or (wr, c) in static_barriers) and (
                        wr < 0 or wr >= H or (wr, c + dc) in static_barriers
                    ):
                        return True

        # Check vertical neighbor block
        for dr in (-1, 1):
            nbr = (r + dr, c)
            if nbr in all_blocks and nbr not in goals:
                # If both are backed by a wall left or right
                for dc in (-1, 1):
                    wc = c + dc
                    if (wc < 0 or wc >= W or (r, wc) in static_barriers) and (
                        wc < 0 or wc >= W or (r + dr, wc) in static_barriers
                    ):
                        return True

        return False

    @classmethod
    def evaluate_deadlock(
        cls,
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        all_blocks: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> DeadlockEvaluation:
        """Run comprehensive OFC counterfactual evaluation on a target block position."""
        if pos in goals:
            return DeadlockEvaluation(False, "none", "Block is in goal receptacle")

        if cls.is_corner_deadlock(pos, goals, static_barriers, grid_shape):
            return DeadlockEvaluation(
                True, "corner", f"Block at {pos} trapped in an orthogonal corner"
            )

        if cls.is_square_deadlock(pos, goals, static_barriers, all_blocks, grid_shape):
            return DeadlockEvaluation(
                True, "square_2x2", f"Block at {pos} frozen in a 2x2 solid obstacle square"
            )

        if cls.is_line_freeze_deadlock(pos, goals, static_barriers, all_blocks, grid_shape):
            return DeadlockEvaluation(
                True, "line_freeze", f"Block at {pos} frozen against wall with adjacent block"
            )

        if cls.is_wall_deadlock(pos, goals, static_barriers, grid_shape):
            return DeadlockEvaluation(
                True,
                "interior_wall",
                f"Block at {pos} trapped against continuous dead wall segment",
            )

        return DeadlockEvaluation(False, "none", "State is reversible")
