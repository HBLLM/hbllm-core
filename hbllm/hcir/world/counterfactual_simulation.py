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
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class DeadlockEvaluation(NamedTuple):
    """Result of an OFC counterfactual deadlock evaluation."""

    is_deadlock: bool
    deadlock_type: str
    description: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Export deadlock evaluation as a declarative causal predicate."""
        return CausalPredicate(
            variable="deadlock_evaluation",
            operator="==",
            value=self.is_deadlock,
            relation=self.deadlock_type,
            target_entity=self.description,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording deadlock state if deadlock detected."""
        if not self.is_deadlock:
            return []
        return [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=f"deadlock_{self.deadlock_type}",
                node_data={
                    "id": f"deadlock_{self.deadlock_type}",
                    "node_type": "hypothesis",
                    "category": "epistemology",
                    "is_deadlock": self.is_deadlock,
                    "deadlock_type": self.deadlock_type,
                    "description": self.description,
                },
            )
        ]


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

    @classmethod
    def evaluate_and_transact(
        cls,
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        all_blocks: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> tuple[DeadlockEvaluation, list[TransactionOperation]]:
        """Run deadlock evaluation and propose atomic state transaction if deadlock detected."""
        eval_res = cls.evaluate_deadlock(pos, goals, static_barriers, all_blocks, grid_shape)
        return eval_res, eval_res.to_transaction_operations()


# ═══════════════════════════════════════════════════════════════════════════
# Pearl Level-3 Counterfactual Simulation & Regret Minimization
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CounterfactualComparison:
    """Evaluation of an observed action against a counterfactual alternative."""

    actual_action: Any
    actual_outcome_value: float
    counterfactual_action: Any
    counterfactual_outcome_value: float
    regret: float
    is_counterfactually_superior: bool
    justification: str = ""

    def to_causal_predicate(self) -> CausalPredicate:
        """Export counterfactual comparison as a declarative causal predicate."""
        return CausalPredicate(
            variable="counterfactual:regret",
            operator="==",
            value=f"regret:{self.regret:.3f}",
            relation=f"actual:{self.actual_action}:counterfactual:{self.counterfactual_action}:superior:{self.is_counterfactually_superior}",
            target_entity=str(self.actual_action),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording counterfactual evaluation."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"counterfactual_{self.actual_action}_{self.counterfactual_action}",
                node_data={
                    "actual_action": str(self.actual_action),
                    "actual_outcome_value": float(self.actual_outcome_value),
                    "counterfactual_action": str(self.counterfactual_action),
                    "counterfactual_outcome_value": float(self.counterfactual_outcome_value),
                    "regret": float(self.regret),
                    "is_counterfactually_superior": self.is_counterfactually_superior,
                    "justification": self.justification,
                },
            )
        ]


class CounterfactualRegretEvaluator:
    """Orbitofrontal counterfactual regret evaluator (Pearl Level 3 Counterfactual Reasoning).

    Biologically modeled on Camille et al. (2004) & Marchiori & Warglien (2008):
    Evaluates: "What would the outcome value have been if alternative action a' was chosen instead of a?"
    Quantifies counterfactual regret: Regret(a, a') = max(0, V(a') - V(a)).
    When regret exceeds the threshold, flags counterfactually superior branches to update the causal theory.
    """

    def __init__(self, regret_threshold: float = 0.10) -> None:
        self.regret_threshold = regret_threshold

    def evaluate_counterfactuals(
        self,
        actual_action: Any,
        actual_reward: float,
        current_pos: tuple[int, int] | None,
        actual_next_pos: tuple[int, int] | None,
        candidate_actions: Sequence[Any],
        grid: np.ndarray,
        goal_positions: set[tuple[int, int]] | None = None,
        static_barriers: set[tuple[int, int]] | None = None,
        lethal_positions: set[tuple[int, int]] | None = None,
        action_deltas: dict[Any, tuple[int, int]] | None = None,
    ) -> list[CounterfactualComparison]:
        """Compute counterfactual regret across all unchosen candidate actions."""
        H, W = grid.shape
        max_dist = math.sqrt(H * H + W * W) or 1.0

        goals = goal_positions or set()
        barriers = static_barriers or set()
        lethals = lethal_positions or set()
        deltas = action_deltas or {}

        # 1. Baseline value of actual outcome
        actual_val = actual_reward
        if actual_next_pos is not None:
            if actual_next_pos in lethals:
                actual_val -= 2.0
            elif actual_next_pos in goals:
                actual_val += 2.0
            elif goals:
                min_g_dist = min(
                    math.sqrt((actual_next_pos[0] - g[0]) ** 2 + (actual_next_pos[1] - g[1]) ** 2)
                    for g in goals
                )
                actual_val += max(0.0, 1.0 - (min_g_dist / max_dist))

        comparisons: list[CounterfactualComparison] = []

        if current_pos is None:
            return comparisons

        ar, ac = current_pos

        for alt_action in candidate_actions:
            if alt_action == actual_action:
                continue

            dr, dc = deltas.get(alt_action, (0, 0))
            alt_r, alt_c = ar + dr, ac + dc

            if not (0 <= alt_r < H and 0 <= alt_c < W) or (alt_r, alt_c) in barriers:
                cf_val = -1.0
                justification = f"Counterfactual action {alt_action} blocked by boundary or barrier"
            elif (alt_r, alt_c) in lethals:
                cf_val = -2.0
                justification = f"Counterfactual action {alt_action} leads to lethal hazard"
            elif (alt_r, alt_c) in goals:
                cf_val = 2.0
                justification = f"Counterfactual action {alt_action} reaches primary goal directly"
            else:
                cf_val = 0.0
                if goals:
                    alt_dist = min(
                        math.sqrt((alt_r - g[0]) ** 2 + (alt_c - g[1]) ** 2) for g in goals
                    )
                    cf_val = max(0.0, 1.0 - (alt_dist / max_dist))
                justification = f"Counterfactual progress toward goal (val={cf_val:.2f})"

            regret = max(0.0, cf_val - actual_val)
            is_superior = regret >= self.regret_threshold

            comp = CounterfactualComparison(
                actual_action=actual_action,
                actual_outcome_value=actual_val,
                counterfactual_action=alt_action,
                counterfactual_outcome_value=cf_val,
                regret=regret,
                is_counterfactually_superior=is_superior,
                justification=justification,
            )
            comparisons.append(comp)

        comparisons.sort(key=lambda c: c.regret, reverse=True)
        return comparisons
