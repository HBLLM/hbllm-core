"""ARC-AGI grid-puzzle planners (WP3.5).

Puzzle-specific planning primitives relocated from the task-agnostic core
(``hbllm.hcir.skills.common_subskills``) in accordance with Invariant 1.7.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.skills.common_subskills import (
    DiscreteVectorTranslator,
    LatticeNavigator,
)


class BlockPushPlanner:
    """Plans kinematic push alignments to push target blocks across grid lattices."""

    @staticmethod
    def get_push_stance(
        block_pos: tuple[int, int],
        push_dir: tuple[int, int],
        block_size: int = 1,
    ) -> tuple[int, int]:
        """Returns the coordinate the agent must occupy to push the block in push_dir."""
        bx, by = block_pos
        pdx, pdy = push_dir
        return (bx - pdx * block_size, by - pdy * block_size)

    @classmethod
    def plan_push_maneuver(
        cls,
        agent_pos: tuple[int, int],
        block_pos: tuple[int, int],
        push_dir: tuple[int, int],
        push_distance: int = 1,
        walkable: set[tuple[int, int]] | None = None,
        step_size: int = 1,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Plans navigation to the push stance, followed by pushing the block."""
        stance = cls.get_push_stance(block_pos, push_dir, block_size=step_size)
        actions: list[tuple[int, dict[str, int] | None]] = []

        if walkable is not None:
            path = LatticeNavigator.bfs_path(agent_pos, stance, walkable, step_size=step_size)
            if path is not None:
                actions.extend(DiscreteVectorTranslator.path_to_actions(path, step_size=step_size))
            else:
                actions.extend(
                    DiscreteVectorTranslator.points_to_actions(
                        agent_pos, stance, step_size=step_size
                    )
                )
        else:
            actions.extend(
                DiscreteVectorTranslator.points_to_actions(agent_pos, stance, step_size=step_size)
            )

        pdx, pdy = push_dir
        push_actions = DiscreteVectorTranslator.delta_to_actions(
            pdx * push_distance, pdy * push_distance, step_size=step_size
        )
        actions.extend(push_actions)
        return actions


class GF2LinearSolver:
    """Solves systems of linear congruences A * x = b (mod 2) using Gaussian elimination."""

    @classmethod
    def solve(cls, A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        """Solve A * x = b (mod 2).

        Args:
            A: Binary matrix of shape (M, N) with values in {0, 1}.
            b: Binary vector of length M with values in {0, 1}.

        Returns:
            Binary solution vector x of length N, or None if system is inconsistent.
        """
        M, N = A.shape
        if M > 256 or N > 256:
            return None

        aug = np.zeros((M, N + 1), dtype=np.uint8)
        aug[:, :N] = (A % 2).astype(np.uint8)
        aug[:, N] = (b % 2).astype(np.uint8)

        row = 0
        pivot_cols: list[int] = []

        for col in range(N):
            if row >= M:
                break

            col_vals = aug[row:M, col]
            rel_piv = int(np.argmax(col_vals))
            if col_vals[rel_piv] == 0:
                continue

            pivot_row = row + rel_piv
            if pivot_row != row:
                aug[[row, pivot_row]] = aug[[pivot_row, row]]

            mask = aug[:, col] == 1
            mask[row] = False
            aug[mask] ^= aug[row]

            pivot_cols.append(col)
            row += 1

        for r in range(row, M):
            if aug[r, N] == 1:
                return None

        x = np.zeros(N, dtype=np.uint8)
        for r, col in enumerate(pivot_cols):
            x[col] = aug[r, N]

        return x
