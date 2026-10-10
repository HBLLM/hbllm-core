"""Inductive Constraint Satisfaction & Subitizing System for HCIR World Kernel.

Modeled on primate dorsolateral prefrontal cortex (dlPFC) working memory constraints
and numerical subitizing (rapid cardinality judgment of small sets):
1. Local Cardinality Constraints: Captures relationships like sum(hazards in N(r, c)) = K.
2. Arc-Consistency & Unit Propagation: Deductively eliminates candidate cells into
   guaranteed-safe vs. guaranteed-lethal sets without search.
3. Subitizing & Feature Counting: Maps visual feature configurations to integer cardinalities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LocalCardinalityConstraint:
    """A local numerical constraint over a neighborhood of cells."""

    center: tuple[int, int]
    radius: int
    required_count: int
    scope: set[tuple[int, int]]  # Neighbor coordinates subject to constraint
    confidence: float = 1.0


class SymbolicConstraintSolver:
    """dlPFC-inspired constraint propagation solver for Minesweeper, counting, and logic grids."""

    def __init__(self) -> None:
        self.constraints: list[LocalCardinalityConstraint] = []
        self.known_safe_cells: set[tuple[int, int]] = set()
        self.known_hazard_cells: set[tuple[int, int]] = set()
        self.revealed_cells: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        """Reset transient episode constraints while preserving long-term symbolic rules."""
        self.constraints.clear()
        self.known_safe_cells.clear()
        self.known_hazard_cells.clear()
        self.revealed_cells.clear()

    def register_observation(
        self,
        center: tuple[int, int],
        count: int,
        grid_shape: tuple[int, int],
        radius: int = 1,
    ) -> None:
        """Register an observed numerical count constraint at center coordinate."""
        H, W = grid_shape
        r, c = center
        self.revealed_cells.add(center)
        self.known_safe_cells.add(center)

        scope: set[tuple[int, int]] = set()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    scope.add((nr, nc))

        # Avoid duplicate identical constraints
        if not any(c.center == center and c.radius == radius for c in self.constraints):
            constraint = LocalCardinalityConstraint(
                center=center,
                radius=radius,
                required_count=count,
                scope=scope,
            )
            self.constraints.append(constraint)

    def propagate_constraints(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Run arc-consistency / unit propagation until reaching a fixed point.

        Returns:
            Tuple of (newly_deduced_safe_cells, newly_deduced_hazard_cells).
        """
        new_safe: set[tuple[int, int]] = set()
        new_hazards: set[tuple[int, int]] = set()

        changed = True
        iterations = 0
        max_iters = 20

        while changed and iterations < max_iters:
            changed = False
            iterations += 1

            for c in self.constraints:
                # Remaining needed hazards
                already_known_hazards = c.scope & self.known_hazard_cells
                remaining_needed = c.required_count - len(already_known_hazards)

                # Unassigned cells in scope (not known safe and not known hazard)
                unassigned = c.scope - self.known_safe_cells - self.known_hazard_cells

                # Rule 1: If remaining needed hazards equals unassigned count,
                # then ALL unassigned cells are guaranteed hazards!
                if len(unassigned) > 0 and len(unassigned) == remaining_needed:
                    for cell in unassigned:
                        self.known_hazard_cells.add(cell)
                        new_hazards.add(cell)
                        changed = True
                    logger.debug(
                        "SymbolicConstraintSolver: Deduced hazards at %s from center %s",
                        unassigned,
                        c.center,
                    )

                # Rule 2: If remaining needed hazards is 0,
                # then ALL unassigned cells are guaranteed safe!
                elif remaining_needed == 0 and len(unassigned) > 0:
                    for cell in unassigned:
                        self.known_safe_cells.add(cell)
                        new_safe.add(cell)
                        changed = True
                    logger.debug(
                        "SymbolicConstraintSolver: Deduced safe cells at %s from center %s",
                        unassigned,
                        c.center,
                    )

        return new_safe, new_hazards

    def get_unrevealed_safe_cells(self) -> set[tuple[int, int]]:
        """Return cells deduced to be safe that haven't been visited / revealed yet."""
        return self.known_safe_cells - self.revealed_cells
