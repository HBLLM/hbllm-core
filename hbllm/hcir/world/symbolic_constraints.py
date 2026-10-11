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

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class LocalCardinalityConstraint:
    """A local numerical constraint over a neighborhood of cells."""

    center: tuple[int, int]
    radius: int
    required_count: int
    scope: set[tuple[int, int]]  # Neighbor coordinates subject to constraint
    confidence: float = 1.0

    def to_causal_predicate(self) -> CausalPredicate:
        """Export local cardinality constraint as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"constraint_cardinality:{self.center}",
            operator="==",
            value=self.required_count,
            relation=f"radius_{self.radius}",
            target_entity=f"scope_size:{len(self.scope)}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording local cardinality constraint."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"cardinality_constraint_{self.center[0]}_{self.center[1]}",
                node_data={
                    "center": self.center,
                    "radius": self.radius,
                    "required_count": self.required_count,
                    "scope": sorted(list(self.scope)),
                    "confidence": self.confidence,
                },
            )
        ]


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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export constraint solver state as a declarative causal predicate."""
        return CausalPredicate(
            variable="symbolic_constraint_solver",
            operator="==",
            value=len(self.constraints),
            relation=f"safe:{len(self.known_safe_cells)}|hazards:{len(self.known_hazard_cells)}",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all solver state and local cardinality constraint predicates."""
        preds = [self.to_causal_predicate()]
        for c in self.constraints:
            preds.append(c.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting solver state and active constraints."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="symbolic_constraint_solver_state",
                node_data={
                    "constraint_count": len(self.constraints),
                    "known_safe_cells": sorted(list(self.known_safe_cells)),
                    "known_hazard_cells": sorted(list(self.known_hazard_cells)),
                    "revealed_cells": sorted(list(self.revealed_cells)),
                },
            )
        ]
        for c in self.constraints:
            ops.extend(c.to_transaction_operations())
        return ops
