"""Global Scalar Invariant Inductive Discovery Engine (Milestone 3 Phase 1).

Discovers global modular scalar invariants from demonstration residual observations:
1. Extracts candidate scalar properties (connected object count, pixel count, color count).
2. Induces modular equivalence classes (modulus k in {2, 3}) mapping scalar residues to transformation behaviors.
3. Synthesizes a Parameterized GlobalScalarConditionedOperator that passes out-of-construction validation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.grid_operator import GridOperator, OperatorBinding

logger = logging.getLogger(__name__)


class GlobalScalarPropertyExtractor:
    """Extracts topological and structural scalar metrics from grid matrices."""

    @staticmethod
    def extract_object_count(grid: np.ndarray) -> int:
        """Count 4-connected components of non-background (non-zero) cells."""
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        count = 0
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 0 and not visited[r, c]:
                    count += 1
                    visited[r, c] = True
                    queue = [(r, c)]
                    color = grid[r, c]
                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if (
                                0 <= nr < H
                                and 0 <= nc < W
                                and not visited[nr, nc]
                                and grid[nr, nc] == color
                            ):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
        return count

    @staticmethod
    def extract_pixel_count(grid: np.ndarray) -> int:
        """Total number of non-zero foreground pixels."""
        return int(np.sum(grid != 0))

    @staticmethod
    def extract_color_count(grid: np.ndarray) -> int:
        """Number of distinct non-zero colors present in the grid."""
        unique_colors = set(np.unique(grid)) - {0}
        return len(unique_colors)

    @classmethod
    def get_extractor(cls, prop_name: str) -> Callable[[np.ndarray], int]:
        """Lookup scalar extraction function by name."""
        if prop_name == "object_count":
            return cls.extract_object_count
        elif prop_name == "pixel_count":
            return cls.extract_pixel_count
        elif prop_name == "color_count":
            return cls.extract_color_count
        else:
            raise ValueError(f"Unknown scalar property: {prop_name}")


@dataclass(frozen=True)
class BranchRule:
    """Transformation rule associated with a specific modular residue."""

    rule_type: str  # 'identity', 'recolor', 'flip_h', 'flip_v', 'rot90'
    params: dict[str, int]

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Execute branch rule on an input grid."""
        if self.rule_type == "identity":
            return np.copy(grid)
        elif self.rule_type == "recolor":
            src = self.params.get("src_color", 0)
            dst = self.params.get("dst_color", 0)
            out = np.copy(grid)
            if src == -1:  # Recolor all foreground pixels
                out[grid != 0] = dst
            else:
                out[grid == src] = dst
            return out
        elif self.rule_type == "flip_h":
            return np.fliplr(grid).copy()
        elif self.rule_type == "flip_v":
            return np.flipud(grid).copy()
        elif self.rule_type == "rot90":
            return np.rot90(grid, k=1).copy()
        else:
            return np.copy(grid)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export branch rule as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"branch_rule:{self.rule_type}",
            operator="==",
            value=self.rule_type,
            relation=f"params:{sorted(self.params.items())}",
            target_entity=self.rule_type,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording branch rule."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"branch_rule_{self.rule_type}",
                node_data={
                    "rule_type": self.rule_type,
                    "params": dict(self.params),
                },
            )
        ]


class GlobalScalarConditionedOperator(GridOperator):
    """Operator that conditions spatial transformation on a global modular scalar invariant."""

    def __init__(
        self,
        name: str,
        property_name: str,
        modulus: int,
        branches: dict[int, BranchRule],
        complexity: float = 2.5,
    ) -> None:
        self.name = name
        self.property_name = property_name
        self.modulus = modulus
        self.branches = branches
        self.complexity = complexity
        self._extractor = GlobalScalarPropertyExtractor.get_extractor(property_name)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export global invariant operator as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"global_invariant:{self.property_name}",
            operator="==",
            value=self.modulus,
            relation=f"branches:{len(self.branches)}",
            target_entity=self.name,
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export operator predicate and all branch rule predicates."""
        preds = [self.to_causal_predicate()]
        for b in self.branches.values():
            preds.append(b.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting global invariant operator."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"op_{self.name}",
                node_data={
                    "name": self.name,
                    "property_name": self.property_name,
                    "modulus": self.modulus,
                    "branch_count": len(self.branches),
                    "complexity": self.complexity,
                },
            )
        ]
        for b in self.branches.values():
            ops.extend(b.to_transaction_operations())
        return ops

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        return [
            OperatorBinding(
                operator_name=self.name,
                params={
                    "property_name": self.property_name,
                    "modulus": self.modulus,
                },
                description=f"{self.name}({self.property_name}, mod={self.modulus})",
                complexity=self.complexity,
            )
        ]

    def apply(self, grid: np.ndarray, binding: OperatorBinding | dict[str, Any]) -> np.ndarray:
        scalar = self._extractor(grid)
        residue = scalar % self.modulus
        branch = self.branches.get(residue)
        if branch is None:
            # Fallback to identity if residue branch is unspecified

            return np.copy(grid)
        return branch.apply(grid)


class InductiveGlobalInvariantSynthesizer:
    """Synthesizes global modular invariants directly from demonstration input-output residuals."""

    SUPPORTED_PROPERTIES = ["object_count", "pixel_count", "color_count"]
    SUPPORTED_MODULI = [2, 3]

    @classmethod
    def _infer_pair_transformation(cls, x: np.ndarray, y: np.ndarray) -> BranchRule | None:
        """Deduce the local/global transformation rule mapping x to y."""
        if np.array_equal(x, y):
            return BranchRule(rule_type="identity", params={})

        if x.shape == y.shape:
            # Test affine transforms
            if np.array_equal(np.fliplr(x), y):
                return BranchRule(rule_type="flip_h", params={})
            if np.array_equal(np.flipud(x), y):
                return BranchRule(rule_type="flip_v", params={})
            if np.array_equal(np.rot90(x, 1), y):
                return BranchRule(rule_type="rot90", params={})

            # Test uniform foreground recoloring
            # Single color swap test (including background or foreground swap)
            diff_mask = x != y
            src_colors = np.unique(x[diff_mask])
            dst_colors = np.unique(y[diff_mask])
            if len(src_colors) == 1 and len(dst_colors) == 1:
                return BranchRule(
                    rule_type="recolor",
                    params={"src_color": int(src_colors[0]), "dst_color": int(dst_colors[0])},
                )

            # Test uniform foreground recoloring
            fg_x = x != 0
            fg_y = y != 0
            if np.array_equal(fg_x, fg_y):
                # Shape preserved, colors changed
                unique_y_fg = np.unique(y[fg_y])
                if len(unique_y_fg) == 1:
                    dst = int(unique_y_fg[0])
                    return BranchRule(
                        rule_type="recolor", params={"src_color": -1, "dst_color": dst}
                    )

        return None

    @classmethod
    def synthesize_from_demonstrations(
        cls, train_pairs: list[tuple[np.ndarray, np.ndarray]]
    ) -> GlobalScalarConditionedOperator | None:
        """Induce a global modular invariant operator explaining all training demonstrations."""
        if len(train_pairs) < 2:
            return None

        # Step 1: Infer per-demonstration transformation rules
        pair_rules: list[BranchRule] = []
        for x_i, y_i in train_pairs:
            rule = cls._infer_pair_transformation(x_i, y_i)
            if rule is None:
                return None
            pair_rules.append(rule)

        # If all demonstrations use the exact same transformation, no modular conditioning is needed
        if all(r == pair_rules[0] for r in pair_rules):
            return None

        # Step 2: Search candidate scalar properties and moduli
        for prop_name in cls.SUPPORTED_PROPERTIES:
            extractor = GlobalScalarPropertyExtractor.get_extractor(prop_name)
            scalars = [extractor(x_i) for x_i, _ in train_pairs]

            for mod in cls.SUPPORTED_MODULI:
                residues = [s % mod for s in scalars]

                # Check consistency: identical residue MUST map to identical rule
                residue_to_rule: dict[int, BranchRule] = {}
                is_consistent = True

                for res, rule in zip(residues, pair_rules, strict=True):
                    if res in residue_to_rule:
                        if residue_to_rule[res] != rule:
                            is_consistent = False
                            break
                    else:
                        residue_to_rule[res] = rule

                if not is_consistent:
                    continue

                # Ensure at least 2 distinct residues are represented across demonstrations
                if len(residue_to_rule) < 2:
                    continue

                # Step 3: Construct candidate operator
                op_name = f"synthesized_global_invariant_{prop_name}_mod_{mod}"
                candidate_op = GlobalScalarConditionedOperator(
                    name=op_name,
                    property_name=prop_name,
                    modulus=mod,
                    branches=residue_to_rule,
                    complexity=2.2 + 0.3 * mod,
                )

                # Step 4: Full verification across all training pairs
                verified = True
                binding = candidate_op.propose({}, {"train_pairs": train_pairs})[0]
                for x_i, y_i in train_pairs:
                    pred_i = candidate_op.apply(x_i, binding)
                    if not np.array_equal(pred_i, y_i):
                        verified = False
                        break

                if verified:
                    logger.info(
                        "Successfully induced global invariant operator: %s (modulus %d on %s)",
                        op_name,
                        mod,
                        prop_name,
                    )
                    return candidate_op

        return None
