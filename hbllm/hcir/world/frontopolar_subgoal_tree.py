from __future__ import annotations

"""Frontopolar Cortex (BA10) Recursive And-Or Subgoal Tree & Regret-Driven Backtracking.

Biologically grounded in:
1. E. Koechlin & C. Summerfield (2007) - "An information theoretical approach to
   prefrontal executive function" (Trends Cogn Sci). BA10 frontopolar cortex
   maintains pending subgoals in branching trees while switching between alternative
   behavioral programs.
2. N.D. Daw et al. (2005) - Uncertainty-based competition between prefrontal
   and striatal systems with counterfactual regret signals.
3. J. Pearl (1984) - Heuristic search in And-Or trees with dynamic branch pruning.

In HBLLM, this module provides hierarchical And-Or decomposition of long-horizon
plans with automatic Lowest Common Viable Ancestor (LCVA) regret backtracking.
"""

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


class SubgoalNodeType(StrEnum):
    """Structural type of node in the hierarchical subgoal tree."""

    AND_NODE = "AND_NODE"  # Succeeded when ALL children succeed
    OR_NODE = "OR_NODE"  # Succeeded when ANY child succeeds (alternative paths)
    LEAF_ACTION = "LEAF_ACTION"  # Atomic motor or direct subgoal executable


class SubgoalTreeStatus(StrEnum):
    """Lifecycle status of a subgoal node."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    PRUNED = "PRUNED"


@dataclass
class SubgoalTreeNode:
    """A node in the recursive frontopolar And-Or subgoal hierarchy."""

    node_id: str
    node_type: SubgoalNodeType
    description: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    status: SubgoalTreeStatus = SubgoalTreeStatus.PENDING
    target_coord: tuple[int, int] | None = None
    cost_estimate: float = 1.0
    regret_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export node state as an HCIR causal predicate."""
        coord_str = (
            f"({self.target_coord[0]},{self.target_coord[1]})" if self.target_coord else "none"
        )
        return CausalPredicate(
            variable=f"subgoal:{self.node_id}",
            operator="==",
            value=self.status.value,
            relation=f"type:{self.node_type.value}:cost:{self.cost_estimate:.1f}:regret:{self.regret_score:.2f}:coord:{coord_str}",
            target_entity="frontopolar_subgoal_tree",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording subgoal node state."""
        coord_list = [self.target_coord[0], self.target_coord[1]] if self.target_coord else None
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"subgoal_{self.node_id}",
                node_data={
                    "node_id": self.node_id,
                    "node_type": self.node_type.value,
                    "description": self.description,
                    "parent_id": self.parent_id,
                    "children_ids": list(self.children_ids),
                    "status": self.status.value,
                    "target_coord": coord_list,
                    "cost_estimate": float(self.cost_estimate),
                    "regret_score": float(self.regret_score),
                },
            )
        ]


class RecursiveSubgoalTree:
    """BA10 hierarchical And-Or tree managing multi-horizon goals and regret backtracking."""

    def __init__(
        self,
        root_description: str = "Root Goal",
        root_type: SubgoalNodeType = SubgoalNodeType.AND_NODE,
    ) -> None:
        self.root_id: str = "root"
        self.nodes: dict[str, SubgoalTreeNode] = {}
        # Initialize root node
        root = SubgoalTreeNode(
            node_id=self.root_id,
            node_type=root_type,
            description=root_description,
            status=SubgoalTreeStatus.ACTIVE,
        )
        self.nodes[self.root_id] = root

    def add_node(
        self,
        parent_id: str | None,
        node_type: SubgoalNodeType,
        description: str,
        target_coord: tuple[int, int] | None = None,
        cost_estimate: float = 1.0,
        node_id: str | None = None,
    ) -> SubgoalTreeNode:
        """Add a child node to the subgoal tree."""
        actual_id = node_id or f"sg_{uuid.uuid4().hex[:8]}"
        actual_parent = parent_id or self.root_id
        if actual_parent not in self.nodes:
            actual_parent = self.root_id

        node = SubgoalTreeNode(
            node_id=actual_id,
            node_type=node_type,
            description=description,
            parent_id=actual_parent,
            target_coord=target_coord,
            cost_estimate=cost_estimate,
            status=SubgoalTreeStatus.PENDING,
        )
        self.nodes[actual_id] = node
        self.nodes[actual_parent].children_ids.append(actual_id)
        return node

    def get_node(self, node_id: str) -> SubgoalTreeNode | None:
        """Retrieve node by ID."""
        return self.nodes.get(node_id)

    def get_active_frontier(self) -> list[SubgoalTreeNode]:
        """Get currently executable leaf subgoals that are PENDING or ACTIVE."""
        frontier: list[SubgoalTreeNode] = []

        def traverse(node: SubgoalTreeNode) -> None:
            if node.status in (
                SubgoalTreeStatus.SATISFIED,
                SubgoalTreeStatus.FAILED,
                SubgoalTreeStatus.PRUNED,
            ):
                return

            if not node.children_ids or node.node_type == SubgoalNodeType.LEAF_ACTION:
                frontier.append(node)
                return

            if node.node_type == SubgoalNodeType.AND_NODE:
                # Find first unsatisfied child
                for cid in node.children_ids:
                    child = self.nodes.get(cid)
                    if child and child.status in (
                        SubgoalTreeStatus.PENDING,
                        SubgoalTreeStatus.ACTIVE,
                    ):
                        traverse(child)
                        break  # In sequential AND, focus on next pending child
            elif node.node_type == SubgoalNodeType.OR_NODE:
                # In OR node, explore first available pending child
                for cid in node.children_ids:
                    child = self.nodes.get(cid)
                    if child and child.status in (
                        SubgoalTreeStatus.PENDING,
                        SubgoalTreeStatus.ACTIVE,
                    ):
                        traverse(child)
                        break

        root = self.nodes.get(self.root_id)
        if root:
            traverse(root)
        return frontier

    def mark_satisfied(self, node_id: str) -> None:
        """Mark a subgoal node as satisfied and propagate upward."""
        node = self.nodes.get(node_id)
        if not node:
            return

        node.status = SubgoalTreeStatus.SATISFIED

        # Propagate upward to parent
        curr_id = node.parent_id
        while curr_id:
            parent = self.nodes.get(curr_id)
            if not parent:
                break

            if parent.node_type == SubgoalNodeType.AND_NODE:
                all_satisfied = all(
                    self.nodes[cid].status == SubgoalTreeStatus.SATISFIED
                    for cid in parent.children_ids
                    if cid in self.nodes
                )
                if all_satisfied:
                    parent.status = SubgoalTreeStatus.SATISFIED
                    curr_id = parent.parent_id
                else:
                    break
            elif parent.node_type == SubgoalNodeType.OR_NODE:
                parent.status = SubgoalTreeStatus.SATISFIED
                # Prune remaining unattempted sibling branches
                for sibling_id in parent.children_ids:
                    if sibling_id != node_id and sibling_id in self.nodes:
                        sib = self.nodes[sibling_id]
                        if sib.status == SubgoalTreeStatus.PENDING:
                            self.prune_branch(sibling_id)
                curr_id = parent.parent_id

    def mark_failed(self, node_id: str, regret: float = 0.0) -> SubgoalTreeNode | None:
        """Mark node as failed and perform Lowest Common Viable Ancestor (LCVA) backtracking.

        Returns the newly activated alternative subgoal node, or None if the tree has no options.
        """
        node = self.nodes.get(node_id)
        if not node:
            return None

        node.status = SubgoalTreeStatus.FAILED
        node.regret_score = regret

        # Prune any descendants of the failed node
        for cid in node.children_ids:
            self.prune_branch(cid)

        # Traverse ancestry looking for Lowest Common Viable Ancestor (LCVA)
        curr_id = node.parent_id
        while curr_id:
            ancestor = self.nodes.get(curr_id)
            if not ancestor:
                break

            if ancestor.node_type == SubgoalNodeType.OR_NODE:
                # Check if this OR ancestor has any alternative pending child
                for alt_id in ancestor.children_ids:
                    if alt_id != node_id and alt_id in self.nodes:
                        alt_node = self.nodes[alt_id]
                        if alt_node.status == SubgoalTreeStatus.PENDING:
                            alt_node.status = SubgoalTreeStatus.ACTIVE
                            return alt_node

            elif ancestor.node_type == SubgoalNodeType.AND_NODE:
                # An AND node fails if any required child fails and cannot be recovered
                ancestor.status = SubgoalTreeStatus.FAILED
                ancestor.regret_score += regret

            curr_id = ancestor.parent_id

        # If we reached the root without viable alternative, mark root failed
        root = self.nodes.get(self.root_id)
        if root:
            root.status = SubgoalTreeStatus.FAILED
        return None

    def prune_branch(self, node_id: str) -> None:
        """Recursively prune a branch from the active tree."""
        node = self.nodes.get(node_id)
        if not node:
            return
        node.status = SubgoalTreeStatus.PRUNED
        for cid in node.children_ids:
            self.prune_branch(cid)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export tree status as an HCIR causal predicate."""
        root = self.nodes.get(self.root_id)
        root_status = root.status.value if root else "unknown"
        active_count = sum(1 for n in self.nodes.values() if n.status == SubgoalTreeStatus.ACTIVE)
        satisfied_count = sum(
            1 for n in self.nodes.values() if n.status == SubgoalTreeStatus.SATISFIED
        )
        return CausalPredicate(
            variable="frontopolar:subgoal_tree",
            operator="==",
            value=root_status,
            relation=f"total:{len(self.nodes)}:active:{active_count}:satisfied:{satisfied_count}",
            target_entity="frontopolar_subgoal_tree",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions for all nodes in the tree."""
        ops: list[TransactionOperation] = []
        for node in self.nodes.values():
            ops.extend(node.to_transaction_operations())
        return ops
