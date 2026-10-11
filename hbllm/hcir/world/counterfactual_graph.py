"""Counterfactual Graph — Dedicated Structure for What-If Counterfactual Simulation Branches.

Manages counterfactual simulation branches, mental what-if rollouts, lifted to
declarative CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
import time

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.world_context import BranchMode

logger = logging.getLogger(__name__)


class CounterfactualBranchNode:
    """Branch node representing a counterfactual simulation variation."""

    def __init__(
        self, branch_id: str, parent_branch_id: str | None = None, action_intent: str = ""
    ) -> None:
        self.branch_id = branch_id
        self.parent_branch_id = parent_branch_id
        self.action_intent = action_intent
        self.branch_mode = BranchMode.COUNTERFACTUAL
        self.created_at = time.time()

    def to_causal_predicate(self) -> CausalPredicate:
        """Export counterfactual branch as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"cf_branch:{self.branch_id}",
            operator="==",
            value=self.action_intent,
            relation=self.parent_branch_id or "root",
            target_entity=self.branch_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording simulation branch."""
        return [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=f"cf_branch_{self.branch_id}",
                node_data={
                    "branch_id": self.branch_id,
                    "parent_branch_id": self.parent_branch_id,
                    "action_intent": self.action_intent,
                    "branch_mode": self.branch_mode.value,
                    "created_at": self.created_at,
                },
            )
        ]


class CounterfactualGraph:
    """Graph structure managing counterfactual branches."""

    def __init__(self, world_id: str = "default_world") -> None:
        self.world_id = world_id
        self._branches: dict[str, CounterfactualBranchNode] = {}

    def create_counterfactual_branch(
        self,
        branch_id: str,
        action_intent: str,
        parent_branch_id: str | None = None,
    ) -> CounterfactualBranchNode:
        """Create counterfactual simulation branch."""
        node = CounterfactualBranchNode(
            branch_id=branch_id, parent_branch_id=parent_branch_id, action_intent=action_intent
        )
        self._branches[branch_id] = node
        logger.debug(
            "CounterfactualGraph [%s] created branch '%s' for action '%s'",
            self.world_id,
            branch_id,
            action_intent,
        )
        return node

    def get_branch(self, branch_id: str) -> CounterfactualBranchNode | None:
        """Retrieve counterfactual branch node."""
        return self._branches.get(branch_id)

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all branches as declarative causal predicates."""
        return [b.to_causal_predicate() for b in self._branches.values()]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting counterfactual graph."""
        ops: list[TransactionOperation] = []
        for b in self._branches.values():
            ops.extend(b.to_transaction_operations())
        return ops
