"""
Counterfactual Graph — Dedicated Structure for What-If Counterfactual Simulation Branches.
"""

from __future__ import annotations

import logging
import time

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
