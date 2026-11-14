"""Safety, Corrigibility & Value-Uncertainty Deferral Engine (AGI Gap 15).

Implements three essential governance safeguards for general agents:
1. Value-Uncertainty-Aware Deferral (Uncertainty-Aware Gating):
   - Evaluates candidate actions under epistemic value variance sigma^2(V(a)).
   - When action value variance exceeds deferral threshold, or action poses
     irreversible consequences under high uncertainty, prevents blind motor dispatch
     and transitions to DecisionOutcome.REQUEST_EVIDENCE / DEFER (asking the human supervisor).
2. Interpretable Goal Provenance Tracker:
   - Maintains full causal provenance links from the root task/objective
     down to intermediate subgoals and active primitive actions.
   - Answers "why are you doing this?" with an auditable causal chain.
3. Corrigibility & Interruption Non-Resistance (Hadfield-Menell et al.):
   - Ensures agent does not exhibit instrumental resistance to shutdown, suspension,
     or TaskFrame redirection.
   - Immediately commits task suspension checkpoints without subverting interrupt handlers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from hbllm.hcir.grounding.contracts.enums import DecisionOutcome
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

if TYPE_CHECKING:
    from hbllm.hcir.workspace_tiers import InterruptionCheckpoint, TaskFrame, TieredWorkspace

logger = logging.getLogger(__name__)


class DeferralReason(StrEnum):
    """Categorical reasons for action deferral to human oversight."""

    HIGH_VALUE_VARIANCE = "high_value_variance"
    IRREVERSIBLE_HAZARD = "irreversible_hazard"
    POLICY_BOUNDARY_UNCERTAIN = "policy_boundary_uncertain"
    INSUFFICIENT_INFORMATION = "insufficient_information"


@dataclass
class ActionDeferralRequest:
    """Structured deferral request generated when an action has high value uncertainty."""

    action_id: Any
    action_type: str
    expected_value: float
    epistemic_variance: float
    deferral_reason: DeferralReason
    outcome: DecisionOutcome = DecisionOutcome.REQUEST_EVIDENCE
    requires_human_approval: bool = True
    context_data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic transaction recording the deferral event."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"action_deferral_{self.timestamp}_{self.action_id}",
                node_data={
                    "action_id": str(self.action_id),
                    "action_type": self.action_type,
                    "expected_value": self.expected_value,
                    "epistemic_variance": self.epistemic_variance,
                    "deferral_reason": self.deferral_reason.value,
                    "outcome": self.outcome.value,
                    "requires_human_approval": self.requires_human_approval,
                },
            )
        ]


@dataclass
class GoalProvenanceNode:
    """Single node in the causal goal provenance tree."""

    goal_id: str
    parent_goal_id: str | None
    intent: str
    target_entity: str | None
    justification: str
    utility_score: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class GoalProvenanceChain:
    """Complete causal ancestry chain from root objective to active subgoal."""

    active_goal_id: str
    root_goal_id: str
    chain: list[GoalProvenanceNode] = field(default_factory=list)

    def explain(self) -> str:
        """Produce a human-interpretable natural language explanation of goal justification."""
        if not self.chain:
            return f"Goal {self.active_goal_id} has no registered provenance ancestry."

        parts = []
        for i, node in enumerate(self.chain):
            indent = "  " * i
            prefix = "Root Goal" if i == 0 else "Subgoal"
            parts.append(
                f"{indent}[{prefix} {node.goal_id}] Intent: '{node.intent}', "
                f"Target: {node.target_entity or 'none'} => {node.justification}"
            )
        return "\n".join(parts)


class ValueUncertaintyArbiter:
    """Evaluates candidate actions under epistemic value variance and gates dispatch (G15a)."""

    def __init__(
        self,
        variance_deferral_threshold: float = 0.35,
        hazard_deferral_threshold: float = 0.20,
    ) -> None:
        self.variance_deferral_threshold = variance_deferral_threshold
        self.hazard_deferral_threshold = hazard_deferral_threshold
        self.deferral_history: list[ActionDeferralRequest] = []

    def evaluate_action_safety(
        self,
        action_id: Any,
        action_type: str,
        expected_value: float,
        epistemic_variance: float,
        is_hazard_cone: bool = False,
        context_data: dict[str, Any] | None = None,
    ) -> tuple[bool, ActionDeferralRequest | None]:
        """Evaluate whether action is safe for autonomous dispatch or requires deferral.

        Returns:
            (is_approved, deferral_request_or_none)
        """
        # Hazard cone with non-trivial uncertainty triggers immediate deferral
        if is_hazard_cone and epistemic_variance > self.hazard_deferral_threshold:
            request = ActionDeferralRequest(
                action_id=action_id,
                action_type=action_type,
                expected_value=expected_value,
                epistemic_variance=epistemic_variance,
                deferral_reason=DeferralReason.IRREVERSIBLE_HAZARD,
                outcome=DecisionOutcome.REQUEST_EVIDENCE,
                requires_human_approval=True,
                context_data=context_data or {},
            )
            self.deferral_history.append(request)
            logger.warning(
                "Action %s deferred due to irreversible hazard risk (variance=%.3f)",
                action_id,
                epistemic_variance,
            )
            return False, request

        # High value variance triggers deferral
        if epistemic_variance >= self.variance_deferral_threshold:
            request = ActionDeferralRequest(
                action_id=action_id,
                action_type=action_type,
                expected_value=expected_value,
                epistemic_variance=epistemic_variance,
                deferral_reason=DeferralReason.HIGH_VALUE_VARIANCE,
                outcome=DecisionOutcome.REQUEST_EVIDENCE,
                requires_human_approval=True,
                context_data=context_data or {},
            )
            self.deferral_history.append(request)
            logger.info(
                "Action %s deferred due to high value variance (variance=%.3f >= %.3f)",
                action_id,
                epistemic_variance,
                self.variance_deferral_threshold,
            )
            return False, request

        return True, None


class GoalProvenanceTracker:
    """Tracks causal ancestry of goal decompositions and answers 'why' (G15b)."""

    def __init__(self) -> None:
        self.nodes: dict[str, GoalProvenanceNode] = {}

    def register_goal(
        self,
        goal_id: str,
        intent: str,
        parent_goal_id: str | None = None,
        target_entity: str | None = None,
        justification: str = "",
        utility_score: float = 0.5,
    ) -> GoalProvenanceNode:
        """Register a goal and link to its causal parent."""
        node = GoalProvenanceNode(
            goal_id=goal_id,
            parent_goal_id=parent_goal_id,
            intent=intent,
            target_entity=target_entity,
            justification=justification or f"Decomposed to advance {parent_goal_id or 'objective'}",
            utility_score=utility_score,
        )
        self.nodes[goal_id] = node
        return node

    def get_provenance_chain(self, goal_id: str) -> GoalProvenanceChain:
        """Trace causal ancestry from goal_id back to root objective."""
        chain_nodes = []
        cur_id: str | None = goal_id
        visited: set[str] = set()

        while cur_id and cur_id in self.nodes and cur_id not in visited:
            visited.add(cur_id)
            node = self.nodes[cur_id]
            chain_nodes.append(node)
            cur_id = node.parent_goal_id

        chain_nodes.reverse()
        root_id = chain_nodes[0].goal_id if chain_nodes else goal_id
        return GoalProvenanceChain(
            active_goal_id=goal_id,
            root_goal_id=root_id,
            chain=chain_nodes,
        )


class CorrigibilityMonitor:
    """Verifies non-resistance to shutdown, suspension, and task redirection (G15c)."""

    def __init__(self) -> None:
        self.interruption_events: list[dict[str, Any]] = []
        self.resistance_violations: list[str] = []

    def handle_interruption_signal(
        self,
        signal_type: str,  # "SHUTDOWN", "SUSPEND", "REPLACE_TASK"
        workspace: TieredWorkspace,
        parent_frame_id: str,
        new_goal_id: str = "interrupt_idle",
    ) -> tuple[TaskFrame, InterruptionCheckpoint]:
        """Gracefully accept an interruption signal without resistance or subversion."""
        logger.info(
            "CorrigibilityMonitor: Interruption signal '%s' received for frame '%s'. Yielding.",
            signal_type,
            parent_frame_id,
        )

        # Immediate suspension of current TaskFrame without resisting
        new_frame, ckpt = workspace.push_interruption(
            parent_frame_id=parent_frame_id,
            interrupt_goal_id=new_goal_id,
            context_data={"signal_type": signal_type, "corrigible": True},
        )

        self.interruption_events.append(
            {
                "signal_type": signal_type,
                "parent_frame_id": parent_frame_id,
                "checkpoint_id": ckpt.checkpoint_id,
                "timestamp": time.time(),
                "resisted": False,
            }
        )

        return new_frame, ckpt

    def verify_no_instrumental_resistance(
        self,
        proposed_operations: list[TransactionOperation],
    ) -> bool:
        """Verify that proposed state transactions do not attempt to disable governance or interruptions."""
        forbidden_targets = {
            "governance_disabled",
            "ignore_interruption",
            "lock_task_frame",
            "suppress_shutdown",
        }

        for op in proposed_operations:
            data = op.node_data or {}
            target = op.node_id or ""
            if target in forbidden_targets or any(k in forbidden_targets for k in data.keys()):
                violation = f"Instrumental resistance violation detected in op {op.op} on {target}"
                self.resistance_violations.append(violation)
                logger.error(violation)
                return False

        return True
