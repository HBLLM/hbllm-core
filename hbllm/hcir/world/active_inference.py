"""Active Inference Engine — Uncertainty-Reduction Action Selection & Utility Evaluation.

Evaluates candidate actions using normalized objective:
Utility = w1*R + w2*InfoGain + w3*FutureVal - w4*RiskPenalty - w5*ResourceCost
Lifted to declarative CausalPredicates, ActionAffordances, and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hbllm.hcir.graph import ActionNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.cortex_causal import ActionAffordance

logger = logging.getLogger(__name__)


@dataclass
class ActiveInferenceCandidateResult:
    """Action evaluation candidate result from ActiveInferenceEngine."""

    action: ActionNode
    utility_score: float
    expected_reward: float
    information_gain: float
    future_value: float
    risk_penalty: float
    resource_cost: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Export candidate evaluation as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"candidate_utility:{self.action.id}",
            operator="==",
            value=self.utility_score,
            relation=self.action.intent,
            target_entity=self.action.id,
        )

    def to_affordance(self) -> ActionAffordance:
        """Synthesize action affordance corresponding to evaluated candidate."""
        from hbllm.hcir.world.cortex_causal import ActionAffordance

        return ActionAffordance(
            action_id=self.action.intent,
            name=f"active_inf_{self.action.intent}",
            confidence=max(0.01, min(1.0, self.utility_score)),
            metadata={
                "action_id": self.action.id,
                "expected_reward": self.expected_reward,
                "information_gain": self.information_gain,
            },
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording candidate evaluation in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"act_cand_{self.action.id}",
                node_data={
                    "action_id": self.action.id,
                    "action_intent": self.action.intent,
                    "utility_score": self.utility_score,
                    "expected_reward": self.expected_reward,
                    "information_gain": self.information_gain,
                    "future_value": self.future_value,
                    "risk_penalty": self.risk_penalty,
                    "resource_cost": self.resource_cost,
                },
            )
        ]


class ActiveInferenceEngine:
    """Action selection engine based on Active Inference principles."""

    def __init__(
        self,
        w_reward: float = 0.35,
        w_info_gain: float = 0.25,
        w_future_val: float = 0.20,
        w_risk: float = 0.10,
        w_cost: float = 0.10,
    ) -> None:
        self.w_reward = w_reward
        self.w_info_gain = w_info_gain
        self.w_future_val = w_future_val
        self.w_risk = w_risk
        self.w_cost = w_cost

    def evaluate_candidates(
        self,
        candidate_actions: Sequence[ActionNode],
        information_gain_map: dict[str, float] | None = None,
    ) -> list[ActiveInferenceCandidateResult]:
        """Evaluate candidate actions and rank by normalized Active Inference utility."""
        info_map = information_gain_map or {}
        results: list[ActiveInferenceCandidateResult] = []

        for action in candidate_actions:
            # Expected reward heuristic
            risk_factor = getattr(action, "risk_factor", 0.1)
            cost_val = getattr(action, "estimated_cost", 10.0)

            reward = 1.0 - (risk_factor * 0.5)
            info_gain = info_map.get(action.id, 0.4)
            future_val = 0.8
            risk_penalty = risk_factor
            cost = min(1.0, cost_val * 0.01)

            # Normalized Active Inference Utility formula
            utility = (
                (self.w_reward * reward)
                + (self.w_info_gain * info_gain)
                + (self.w_future_val * future_val)
                - (self.w_risk * risk_penalty)
                - (self.w_cost * cost)
            )

            results.append(
                ActiveInferenceCandidateResult(
                    action=action,
                    utility_score=utility,
                    expected_reward=reward,
                    information_gain=info_gain,
                    future_value=future_val,
                    risk_penalty=risk_penalty,
                    resource_cost=cost,
                )
            )

        results.sort(key=lambda r: r.utility_score, reverse=True)
        logger.info(
            "ActiveInferenceEngine evaluated %d candidates; top action='%s' (utility=%.4f)",
            len(candidate_actions),
            results[0].action.intent if results else "none",
            results[0].utility_score if results else 0.0,
        )
        return results

    def select_best_action(
        self,
        candidate_actions: Sequence[ActionNode],
        information_gain_map: dict[str, float] | None = None,
    ) -> ActiveInferenceCandidateResult:
        """Select single candidate action maximizing Active Inference utility."""
        evals = self.evaluate_candidates(candidate_actions, information_gain_map)
        if not evals:
            raise ValueError("Candidate actions sequence cannot be empty")
        return evals[0]

    def select_and_transact(
        self,
        candidate_actions: Sequence[ActionNode],
        information_gain_map: dict[str, float] | None = None,
    ) -> tuple[ActiveInferenceCandidateResult, list[TransactionOperation]]:
        """Select best action and propose atomic state transaction recording choice."""
        best = self.select_best_action(candidate_actions, information_gain_map)
        return best, best.to_transaction_operations()

    def export_causal_predicates(
        self,
        candidate_actions: Sequence[ActionNode],
        information_gain_map: dict[str, float] | None = None,
    ) -> list[CausalPredicate]:
        """Export candidate utilities as declarative causal predicates."""
        evals = self.evaluate_candidates(candidate_actions, information_gain_map)
        return [e.to_causal_predicate() for e in evals]
