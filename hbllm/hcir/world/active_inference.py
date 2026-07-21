"""
Active Inference Engine — Uncertainty-Reduction Action Selection & Utility Evaluation.

Evaluates candidate actions using normalized objective:
Utility = w1*R + w2*InfoGain + w3*FutureVal - w4*RiskPenalty - w5*ResourceCost
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from hbllm.hcir.graph import ActionNode

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
