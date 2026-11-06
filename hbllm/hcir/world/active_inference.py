"""Active Inference Engine — Uncertainty-Reduction Action Selection & Utility Evaluation.

Evaluates candidate actions using normalized objective:
Utility = w1*R + w2*InfoGain + w3*FutureVal - w4*RiskPenalty - w5*ResourceCost
Lifted to declarative CausalPredicates, ActionAffordances, and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

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


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Variational Free Energy Minimization (Karl Friston)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HierarchicalFreeEnergyDecomposition:
    """Detailed 3-tier variational free energy decomposition.

    Level 0 (Sensory): Observation prediction error and precision Pi_s.
    Level 1 (Kinematic): Dynamic transition prediction error and precision Pi_mu.
    Level 2 (Policy/EFE): Expected Free Energy G(pi) balancing epistemic uncertainty
    reduction and pragmatic goal convergence.
    """

    sensory_free_energy_f0: float
    dynamic_free_energy_f1: float
    expected_free_energy_g: float
    epistemic_information_gain: float
    extrinsic_goal_value: float
    sensory_precision: float
    dynamic_precision: float
    total_variational_energy: float

    def to_causal_predicate(self, action_id: str) -> CausalPredicate:
        """Export hierarchical free energy evaluation as a formal causal predicate."""
        return CausalPredicate(
            variable=f"hierarchical_free_energy:{action_id}",
            operator="==",
            value=round(self.total_variational_energy, 4),
            relation=f"G:{self.expected_free_energy_g:.3f}:F0:{self.sensory_free_energy_f0:.3f}:F1:{self.dynamic_free_energy_f1:.3f}",
            target_entity=action_id,
        )


@dataclass
class HierarchicalInferenceCandidateResult:
    """Action candidate evaluated via Hierarchical Variational Free Energy minimization."""

    action: ActionNode
    decomposition: HierarchicalFreeEnergyDecomposition
    rank: int = 0

    @property
    def free_energy(self) -> float:
        return self.decomposition.total_variational_energy

    @property
    def total_variational_energy(self) -> float:
        return self.decomposition.total_variational_energy

    @property
    def expected_free_energy_g(self) -> float:
        return self.decomposition.expected_free_energy_g

    def to_causal_predicate(self) -> CausalPredicate:
        return self.decomposition.to_causal_predicate(self.action.id)

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR graph mutation logging candidate free energy in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"hfe_eval_{self.action.id}",
                node_data={
                    "action_id": self.action.id,
                    "action_intent": self.action.intent,
                    "total_free_energy": self.free_energy,
                    "expected_free_energy_g": self.expected_free_energy_g,
                    "sensory_f0": self.decomposition.sensory_free_energy_f0,
                    "dynamic_f1": self.decomposition.dynamic_free_energy_f1,
                    "epistemic_gain": self.decomposition.epistemic_information_gain,
                    "extrinsic_val": self.decomposition.extrinsic_goal_value,
                    "sensory_precision": self.decomposition.sensory_precision,
                    "dynamic_precision": self.decomposition.dynamic_precision,
                },
            )
        ]


class HierarchicalActiveInferenceEngine:
    """Karl Friston's 3-Tier Hierarchical Variational Free Energy Minimization Engine.

    Minimizes surprise across the hierarchical active inference architecture:
    1. Level 0 (Sensory precision & observation error):
       F_0 = 0.5 * Pi_s * ||y - g(mu)||^2 - 0.5 * ln(Pi_s)
    2. Level 1 (Kinematic dynamic affordances & transition error):
       F_1 = 0.5 * Pi_mu * ||mu' - f(mu, a)||^2 - 0.5 * ln(Pi_mu)
    3. Level 2 (Policy optimization via Expected Free Energy G(pi)):
       G(pi) = PragmaticDivergence(s', goal) - EpistemicInformationGain(pi) + RiskPenalty(pi)
    """

    def __init__(
        self,
        sensory_precision: float = 1.0,
        dynamic_precision: float = 1.0,
        lambda_epistemic: float = 0.4,
        lambda_extrinsic: float = 0.6,
        lambda_risk: float = 0.5,
    ) -> None:
        self.sensory_precision = sensory_precision
        self.dynamic_precision = dynamic_precision
        self.lambda_epistemic = lambda_epistemic
        self.lambda_extrinsic = lambda_extrinsic
        self.lambda_risk = lambda_risk

    def compute_sensory_free_energy(
        self,
        current_obs: Any | None,
        predicted_obs: Any | None,
    ) -> float:
        """Compute Level 0 sensory variational free energy F0."""
        if current_obs is None or predicted_obs is None:
            return 0.1

        try:
            arr_curr = np.asarray(current_obs, dtype=float)
            arr_pred = np.asarray(predicted_obs, dtype=float)
            if arr_curr.shape == arr_pred.shape:
                diff = arr_curr - arr_pred
                error_sq = float(np.mean(diff**2))
            else:
                error_sq = 1.0
        except Exception:
            error_sq = 0.5 if current_obs != predicted_obs else 0.0

        p = max(0.01, self.sensory_precision)
        f0 = 0.5 * p * error_sq - 0.5 * math.log(p)
        return float(f0)

    def compute_dynamic_free_energy(
        self,
        current_state: Any | None,
        predicted_state: Any | None,
    ) -> float:
        """Compute Level 1 kinematic dynamic variational free energy F1."""
        if current_state is None or predicted_state is None:
            return 0.1

        try:
            s_curr = np.asarray(current_state, dtype=float)
            s_pred = np.asarray(predicted_state, dtype=float)
            if s_curr.shape == s_pred.shape:
                diff = s_curr - s_pred
                error_sq = float(np.sum(diff**2))
            else:
                error_sq = 1.0
        except Exception:
            error_sq = 0.5 if current_state != predicted_state else 0.0

        p = max(0.01, self.dynamic_precision)
        f1 = 0.5 * p * error_sq - 0.5 * math.log(p)
        return float(f1)

    def compute_expected_free_energy_g(
        self,
        action: ActionNode,
        epistemic_gain: float,
        goal_divergence: float,
        risk_penalty: float,
    ) -> float:
        """Compute Level 2 Expected Free Energy G(pi) for policy candidate.

        Lower is preferred (minimization of expected free energy).
        G = lambda_extrinsic * goal_divergence - lambda_epistemic * epistemic_gain + lambda_risk * risk
        """
        g = (
            (self.lambda_extrinsic * goal_divergence)
            - (self.lambda_epistemic * epistemic_gain)
            + (self.lambda_risk * risk_penalty)
        )
        return float(g)

    def evaluate_candidates_hierarchical(
        self,
        candidate_actions: Sequence[ActionNode],
        current_obs: Any | None = None,
        predicted_obs: Any | None = None,
        current_state: Any | None = None,
        predicted_state: Any | None = None,
        goal_pos: tuple[float, float] | None = None,
        predicted_positions: dict[str, tuple[float, float]] | None = None,
        epistemic_gain_map: dict[str, float] | None = None,
    ) -> list[HierarchicalInferenceCandidateResult]:
        """Evaluate candidate actions across the 3-level variational hierarchy."""
        f0 = self.compute_sensory_free_energy(current_obs, predicted_obs)
        f1 = self.compute_dynamic_free_energy(current_state, predicted_state)

        info_map = epistemic_gain_map or {}
        pos_map = predicted_positions or {}
        results: list[HierarchicalInferenceCandidateResult] = []

        for action in candidate_actions:
            info_gain = info_map.get(action.id, 0.4)
            risk = getattr(action, "risk_factor", 0.1)

            # Goal divergence heuristic
            goal_div = 0.5
            if goal_pos is not None and action.id in pos_map:
                pred_p = pos_map[action.id]
                dist = math.hypot(pred_p[0] - goal_pos[0], pred_p[1] - goal_pos[1])
                goal_div = min(2.0, dist / 10.0)

            g = self.compute_expected_free_energy_g(
                action=action,
                epistemic_gain=info_gain,
                goal_divergence=goal_div,
                risk_penalty=risk,
            )

            total_energy = f0 + f1 + g

            decomp = HierarchicalFreeEnergyDecomposition(
                sensory_free_energy_f0=f0,
                dynamic_free_energy_f1=f1,
                expected_free_energy_g=g,
                epistemic_information_gain=info_gain,
                extrinsic_goal_value=goal_div,
                sensory_precision=self.sensory_precision,
                dynamic_precision=self.dynamic_precision,
                total_variational_energy=total_energy,
            )
            results.append(
                HierarchicalInferenceCandidateResult(
                    action=action,
                    decomposition=decomp,
                )
            )

        # In Free Energy Minimization, lower variational energy is optimal
        results.sort(key=lambda r: r.total_variational_energy)
        for rank, res in enumerate(results):
            res.rank = rank

        return results

    def select_min_free_energy_action(
        self,
        candidate_actions: Sequence[ActionNode],
        current_obs: Any | None = None,
        predicted_obs: Any | None = None,
        current_state: Any | None = None,
        predicted_state: Any | None = None,
        goal_pos: tuple[float, float] | None = None,
        predicted_positions: dict[str, tuple[float, float]] | None = None,
        epistemic_gain_map: dict[str, float] | None = None,
    ) -> HierarchicalInferenceCandidateResult:
        """Select candidate action that minimizes total hierarchical free energy."""
        evals = self.evaluate_candidates_hierarchical(
            candidate_actions=candidate_actions,
            current_obs=current_obs,
            predicted_obs=predicted_obs,
            current_state=current_state,
            predicted_state=predicted_state,
            goal_pos=goal_pos,
            predicted_positions=predicted_positions,
            epistemic_gain_map=epistemic_gain_map,
        )
        if not evals:
            raise ValueError("Candidate actions cannot be empty")
        return evals[0]

    def select_and_transact(
        self,
        candidate_actions: Sequence[ActionNode],
        **kwargs: Any,
    ) -> tuple[HierarchicalInferenceCandidateResult, list[TransactionOperation]]:
        """Select best candidate action and propose atomic state transaction."""
        best = self.select_min_free_energy_action(candidate_actions, **kwargs)
        return best, best.to_transaction_operations()

    def export_causal_predicates(
        self,
        candidate_actions: Sequence[ActionNode],
        **kwargs: Any,
    ) -> list[CausalPredicate]:
        """Export candidate hierarchical evaluations as causal predicates."""
        evals = self.evaluate_candidates_hierarchical(candidate_actions, **kwargs)
        return [e.to_causal_predicate() for e in evals]
