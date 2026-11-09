"""Superposition Counterfactual Tree Search (Quantum-Inspired OFC Rollouts).

Biologically grounded in Orbitofrontal Cortex (OFC) state representations under
partial observability and unobserved latent causes (Wilson et al., 2014; Schuck et al., 2016):
1. Superposition Belief State:
   - When entities possess ambiguous causal roles (e.g., whether a switch opens a door,
     reverses conveyor direction, or triggers a hazard), maintains a superposition
     probability distribution across plausible hypotheses:
       |Psi> = sum_k p_k |H_k>,  sum_k p_k = 1.0.
2. Parallel Superposition Mental Simulation:
   - Rollouts evaluate candidate motor policies across all hypotheses simultaneously.
   - Computes Expected Multi-World Utility: E[U(a)] = sum_k p_k * U(a | H_k).
3. Universal Invariant Action Identification:
   - Discovers actions that are mutually safe and progress-optimal across ALL hypotheses.
   - Allows decisive action execution without requiring premature epistemic resolution.
4. Active Epistemic Probing:
   - Identifies actions that maximize mutual information I(Action; Hypotheses),
     deliberately separating hypotheses with minimal risk.
5. Epistemic Wavefunction Collapse:
   - Upon receiving actual empirical observation diffs, Pearl's causal calculus
     updates hypothesis posteriors, collapsing the superposition state and pruning
     incompatible tree search branches instantaneously.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class HypothesisState:
    """A single plausible latent world configuration in the superposition."""

    hypothesis_id: str
    description: str
    probability: float
    latent_variables: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export latent hypothesis as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"superposition_hypothesis:{self.hypothesis_id}",
            operator="==",
            value=round(self.probability, 4),
            relation=f"latent_vars:{len(self.latent_variables)}",
            target_entity=self.hypothesis_id,
        )


@dataclass
class SuperpositionEvaluationResult:
    """Action evaluation across the superposition belief distribution."""

    action_id: Any
    expected_utility: float
    is_universal_invariant: bool
    information_gain_divergence: float
    hypothesis_utilities: dict[str, float]

    def to_causal_predicate(self) -> CausalPredicate:
        """Export action superposition evaluation as a formal causal predicate."""
        return CausalPredicate(
            variable=f"superposition_eval:{self.action_id}",
            operator="==",
            value=round(self.expected_utility, 4),
            relation=f"invariant:{self.is_universal_invariant}:infogain:{self.information_gain_divergence:.3f}",
            target_entity=str(self.action_id),
        )

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Propose atomic HCIR transaction logging superposition evaluation in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"sup_eval_{self.action_id}",
                node_data={
                    "action_id": self.action_id,
                    "expected_utility": float(self.expected_utility),
                    "is_universal_invariant": bool(self.is_universal_invariant),
                    "information_gain_divergence": float(self.information_gain_divergence),
                    "hypothesis_utilities": self.hypothesis_utilities,
                    "tenant_id": tenant_id,
                },
            )
        ]


class SuperpositionPlanner:
    """Orbitofrontal mental simulation planner executing over superposition belief states."""

    def __init__(self, min_probability_threshold: float = 0.05) -> None:
        self.hypotheses: dict[str, HypothesisState] = {}
        self.min_probability_threshold = min_probability_threshold
        self.evaluation_history: list[SuperpositionEvaluationResult] = []

    def add_hypothesis(
        self,
        hypothesis_id: str,
        probability: float,
        description: str = "",
        latent_variables: dict[str, Any] | None = None,
    ) -> HypothesisState:
        """Add a candidate latent causal configuration to the superposition belief state."""
        h = HypothesisState(
            hypothesis_id=hypothesis_id,
            description=description or f"Hypothesis {hypothesis_id}",
            probability=float(probability),
            latent_variables=dict(latent_variables or {}),
        )
        self.hypotheses[hypothesis_id] = h
        self._normalize_probabilities()
        return h

    def _normalize_probabilities(self) -> None:
        """Normalize hypothesis probabilities across the simplex."""
        total = sum(h.probability for h in self.hypotheses.values())
        if total > 0.0:
            for h in self.hypotheses.values():
                h.probability /= total

    def evaluate_superposition_actions(
        self,
        candidate_actions: list[Any],
        simulator_fn: Callable[[Any, dict[str, Any]], float],
    ) -> list[SuperpositionEvaluationResult]:
        """Evaluate actions across all superposition states simultaneously.

        Args:
            candidate_actions: List of action identifiers or nodes.
            simulator_fn: Callable(action, latent_variables) -> utility_score in [0.0, 1.0].
        """
        if not self.hypotheses:
            # Trivial case: single deterministic reality
            return [
                SuperpositionEvaluationResult(
                    action_id=a,
                    expected_utility=1.0,
                    is_universal_invariant=True,
                    information_gain_divergence=0.0,
                    hypothesis_utilities={"default": 1.0},
                )
                for a in candidate_actions
            ]

        results: list[SuperpositionEvaluationResult] = []

        for action in candidate_actions:
            utils: dict[str, float] = {}
            expected_u = 0.0
            for hid, h in self.hypotheses.items():
                u = simulator_fn(action, h.latent_variables)
                utils[hid] = float(u)
                expected_u += h.probability * u

            # Check if universal invariant: safe/rewarding across all branches (all u >= 0.5)
            is_invariant = all(u >= 0.5 for u in utils.values())

            # Information gain divergence: variance across hypothesis outcomes
            # Actions that yield very different outcomes under different hypotheses provide high information gain
            u_vals = list(utils.values())
            mean_u = sum(u_vals) / len(u_vals)
            divergence = sum((u - mean_u) ** 2 for u in u_vals) / len(u_vals)
            info_gain = math.sqrt(divergence)

            res = SuperpositionEvaluationResult(
                action_id=action,
                expected_utility=expected_u,
                is_universal_invariant=is_invariant,
                information_gain_divergence=info_gain,
                hypothesis_utilities=utils,
            )
            results.append(res)

        # Primary sort: universal invariants first, then highest expected utility
        results.sort(
            key=lambda r: (1 if r.is_universal_invariant else 0, r.expected_utility),
            reverse=True,
        )
        self.evaluation_history = results
        return results

    def select_best_action(
        self,
        candidate_actions: list[Any],
        simulator_fn: Callable[[Any, dict[str, Any]], float],
        prefer_epistemic_probe_when_ambiguous: bool = True,
    ) -> SuperpositionEvaluationResult:
        """Select action: if universal invariant exists, execute it; else probe highest info gain."""
        evals = self.evaluate_superposition_actions(candidate_actions, simulator_fn)
        if not evals:
            raise ValueError("No candidate actions provided")

        # If top action is invariant or if hypotheses are already nearly collapsed, proceed pragmatically
        max_prob = max((h.probability for h in self.hypotheses.values()), default=1.0)
        if (
            evals[0].is_universal_invariant
            or max_prob > 0.85
            or not prefer_epistemic_probe_when_ambiguous
        ):
            return evals[0]

        # Under high ambiguity without invariant safe action, pick maximal information gain probe
        probe_sorted = sorted(evals, key=lambda r: r.information_gain_divergence, reverse=True)
        return probe_sorted[0]

    def collapse_superposition(
        self,
        observation_evidence: dict[str, Any],
        likelihood_fn: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
    ) -> str:
        """Collapse superposition belief state upon receiving empirical feedback.

        Updates Bayesian posterior probabilities:
            P(H_k | E) proportional to P(E | H_k) * P(H_k)
        Prunes zero-probability branches instantaneously.
        Returns the ID of the most probable surviving hypothesis.
        """
        for hid, h in self.hypotheses.items():
            if likelihood_fn is not None:
                likelihood = likelihood_fn(observation_evidence, h.latent_variables)
            else:
                # Default match check on evidence keys
                match_count = sum(
                    1 for k, v in observation_evidence.items() if h.latent_variables.get(k) == v
                )
                likelihood = math.exp(match_count) if match_count > 0 else 0.1

            h.probability *= float(likelihood)

        self._normalize_probabilities()

        # Prune branches below minimum threshold
        surviving = {
            hid: h
            for hid, h in self.hypotheses.items()
            if h.probability >= self.min_probability_threshold
        }
        if surviving:
            self.hypotheses = surviving
            self._normalize_probabilities()

        winner_id = max(self.hypotheses.keys(), key=lambda k: self.hypotheses[k].probability)
        logger.info(
            "Superposition collapsed: Winner '%s' (P=%.3f, remaining=%d)",
            winner_id,
            self.hypotheses[winner_id].probability,
            len(self.hypotheses),
        )
        return winner_id

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export current superposition state as declarative causal predicates."""
        predicates = [
            CausalPredicate(
                variable="superposition_belief_state",
                operator="==",
                value=f"hypotheses:{len(self.hypotheses)}",
                relation="quantum_inspired_ofc",
                target_entity="superposition_planner",
            )
        ]
        for h in self.hypotheses.values():
            predicates.append(h.to_causal_predicate())
        return predicates
