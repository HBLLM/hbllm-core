"""Orbitofrontal Cortex (OFC) Counterfactual Credit Assignment & Inferred State Value Updating.

Biological Foundations:
- Walton, Behrens, Buckley, Rudebeck, Rushworth (2010): "Separable learning systems
  in the macaque brain and the role of orbitofrontal cortex in contingent learning"
  (Neuron 65: 92-104).
- Boorman, Behrens, Woolrich, Rushworth (2009): "How green is the grass on the other side?
  Frontopolar cortex and the evidence in favour of alternative courses of action"
  (Neuron 62: 733-743).
- Abe & Lee (2011): "Distributed coding of actual and counterfactual outcomes in the
  primate prefrontal cortex" (Neuron).

Key Cognitive Capabilities:
1. Counterfactual Credit Propagation:
   - When an outcome (reward, deadlock, hazard) is experienced from choosing action A,
     the OFC infers what would have happened had alternative actions B or C been chosen,
     without requiring direct physical exploration.
2. Inferred Alternative Value Updating:
   - Updates Q(s, a_alt) using relational world structure, symmetry, and topological
     correlations (e.g. if moving into one narrow dead-end cavity caused a trap, other
     symmetric dead-ends are penalized counterfactually).
3. Counterfactual Regret Metric:
   - Computes regret R = max_a Q(s, a) - Q(s, a_chosen). When regret exceeds a threshold,
     it triggers frontopolar branch switching or task-set re-arbitration.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass(frozen=True)
class InferredBranchValue:
    """Estimated value and credit assigned to a counterfactual (unchosen) action branch."""

    action_id: str | int
    prior_value: float
    updated_value: float
    counterfactual_delta: float
    relational_similarity: float


@dataclass(frozen=True)
class CounterfactualCreditReport:
    """Consolidated counterfactual evaluation report emitted by the orbitofrontal cortex."""

    state_key: str
    chosen_action: str | int
    observed_outcome: float
    chosen_updated_value: float
    best_alternative_action: str | int
    best_alternative_value: float
    counterfactual_regret: float
    branch_updates: tuple[InferredBranchValue, ...]

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert counterfactual credit report into an HCIR CausalPredicate."""
        return CausalPredicate(
            variable="orbitofrontal_counterfactual_credit",
            operator="==",
            value=f"state:{self.state_key}:chosen:{self.chosen_action}:regret:{self.counterfactual_regret:.2f}:best_alt:{self.best_alternative_action}:alt_val:{self.best_alternative_value:.2f}",
            relation="ofc_contingent_credit_assignment",
            target_entity="orbitofrontal_cortex",
        )


class OrbitofrontalCreditAssignmentEngine:
    """Orchestrator for OFC counterfactual value updates and regret computation."""

    def __init__(
        self,
        learning_rate: float = 0.25,
        counterfactual_generalization: float = 0.50,
        regret_threshold: float = 0.35,
    ) -> None:
        """Initialize orbitofrontal credit assignment engine.

        Args:
            learning_rate: Base learning rate for direct experiential value updates.
            counterfactual_generalization: Weight given to relational similarity when propagating credit.
            regret_threshold: Regret level above which branch switching is recommended.
        """
        self.learning_rate: float = max(0.01, min(1.0, float(learning_rate)))
        self.generalization: float = max(0.0, min(1.0, float(counterfactual_generalization)))
        self.regret_threshold: float = max(0.05, float(regret_threshold))

        # Q-table: state_key -> {action_id: value}
        self._q_values: dict[str, dict[str, float]] = {}

    def get_action_values(self, state_key: str) -> dict[str, float]:
        """Retrieve estimated Q-values for a given state."""
        return dict(self._q_values.get(str(state_key), {}))

    def update_contingent_credit(
        self,
        state_key: str,
        chosen_action: str | int,
        observed_outcome: float,
        available_actions: Collection[str | int],
        relational_similarities: Mapping[str, float]
        | Mapping[int, float]
        | Mapping[str | int, float]
        | None = None,
    ) -> CounterfactualCreditReport:
        """Assign experiential credit to chosen action and propagate counterfactual credit to alternatives.

        Args:
            state_key: Symbolic identifier or hash of the current state.
            chosen_action: The action that was physically executed.
            observed_outcome: Realized outcome / reward / penalty score in [-1.0, 1.0].
            available_actions: Complete set of candidate actions available at this state.
            relational_similarities: Optional map of similarity [0.0, 1.0] between chosen and alternative actions.

        Returns:
            CounterfactualCreditReport detailing updated values, counterfactual deltas, and regret.
        """
        sk = str(state_key)
        ca_str = str(chosen_action)
        outcome = float(observed_outcome)
        sim_map = {str(k): float(v) for k, v in (relational_similarities or {}).items()}

        if sk not in self._q_values:
            self._q_values[sk] = {str(a): 0.0 for a in available_actions}

        state_qs = self._q_values[sk]
        for a in available_actions:
            a_str = str(a)
            if a_str not in state_qs:
                state_qs[a_str] = 0.0

        # 1. Direct Experiential Update of Chosen Action
        prior_chosen = state_qs[ca_str]
        chosen_delta = self.learning_rate * (outcome - prior_chosen)
        updated_chosen = prior_chosen + chosen_delta
        state_qs[ca_str] = updated_chosen

        # 2. Counterfactual Inferred Updates on Alternative Branches
        branch_updates: list[InferredBranchValue] = []
        for a in available_actions:
            a_str = str(a)
            if a_str == ca_str:
                continue

            prior_alt = state_qs[a_str]
            similarity = sim_map.get(a_str, 0.0)

            # If outcome is negative (e.g. hit wall, trap), similar actions are penalized counterfactually
            # If outcome is positive, different actions might yield lower value or similar actions benefit
            if outcome < 0.0:
                # Negative transfer: similar actions share the penalty
                cf_delta = self.learning_rate * self.generalization * similarity * outcome
            else:
                # Positive transfer: moderate positive generalization to similar affordances
                cf_delta = self.learning_rate * self.generalization * similarity * (outcome * 0.5)

            updated_alt = prior_alt + cf_delta
            state_qs[a_str] = updated_alt

            branch_updates.append(
                InferredBranchValue(
                    action_id=a,
                    prior_value=prior_alt,
                    updated_value=updated_alt,
                    counterfactual_delta=cf_delta,
                    relational_similarity=similarity,
                )
            )

        # 3. Find Best Alternative & Compute Regret
        alt_values = [(str(b.action_id), b.updated_value) for b in branch_updates]
        if alt_values:
            best_alt, best_alt_val = max(alt_values, key=lambda item: item[1])
        else:
            best_alt, best_alt_val = ca_str, updated_chosen

        # Regret R = max(0.0, best_alt_value - chosen_outcome/updated_chosen)
        regret = max(0.0, best_alt_val - updated_chosen)

        return CounterfactualCreditReport(
            state_key=sk,
            chosen_action=chosen_action,
            observed_outcome=outcome,
            chosen_updated_value=updated_chosen,
            best_alternative_action=best_alt,
            best_alternative_value=best_alt_val,
            counterfactual_regret=regret,
            branch_updates=tuple(branch_updates),
        )

    def is_branch_switch_indicated(self, report: CounterfactualCreditReport) -> bool:
        """Determine if counterfactual regret exceeds the policy re-arbitration threshold."""
        return report.counterfactual_regret >= self.regret_threshold
