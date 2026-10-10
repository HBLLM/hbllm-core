"""Domain-General Interactive Action Discovery & Constrained Active Experimentation Engine.

Implements capabilities:
- W161: Ungrounded Action Semantics Inference with Bayesian Uncertainty
- W162: Action-Effect Distribution & Feature Mutation Modeling
- W163: Open-Set Unknown-Mechanics & Affordance Ledger
- W164: Active Experiment Selection under Expected Free Energy
- W165: Constrained Expected Information Gain (EIG) subject to Cost, Risk, and Goal Progress
- W171: Action Precondition Induction & Contrastive Version Spaces
- W187: Action Budget Governor & Epistemic Exploration Throttling
- W188: Habenular Inhibition of Return (IOR) for Experimentation Avoidance
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class ActionEffectType(StrEnum):
    """Categorical characterization of an action's emergent empirical effect profile."""

    SPATIAL_DISPLACEMENT = "SPATIAL_DISPLACEMENT"
    STATE_MUTATION = "STATE_MUTATION"
    CONTACT_INTERACTION = "CONTACT_INTERACTION"
    QUIESCENT_NOOP = "QUIESCENT_NOOP"
    STOCHASTIC_OR_MULTIMODAL = "STOCHASTIC_OR_MULTIMODAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ActionEffectProfile:
    """Empirical effect profile for an ungrounded action token with Bayesian uncertainty (W161, W162)."""

    action_id: Any
    total_observations: int = 0
    quiescent_count: int = 0
    # Spatial displacement moments: delta_x = (delta_r, delta_c)
    mean_displacement: tuple[float, float] = (0.0, 0.0)
    displacement_variance: float = 0.0
    displacement_counts: dict[tuple[int, int], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # State mutations observed (attribute changes or feature edits)
    mutation_frequency: float = 0.0
    mutation_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Categorical classification and uncertainty
    effect_type: ActionEffectType = ActionEffectType.UNKNOWN
    uncertainty: float = 1.0  # Epistemic entropy / variance in [0.0, 1.0]

    def update(
        self,
        delta_pos: tuple[int, int] | None = None,
        mutations: dict[str, Any] | None = None,
    ) -> None:
        """Update posterior empirical moments upon observing a transition under this action."""
        self.total_observations += 1
        n = self.total_observations

        is_quiescent = True

        if delta_pos is not None:
            dr, dc = delta_pos
            if dr != 0 or dc != 0:
                is_quiescent = False
            self.displacement_counts[(dr, dc)] += 1
            # Welford's online mean and variance update for 2D vectors
            old_mean = self.mean_displacement
            new_r = old_mean[0] + (dr - old_mean[0]) / n
            new_c = old_mean[1] + (dc - old_mean[1]) / n
            self.mean_displacement = (new_r, new_c)

            # Empirical variance across observed displacement vectors
            sq_dists = [
                count * ((pos[0] - new_r) ** 2 + (pos[1] - new_c) ** 2)
                for pos, count in self.displacement_counts.items()
            ]
            self.displacement_variance = sum(sq_dists) / n if n > 0 else 0.0

        if mutations:
            is_quiescent = False
            for k, count in mutations.items():
                self.mutation_counts[k] += int(count)
            self.mutation_frequency = sum(self.mutation_counts.values()) / float(n)

        if is_quiescent:
            self.quiescent_count += 1

        # Classify effect profile
        p_quiescent = self.quiescent_count / float(n)
        if p_quiescent >= 0.85:
            self.effect_type = ActionEffectType.QUIESCENT_NOOP
        elif self.displacement_variance < 0.25 and (
            abs(self.mean_displacement[0]) > 0.1 or abs(self.mean_displacement[1]) > 0.1
        ):
            self.effect_type = ActionEffectType.SPATIAL_DISPLACEMENT
        elif self.mutation_frequency >= 0.50:
            self.effect_type = ActionEffectType.STATE_MUTATION
        elif len(self.displacement_counts) > 2:
            self.effect_type = ActionEffectType.STOCHASTIC_OR_MULTIMODAL
        else:
            self.effect_type = ActionEffectType.CONTACT_INTERACTION

        # Epistemic uncertainty decreases with sample size: normalized Dirichlet-multinomial entropy
        self.uncertainty = 1.0 / (1.0 + math.sqrt(n))


class UngroundedActionDiscoveryEngine:
    """Infers action transition distributions and semantics from ungrounded interaction tokens (W161, W162)."""

    def __init__(self, action_space: list[Any] | None = None) -> None:
        self.action_profiles: dict[Any, ActionEffectProfile] = {}
        if action_space:
            for a in action_space:
                self.action_profiles[a] = ActionEffectProfile(action_id=a)

    def register_action(self, action_id: Any) -> ActionEffectProfile:
        """Register a new action symbol into the discovery engine."""
        if action_id not in self.action_profiles:
            self.action_profiles[action_id] = ActionEffectProfile(action_id=action_id)
        return self.action_profiles[action_id]

    def record_transition(
        self,
        action_id: Any,
        prior_state: dict[str, Any] | np.ndarray,
        next_state: dict[str, Any] | np.ndarray,
        avatar_delta: tuple[int, int] | None = None,
    ) -> ActionEffectProfile:
        """Process an observed transition (s, a, s') and update action posterior."""
        profile = self.register_action(action_id)

        # Infer displacement if not explicitly provided
        delta_pos = avatar_delta
        if delta_pos is None and isinstance(prior_state, dict) and isinstance(next_state, dict):
            p_pos = prior_state.get("avatar_pos")
            n_pos = next_state.get("avatar_pos")
            if p_pos and n_pos:
                delta_pos = (int(n_pos[0] - p_pos[0]), int(n_pos[1] - p_pos[1]))

        # Infer state mutations
        mutations: dict[str, Any] = {}
        if isinstance(prior_state, dict) and isinstance(next_state, dict):
            for k in set(prior_state.keys()) | set(next_state.keys()):
                if k != "avatar_pos" and prior_state.get(k) != next_state.get(k):
                    mutations[k] = 1
        elif isinstance(prior_state, np.ndarray) and isinstance(next_state, np.ndarray):
            if prior_state.shape == next_state.shape:
                diff_count = int(np.count_nonzero(prior_state != next_state))
                if diff_count > 0:
                    mutations["grid_cells_mutated"] = diff_count

        profile.update(delta_pos=delta_pos, mutations=mutations)
        return profile

    def get_most_uncertain_action(self) -> Any:
        """Returns action symbol with highest epistemic uncertainty."""
        if not self.action_profiles:
            return None
        return max(self.action_profiles.items(), key=lambda kv: kv[1].uncertainty)[0]


# ── W163: Unknown Mechanics & Affordance Ledger ──────────────────────────────


@dataclass
class UnknownMechanicEntry:
    """Record of an unconfirmed or unprobed world mechanic (W163)."""

    entity_signature: str
    target_action: Any
    hypothesis_description: str
    probes_conducted: int = 0
    confidence: float = 0.0
    confirmed: bool = False
    refuted: bool = False


class UnknownMechanicsLedger:
    """Maintains an open-set ledger of untested affordances, entities, and dynamics (W163)."""

    def __init__(self) -> None:
        self.entries: dict[str, UnknownMechanicEntry] = {}

    def register_untested_affordance(
        self,
        entity_signature: str,
        action_id: Any,
        description: str,
    ) -> str:
        """Register a new unprobed affordance hypothesis."""
        key = f"{entity_signature}::act_{action_id}"
        if key not in self.entries:
            self.entries[key] = UnknownMechanicEntry(
                entity_signature=entity_signature,
                target_action=action_id,
                hypothesis_description=description,
            )
        return key

    def record_probe_outcome(
        self,
        entry_key: str,
        produced_expected_effect: bool,
    ) -> None:
        """Update confidence after conducting an interventional probe."""
        if entry_key not in self.entries:
            return
        entry = self.entries[entry_key]
        entry.probes_conducted += 1
        if produced_expected_effect:
            entry.confidence = min(1.0, entry.confidence + 0.35)
            if entry.confidence >= 0.70:
                entry.confirmed = True
        else:
            entry.confidence = max(0.0, entry.confidence - 0.40)
            if entry.probes_conducted >= 2 and entry.confidence <= 0.20:
                entry.refuted = True

    def get_pending_hypotheses(self) -> list[UnknownMechanicEntry]:
        """Return all active, non-refuted and non-confirmed mechanics awaiting test."""
        return [e for e in self.entries.values() if not e.confirmed and not e.refuted]


# ── W164, W165: Constrained Expected Free Energy Active Experimentation ──────


@dataclass
class ExperimentValue:
    """Decomposition of epistemic experiment value under constrained Expected Free Energy (W164, W165)."""

    action_id: Any
    expected_info_gain: float  # EIG in bits
    action_cost: float  # Resource / step cost
    risk_penalty: float  # Hazard / irreversibility penalty
    expected_goal_progress: float  # Alignment with goal hypothesis
    net_value: float  # G(a) = EIG - lambda_c * C - lambda_r * R + lambda_g * Prog


class ConstrainedActiveExperimenter:
    """Optimizes expected information gain subject to cost, hazard risk, and goal progress (W164, W165)."""

    def __init__(
        self,
        cost_weight: float = 0.20,
        risk_weight: float = 1.00,
        goal_weight: float = 0.50,
    ) -> None:
        self.cost_weight = cost_weight
        self.risk_weight = risk_weight
        self.goal_weight = goal_weight

    def evaluate_experiment(
        self,
        action_id: Any,
        profile: ActionEffectProfile,
        predicted_hazard_prob: float = 0.0,
        predicted_goal_progress: float = 0.0,
        action_cost: float = 1.0,
    ) -> ExperimentValue:
        """Compute net epistemic value G(a) for testing an action."""
        # Expected information gain is proportional to current parameter uncertainty
        eig = profile.uncertainty * math.log2(1.0 + 1.0 / max(0.1, profile.uncertainty))

        # Risk penalty is high if hazard probability is significant
        risk = predicted_hazard_prob * 3.0

        # Net objective: maximize information, minimize cost and danger, maximize goal progress
        net_val = (
            eig
            - self.cost_weight * action_cost
            - self.risk_weight * risk
            + self.goal_weight * predicted_goal_progress
        )

        return ExperimentValue(
            action_id=action_id,
            expected_info_gain=round(eig, 4),
            action_cost=round(action_cost, 4),
            risk_penalty=round(risk, 4),
            expected_goal_progress=round(predicted_goal_progress, 4),
            net_value=round(net_val, 4),
        )

    def select_best_action(
        self,
        available_actions: list[Any],
        discovery_engine: UngroundedActionDiscoveryEngine,
        hazard_oracle: dict[Any, float] | None = None,
        progress_oracle: dict[Any, float] | None = None,
    ) -> tuple[Any, ExperimentValue]:
        """Select action that maximizes constrained expected free energy."""
        best_act = available_actions[0] if available_actions else None
        best_eval: ExperimentValue | None = None

        for a in available_actions:
            prof = discovery_engine.register_action(a)
            h_prob = (hazard_oracle or {}).get(a, 0.0)
            g_prog = (progress_oracle or {}).get(a, 0.0)

            val = self.evaluate_experiment(
                action_id=a,
                profile=prof,
                predicted_hazard_prob=h_prob,
                predicted_goal_progress=g_prog,
            )

            if best_eval is None or val.net_value > best_eval.net_value:
                best_eval = val
                best_act = a

        if best_eval is None:
            best_eval = ExperimentValue(
                action_id=best_act,
                expected_info_gain=0.0,
                action_cost=1.0,
                risk_penalty=0.0,
                expected_goal_progress=0.0,
                net_value=0.0,
            )

        return best_act, best_eval


# ── W171: Action Precondition Induction & Contrastive Version Spaces ────────


@dataclass
class PreconditionRule:
    """Induced symbolic precondition enabling an action effect (W171)."""

    action_id: Any
    effect_name: str
    required_features: dict[str, Any]  # e.g., {"adjacent_to_key": True, "has_charge": True}
    confidence: float
    support_count: int


class ActionPreconditionLearner:
    """Induces necessary and sufficient enabling preconditions for actions (W171)."""

    def __init__(self) -> None:
        self.positive_examples: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
        self.negative_examples: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)

    def record_outcome(
        self,
        action_id: Any,
        effect_name: str,
        state_features: dict[str, Any],
        effect_occurred: bool,
    ) -> None:
        """Record state features where action either produced effect or failed to produce effect."""
        key = (action_id, effect_name)
        if effect_occurred:
            self.positive_examples[key].append(dict(state_features))
        else:
            self.negative_examples[key].append(dict(state_features))

    def induce_preconditions(
        self,
        action_id: Any,
        effect_name: str,
    ) -> PreconditionRule | None:
        """Find feature predicates true across all positive examples and absent in negative examples."""
        key = (action_id, effect_name)
        pos = self.positive_examples.get(key, [])
        neg = self.negative_examples.get(key, [])

        if not pos:
            return None

        # Candidate features shared across all positive instances
        shared_keys = set(pos[0].keys())
        for p in pos[1:]:
            shared_keys &= set(p.keys())

        induced_features: dict[str, Any] = {}
        for k in shared_keys:
            val = pos[0][k]
            if all(p.get(k) == val for p in pos):
                # Check discriminative power against negative instances
                if neg:
                    neg_match = sum(1 for n in neg if n.get(k) == val)
                    if neg_match < len(neg):
                        induced_features[k] = val
                else:
                    induced_features[k] = val

        confidence = len(pos) / float(len(pos) + len(neg)) if (len(pos) + len(neg)) > 0 else 0.5
        return PreconditionRule(
            action_id=action_id,
            effect_name=effect_name,
            required_features=induced_features,
            confidence=round(confidence, 4),
            support_count=len(pos),
        )


# ── W187, W188: Budget Governor & Habenular Inhibition of Return ─────────────


class ActionBudgetGovernor:
    """Manages episode budget consumption and controls exploration vs. exploitation throttling (W187, W188)."""

    def __init__(self, max_budget: int = 1000) -> None:
        self.max_budget = max_budget
        self.steps_consumed = 0
        self.ior_cache: dict[str, int] = {}  # Habenular Inhibition of Return (key -> step tested)

    def reset(self, max_budget: int | None = None) -> None:
        """Reset budget tracker for a new episode."""
        if max_budget is not None:
            self.max_budget = max_budget
        self.steps_consumed = 0
        self.ior_cache.clear()

    def record_step(self) -> int:
        """Increment consumed steps and return remaining budget."""
        self.steps_consumed += 1
        return self.get_remaining_budget()

    def get_remaining_budget(self) -> int:
        """Return remaining step budget."""
        return max(0, self.max_budget - self.steps_consumed)

    def get_exploration_weight(self) -> float:
        """Dynamically scales epistemic exploration drive based on remaining budget (W187)."""
        rem_ratio = self.get_remaining_budget() / float(max(1, self.max_budget))
        if rem_ratio > 0.60:
            return 1.0  # Full active exploration
        elif rem_ratio > 0.25:
            return rem_ratio  # Linear throttle
        else:
            return 0.10  # Emergency exploitation mode

    def register_experiment_trial(self, experiment_signature: str) -> None:
        """Record that an experiment was executed at the current step (W188 IOR)."""
        self.ior_cache[experiment_signature] = self.steps_consumed

    def is_inhibited_by_ior(self, experiment_signature: str, refractory_period: int = 15) -> bool:
        """True if experiment was recently tested and is currently inhibited by Habenular IOR (W188)."""
        last_step = self.ior_cache.get(experiment_signature)
        if last_step is None:
            return False
        return (self.steps_consumed - last_step) < refractory_period
