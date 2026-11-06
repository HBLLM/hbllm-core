"""Domain-General Interactive Goal Induction, Deadlock Detection, and Progress Evaluator.

Implements capabilities:
- W168: Open-Set Bayesian Goal Hypothesis Induction over Latent Objectives
- W169: Goal Confirmation Gate & Anti-Lure Discrimination
- W170: Metric and Discrete Progress Estimation
- W174: Model-Based Forward Reachability Deadlock Detection
- W175: Counterfactual Recovery Planning and Branch Exploration
- W181: Terminal-State Classification (Win, Loss, Quiescence, Incomplete)
- W182: Multi-Channel Environmental Feedback Interpretation
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate


class TerminalStateCategory(StrEnum):
    """Categorical classification of environmental episode termination (W181)."""

    WIN = "WIN"
    DEATH_OR_LOSS = "DEATH_OR_LOSS"
    DEADLOCK_OR_TRAP = "DEADLOCK_OR_TRAP"
    BUDGET_EXHAUSTION = "BUDGET_EXHAUSTION"
    NOT_TERMINAL = "NOT_TERMINAL"


class GoalHypothesisType(StrEnum):
    """Open-set taxonomy of candidate environmental win conditions (W168)."""

    BOUNDARY_CROSSING = "BOUNDARY_CROSSING"
    CONSUMABLE_CLEARANCE = "CONSUMABLE_CLEARANCE"
    TOPOLOGICAL_DOCKING = "TOPOLOGICAL_DOCKING"
    PATTERN_SYMMETRY = "PATTERN_SYMMETRY"
    STATE_REGISTER_THRESHOLD = "STATE_REGISTER_THRESHOLD"
    CUSTOM_PREDICATE = "CUSTOM_PREDICATE"


@dataclass
class GoalHypothesis:
    """A testable latent environmental goal hypothesis with Bayesian posterior confidence (W168)."""

    goal_id: str
    goal_type: GoalHypothesisType
    description: str
    target_criteria: dict[str, Any]
    prior_probability: float = 0.20
    posterior_probability: float = 0.20
    positive_evidence_count: int = 0
    negative_evidence_count: int = 0
    is_confirmed: bool = False
    is_refuted: bool = False

    def evaluate_state_satisfaction(self, state: dict[str, Any] | np.ndarray) -> bool:
        """Evaluate if the given state satisfies this goal hypothesis."""
        if self.goal_type == GoalHypothesisType.CONSUMABLE_CLEARANCE:
            feat_name = self.target_criteria.get("feature_name", "consumables_count")
            if isinstance(state, dict):
                val = state.get(feat_name, 1)
                return val == 0
            elif isinstance(state, np.ndarray):
                target_val = self.target_criteria.get("target_value", 2)
                return int(np.count_nonzero(state == target_val)) == 0

        elif self.goal_type == GoalHypothesisType.BOUNDARY_CROSSING:
            target_region = self.target_criteria.get("target_region", set())
            if isinstance(state, dict):
                pos = state.get("avatar_pos")
                return pos in target_region if pos else False

        elif self.goal_type == GoalHypothesisType.TOPOLOGICAL_DOCKING:
            if isinstance(state, dict):
                docked = state.get("docked_count", 0)
                required = self.target_criteria.get("required_docked", 1)
                return docked >= required

        elif self.goal_type == GoalHypothesisType.PATTERN_SYMMETRY:
            if isinstance(state, np.ndarray):
                axis = self.target_criteria.get("axis", "vertical")
                if axis == "horizontal":
                    return np.array_equal(state, np.flipud(state))
                return np.array_equal(state, np.fliplr(state))

        elif self.goal_type == GoalHypothesisType.STATE_REGISTER_THRESHOLD:
            if isinstance(state, dict):
                reg_name = self.target_criteria.get("register_name", "score")
                thresh = self.target_criteria.get("threshold", 1)
                return float(state.get(reg_name, 0)) >= float(thresh)

        elif self.goal_type == GoalHypothesisType.CUSTOM_PREDICATE:
            pred_fn = self.target_criteria.get("predicate_fn")
            if callable(pred_fn):
                return bool(pred_fn(state))
            causal_pred = self.target_criteria.get("causal_predicate")
            if isinstance(causal_pred, CausalPredicate) and isinstance(state, dict):
                return causal_pred.evaluate(state)

        return False

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert goal hypothesis into a declarative HCIR CausalPredicate."""
        if self.goal_type == GoalHypothesisType.CONSUMABLE_CLEARANCE:
            feat_name = self.target_criteria.get("feature_name", "consumables_count")
            return CausalPredicate(variable=feat_name, operator="==", value=0)
        elif self.goal_type == GoalHypothesisType.BOUNDARY_CROSSING:
            target_region = self.target_criteria.get("target_region", set())
            return CausalPredicate(variable="avatar_pos", operator="in", value=target_region)
        elif self.goal_type == GoalHypothesisType.TOPOLOGICAL_DOCKING:
            required = self.target_criteria.get("required_docked", 1)
            return CausalPredicate(variable="docked_count", operator=">=", value=required)
        elif self.goal_type == GoalHypothesisType.PATTERN_SYMMETRY:
            return CausalPredicate(variable="grid_symmetry", operator="==", value=True)
        elif self.goal_type == GoalHypothesisType.STATE_REGISTER_THRESHOLD:
            reg_name = self.target_criteria.get("register_name", "score")
            thresh = self.target_criteria.get("threshold", 1)
            return CausalPredicate(variable=reg_name, operator=">=", value=thresh)
        elif self.goal_type == GoalHypothesisType.CUSTOM_PREDICATE:
            causal_pred = self.target_criteria.get("causal_predicate")
            if isinstance(causal_pred, CausalPredicate):
                return causal_pred
        return CausalPredicate(variable=f"goal_{self.goal_id}", operator="==", value=True)


@dataclass
class GoalInductionObservationResult:
    """Structured result of Bayesian goal belief updating under environmental feedback."""

    confirmed_goals: list[GoalHypothesis] = field(default_factory=list)
    refuted_goals: list[GoalHypothesis] = field(default_factory=list)
    active_hypotheses: list[GoalHypothesis] = field(default_factory=list)
    dominant_goal: GoalHypothesis | None = None
    goal_entropy: float = 0.0


class BayesianGoalInductionEngine:
    """Maintains open-set hypotheses over latent goal criteria and updates beliefs from feedback (W168, W182)."""

    def __init__(self) -> None:
        self.hypotheses: dict[str, GoalHypothesis] = {}
        self._initialize_default_priors()

    def _initialize_default_priors(self) -> None:
        """Seed open-set hypothesis priors for common cognitive objective patterns."""
        self.register_hypothesis(
            GoalHypothesis(
                goal_id="clear_all_targets",
                goal_type=GoalHypothesisType.CONSUMABLE_CLEARANCE,
                description="Eliminate all remaining consumable target tokens",
                target_criteria={"feature_name": "remaining_items"},
                prior_probability=0.30,
                posterior_probability=0.30,
            )
        )
        self.register_hypothesis(
            GoalHypothesis(
                goal_id="reach_perimeter_exit",
                goal_type=GoalHypothesisType.BOUNDARY_CROSSING,
                description="Navigate avatar to environmental perimeter exit zone",
                target_criteria={"feature_name": "exit_portal"},
                prior_probability=0.25,
                posterior_probability=0.25,
            )
        )
        self.register_hypothesis(
            GoalHypothesis(
                goal_id="dock_cargo_in_receptacle",
                goal_type=GoalHypothesisType.TOPOLOGICAL_DOCKING,
                description="Align portable cargo entities with target receptors",
                target_criteria={"required_docked": 1},
                prior_probability=0.25,
                posterior_probability=0.25,
            )
        )
        self.register_hypothesis(
            GoalHypothesis(
                goal_id="restore_canvas_symmetry",
                goal_type=GoalHypothesisType.PATTERN_SYMMETRY,
                description="Transform grid canvas to complete reflective symmetry",
                target_criteria={},
                prior_probability=0.20,
                posterior_probability=0.20,
            )
        )

    def register_hypothesis(self, hypothesis: GoalHypothesis) -> None:
        """Register a new candidate goal hypothesis."""
        self.hypotheses[hypothesis.goal_id] = hypothesis
        self._normalize_posteriors()

    def _normalize_posteriors(self) -> None:
        """Ensure posterior probabilities form a valid distribution across non-refuted hypotheses."""
        active = [h for h in self.hypotheses.values() if not h.is_refuted]
        total = sum(h.posterior_probability for h in active)
        if total > 0.0:
            for h in active:
                h.posterior_probability = h.posterior_probability / total

    def compute_goal_entropy(self) -> float:
        """Calculate Shannon entropy over active goal hypotheses in bits."""
        active = [h for h in self.hypotheses.values() if not h.is_refuted]
        if not active:
            return 0.0
        entropy = 0.0
        for h in active:
            p = h.posterior_probability
            if p > 1e-9:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def observe_feedback(
        self,
        current_state: dict[str, Any] | np.ndarray,
        level_incremented: bool = False,
        reward_received: float = 0.0,
        is_death_or_reset: bool = False,
    ) -> list[GoalHypothesis]:
        """Update Bayesian posteriors upon receiving multi-channel environmental feedback (W182)."""
        result = self.observe_feedback_detailed(
            current_state,
            level_incremented=level_incremented,
            reward_received=reward_received,
            is_death_or_reset=is_death_or_reset,
        )
        return result.confirmed_goals

    def observe_feedback_detailed(
        self,
        current_state: dict[str, Any] | np.ndarray,
        level_incremented: bool = False,
        reward_received: float = 0.0,
        is_death_or_reset: bool = False,
    ) -> GoalInductionObservationResult:
        """Structured update of Bayesian goal beliefs with complete diagnostic accounting."""
        confirmed_goals: list[GoalHypothesis] = []
        refuted_goals: list[GoalHypothesis] = []

        for h in list(self.hypotheses.values()):
            if h.is_refuted:
                continue

            satisfied = h.evaluate_state_satisfaction(current_state)

            if level_incremented or reward_received > 0.0:
                if satisfied:
                    h.positive_evidence_count += 1
                    # Bayesian likelihood boost: P(E | H) = 0.95
                    h.posterior_probability *= 4.0
                    if h.positive_evidence_count >= 1:
                        h.is_confirmed = True
                        confirmed_goals.append(h)
                else:
                    # Penalize hypotheses not satisfied upon goal trigger
                    h.posterior_probability *= 0.3
            elif is_death_or_reset:
                if satisfied:
                    # Satisfied when death occurred: likely a lethal trap / hazard, NOT a goal!
                    h.negative_evidence_count += 1
                    h.posterior_probability *= 0.1
                    if h.negative_evidence_count >= 2:
                        h.is_refuted = True
                        refuted_goals.append(h)

        self._normalize_posteriors()
        active = [h for h in self.hypotheses.values() if not h.is_refuted]
        dominant = max(active, key=lambda h: h.posterior_probability) if active else None

        return GoalInductionObservationResult(
            confirmed_goals=confirmed_goals,
            refuted_goals=refuted_goals,
            active_hypotheses=active,
            dominant_goal=dominant,
            goal_entropy=self.compute_goal_entropy(),
        )

    def get_dominant_goal(self) -> GoalHypothesis | None:
        """Return the highest-probability active goal hypothesis."""
        active = [h for h in self.hypotheses.values() if not h.is_refuted]
        if not active:
            return None
        return max(active, key=lambda h: h.posterior_probability)

    def to_symbolic_world_theory(self) -> list[CausalPredicate]:
        """Export all confirmed goal hypotheses as declarative HCIR CausalPredicates."""
        return [h.to_causal_predicate() for h in self.hypotheses.values() if h.is_confirmed]

    def reset(self) -> None:
        """Reset goal engine back to initial priors."""
        self.hypotheses.clear()
        self._initialize_default_priors()


# ── W169, W181: Goal Confirmation Gate & Terminal Classification ─────────────


class GoalConfirmationGate:
    """Discriminates genuine goals from lethal lures and classifies terminal states (W169, W181)."""

    @staticmethod
    def classify_terminal_event(
        is_finished: bool,
        levels_completed_delta: int,
        is_death: bool,
        steps_left: int,
    ) -> TerminalStateCategory:
        """Classifies episode boundary events."""
        if levels_completed_delta > 0:
            return TerminalStateCategory.WIN
        if is_death:
            return TerminalStateCategory.DEATH_OR_LOSS
        if steps_left <= 0:
            return TerminalStateCategory.BUDGET_EXHAUSTION
        return TerminalStateCategory.NOT_TERMINAL

    @staticmethod
    def evaluate_anti_lure_hazard(
        hypothesis: GoalHypothesis,
    ) -> float:
        """Anti-lure discrimination score in [0.0, 1.0]. High score indicates a lethal lure."""
        total = hypothesis.positive_evidence_count + hypothesis.negative_evidence_count
        if total == 0:
            return 0.0
        return hypothesis.negative_evidence_count / float(total)


# ── W170: Progress Estimation ───────────────────────────────────────────────


class DynamicProgressEstimator:
    """Computes continuous and discrete progress metrics toward hypothesized goals (W170)."""

    @staticmethod
    def estimate_progress(
        current_distance: float,
        initial_distance: float,
        subgoals_achieved: int = 0,
        total_subgoals: int = 1,
        spatial_weight: float = 0.6,
    ) -> float:
        """Compute normalized progress in [0.0, 1.0]."""
        if initial_distance <= 0.0:
            spatial_prog = 1.0
        else:
            spatial_prog = max(0.0, min(1.0, 1.0 - (current_distance / initial_distance)))

        subgoal_weight = 1.0 - spatial_weight
        subgoal_prog = subgoals_achieved / float(total_subgoals) if total_subgoals > 0 else 0.0

        return spatial_weight * spatial_prog + subgoal_weight * subgoal_prog


# ── W174, W175: Model-Based Reachability Deadlock Detection & Recovery ──────


def canonical_state_hash(state: Any) -> Any:
    """Compute deterministic hashable representation for graph reachability search."""
    if isinstance(state, (int, float, str, bool)):
        return state
    elif isinstance(state, tuple):
        return tuple(canonical_state_hash(x) for x in state)
    elif isinstance(state, list):
        return tuple(canonical_state_hash(x) for x in state)
    elif isinstance(state, np.ndarray):
        return (state.shape, state.tobytes())
    elif isinstance(state, dict):
        return tuple(sorted((k, canonical_state_hash(v)) for k, v in state.items()))
    return str(state)


@dataclass
class DeadlockAnalysisResult:
    """Result of forward reachability graph search under learned transition dynamics (W174)."""

    is_deadlocked: bool
    reachable_state_count: int
    goal_reachable: bool
    recovery_branch_state: Any | None = None
    steps_to_recovery: int = -1
    cycle_detected: bool = False


class ModelBasedDeadlockDetector:
    """Detects absorbing non-goal states via forward reachability analysis on the learned model (W174, W175)."""

    def __init__(self, max_search_depth: int = 25) -> None:
        self.max_search_depth = max_search_depth

    def analyze_reachability(
        self,
        current_state: Any,
        transition_fn: Callable[[Any, Any], Any],
        available_actions: Sequence[Any],
        is_goal_fn: Callable[[Any], bool],
        is_loss_fn: Callable[[Any], bool] | None = None,
    ) -> DeadlockAnalysisResult:
        """Performs forward reachability search to prove whether any path to goal exists.

        A state is deadlocked iff no reachable trajectory leads to goal satisfaction.
        """
        # BFS forward expansion
        visited: set[Any] = set()
        queue: deque[tuple[Any, int, list[Any]]] = deque([(current_state, 0, [])])
        visited.add(canonical_state_hash(current_state))

        goal_found = False
        cycle_found = False
        state_count = 0

        while queue and state_count < 1000:
            state, depth, path = queue.popleft()
            state_count += 1

            if is_goal_fn(state):
                goal_found = True
                break

            if is_loss_fn and is_loss_fn(state):
                continue

            if depth >= self.max_search_depth:
                continue

            for a in available_actions:
                try:
                    next_s = transition_fn(state, a)
                except Exception:
                    continue

                if next_s is None:
                    continue

                s_key = canonical_state_hash(next_s)
                if s_key in visited:
                    cycle_found = True
                    continue

                visited.add(s_key)
                queue.append((next_s, depth + 1, path + [a]))

        is_deadlocked = not goal_found
        return DeadlockAnalysisResult(
            is_deadlocked=is_deadlocked,
            reachable_state_count=state_count,
            goal_reachable=goal_found,
            cycle_detected=cycle_found,
        )

    def plan_recovery_action(
        self,
        history_trace: Sequence[tuple[Any, Any]],  # [(state, action)]
        transition_fn: Callable[[Any, Any], Any],
        available_actions: Sequence[Any],
        is_goal_fn: Callable[[Any], bool],
    ) -> Any | None:
        """Finds the most recent historical state that had an alternative non-deadlocked branch (W175)."""
        for state, prior_act in reversed(history_trace):
            for alt_act in available_actions:
                if alt_act == prior_act:
                    continue
                try:
                    candidate_s = transition_fn(state, alt_act)
                    reach = self.analyze_reachability(
                        candidate_s, transition_fn, available_actions, is_goal_fn
                    )
                    if reach.goal_reachable:
                        return alt_act
                except Exception:
                    continue
        return None
