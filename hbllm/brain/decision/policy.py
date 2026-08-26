"""Multi-Criteria Decision Policy and Rational Inaction for A19.

Evaluates hierarchical candidates (Goal Actions vs Epistemic Probes) under a unified
Expected Utility objective:
    EU(a) = w_g * G(a) + w_i * VoI(a) - w_r * R(a) - w_c * C(a) + w_v * V(a)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CandidateKind(str, Enum):
    """The purposeful origin of a decision candidate."""

    GOAL_ACTION = "goal_action"        # Direct progress toward extrinsic task
    EPISTEMIC_PROBE = "epistemic_probe"# Information-seeking probe to resolve uncertainty
    CLARIFICATION = "clarification"    # Linguistic dialog probe
    WAIT = "wait"                      # Temporary delay / wait for environment


class DecisionType(str, Enum):
    """The classified decision outcome."""

    ACTION = "action"
    PROBE = "probe"
    CLARIFICATION = "clarification"
    WAIT = "wait"
    INACTION = "inaction"  # Rational refusal to act when cost/risk exceeds value


@dataclass
class DecisionCandidate:
    """A proposed course of action with associated multi-criteria valuation metrics."""

    candidate_id: str = field(default_factory=lambda: f"cand_{uuid.uuid4().hex[:8]}")
    candidate_kind: CandidateKind = CandidateKind.GOAL_ACTION
    action_sequence: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    description: str = ""
    target_goal_ids: list[str] = field(default_factory=list)
    target_gap_ids: list[str] = field(default_factory=list)

    # Multi-criteria valuation metrics (0.0 to 1.0)
    goal_progress: float = 0.0      # G(a)
    information_gain: float = 0.0   # IG(a)
    value_of_information: float = 0.0 # VoI(a)
    predicted_risk: float = 0.0     # R(a)
    action_cost: float = 0.0        # C(a)
    reversibility: float = 1.0      # V(a)

    expected_utility: float = 0.0


@dataclass
class DecisionResult:
    """The final outcome of the cognitive decision process."""

    selected_candidate: DecisionCandidate | None
    decision_type: DecisionType
    expected_utility: float
    rejected_candidates: list[DecisionCandidate] = field(default_factory=list)
    rationale: str = ""


class DecisionEngine:
    """Multi-criteria expected utility decision policy."""

    def __init__(
        self,
        weight_goal: float = 1.0,
        weight_info: float = 0.8,
        weight_risk: float = 1.2,
        weight_cost: float = 0.5,
        weight_reversibility: float = 0.3,
        inaction_threshold: float = 0.05,
    ) -> None:
        self.w_g = weight_goal
        self.w_i = weight_info
        self.w_r = weight_risk
        self.w_c = weight_cost
        self.w_v = weight_reversibility
        self.inaction_threshold = inaction_threshold

    def evaluate_candidate_utility(self, candidate: DecisionCandidate) -> float:
        """Compute composite expected utility:

        EU(a) = (w_g * G + w_i * VoI) * (1 + w_v * V) - w_r * R - w_c * C * (2 - V)
        """
        # VoI takes precedence over raw IG if present
        info_val = candidate.value_of_information if candidate.value_of_information > 0.0 else candidate.information_gain
        base_value = (self.w_g * candidate.goal_progress) + (self.w_i * info_val)

        if base_value > 0.02:
            eu = (
                (base_value * (1.0 + (self.w_v * candidate.reversibility)))
                - (self.w_r * candidate.predicted_risk)
                - (self.w_c * candidate.action_cost * (2.0 - candidate.reversibility))
            )
        else:
            # Action with negligible/zero info or goal progress can only have zero or negative utility
            eu = base_value - (self.w_r * candidate.predicted_risk) - (self.w_c * candidate.action_cost)

        candidate.expected_utility = round(eu, 4)
        return candidate.expected_utility

    def select_best_decision(self, candidates: list[DecisionCandidate]) -> DecisionResult:
        """Rank candidates by expected utility and evaluate against rational inaction."""
        if not candidates:
            return DecisionResult(
                selected_candidate=None,
                decision_type=DecisionType.INACTION,
                expected_utility=0.0,
                rationale="No candidate actions available.",
            )

        # Score all candidates
        for c in candidates:
            self.evaluate_candidate_utility(c)

        # Sort descending by expected utility
        sorted_candidates = sorted(candidates, key=lambda c: c.expected_utility, reverse=True)
        top = sorted_candidates[0]

        # Check against rational inaction threshold
        if top.expected_utility < self.inaction_threshold or top.predicted_risk >= 0.80:
            return DecisionResult(
                selected_candidate=None,
                decision_type=DecisionType.INACTION,
                expected_utility=top.expected_utility,
                rejected_candidates=sorted_candidates,
                rationale=(
                    f"Rationally abstained from action. Best candidate '{top.description}' "
                    f"has insufficient net expected utility ({top.expected_utility:.2f} < threshold {self.inaction_threshold:.2f}) "
                    f"or excessive risk ({top.predicted_risk:.2f})."
                ),
            )

        # Map candidate kind to decision type
        dtype_map = {
            CandidateKind.GOAL_ACTION: DecisionType.ACTION,
            CandidateKind.EPISTEMIC_PROBE: DecisionType.PROBE,
            CandidateKind.CLARIFICATION: DecisionType.CLARIFICATION,
            CandidateKind.WAIT: DecisionType.WAIT,
        }
        dec_type = dtype_map.get(top.candidate_kind, DecisionType.ACTION)

        return DecisionResult(
            selected_candidate=top,
            decision_type=dec_type,
            expected_utility=top.expected_utility,
            rejected_candidates=sorted_candidates[1:],
            rationale=(
                f"Selected {dec_type.value.upper()} '{top.description}' with expected utility {top.expected_utility:.2f} "
                f"(Goal: {top.goal_progress:.2f}, InfoVal: {top.value_of_information:.2f}, "
                f"Risk: {top.predicted_risk:.2f}, Cost: {top.action_cost:.2f})."
            ),
        )
