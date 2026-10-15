from __future__ import annotations

"""Prefrontal Cortex Inductive Logic & Relational Hypothesis Testing Faculty.

Biologically modeled on mammalian prefrontal cortex inductive hypothesis generation,
Popperian counterexample refutation, and causal rule abstraction:
1. Relational Rule Generation:
   - Observes state transitions and abstracts candidate symbolic rules:
     - Contact transformations (e.g. avatar stepping on switch clears barrier).
     - Action mutations (e.g. effector action toggles feature X -> Y).
     - Remote mechanism bindings (key of feature K unlocks door D).
2. Popperian Counterexample Refutation:
   - Tracks positive empirical support vs falsifications.
   - Immediately refutes hypotheses upon counterexample observation.
3. Invariant Retention:
   - Retains verified causal invariants across retries and episodes.
4. Frontopolar Subgoal Synergy:
   - Informs subgoal planning of confirmed keys, switches, and triggers.
5. Neuro-Symbolic Predicate Grounding:
   - Compiles empirical rules into formal first-order CausalPredicates.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.causal_discovery import BeliefTransitionType, CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

logger = logging.getLogger(__name__)


class HypothesisType(StrEnum):
    """Categorization of relational causal hypotheses."""

    CONTACT_TRANSFORM = "CONTACT_TRANSFORM"
    ACTION_MUTATION = "ACTION_MUTATION"
    REMOTE_MECHANISM = "REMOTE_MECHANISM"
    HAZARD_CONTACT = "HAZARD_CONTACT"


class HypothesisStatus(StrEnum):
    """Lifecycle state of an inductive hypothesis."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"


@dataclass(frozen=True)
class HypothesisTransitionEvent:
    """Immutable audit event recording hypothesis lifecycle state changes."""

    event_type: BeliefTransitionType
    rule_id: str
    rule_type: HypothesisType
    timestamp: float
    confidence_before: float
    confidence_after: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationalRuleHypothesis:
    """Symbolic relational causal rule hypothesis."""

    rule_id: str
    rule_type: HypothesisType
    premise_feature: int | None = None
    target_feature: int | None = None
    result_feature: int | None = None
    action_id: Any = None
    positive_support: int = 1
    counterexamples: int = 0
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.TENTATIVE
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_support(
        self, weight: float = 1.0, min_support: int = 2
    ) -> HypothesisTransitionEvent:
        """Record positive empirical evidence supporting this rule."""
        conf_before = self.confidence
        self.positive_support += int(weight)
        self.confidence = min(0.99, self.confidence + 0.15 * weight)
        event_type = BeliefTransitionType.HYPOTHESIS_TESTED
        reason = f"Empirical support observed (+{weight:.2f})"

        if self.positive_support >= min_support and self.counterexamples == 0:
            if self.status != HypothesisStatus.CONFIRMED:
                self.status = HypothesisStatus.CONFIRMED
                event_type = BeliefTransitionType.HYPOTHESIS_CONFIRMED
                reason = f"Confirmed with {self.positive_support} supporting observations"

        return HypothesisTransitionEvent(
            event_type=event_type,
            rule_id=self.rule_id,
            rule_type=self.rule_type,
            timestamp=time.time(),
            confidence_before=conf_before,
            confidence_after=self.confidence,
            reason=reason,
        )

    def record_counterexample(self, weight: float = 1.0) -> HypothesisTransitionEvent:
        """Record a falsifying counterexample (Popperian refutation)."""
        conf_before = self.confidence
        self.counterexamples += int(weight)
        self.confidence = max(0.01, self.confidence - 0.4 * weight)
        event_type = BeliefTransitionType.HYPOTHESIS_TESTED
        reason = f"Counterexample observed (+{weight:.2f})"

        if self.counterexamples >= 1:
            if self.status != HypothesisStatus.REFUTED:
                self.status = HypothesisStatus.REFUTED
                event_type = BeliefTransitionType.HYPOTHESIS_FALSIFIED
                reason = f"Falsified by counterexample (count={self.counterexamples})"

        return HypothesisTransitionEvent(
            event_type=event_type,
            rule_id=self.rule_id,
            rule_type=self.rule_type,
            timestamp=time.time(),
            confidence_before=conf_before,
            confidence_after=self.confidence,
            reason=reason,
        )

    @property
    def is_valid(self) -> bool:
        """True if the rule is confirmed or tentative without refutations."""
        return self.status != HypothesisStatus.REFUTED and self.counterexamples == 0

    def to_causal_predicate(self) -> CausalPredicate:
        """Compile this hypothesis into a formal first-order CausalPredicate."""
        if self.rule_type == HypothesisType.REMOTE_MECHANISM:
            return CausalPredicate(
                variable="contact_feature",
                operator="==",
                value=self.premise_feature,
                relation="clears_barrier",
                target_entity=str(self.target_feature) if self.target_feature is not None else "",
            )
        elif self.rule_type == HypothesisType.HAZARD_CONTACT:
            return CausalPredicate(
                variable="feature_id",
                operator="==",
                value=self.premise_feature,
                relation="lethal_hazard",
            )
        elif self.rule_type == HypothesisType.ACTION_MUTATION:
            return CausalPredicate(
                variable="action_id",
                operator="==",
                value=self.action_id,
                relation="transforms_to",
                target_entity=str(self.result_feature) if self.result_feature is not None else "",
            )
        else:
            return CausalPredicate(
                variable="feature_id",
                operator="==",
                value=self.premise_feature,
                relation="contact_transform",
                target_entity=str(self.result_feature) if self.result_feature is not None else "",
            )


@dataclass(frozen=True)
class TransitionObservationContext:
    """Immutable snapshot of an embodied state transition."""

    prev_grid: np.ndarray
    action: Any
    curr_grid: np.ndarray
    prev_avatar_pos: tuple[int, int] | None = None
    curr_avatar_pos: tuple[int, int] | None = None
    is_dead: bool = False
    is_won: bool = False
    background_feature: int = 0

    @property
    def diff_mask(self) -> np.ndarray:
        return self.prev_grid != self.curr_grid

    @property
    def changed_count(self) -> int:
        return int(np.count_nonzero(self.diff_mask))


class HypothesisObserver(ABC):
    """Abstract observer extracting relational hypotheses from state transitions."""

    @abstractmethod
    def observe(
        self,
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
        min_support_to_confirm: int,
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        """Observe state transition, formulate or support candidate hypotheses."""
        ...


class HazardContactObserver(HypothesisObserver):
    """Observes agent mortality and correlates it with adjacent contact features."""

    def observe(
        self,
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
        min_support_to_confirm: int,
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        if not context.is_dead or context.prev_avatar_pos is None:
            return []

        H, W = context.curr_grid.shape
        pr, pc = context.prev_avatar_pos
        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < H and 0 <= nc < W:
                feat = int(context.prev_grid[nr, nc])
                if feat != context.background_feature:
                    hid = f"hazard_contact_{feat}"
                    if hid in hypotheses:
                        hyp = hypotheses[hid]
                        event = hyp.record_support(min_support=min_support_to_confirm)
                        results.append((hyp, event))
                    else:
                        hyp = RelationalRuleHypothesis(
                            rule_id=hid,
                            rule_type=HypothesisType.HAZARD_CONTACT,
                            premise_feature=feat,
                            positive_support=1,
                            confidence=0.7,
                            description=f"Feature {feat} contact causes death",
                        )
                        event = HypothesisTransitionEvent(
                            event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                            rule_id=hid,
                            rule_type=hyp.rule_type,
                            timestamp=time.time(),
                            confidence_before=0.0,
                            confidence_after=hyp.confidence,
                            reason="Mortality correlation near feature",
                        )
                        results.append((hyp, event))

        return results


class RemoteMechanismObserver(HypothesisObserver):
    """Observes stepping on a trigger feature correlating with distal barrier clearing."""

    def observe(
        self,
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
        min_support_to_confirm: int,
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        if context.curr_avatar_pos is None or context.prev_avatar_pos is None:
            return []

        cr, cc = context.curr_avatar_pos
        H, W = context.prev_grid.shape
        if not (0 <= cr < H and 0 <= cc < W):
            return []
        contact_feat = int(context.prev_grid[cr, cc])
        if contact_feat == context.background_feature:
            return []

        diff_mask = context.diff_mask
        old_diff_feats = set(context.prev_grid[diff_mask].tolist())
        new_diff_feats = set(context.curr_grid[diff_mask].tolist())
        cleared_feats = old_diff_feats - new_diff_feats

        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []
        for cleared in cleared_feats:
            if cleared != contact_feat and cleared != context.background_feature:
                hid = f"remote_mech_{contact_feat}_clears_{cleared}"
                if hid in hypotheses:
                    hyp = hypotheses[hid]
                    event = hyp.record_support(min_support=min_support_to_confirm)
                    results.append((hyp, event))
                else:
                    hyp = RelationalRuleHypothesis(
                        rule_id=hid,
                        rule_type=HypothesisType.REMOTE_MECHANISM,
                        premise_feature=contact_feat,
                        target_feature=cleared,
                        positive_support=1,
                        confidence=0.6,
                        description=f"Contact with feature {contact_feat} clears barrier {cleared}",
                    )
                    event = HypothesisTransitionEvent(
                        event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                        rule_id=hid,
                        rule_type=hyp.rule_type,
                        timestamp=time.time(),
                        confidence_before=0.0,
                        confidence_after=hyp.confidence,
                        reason="Distal feature clearance upon trigger contact",
                    )
                    results.append((hyp, event))

        return results


class ActionMutationObserver(HypothesisObserver):
    """Observes direct action transformations modifying features in-place."""

    def observe(
        self,
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
        min_support_to_confirm: int,
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        if context.changed_count not in (1, 2) or context.action is None:
            return []

        changed_coords = np.argwhere(context.diff_mask)
        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        for r, c in changed_coords:
            old_f = int(context.prev_grid[r, c])
            new_f = int(context.curr_grid[r, c])
            if old_f != new_f:
                hid = f"action_{context.action}_mutates_{old_f}_to_{new_f}"
                if hid in hypotheses:
                    hyp = hypotheses[hid]
                    event = hyp.record_support(min_support=min_support_to_confirm)
                    results.append((hyp, event))
                else:
                    hyp = RelationalRuleHypothesis(
                        rule_id=hid,
                        rule_type=HypothesisType.ACTION_MUTATION,
                        premise_feature=old_f,
                        result_feature=new_f,
                        action_id=context.action,
                        positive_support=1,
                        confidence=0.55,
                        description=f"Action {context.action} transforms feature {old_f} into {new_f}",
                    )
                    event = HypothesisTransitionEvent(
                        event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                        rule_id=hid,
                        rule_type=hyp.rule_type,
                        timestamp=time.time(),
                        confidence_before=0.0,
                        confidence_after=hyp.confidence,
                        reason="In-place feature transformation by action",
                    )
                    results.append((hyp, event))

        return results


class PopperianFalsifier:
    """Evaluates active hypotheses against counterexamples to immediately falsify refutations."""

    @staticmethod
    def falsify(
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        if context.curr_avatar_pos is None or context.is_dead:
            return []

        cr, cc = context.curr_avatar_pos
        if not (0 <= cr < context.prev_grid.shape[0] and 0 <= cc < context.prev_grid.shape[1]):
            return []
        c_feat = int(context.prev_grid[cr, cc])
        falsified: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        for hyp in list(hypotheses.values()):
            if hyp.rule_type == HypothesisType.REMOTE_MECHANISM:
                if hyp.premise_feature == c_feat:
                    target_f = hyp.target_feature
                    if target_f is not None and target_f in context.curr_grid:
                        prev_target_count = int(np.count_nonzero(context.prev_grid == target_f))
                        curr_target_count = int(np.count_nonzero(context.curr_grid == target_f))
                        if curr_target_count >= prev_target_count:
                            event = hyp.record_counterexample()
                            falsified.append((hyp, event))

        return falsified


class InductiveHypothesisEngine:
    """Prefrontal Cortex Inductive Logic and Relational Rule Discovery Engine."""

    def __init__(
        self,
        min_support_to_confirm: int = 2,
        observers: list[HypothesisObserver] | None = None,
    ) -> None:
        self.min_support_to_confirm = min_support_to_confirm
        self.hypotheses: dict[str, RelationalRuleHypothesis] = {}
        self.event_log: list[HypothesisTransitionEvent] = []
        self.observers: list[HypothesisObserver] = (
            observers
            if observers is not None
            else [
                HazardContactObserver(),
                RemoteMechanismObserver(),
                ActionMutationObserver(),
            ]
        )
        self.falsifier: PopperianFalsifier = PopperianFalsifier()

    def observe_transition(
        self,
        prev_grid: np.ndarray,
        action: Any,
        curr_grid: np.ndarray,
        prev_avatar_pos: tuple[int, int] | None = None,
        curr_avatar_pos: tuple[int, int] | None = None,
        is_dead: bool = False,
        is_won: bool = False,
        background_feature: int = 0,
    ) -> list[RelationalRuleHypothesis]:
        """Observe state transition, formulate new hypotheses, and test active ones.

        Returns:
            List of hypotheses updated or created during this transition.
        """
        if prev_grid.shape != curr_grid.shape:
            return []

        context = TransitionObservationContext(
            prev_grid=prev_grid,
            action=action,
            curr_grid=curr_grid,
            prev_avatar_pos=prev_avatar_pos,
            curr_avatar_pos=curr_avatar_pos,
            is_dead=is_dead,
            is_won=is_won,
            background_feature=background_feature,
        )

        updated_rules: list[RelationalRuleHypothesis] = []

        # 1. Dispatch inductive observers for rule discovery and positive support
        for observer in self.observers:
            obs_results = observer.observe(
                context=context,
                hypotheses=self.hypotheses,
                min_support_to_confirm=self.min_support_to_confirm,
            )
            for hyp, event in obs_results:
                self.hypotheses[hyp.rule_id] = hyp
                self.event_log.append(event)
                updated_rules.append(hyp)

        # 2. Popperian Counterexample Check on active hypotheses
        falsified_results = self.falsifier.falsify(
            context=context,
            hypotheses=self.hypotheses,
        )
        for hyp, event in falsified_results:
            self.event_log.append(event)
            updated_rules.append(hyp)

        return updated_rules

    def get_confirmed_rules(self) -> list[RelationalRuleHypothesis]:
        """Return all empirically confirmed relational rules."""
        return [
            h
            for h in self.hypotheses.values()
            if h.status == HypothesisStatus.CONFIRMED and h.is_valid
        ]

    def get_rule_for_barrier(self, barrier_feat: int) -> RelationalRuleHypothesis | None:
        """Find confirmed or highest-confidence trigger rule for opening barrier."""
        candidates = [
            h for h in self.hypotheses.values() if h.target_feature == barrier_feat and h.is_valid
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda h: (h.status == HypothesisStatus.CONFIRMED, h.confidence), reverse=True
        )
        return candidates[0]

    def get_hazard_rule(self, feature_id: int) -> RelationalRuleHypothesis | None:
        """Find confirmed or tentative hazard rule for a lethal feature."""
        hid = f"hazard_contact_{feature_id}"
        hyp = self.hypotheses.get(hid)
        if hyp is not None and hyp.is_valid:
            return hyp
        return None

    def export_to_world_theory(self, theory: HCIRSymbolicWorldTheory) -> None:
        """Ground confirmed inductive rules into first-order HCIR CausalPredicates."""
        for hyp in self.get_confirmed_rules():
            pred = hyp.to_causal_predicate()
            if hyp.rule_type == HypothesisType.REMOTE_MECHANISM:
                if hyp.target_feature is not None:
                    theory.barrier_features.add(hyp.target_feature)
                    theory.barrier_predicates.append(
                        CausalPredicate(
                            variable="feature_id",
                            operator="==",
                            value=hyp.target_feature,
                            relation="remote_barrier",
                            target_entity=str(hyp.premise_feature),
                        )
                    )
            elif hyp.rule_type == HypothesisType.HAZARD_CONTACT:
                theory.barrier_predicates.append(pred)

    def reset_episode(self, retain_confirmed: bool = True) -> None:
        """Reset episodic hypotheses between levels or attempts."""
        if retain_confirmed:
            self.hypotheses = {
                k: h
                for k, h in self.hypotheses.items()
                if h.status == HypothesisStatus.CONFIRMED and h.is_valid
            }
        else:
            self.hypotheses.clear()
