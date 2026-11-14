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
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionType,
    CausalPredicate,
    LogicOperator,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

logger = logging.getLogger(__name__)


class HypothesisType(StrEnum):
    """Categorization of relational causal hypotheses."""

    CONTACT_TRANSFORM = "CONTACT_TRANSFORM"
    ACTION_MUTATION = "ACTION_MUTATION"
    REMOTE_MECHANISM = "REMOTE_MECHANISM"
    HAZARD_CONTACT = "HAZARD_CONTACT"
    COMPOSITIONAL_PROGRAM = "COMPOSITIONAL_PROGRAM"


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
        if self.rule_type == HypothesisType.COMPOSITIONAL_PROGRAM:
            ast_obj = self.metadata.get("ast")
            if isinstance(ast_obj, SynthesizedRuleAST):
                return CausalPredicate(
                    variable=f"inductive_program:{self.rule_id}",
                    operator="==",
                    value=ast_obj.effect_operator,
                    relation="synthesized_transition_rule",
                    target_entity=ast_obj.precondition.describe(),
                    logic_op=ast_obj.precondition.logic_op,
                    children=list(ast_obj.precondition.children),
                    causal_status=self.status.value,
                    confidence=self.confidence,
                    metadata={"effect_params": dict(ast_obj.effect_params)},
                )
        if self.rule_type == HypothesisType.REMOTE_MECHANISM:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.premise_feature,
                relation="clears_barrier",
                target_entity=str(self.target_feature)
                if self.target_feature is not None
                else self.rule_id,
            )
        elif self.rule_type == HypothesisType.HAZARD_CONTACT:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.premise_feature,
                relation="lethal_hazard",
                target_entity=str(self.premise_feature)
                if self.premise_feature is not None
                else self.rule_id,
            )
        elif self.rule_type == HypothesisType.ACTION_MUTATION:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.action_id,
                relation="transforms_to",
                target_entity=str(self.result_feature)
                if self.result_feature is not None
                else self.rule_id,
            )
        else:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.premise_feature,
                relation="contact_transform",
                target_entity=str(self.result_feature)
                if self.result_feature is not None
                else self.rule_id,
            )


@dataclass(frozen=True)
class TransitionObservationContext:
    """Immutable snapshot of an embodied state transition."""

    prev_grid: np.ndarray
    action: Any
    curr_grid: np.ndarray
    prev_avatar_pos: tuple[int, int] | None = None
    curr_avatar_pos: tuple[int, int] | None = None
    avatar_h: int = 1
    avatar_w: int = 1
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
        if (
            context.is_dead
            or context.changed_count == 0
            or context.curr_avatar_pos is None
            or context.prev_avatar_pos is None
        ):
            return []

        H, W = context.prev_grid.shape
        cr, cc = context.curr_avatar_pos
        if not (0 <= cr < H and 0 <= cc < W):
            return []

        contact_feat: int | None = int(context.prev_grid[cr, cc])
        av_feat = (
            int(context.prev_grid[context.prev_avatar_pos[0], context.prev_avatar_pos[1]])
            if context.prev_avatar_pos is not None
            else None
        )
        if contact_feat == av_feat or contact_feat == context.background_feature:
            contact_feat = None
            # Only attribute an adjacent contact when performing an in-place
            # interaction (standing still); walking onto empty floor is not contact.
            if context.curr_avatar_pos == context.prev_avatar_pos:
                changed_adj: list[int] = []
                static_adj: list[int] = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        cand_f = int(context.prev_grid[nr, nc])
                        if cand_f != context.background_feature and cand_f != av_feat:
                            if bool(context.diff_mask[nr, nc]):
                                changed_adj.append(cand_f)
                            else:
                                static_adj.append(cand_f)
                if len(set(changed_adj)) == 1:
                    contact_feat = changed_adj[0]
                elif not changed_adj and len(set(static_adj)) == 1:
                    contact_feat = static_adj[0]
        if contact_feat is None or contact_feat == context.background_feature:
            return []

        diff_mask = context.diff_mask
        old_diff_feats = set(context.prev_grid[diff_mask].tolist())
        new_diff_feats = set(context.curr_grid[diff_mask].tolist())
        cleared_feats = old_diff_feats - new_diff_feats

        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []
        for cleared in cleared_feats:
            if (
                cleared != contact_feat
                and cleared != context.background_feature
                and cleared != av_feat
            ):
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
        if not (1 <= context.changed_count <= 16) or context.action is None:
            return []

        changed_coords = np.argwhere(context.diff_mask)
        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        radius_r = max(1, context.avatar_h // 2)
        radius_c = max(1, context.avatar_w // 2)
        for r, c in changed_coords:
            # Skip cells within avatar bounding areas during movement
            if context.prev_avatar_pos is not None:
                if (
                    abs(r - context.prev_avatar_pos[0]) <= radius_r
                    and abs(c - context.prev_avatar_pos[1]) <= radius_c
                ):
                    continue
            if context.curr_avatar_pos is not None:
                if (
                    abs(r - context.curr_avatar_pos[0]) <= radius_r
                    and abs(c - context.curr_avatar_pos[1]) <= radius_c
                ):
                    continue

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


class RelationalCueObserver(HypothesisObserver):
    """Observes relational contextual cue pairs where interacting adjacent to a paired feature clears a distal barrier."""

    def observe(
        self,
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
        min_support_to_confirm: int,
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        if (
            context.curr_avatar_pos is None
            or context.prev_avatar_pos != context.curr_avatar_pos
            or not (1 <= context.changed_count <= 8)
        ):
            return []

        H, W = context.prev_grid.shape
        ar, ac = context.curr_avatar_pos
        bg = context.background_feature
        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        adj_paired_feats: set[int] = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < H and 0 <= nc < W:
                f = int(context.prev_grid[nr, nc])
                if f != bg and int(np.count_nonzero(context.prev_grid == f)) == 2:
                    adj_paired_feats.add(f)

        if not adj_paired_feats:
            return []

        for mr, mc in np.argwhere(context.diff_mask):
            if abs(int(mr) - ar) + abs(int(mc) - ac) <= 1:
                continue
            old_f = int(context.prev_grid[mr, mc])
            new_f = int(context.curr_grid[mr, mc])
            if old_f != bg and new_f == bg:
                for pf in adj_paired_feats:
                    hid = f"relational_cue_{pf}_clears_{old_f}"
                    if hid in hypotheses:
                        hyp = hypotheses[hid]
                        event = hyp.record_support(min_support=min_support_to_confirm)
                        results.append((hyp, event))
                    else:
                        hyp = RelationalRuleHypothesis(
                            rule_id=hid,
                            rule_type=HypothesisType.REMOTE_MECHANISM,
                            premise_feature=pf,
                            target_feature=old_f,
                            action_id=context.action,
                            positive_support=1,
                            confidence=0.85,
                            description=f"Relational cue-paired feature {pf} clears distal barrier {old_f}",
                        )
                        event = HypothesisTransitionEvent(
                            event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                            rule_id=hid,
                            rule_type=hyp.rule_type,
                            timestamp=time.time(),
                            confidence_before=0.0,
                            confidence_after=hyp.confidence,
                            reason="Relational contextual cue pair remote clearance",
                        )
                        results.append((hyp, event))

        return results


@dataclass(frozen=True)
class SynthesizedRuleAST:
    """Open-ended symbolic transition rule T(s, a) -> delta_s represented as a program AST."""

    precondition: CausalPredicate
    effect_operator: str
    effect_params: dict[str, Any] = field(default_factory=dict)
    ast_depth: int = 1
    productions_used: tuple[str, ...] = ()

    @staticmethod
    def extract_transition_features(context: TransitionObservationContext) -> dict[str, Any]:
        """Extract observable transition features without environment-specific labels."""
        H, W = context.prev_grid.shape
        contact_feat: int | None = None
        if context.curr_avatar_pos is not None:
            cr, cc = context.curr_avatar_pos
            if 0 <= cr < H and 0 <= cc < W:
                contact_feat = int(context.prev_grid[cr, cc])
        adj_feats: list[int] = []
        if context.prev_avatar_pos is not None:
            pr, pc = context.prev_avatar_pos
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    f = int(context.prev_grid[nr, nc])
                    if f != context.background_feature and f not in adj_feats:
                        adj_feats.append(f)
        moved = (
            context.prev_avatar_pos is not None
            and context.curr_avatar_pos is not None
            and context.prev_avatar_pos != context.curr_avatar_pos
        )
        return {
            "action_id": context.action,
            "contact_feature": contact_feat,
            "adjacent_features": adj_feats,
            "changed_count": context.changed_count,
            "avatar_moved": moved,
            "is_dead": context.is_dead,
            "is_won": context.is_won,
        }

    def evaluate_precondition(self, context: TransitionObservationContext) -> bool:
        """Evaluate whether this AST precondition matches the current transition."""
        feats = self.extract_transition_features(context)
        return bool(self.precondition.evaluate(feats))

    def evaluate_effect(self, context: TransitionObservationContext) -> bool:
        """Evaluate whether the transition delta conforms to this AST effect operator."""
        diff_mask = context.diff_mask
        if self.effect_operator == "modulate_terminal":
            target = str(self.effect_params.get("terminal", ""))
            if target == "dead":
                return bool(context.is_dead)
            if target == "won":
                return bool(context.is_won)
            return False
        if not np.any(diff_mask):
            return False
        if self.effect_operator == "clear_distal_feature":
            tf = self.effect_params.get("target_feature")
            if tf is None:
                return False
            prev_n = int(np.count_nonzero(context.prev_grid == tf))
            curr_n = int(np.count_nonzero(context.curr_grid == tf))
            return curr_n < prev_n
        if self.effect_operator == "transform_feature":
            src = self.effect_params.get("src_feature")
            dst = self.effect_params.get("dst_feature")
            if src is None or dst is None:
                return False
            coords = np.argwhere(diff_mask)
            return any(
                int(context.prev_grid[r, c]) == src and int(context.curr_grid[r, c]) == dst
                for r, c in coords
            )
        if self.effect_operator == "translate_entity":
            feat = self.effect_params.get("feature")
            delta = self.effect_params.get("delta")
            if feat is None or not isinstance(delta, tuple) or len(delta) != 2:
                return False
            dr, dc = int(delta[0]), int(delta[1])
            H, W = context.curr_grid.shape
            for r, c in np.argwhere(diff_mask):
                if int(context.prev_grid[r, c]) == feat and int(context.curr_grid[r, c]) != feat:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < H and 0 <= nc < W and int(context.curr_grid[nr, nc]) == feat:
                        return True
            return False
        if self.effect_operator == "spawn_feature":
            dst = self.effect_params.get("dst_feature")
            if dst is None:
                return False
            return bool(np.any(context.curr_grid[diff_mask] == dst))
        return False


class RulePCFGPrior:
    """Learned Dirichlet-Laplace production prior over open-ended transition rule ASTs."""

    def __init__(self, complexity_penalty: float = 0.25, alpha: float = 1.0) -> None:
        self.complexity_penalty = complexity_penalty
        self.alpha = alpha
        self.production_counts: dict[str, float] = {
            "precondition:AND": 2.0,
            "precondition:OR": 1.0,
            "precondition:NOT": 1.0,
            "precondition:contact_feature": 3.0,
            "precondition:adjacent_features": 2.0,
            "precondition:action_id": 2.0,
            "precondition:avatar_moved": 1.5,
            "effect:transform_feature": 2.0,
            "effect:clear_distal_feature": 2.0,
            "effect:translate_entity": 2.0,
            "effect:spawn_feature": 1.5,
            "effect:modulate_terminal": 1.5,
        }

    def prior_probability(self, ast: SynthesizedRuleAST) -> float:
        """Compute MDL-regularized PCFG prior probability P(AST) in (0, 1]."""
        prods = ast.productions_used or (
            "precondition:contact_feature",
            f"effect:{ast.effect_operator}",
        )
        vocab_size = max(1, len(self.production_counts))
        total = sum(self.production_counts.values()) + self.alpha * vocab_size
        log_p = -self.complexity_penalty * max(0, ast.ast_depth - 1)
        for p in prods:
            count = self.production_counts.get(p, 0.5)
            prob = (count + self.alpha) / max(1e-9, total)
            log_p += math.log(max(1e-9, prob))
        return float(min(1.0, math.exp(max(-30.0, log_p))))

    def update_from_outcome(self, ast: SynthesizedRuleAST, confirmed: bool) -> None:
        """Update production weights via Bayesian reinforcement or falsification decay."""
        prods = ast.productions_used or (
            "precondition:contact_feature",
            f"effect:{ast.effect_operator}",
        )
        for p in prods:
            cur = self.production_counts.get(p, 1.0)
            if confirmed:
                self.production_counts[p] = min(1000.0, cur + 1.0)
            else:
                self.production_counts[p] = max(0.1, cur * 0.85)

    def sample_or_rank_proposals(
        self,
        candidates: Sequence[SynthesizedRuleAST],
    ) -> list[tuple[SynthesizedRuleAST, float]]:
        """Rank candidate AST programs by PCFG prior * structural simplicity."""
        scored = [(cand, self.prior_probability(cand)) for cand in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


class CompositionalProgramObserver(HypothesisObserver):
    """Synthesizes open-ended compositional AST transition rules ranked by a learned PCFG prior."""

    def __init__(self, pcfg_prior: RulePCFGPrior | None = None) -> None:
        self.pcfg_prior = pcfg_prior or RulePCFGPrior()

    def observe(
        self,
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
        min_support_to_confirm: int,
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        if context.changed_count == 0 and not context.is_dead and not context.is_won:
            return []

        feats = SynthesizedRuleAST.extract_transition_features(context)
        candidates: list[SynthesizedRuleAST] = []
        bg = context.background_feature
        contact_f = feats.get("contact_feature")
        adj_feats: list[int] = list(feats.get("adjacent_features", []))

        # Candidate ASTs for entity translation (e.g., pushing cargo along movement delta)
        if (
            context.changed_count > 0
            and context.changed_count <= 8
            and context.prev_avatar_pos is not None
            and context.curr_avatar_pos is not None
        ):
            dr = int(context.curr_avatar_pos[0] - context.prev_avatar_pos[0])
            dc = int(context.curr_avatar_pos[1] - context.prev_avatar_pos[1])
            if (dr != 0 or dc != 0) and (abs(dr) + abs(dc) <= 2):
                H, W = context.curr_grid.shape
                diff_coords = np.argwhere(context.diff_mask)[:16]
                for r, c in diff_coords:
                    ir, ic = int(r), int(c)
                    old_f = int(context.prev_grid[ir, ic])
                    new_f = int(context.curr_grid[ir, ic])
                    if old_f != bg and new_f != old_f and (ir, ic) != context.prev_avatar_pos:
                        nr, nc = ir + dr, ic + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and int(context.curr_grid[nr, nc]) == old_f
                            and int(context.prev_grid[nr, nc]) != old_f
                        ):
                            pre = CausalPredicate(
                                logic_op=LogicOperator.AND,
                                children=[
                                    CausalPredicate(
                                        variable="action_id",
                                        operator="==",
                                        value=context.action,
                                    ),
                                    CausalPredicate(
                                        variable="contact_feature",
                                        operator="==",
                                        value=old_f,
                                    ),
                                ],
                            )
                            candidates.append(
                                SynthesizedRuleAST(
                                    precondition=pre,
                                    effect_operator="translate_entity",
                                    effect_params={"feature": old_f, "delta": (dr, dc)},
                                    ast_depth=2,
                                    productions_used=(
                                        "precondition:AND",
                                        "precondition:action_id",
                                        "precondition:contact_feature",
                                        "effect:translate_entity",
                                    ),
                                )
                            )
                            break

        # Candidate AST for distal clearing / feature transform
        if (
            not context.is_dead
            and 1 <= context.changed_count <= 16
            and (contact_f is not None and contact_f != bg or adj_feats)
        ):
            premise_var = (
                "contact_feature"
                if (contact_f is not None and contact_f != bg)
                else "adjacent_features"
            )
            premise_val: Any = contact_f if premise_var == "contact_feature" else adj_feats[0]
            op_name = "==" if premise_var == "contact_feature" else "contains"
            diff_coords = np.argwhere(context.diff_mask)[:16]
            for r, c in diff_coords:
                ir, ic = int(r), int(c)
                if context.prev_avatar_pos == (ir, ic) or context.curr_avatar_pos == (ir, ic):
                    continue
                old_f = int(context.prev_grid[ir, ic])
                new_f = int(context.curr_grid[ir, ic])
                if old_f != bg and new_f == bg and old_f != premise_val:
                    pre = CausalPredicate(
                        logic_op=LogicOperator.AND,
                        children=[
                            CausalPredicate(
                                variable=premise_var,
                                operator=op_name,
                                value=premise_val,
                            ),
                            CausalPredicate(
                                variable="action_id",
                                operator="==",
                                value=context.action,
                            ),
                        ],
                    )
                    candidates.append(
                        SynthesizedRuleAST(
                            precondition=pre,
                            effect_operator="clear_distal_feature",
                            effect_params={"premise_feature": premise_val, "target_feature": old_f},
                            ast_depth=2,
                            productions_used=(
                                "precondition:AND",
                                f"precondition:{premise_var}",
                                "precondition:action_id",
                                "effect:clear_distal_feature",
                            ),
                        )
                    )
                    break
                if old_f != new_f:
                    pre = CausalPredicate(
                        variable="action_id",
                        operator="==",
                        value=context.action,
                    )
                    candidates.append(
                        SynthesizedRuleAST(
                            precondition=pre,
                            effect_operator="transform_feature",
                            effect_params={"src_feature": old_f, "dst_feature": new_f},
                            ast_depth=1,
                            productions_used=("precondition:action_id", "effect:transform_feature"),
                        )
                    )
                    break

        if not candidates:
            return []

        ranked = self.pcfg_prior.sample_or_rank_proposals(candidates)
        best_ast, prior_score = ranked[0]
        eff_params = best_ast.effect_params
        prem_f = (
            eff_params.get("premise_feature")
            or eff_params.get("feature")
            or eff_params.get("src_feature")
        )
        tgt_f = eff_params.get("target_feature")
        res_f = eff_params.get("dst_feature")
        hid = f"prog_{best_ast.effect_operator}_{prem_f}_{tgt_f or res_f}_{context.action}"
        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []
        if hid in hypotheses:
            hyp = hypotheses[hid]
            hyp.metadata["prior_score"] = prior_score
            event = hyp.record_support(min_support=min_support_to_confirm)
            if hyp.status == HypothesisStatus.CONFIRMED:
                self.pcfg_prior.update_from_outcome(best_ast, confirmed=True)
            results.append((hyp, event))
        else:
            hyp = RelationalRuleHypothesis(
                rule_id=hid,
                rule_type=HypothesisType.COMPOSITIONAL_PROGRAM,
                premise_feature=int(prem_f) if isinstance(prem_f, int) else None,
                target_feature=int(tgt_f) if isinstance(tgt_f, int) else None,
                result_feature=int(res_f) if isinstance(res_f, int) else None,
                action_id=context.action,
                positive_support=1,
                confidence=round(min(0.85, 0.45 + 0.4 * prior_score), 4),
                description=f"Synthesized AST: IF {best_ast.precondition.describe()} THEN {best_ast.effect_operator}({eff_params})",
                metadata={"ast": best_ast, "prior_score": prior_score},
            )
            event = HypothesisTransitionEvent(
                event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                rule_id=hid,
                rule_type=hyp.rule_type,
                timestamp=time.time(),
                confidence_before=0.0,
                confidence_after=hyp.confidence,
                reason=f"Synthesized open-ended AST under PCFG prior={prior_score:.4f}",
            )
            results.append((hyp, event))
        return results


class PopperianFalsifier:
    """Evaluates active hypotheses against counterexamples to immediately falsify refutations."""

    @staticmethod
    def falsify(
        context: TransitionObservationContext,
        hypotheses: dict[str, RelationalRuleHypothesis],
        pcfg_prior: RulePCFGPrior | None = None,
    ) -> list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]]:
        if context.curr_avatar_pos is None or context.is_dead:
            return []

        H, W = context.prev_grid.shape
        cr, cc = context.curr_avatar_pos
        if not (0 <= cr < H and 0 <= cc < W):
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
            elif hyp.rule_type == HypothesisType.COMPOSITIONAL_PROGRAM:
                ast_obj = hyp.metadata.get("ast")
                if isinstance(ast_obj, SynthesizedRuleAST):
                    if ast_obj.evaluate_precondition(context) and not ast_obj.evaluate_effect(
                        context
                    ):
                        event = hyp.record_counterexample()
                        if pcfg_prior is not None:
                            pcfg_prior.update_from_outcome(ast_obj, confirmed=False)
                        falsified.append((hyp, event))

        return falsified


class InductiveHypothesisEngine:
    """Prefrontal Cortex Inductive Logic and Relational Rule Discovery Engine."""

    def __init__(
        self,
        min_support_to_confirm: int = 2,
        observers: list[HypothesisObserver] | None = None,
        pcfg_prior: RulePCFGPrior | None = None,
    ) -> None:
        self.min_support_to_confirm = min_support_to_confirm
        self.pcfg_prior: RulePCFGPrior = pcfg_prior or RulePCFGPrior()
        self.hypotheses: dict[str, RelationalRuleHypothesis] = {}
        self.event_log: list[HypothesisTransitionEvent] = []
        self.observers: list[HypothesisObserver] = (
            observers
            if observers is not None
            else [
                HazardContactObserver(),
                RemoteMechanismObserver(),
                ActionMutationObserver(),
                RelationalCueObserver(),
                CompositionalProgramObserver(self.pcfg_prior),
            ]
        )
        self.falsifier: PopperianFalsifier = PopperianFalsifier()

    def propose_hypothesis(
        self,
        rule_type: HypothesisType,
        premise_feature: int | None = None,
        target_feature: int | None = None,
        result_feature: int | None = None,
        action_id: Any = None,
        consequence: str = "",
    ) -> RelationalRuleHypothesis:
        """Explicitly propose an inductive relational rule hypothesis."""
        rule_id = f"hyp_{rule_type.value}_{premise_feature}_{target_feature}_{uuid.uuid4().hex[:6]}"
        hyp = RelationalRuleHypothesis(
            rule_id=rule_id,
            rule_type=rule_type,
            premise_feature=premise_feature,
            target_feature=target_feature,
            result_feature=result_feature,
            action_id=action_id,
            description=consequence,
        )
        self.hypotheses[rule_id] = hyp
        return hyp

    def observe_transition(
        self,
        prev_grid: np.ndarray,
        action: Any,
        curr_grid: np.ndarray,
        prev_avatar_pos: tuple[int, int] | None = None,
        curr_avatar_pos: tuple[int, int] | None = None,
        avatar_h: int = 1,
        avatar_w: int = 1,
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
            avatar_h=avatar_h,
            avatar_w=avatar_w,
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
            pcfg_prior=self.pcfg_prior,
        )
        for hyp, event in falsified_results:
            self.event_log.append(event)
            updated_rules.append(hyp)

        # 3. Solved solution induction: rules active during winning trajectories are confirmed
        if context.is_won:
            for h in self.hypotheses.values():
                if h.positive_support >= 1 and h.counterexamples == 0:
                    if h.status != HypothesisStatus.CONFIRMED:
                        h.status = HypothesisStatus.CONFIRMED
                        ast_obj = h.metadata.get("ast")
                        if isinstance(ast_obj, SynthesizedRuleAST):
                            self.pcfg_prior.update_from_outcome(ast_obj, confirmed=True)

        return updated_rules

    def get_confirmed_rules(self) -> list[RelationalRuleHypothesis]:
        """Return all empirically confirmed relational rules."""
        return [
            h
            for h in self.hypotheses.values()
            if (
                h.status == HypothesisStatus.CONFIRMED
                or (
                    h.positive_support >= 1
                    and h.counterexamples == 0
                    and h.rule_type == HypothesisType.REMOTE_MECHANISM
                )
            )
            and h.is_valid
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
        av_feats = getattr(theory, "avatar_features", set())
        confirmed = sorted(
            self.get_confirmed_rules(),
            key=lambda h: (
                h.status == HypothesisStatus.CONFIRMED,
                h.positive_support,
                h.confidence,
            ),
        )
        for hyp in confirmed:
            pred = hyp.to_causal_predicate()
            theory.record_causal_predicate(pred)
            if hyp.rule_type == HypothesisType.REMOTE_MECHANISM:
                if (
                    hyp.target_feature is not None
                    and hyp.premise_feature is not None
                    and hyp.premise_feature != hyp.target_feature
                    and hyp.premise_feature not in av_feats
                    and hyp.target_feature not in av_feats
                    and self.get_hazard_rule(hyp.premise_feature) is None
                ):
                    is_rel_cue = hyp.rule_id.startswith("relational_cue_")
                    theory.induce_conditional_affordance(
                        feature=hyp.target_feature,
                        condition=CausalPredicate(
                            variable="relational_cue" if is_rel_cue else "inventory",
                            operator="==" if is_rel_cue else "contains",
                            value=hyp.premise_feature,
                            relation="cue_actuator_resonance" if is_rel_cue else "",
                        ),
                        evidence=(
                            f"relational_cue:{hyp.premise_feature}_clears_{hyp.target_feature}"
                            if is_rel_cue
                            else f"remote_mech:{hyp.premise_feature}_clears_{hyp.target_feature}"
                        ),
                        prerequisite_feature=hyp.premise_feature,
                    )
                    theory.barrier_predicates.append(
                        CausalPredicate(
                            variable="feature_id",
                            operator="==",
                            value=hyp.target_feature,
                            relation="remote_barrier",
                            target_entity=str(hyp.premise_feature),
                        )
                    )
                    theory.induce_barrier(hyp.target_feature, evidence="remote_mechanism")
                if (
                    hyp.premise_feature is not None
                    and hyp.premise_feature not in av_feats
                    and hyp.premise_feature not in getattr(theory, "walkable_features", set())
                    and self.get_hazard_rule(hyp.premise_feature) is None
                ):
                    theory.induce_actuator(hyp.premise_feature, evidence="remote_mechanism")
            elif hyp.rule_type == HypothesisType.HAZARD_CONTACT:
                theory.barrier_predicates.append(pred)
                if hyp.premise_feature is not None:
                    theory.induce_barrier(hyp.premise_feature, evidence="hazard_contact")
            elif hyp.rule_type == HypothesisType.ACTION_MUTATION:
                # Direct action mutations represent in-place terrain/entity transformations,
                # NOT distal actuators (switches/buttons). The causal predicate already records this.
                pass
            elif hyp.rule_type == HypothesisType.COMPOSITIONAL_PROGRAM:
                ast_obj = hyp.metadata.get("ast")
                if isinstance(ast_obj, SynthesizedRuleAST):
                    if (
                        ast_obj.effect_operator == "clear_distal_feature"
                        and hyp.premise_feature is not None
                        and hyp.target_feature is not None
                        and hyp.premise_feature != hyp.target_feature
                        and hyp.premise_feature not in av_feats
                        and hyp.target_feature not in av_feats
                        and self.get_hazard_rule(hyp.premise_feature) is None
                    ):
                        theory.induce_conditional_affordance(
                            feature=hyp.target_feature,
                            condition=ast_obj.precondition,
                            evidence=f"program:{hyp.rule_id}",
                            prerequisite_feature=hyp.premise_feature,
                        )
                        if hyp.premise_feature not in getattr(theory, "walkable_features", set()):
                            theory.induce_actuator(hyp.premise_feature, evidence="program")
                    elif (
                        ast_obj.effect_operator == "translate_entity"
                        and hyp.premise_feature is not None
                        and hyp.premise_feature not in av_feats
                        and hasattr(theory, "induce_cargo")
                    ):
                        theory.induce_cargo(hyp.premise_feature, evidence="program_translate")

    def retract_refuted_rules(self, theory: HCIRSymbolicWorldTheory) -> list[str]:
        """Remove refuted hypotheses from active symbolic world theory predicates."""
        retracted_ids = []
        for hyp in self.hypotheses.values():
            if hyp.status == HypothesisStatus.REFUTED:
                retracted_ids.append(hyp.rule_id)
                if hyp.rule_type == HypothesisType.HAZARD_CONTACT:
                    if hyp.premise_feature is not None:
                        theory.retract_barrier(hyp.premise_feature, reason="refuted_rule")
                elif hyp.rule_type == HypothesisType.REMOTE_MECHANISM:
                    if hyp.target_feature is not None:
                        theory.retract_barrier(hyp.target_feature, reason="refuted_rule")
                        theory.retract_conditional_affordance(
                            hyp.target_feature, reason="refuted_rule"
                        )
                    if hyp.premise_feature is not None:
                        theory.retract_actuator(hyp.premise_feature, reason="refuted_rule")
                elif hyp.rule_type in (
                    HypothesisType.ACTION_MUTATION,
                    HypothesisType.COMPOSITIONAL_PROGRAM,
                ):
                    if (
                        hyp.rule_type == HypothesisType.COMPOSITIONAL_PROGRAM
                        and hyp.target_feature is not None
                    ):
                        theory.retract_conditional_affordance(
                            hyp.target_feature, reason="refuted_rule"
                        )
                else:
                    if hyp.target_feature is not None:
                        theory.retract_barrier(hyp.target_feature, reason="refuted_rule")
                    if hyp.premise_feature is not None:
                        theory.retract_actuator(hyp.premise_feature, reason="refuted_rule")
                theory.domain_predicates = [
                    p
                    for p in theory.domain_predicates
                    if hyp.rule_id not in p.variable and p.target_entity != hyp.rule_id
                ]
                theory.barrier_predicates = [
                    p
                    for p in theory.barrier_predicates
                    if p.target_entity != str(hyp.premise_feature) and hyp.rule_id not in p.variable
                ]
        return retracted_ids

    def on_boundary(self, boundary) -> None:
        """Episode lifecycle boundary handler (WP0.5). Delegates to the
        deprecated reset_episode for backward-compatible semantics."""
        self.reset_episode()

    @property
    def boundary_retention(self):
        """Declared retention policy: default level-local semantics."""
        from hbllm.hcir.world.percept import RetentionPolicy

        return RetentionPolicy.LEVEL_LOCAL

    def reset_episode(self, retain_confirmed: bool = True) -> None:
        """Reset episodic hypotheses between levels or attempts."""
        if retain_confirmed:
            self.hypotheses = {
                k: h
                for k, h in self.hypotheses.items()
                if (h.status == HypothesisStatus.CONFIRMED or h.positive_support >= 1)
                and h.is_valid
            }
        else:
            self.hypotheses.clear()
