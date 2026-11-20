"""Specialized Inductive Hypothesis Observers (Contact, Mechanism, Mutation)."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from hbllm.hcir.world.causal_discovery import BeliefTransitionType
from hbllm.hcir.world.hypothesis.models import (
    HypothesisTransitionEvent,
    HypothesisType,
    RelationalRuleHypothesis,
    TransitionObservationContext,
)

logger = logging.getLogger(__name__)


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

        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []
        for feat in context.get_prev_contact_and_adjacent_features():
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

        if not context.is_valid_curr_locus():
            return []

        contact_feat: int | None = context.get_contact_feature()
        av_feat = context.get_avatar_feature()
        if (
            contact_feat is None
            or contact_feat == av_feat
            or contact_feat == context.background_feature
        ):
            contact_feat = None
            # Only attribute an adjacent contact when performing an in-place
            # interaction (standing still); walking onto empty floor is not contact.
            if context.curr_avatar_pos == context.prev_avatar_pos:
                changed_adj, static_adj = context.get_curr_changed_and_static_adjacent(av_feat)
                if len(set(changed_adj)) == 1:
                    contact_feat = changed_adj[0]
                elif not changed_adj and len(set(static_adj)) == 1:
                    contact_feat = static_adj[0]
        if contact_feat is None or contact_feat == context.background_feature:
            return []

        old_diff_feats, new_diff_feats = context.get_diff_feature_sets()
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

        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        for delta in context.get_entity_deltas():
            # Skip loci within avatar footprint during movement
            if context.is_within_avatar_footprint(delta):
                continue

            old_f = int(delta.prev_feature)
            new_f = int(delta.curr_feature)
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

        bg = context.background_feature
        results: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        if context.paired_features is not None:
            adj_paired_feats = {
                f
                for f in context.get_curr_adjacent_features()
                if f != bg and f in context.paired_features
            }
        else:
            adj_paired_feats = {
                f
                for f in context.get_curr_adjacent_features()
                if f != bg and context.get_prev_feature_count(f) == 2
            }

        if not adj_paired_feats:
            return []

        for delta in context.get_entity_deltas():
            if delta.locus == context.curr_avatar_pos or delta.adjacent_to_curr_avatar:
                continue
            old_f = int(delta.prev_feature)
            new_f = int(delta.curr_feature)
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
