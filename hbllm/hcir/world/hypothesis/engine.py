"""InductiveHypothesisEngine: Executive Prefrontal Hypothesis Testing Faculty."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.hypothesis.models import (
    EntityStateDelta,
    HypothesisStatus,
    HypothesisTransitionEvent,
    HypothesisType,
    RelationalRuleHypothesis,
    TransitionObservationContext,
)
from hbllm.hcir.world.hypothesis.observers import (
    ActionMutationObserver,
    HazardContactObserver,
    HypothesisObserver,
    RelationalCueObserver,
    RemoteMechanismObserver,
)
from hbllm.hcir.world.hypothesis.pcfg import (
    CompositionalProgramObserver,
    PopperianFalsifier,
    RulePCFGPrior,
    SynthesizedRuleAST,
)
from hbllm.hcir.world.percept import PlaceLocus

if TYPE_CHECKING:
    from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

logger = logging.getLogger(__name__)


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
        prev_grid: np.ndarray | None = None,
        action: Any = None,
        curr_grid: np.ndarray | None = None,
        prev_avatar_pos: PlaceLocus | None = None,
        curr_avatar_pos: PlaceLocus | None = None,
        avatar_h: int = 1,
        avatar_w: int = 1,
        is_dead: bool = False,
        is_won: bool = False,
        background_feature: int = 0,
        *,
        entity_deltas: tuple[EntityStateDelta, ...] | list[EntityStateDelta] | None = None,
        contact_feature: int | None = None,
        avatar_feature: int | None = None,
        prev_adjacent_features: tuple[int, ...] | list[int] | None = None,
        curr_adjacent_features: tuple[int, ...] | list[int] | None = None,
        curr_static_adjacent_features: tuple[int, ...] | list[int] | None = None,
        paired_features: frozenset[int] | set[int] | None = None,
        prev_feature_counts: dict[int, int] | None = None,
        curr_feature_counts: dict[int, int] | None = None,
        context: TransitionObservationContext | None = None,
    ) -> list[RelationalRuleHypothesis]:
        """Observe state transition, formulate new hypotheses, and test active ones.

        Returns:
            List of hypotheses updated or created during this transition.
        """
        if context is None:
            if (
                prev_grid is not None
                and curr_grid is not None
                and prev_grid.shape != curr_grid.shape
            ):
                return []
            if prev_grid is None and entity_deltas is None and not is_dead and not is_won:
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
                entity_deltas=tuple(entity_deltas) if entity_deltas is not None else None,
                contact_feature=contact_feature,
                avatar_feature=avatar_feature,
                prev_adjacent_features=(
                    tuple(prev_adjacent_features) if prev_adjacent_features is not None else None
                ),
                curr_adjacent_features=(
                    tuple(curr_adjacent_features) if curr_adjacent_features is not None else None
                ),
                curr_static_adjacent_features=(
                    tuple(curr_static_adjacent_features)
                    if curr_static_adjacent_features is not None
                    else None
                ),
                paired_features=frozenset(paired_features) if paired_features is not None else None,
                prev_feature_counts=prev_feature_counts,
                curr_feature_counts=curr_feature_counts,
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
                        existing_aff = theory.get_conditional_affordance(hyp.target_feature)
                        if (
                            existing_aff is not None
                            and existing_aff.prerequisite_feature == hyp.premise_feature
                        ):
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
                        existing_aff = theory.get_conditional_affordance(hyp.target_feature)
                        if (
                            existing_aff is not None
                            and existing_aff.prerequisite_feature == hyp.premise_feature
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
        """Declared retention policy: default episode-local semantics."""
        from hbllm.hcir.world.percept import RetentionPolicy

        return RetentionPolicy.EPISODE_LOCAL

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
