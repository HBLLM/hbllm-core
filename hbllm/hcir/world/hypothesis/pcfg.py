"""Synthesized PCFG Rules, Compositional Program Observer, & Popperian Falsifier."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.causal_discovery import BeliefTransitionType, CausalPredicate, LogicOperator
from hbllm.hcir.world.hypothesis.models import (
    HypothesisStatus,
    HypothesisTransitionEvent,
    HypothesisType,
    RelationalRuleHypothesis,
    TransitionObservationContext,
    _shift_numeric_locus,
)
from hbllm.hcir.world.hypothesis.observers import HypothesisObserver

logger = logging.getLogger(__name__)


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
        contact_feat = context.get_contact_feature()
        adj_feats = list(context.get_prev_adjacent_features())
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
        if self.effect_operator == "modulate_terminal":
            target = str(self.effect_params.get("terminal", ""))
            if target == "dead":
                return bool(context.is_dead)
            if target == "won":
                return bool(context.is_won)
            return False
        if context.changed_count == 0:
            return False
        if self.effect_operator == "clear_distal_feature":
            tf = self.effect_params.get("target_feature")
            if tf is None:
                return False
            tf_int = int(tf)
            if context.has_feature_count_summary():
                return context.get_curr_feature_count(tf_int) < context.get_prev_feature_count(
                    tf_int
                )
            deltas = context.get_entity_deltas()
            net_delta = sum(1 for d in deltas if d.curr_feature == tf_int) - sum(
                1 for d in deltas if d.prev_feature == tf_int
            )
            return net_delta < 0
        if self.effect_operator == "transform_feature":
            src = self.effect_params.get("src_feature")
            dst = self.effect_params.get("dst_feature")
            if src is None or dst is None:
                return False
            return any(
                int(d.prev_feature) == src and int(d.curr_feature) == dst
                for d in context.get_entity_deltas()
            )
        if self.effect_operator == "translate_entity":
            feat = self.effect_params.get("feature")
            delta = self.effect_params.get("delta")
            if feat is None or not isinstance(delta, tuple) or not delta:
                return False
            for d in context.get_entity_deltas():
                if d.prev_feature == feat and d.curr_feature != feat:
                    target_locus = _shift_numeric_locus(d.locus, delta)
                    if (
                        target_locus is not None
                        and context.get_curr_feature_at(target_locus) == feat
                    ):
                        return True
            return False
        if self.effect_operator == "spawn_feature":
            dst = self.effect_params.get("dst_feature")
            if dst is None:
                return False
            _, new_diff_feats = context.get_diff_feature_sets()
            return int(dst) in new_diff_feats
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
            0 < context.changed_count <= 8
            and isinstance(context.prev_avatar_pos, tuple)
            and isinstance(context.curr_avatar_pos, tuple)
            and len(context.prev_avatar_pos) == len(context.curr_avatar_pos)
            and all(
                isinstance(p, (int, float)) and isinstance(c, (int, float))
                for p, c in zip(context.prev_avatar_pos, context.curr_avatar_pos)
            )
        ):
            displacement = tuple(
                int(c - p) if isinstance(p, int) and isinstance(c, int) else float(c - p)
                for p, c in zip(context.prev_avatar_pos, context.curr_avatar_pos)
            )
            if any(d != 0 for d in displacement) and sum(abs(d) for d in displacement) <= 2:
                for delta in context.get_entity_deltas()[:16]:
                    old_f = int(delta.prev_feature)
                    new_f = int(delta.curr_feature)
                    if (
                        old_f != bg
                        and new_f != old_f
                        and delta.locus == context.curr_avatar_pos
                        and contact_f == old_f
                    ):
                        target_locus = _shift_numeric_locus(delta.locus, displacement)
                        if (
                            target_locus is not None
                            and context.get_curr_feature_at(target_locus) == old_f
                            and context.get_prev_feature_at(target_locus) != old_f
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
                                    effect_params={"feature": old_f, "delta": displacement},
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
            for delta in context.get_entity_deltas()[:16]:
                if context.prev_avatar_pos == delta.locus or context.curr_avatar_pos == delta.locus:
                    continue
                old_f = int(delta.prev_feature)
                new_f = int(delta.curr_feature)
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
        if not context.is_valid_curr_locus() or context.is_dead:
            return []

        c_feat = context.get_contact_feature()
        if c_feat is None:
            return []
        falsified: list[tuple[RelationalRuleHypothesis, HypothesisTransitionEvent]] = []

        for hyp in list(hypotheses.values()):
            if hyp.rule_type == HypothesisType.REMOTE_MECHANISM:
                if hyp.premise_feature == c_feat:
                    target_f = hyp.target_feature
                    if target_f is not None and context.has_feature_count_summary():
                        curr_target_count = context.get_curr_feature_count(target_f)
                        prev_target_count = context.get_prev_feature_count(target_f)
                        if curr_target_count > 0 and curr_target_count >= prev_target_count:
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
