"""Active Interventional Causal Discovery Engine for HCIR World Kernel."""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Any

from hbllm.hcir.world.causal.mediation import (
    CausalMediationResult,
    MediationIdentifiabilityAssumptions,
)
from hbllm.hcir.world.causal.predicates import (
    BeliefTransitionEvent,
    BeliefTransitionType,
    CausalHypothesis,
    CausalPredicate,
    LogicOperator,
)
from hbllm.hcir.world.world_causal import CausalEdgeType, WorldCausalGraph

logger = logging.getLogger(__name__)


class BaseCausalDiscoveryEngine:
    """Domain-agnostic causal discovery engine using active interventional contrastive inference."""

    def __init__(
        self,
        causal_graph: WorldCausalGraph | None = None,
    ) -> None:
        self.causal_graph = causal_graph or WorldCausalGraph()
        self.hypotheses: list[CausalHypothesis] = []
        self.belief_history: list[BeliefTransitionEvent] = []
        self.interventions_count: int = 0
        self.confirmed_causal_rules: list[dict[str, Any]] = []

    @staticmethod
    def compute_hypothesis_entropy(hypotheses: Sequence[CausalHypothesis]) -> float:
        """Compute Shannon entropy across active non-falsified hypotheses."""
        active_confs = [h.confidence for h in hypotheses if not h.falsified and h.confidence > 0.0]
        total = sum(active_confs)
        if not active_confs or total <= 0.0:
            return 0.0
        probs = [c / total for c in active_confs]
        return -sum(p * math.log2(p) for p in probs if p > 0.0)

    def _compute_hypothesis_entropy(self) -> float:
        """Helper for self-held hypotheses."""
        return self.compute_hypothesis_entropy(self.hypotheses)

    @staticmethod
    def predict_hypothesis(
        h: CausalHypothesis,
        features: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate hypothesis prediction against observed features and optional context."""
        if h.predicate is not None:
            return h.predicate.evaluate(features, context)

        # Legacy atomic attribute evaluation
        val = features.get(h.variable)
        if val is None:
            return False
        if h.operator == "==":
            return str(val) == str(h.value)
        elif h.operator == "!=":
            return str(val) != str(h.value)
        elif h.operator == "in":
            if isinstance(h.value, (list, set, tuple)):
                return val in h.value
            return str(val) in str(h.value)
        elif h.operator == "contains":
            if isinstance(val, (list, set, tuple)):
                return h.value in val
            return str(h.value) in str(val)
        try:
            fval = float(val)
            fthresh = float(h.value)
            if h.operator == "<":
                return fval < fthresh
            elif h.operator == "<=":
                return fval <= fthresh
            elif h.operator == ">":
                return fval > fthresh
            elif h.operator == ">=":
                return fval >= fthresh
        except (ValueError, TypeError):
            return False
        return False

    @staticmethod
    def _predict_hypothesis(
        h: CausalHypothesis,
        features: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Backward-compatible private alias."""
        return BaseCausalDiscoveryEngine.predict_hypothesis(h, features, context)

    def compose_hypotheses(
        self,
        hypotheses: Sequence[CausalHypothesis],
        logic_op: LogicOperator = LogicOperator.AND,
        hypothesis_id: str | None = None,
        action: Any | None = None,
        consequence: str | None = None,
    ) -> CausalHypothesis:
        """Compose multiple hypotheses into a higher-order composite hypothesis."""
        if not hypotheses:
            raise ValueError("Cannot compose empty hypotheses list")

        act = action if action is not None else hypotheses[0].action
        cons = consequence if consequence is not None else hypotheses[0].consequence
        children = [h.get_predicate() for h in hypotheses]
        composite_pred = CausalPredicate(
            logic_op=logic_op,
            children=children,
        )

        if logic_op == LogicOperator.AND:
            prior_conf = math.prod(h.confidence for h in hypotheses)
        elif logic_op == LogicOperator.OR:
            prior_conf = max(h.confidence for h in hypotheses)
        else:
            prior_conf = hypotheses[0].confidence

        comp_hyp = CausalHypothesis(
            hypothesis_id=hypothesis_id or f"hyp_comp_{uuid.uuid4().hex[:6]}",
            action=act,
            variable=hypotheses[0].variable,
            operator=hypotheses[0].operator,
            value=hypotheses[0].value,
            consequence=cons,
            confidence=max(0.05, round(prior_conf, 3)),
            predicate=composite_pred,
        )
        self.hypotheses.append(comp_hyp)
        return comp_hyp

    def specialize_hypothesis(
        self,
        parent_hypothesis: CausalHypothesis,
        discriminating_feature: str,
        operator: str,
        value: Any,
        logic_op: LogicOperator = LogicOperator.AND,
        hypothesis_id: str | None = None,
    ) -> CausalHypothesis:
        """Refine a partially predictive hypothesis by adding a discriminating condition."""
        parent_pred = parent_hypothesis.get_predicate()
        new_pred = CausalPredicate(
            variable=discriminating_feature,
            operator=operator,
            value=value,
        )
        if parent_pred.logic_op == logic_op:
            children = list(parent_pred.children) + [new_pred]
        else:
            children = [parent_pred, new_pred]

        composite_pred = CausalPredicate(
            logic_op=logic_op,
            children=children,
        )
        specialized = CausalHypothesis(
            hypothesis_id=hypothesis_id or f"hyp_spec_{uuid.uuid4().hex[:6]}",
            action=parent_hypothesis.action,
            variable=discriminating_feature,
            operator=operator,
            value=value,
            consequence=parent_hypothesis.consequence,
            confidence=max(0.1, round(parent_hypothesis.confidence * 0.8, 3)),
            predicate=composite_pred,
        )
        self.hypotheses.append(specialized)
        return specialized

    def rank_interventional_candidates(
        self,
        candidate_ids: Sequence[str],
        active_hypotheses: Sequence[CausalHypothesis],
        feature_map: dict[str, dict[str, Any]],
        exploration_penalty: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Rank interventional candidates by expected information gain (hypothesis disagreement score)."""
        active_hyps = [h for h in active_hypotheses if not h.falsified]
        if not active_hyps:
            return [(cid, 0.0) for cid in candidate_ids]

        ranked: list[tuple[str, float]] = []
        for ent_id in candidate_ids:
            percept = feature_map.get(ent_id)
            if not percept:
                continue

            predictions = [self.predict_hypothesis(h, percept, context) for h in active_hyps]

            disagreements = 0
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    if predictions[i] != predictions[j]:
                        disagreements += 1

            times_tested = sum(
                1
                for h in active_hyps
                if ent_id in h.supporting_episodes or ent_id in h.counterexamples
            )
            score = float(disagreements) - exploration_penalty * float(times_tested)
            ranked.append((ent_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def update_hypotheses_from_evidence(
        self,
        hypotheses: list[CausalHypothesis],
        probe_result: dict[str, Any],
        step_index: int = 0,
        confirmation_threshold: float = 0.95,
        confidence_step: float = 0.25,
        context: dict[str, Any] | None = None,
    ) -> list[BeliefTransitionEvent]:
        """Update posterior confidences and falsify invalidated hypotheses on counterexample."""
        did_move = bool(probe_result.get("did_move", False))
        target_id = str(probe_result.get("target_id", ""))
        events: list[BeliefTransitionEvent] = []

        for h in hypotheses:
            if h.falsified:
                continue

            prior_conf = h.confidence
            predicted_move = self.predict_hypothesis(h, probe_result, context)
            h.interventions_tested += 1

            cond_str = (
                h.predicate.describe()
                if h.predicate is not None
                else f"{h.variable} {h.operator} {h.value}"
            )

            if predicted_move != did_move:
                # Prediction Error / Counterexample: Strict Falsification
                h.falsified = True
                h.confidence = 0.0
                h.counterexamples.append(target_id)
                events.append(
                    BeliefTransitionEvent(
                        event_type=BeliefTransitionType.HYPOTHESIS_FALSIFIED,
                        step_index=step_index,
                        hypothesis_id=h.hypothesis_id,
                        variable=h.variable,
                        condition=cond_str,
                        prior_confidence=prior_conf,
                        posterior_confidence=0.0,
                        is_falsified=True,
                        evidence={"counterexample": probe_result},
                    )
                )
                logger.info(
                    "Hypothesis falsified: %s (condition=%s)",
                    h.hypothesis_id,
                    cond_str,
                    extra={"hypothesis_id": h.hypothesis_id, "variable": h.variable},
                )
            else:
                # Evidence consistent with hypothesis
                h.supporting_episodes.append(target_id)
                h.confidence = min(1.0, h.confidence + confidence_step)
                events.append(
                    BeliefTransitionEvent(
                        event_type=BeliefTransitionType.CONFIDENCE_CHANGED,
                        step_index=step_index,
                        hypothesis_id=h.hypothesis_id,
                        variable=h.variable,
                        condition=cond_str,
                        prior_confidence=prior_conf,
                        posterior_confidence=h.confidence,
                        is_falsified=False,
                        evidence={"supporting_evidence": probe_result},
                    )
                )

                if h.confidence >= confirmation_threshold and not h.confirmed:
                    h.confirmed = True
                    events.append(
                        BeliefTransitionEvent(
                            event_type=BeliefTransitionType.HYPOTHESIS_CONFIRMED,
                            step_index=step_index,
                            hypothesis_id=h.hypothesis_id,
                            variable=h.variable,
                            condition=cond_str,
                            prior_confidence=prior_conf,
                            posterior_confidence=h.confidence,
                            is_falsified=False,
                            evidence={"confirmed_causal_law": h.describe()},
                        )
                    )

        return events

    def induce_causal_rule(
        self,
        confirmed_hyp: CausalHypothesis,
        rule_store: list[dict[str, Any]],
        causal_graph: WorldCausalGraph | None = None,
        step_index: int = 0,
    ) -> tuple[dict[str, Any], BeliefTransitionEvent]:
        """Synthesize confirmed hypothesis into a generalized HCIR causal rule."""
        action_val = (
            confirmed_hyp.action.value
            if isinstance(confirmed_hyp.action, Enum)
            else str(confirmed_hyp.action)
        )
        is_composite = (
            confirmed_hyp.predicate is not None and confirmed_hyp.predicate.is_composite()
        )
        cond_str = (
            confirmed_hyp.predicate.describe()
            if confirmed_hyp.predicate is not None
            else f"{confirmed_hyp.variable} {confirmed_hyp.operator} {confirmed_hyp.value}"
        )

        factors: list[str] = []
        if is_composite and confirmed_hyp.predicate:
            factors = [c.describe() for c in confirmed_hyp.predicate.children]
        elif confirmed_hyp.predicate and confirmed_hyp.predicate.is_relational():
            factors = [confirmed_hyp.predicate.describe()]
        else:
            factors = [f"{confirmed_hyp.variable} {confirmed_hyp.operator} {confirmed_hyp.value}"]

        for existing in rule_store:
            # Check atomic match
            if not is_composite and (
                existing.get("action") == action_val
                and existing.get("consequence") == confirmed_hyp.consequence
                and existing.get("precondition", {}).get("property") == confirmed_hyp.variable
                and existing.get("precondition", {}).get("operator") == confirmed_hyp.operator
                and existing.get("precondition", {}).get("value") == confirmed_hyp.value
            ):
                add_count = max(1, len(confirmed_hyp.supporting_episodes))
                existing["empirical_support_count"] = (
                    existing.get("empirical_support_count", 1) + add_count
                )
                existing["confidence"] = min(
                    1.0, max(existing.get("confidence", 0.5), confirmed_hyp.confidence)
                )
                event = BeliefTransitionEvent(
                    event_type=BeliefTransitionType.RULE_REVISED,
                    step_index=step_index,
                    hypothesis_id=confirmed_hyp.hypothesis_id,
                    variable=confirmed_hyp.variable,
                    condition=cond_str,
                    prior_confidence=confirmed_hyp.confidence,
                    posterior_confidence=existing["confidence"],
                    is_falsified=False,
                    evidence={"rule": existing},
                )
                return existing, event
            # Check composite match
            elif (
                is_composite
                and existing.get("action") == action_val
                and existing.get("consequence") == confirmed_hyp.consequence
                and existing.get("is_composite")
                and existing.get("precondition", {}).get("composite_description") == cond_str
            ):
                add_count = max(1, len(confirmed_hyp.supporting_episodes))
                existing["empirical_support_count"] = (
                    existing.get("empirical_support_count", 1) + add_count
                )
                existing["confidence"] = min(
                    1.0, max(existing.get("confidence", 0.5), confirmed_hyp.confidence)
                )
                event = BeliefTransitionEvent(
                    event_type=BeliefTransitionType.RULE_REVISED,
                    step_index=step_index,
                    hypothesis_id=confirmed_hyp.hypothesis_id,
                    variable=confirmed_hyp.variable,
                    condition=cond_str,
                    prior_confidence=confirmed_hyp.confidence,
                    posterior_confidence=existing["confidence"],
                    is_falsified=False,
                    evidence={"rule": existing},
                )
                return existing, event

        rule: dict[str, Any] = {
            "rule_id": f"causal_rule_{len(rule_store) + 1}",
            "action": action_val,
            "is_composite": is_composite,
            "predicate": confirmed_hyp.predicate.to_dict() if confirmed_hyp.predicate else None,
            "precondition": {
                "property": confirmed_hyp.variable,
                "operator": confirmed_hyp.operator,
                "value": confirmed_hyp.value,
                "composite_description": cond_str if is_composite else "",
            },
            "consequence": confirmed_hyp.consequence,
            "confidence": confirmed_hyp.confidence,
            "empirical_support_count": len(confirmed_hyp.supporting_episodes),
        }
        rule_store.append(rule)

        target_graph = causal_graph or self.causal_graph
        if target_graph is not None:
            source_node = (
                cond_str
                if is_composite
                else f"{confirmed_hyp.variable}_{confirmed_hyp.operator}_{confirmed_hyp.value}"
            )
            target_node = str(confirmed_hyp.consequence)
            target_graph.add_causal_relation(
                source_id=source_node,
                target_id=target_node,
                relationship=CausalEdgeType.CAUSES,
                weight=confirmed_hyp.confidence,
                factors=factors,
            )

        event = BeliefTransitionEvent(
            event_type=BeliefTransitionType.RULE_GENERALIZED,
            step_index=step_index,
            hypothesis_id=confirmed_hyp.hypothesis_id,
            variable=confirmed_hyp.variable,
            condition=cond_str,
            prior_confidence=confirmed_hyp.confidence,
            posterior_confidence=1.0,
            is_falsified=False,
            evidence={"generalized_rule": rule},
        )
        return rule, event

    @staticmethod
    def test_generalization(
        test_objects: list[dict[str, Any]],
        confirmed_rules: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired causal rules against held-out entities or worlds."""
        if not confirmed_rules:
            return 0.0, []

        phys_rules = [
            r
            for r in confirmed_rules
            if r.get("precondition", {}).get("property")
            in ("surface_friction", "mass_sensation", "static_threshold", "clearance_diameter")
            or r.get("is_composite")
        ]
        rule = phys_rules[0] if phys_rules else confirmed_rules[0]

        eval_records = []
        correct = 0

        is_composite = bool(rule.get("is_composite")) and bool(rule.get("predicate"))
        comp_predicate: CausalPredicate | None = None
        if is_composite and rule.get("predicate"):
            comp_predicate = CausalPredicate.from_dict(rule["predicate"])

        precond = rule["precondition"]
        prop_key = precond.get("property", "")
        threshold = float(precond.get("value", 0.0)) if precond.get("value") is not None else 0.0
        operator = precond.get("operator", "==")

        for obj_info in test_objects:
            actual_mass = float(obj_info.get("mass", 5.0))
            actual_friction = float(obj_info.get("surface_friction", 1.0))
            actual_static = float(obj_info.get("static_threshold", 0.0))
            actual_clearance = float(obj_info.get("clearance_diameter", 0.4))

            if is_composite and comp_predicate is not None:
                predicted_moves = comp_predicate.evaluate(obj_info)
            else:
                if prop_key == "surface_friction":
                    feat_val: Any = actual_friction
                elif prop_key == "mass_sensation":
                    feat_val = actual_mass
                elif prop_key == "static_threshold":
                    feat_val = actual_static
                elif prop_key == "clearance_diameter":
                    feat_val = actual_clearance
                else:
                    feat_val = obj_info.get(prop_key, "")

                if operator == "<":
                    predicted_moves = float(feat_val) < threshold
                elif operator == "<=":
                    predicted_moves = float(feat_val) <= threshold
                elif operator == ">":
                    predicted_moves = float(feat_val) > threshold
                elif operator == ">=":
                    predicted_moves = float(feat_val) >= threshold
                elif operator == "==":
                    predicted_moves = str(feat_val) == str(threshold)
                else:
                    predicted_moves = False

            if prop_key == "clearance_diameter":
                actual_moves = actual_clearance <= 0.5
            else:
                effective_resistance = max(actual_mass * actual_friction, actual_static)
                actual_moves = 5.0 > effective_resistance

            is_correct = predicted_moves == actual_moves
            if is_correct:
                correct += 1

            eval_records.append(
                {
                    "id": obj_info.get("id", f"eval_obj_{len(eval_records) + 1}"),
                    "color": obj_info.get("color"),
                    "shape": obj_info.get("shape"),
                    "mass": actual_mass,
                    "surface_friction": actual_friction,
                    "predicted_moves": predicted_moves,
                    "actual_moves": actual_moves,
                    "is_correct": is_correct,
                }
            )

        accuracy = round(correct / len(test_objects), 4) if test_objects else 0.0
        return accuracy, eval_records

    def evaluate_generalization(
        self,
        test_objects: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired causal rules against held-out entities or worlds."""
        return self.test_generalization(test_objects, self.confirmed_causal_rules)

    def compute_causal_mediation(
        self,
        observations: list[dict[str, Any]],
        treatment: str,
        mediator: str,
        outcome: str,
        covariates: list[str] | None = None,
        baseline_treatment: Any = 0,
        active_treatment: Any = 1,
        identifiability_assumptions: MediationIdentifiabilityAssumptions | None = None,
    ) -> CausalMediationResult:
        """Compute Pearl's non-parametric causal mediation effects (NDE, NIE, TE, CDE) under DAG identifiability (W079).

        Replaces naive regression decomposition with Pearl's interventional counterfactual
        mediation formula:
            TE = E[Y(x1) - Y(x0)]
            NDE = E[Y(x1, M(x0)) - Y(x0, M(x0))]
            NIE = E[Y(x1, M(x1)) - Y(x1, M(x0))]
            CDE(m) = E[Y(x1, m) - Y(x0, m)]
        """
        cov_keys = tuple(sorted(covariates or []))
        warnings: list[str] = []

        if identifiability_assumptions is None:
            identifiability = MediationIdentifiabilityAssumptions(
                no_treatment_outcome_confounding=True,
                no_mediator_outcome_confounding=True,
                no_treatment_mediator_confounding=True,
                no_cross_world_counterfactual_confounding=True,
                covariates_controlled=list(cov_keys),
            )
            if not cov_keys:
                warnings.append(
                    "No baseline covariates specified: assuming unconfounded observational exchangeability."
                )
        else:
            identifiability = identifiability_assumptions
            if not identifiability.is_fully_identified():
                warnings.append(
                    "Causal identifiability conditions violated: resulting effects represent associative decompositions."
                )

        if not observations:
            return CausalMediationResult(
                treatment=treatment,
                mediator=mediator,
                outcome=outcome,
                total_effect=0.0,
                natural_direct_effect=0.0,
                natural_indirect_effect=0.0,
                controlled_direct_effect={},
                proportion_mediated=0.0,
                identifiability=identifiability,
                is_causally_identified=identifiability.is_fully_identified(),
                unmeasured_confounding_warnings=warnings,
            )

        # 1. Stratify by baseline covariates W
        strata: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for obs in observations:
            w_val = tuple(obs.get(k) for k in cov_keys)
            strata.setdefault(w_val, []).append(obs)

        N_total = len(observations)

        # 2. Extract discrete mediator values
        all_m_vals = sorted(list({obs[mediator] for obs in observations if mediator in obs}))

        # Helper to compute E[Y(x, M(x_prime))] via Pearl's mediation integration
        def compute_counterfactual_expectation(x_val: Any, x_prime_val: Any) -> float:
            total_exp = 0.0
            for w_val, w_obs in strata.items():
                p_w = len(w_obs) / N_total
                if p_w <= 0.0:
                    continue

                # Estimate P(M=m | X=x_prime, W=w)
                x_prime_obs = [o for o in w_obs if o.get(treatment) == x_prime_val]
                p_m_given_x_prime: dict[Any, float] = {}
                if x_prime_obs:
                    for m in all_m_vals:
                        count_m = sum(1 for o in x_prime_obs if o.get(mediator) == m)
                        p_m_given_x_prime[m] = count_m / len(x_prime_obs)
                else:
                    # Fallback to unconditional mediator distribution if stratum sparse
                    for m in all_m_vals:
                        count_m = sum(1 for o in w_obs if o.get(mediator) == m)
                        p_m_given_x_prime[m] = count_m / len(w_obs) if w_obs else 0.0

                # Estimate E[Y | X=x, M=m, W=w]
                stratum_sum = 0.0
                for m, p_m in p_m_given_x_prime.items():
                    if p_m <= 0.0:
                        continue
                    cell_obs = [
                        o for o in w_obs if o.get(treatment) == x_val and o.get(mediator) == m
                    ]
                    if cell_obs:
                        e_y = sum(float(o.get(outcome, 0.0)) for o in cell_obs) / len(cell_obs)
                    else:
                        # Controlled fallback: marginal E[Y | X=x, W=w]
                        x_obs = [o for o in w_obs if o.get(treatment) == x_val]
                        e_y = (
                            sum(float(o.get(outcome, 0.0)) for o in x_obs) / len(x_obs)
                            if x_obs
                            else 0.0
                        )

                    stratum_sum += e_y * p_m

                total_exp += stratum_sum * p_w

            return total_exp

        # 3. Compute counterfactual expectations: E[Y(x1, M(x1))], E[Y(x0, M(x0))], E[Y(x1, M(x0))]
        e_y_11 = compute_counterfactual_expectation(active_treatment, active_treatment)
        e_y_00 = compute_counterfactual_expectation(baseline_treatment, baseline_treatment)
        e_y_10 = compute_counterfactual_expectation(active_treatment, baseline_treatment)

        te = e_y_11 - e_y_00
        nde = e_y_10 - e_y_00
        nie = e_y_11 - e_y_10

        # 4. Controlled Direct Effects CDE(m)
        cde_dict: dict[Any, float] = {}
        for m in all_m_vals:
            # E[Y | X=x1, M=m] - E[Y | X=x0, M=m]
            obs_x1_m = [
                o
                for o in observations
                if o.get(treatment) == active_treatment and o.get(mediator) == m
            ]
            obs_x0_m = [
                o
                for o in observations
                if o.get(treatment) == baseline_treatment and o.get(mediator) == m
            ]
            if obs_x1_m and obs_x0_m:
                e_x1 = sum(float(o.get(outcome, 0.0)) for o in obs_x1_m) / len(obs_x1_m)
                e_x0 = sum(float(o.get(outcome, 0.0)) for o in obs_x0_m) / len(obs_x0_m)
                cde_dict[m] = round(e_x1 - e_x0, 4)

        prop_mediated = nie / te if abs(te) > 1e-6 else 0.0

        return CausalMediationResult(
            treatment=treatment,
            mediator=mediator,
            outcome=outcome,
            total_effect=round(te, 4),
            natural_direct_effect=round(nde, 4),
            natural_indirect_effect=round(nie, 4),
            controlled_direct_effect=cde_dict,
            proportion_mediated=round(prop_mediated, 4),
            identifiability=identifiability,
            is_causally_identified=identifiability.is_fully_identified(),
            unmeasured_confounding_warnings=warnings,
            estimation_regime="pearl_counterfactual_stratification"
            if identifiability.is_fully_identified()
            else "observational_association",
        )
