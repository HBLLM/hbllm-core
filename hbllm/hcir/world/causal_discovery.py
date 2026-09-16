"""Active Interventional Causal Discovery Engine for HCIR World Kernel.

Implements domain-agnostic active hypothesis generation, interventional probing,
strict falsification, Bayesian posterior updating, and causal rule induction under confounding.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.hcir.world.world_causal import CausalEdgeType, WorldCausalGraph

logger = logging.getLogger(__name__)


class BeliefTransitionType(str, Enum):
    """Immutable event types for event-sourced developmental belief logging."""

    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_TESTED = "hypothesis_tested"
    HYPOTHESIS_FALSIFIED = "hypothesis_falsified"
    HYPOTHESIS_CONFIRMED = "hypothesis_confirmed"
    CONFIDENCE_CHANGED = "confidence_changed"
    RULE_GENERALIZED = "rule_generalized"
    RULE_REVISED = "rule_revised"
    AFFORDANCE_DISCOVERED = "affordance_discovered"
    SPATIAL_SCHEMA_INDUCED = "spatial_schema_induced"
    TOOL_COMPOSED = "tool_composed"
    GOAL_SYNTHESIZED = "goal_synthesized"
    PLAN_EXECUTED = "plan_executed"
    PLAN_REPLANNED = "plan_replanned"
    CURIOSITY_EXPLORATION = "curiosity_exploration"
    CONCEPT_INDUCED = "concept_induced"
    LEXICON_GROUNDED = "lexicon_grounded"
    COMPOSITION_PARSED = "composition_parsed"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    METACOGNITION_CALIBRATED = "metacognition_calibrated"
    CROSS_WORLD_TRANSFERRED = "cross_world_transferred"
    CROSS_DOMAIN_TRANSFERRED = "cross_domain_transferred"


@dataclass
class BeliefTransitionEvent:
    """An immutable record of a developmental or cognitive belief transition."""

    event_id: str = field(default_factory=lambda: f"bte_{uuid.uuid4().hex[:8]}")
    event_type: BeliefTransitionType = BeliefTransitionType.HYPOTHESIS_CREATED
    step_index: int = 0
    hypothesis_id: str = ""
    variable: str = ""  # e.g. "color", "mass", "shape", "feature_x"
    condition: str = ""  # e.g. "color == 'red'", "mass < 5.0"
    prior_confidence: float = 0.0
    posterior_confidence: float = 0.0
    is_falsified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CausalHypothesis:
    """State tracker for candidate causal relationships."""

    hypothesis_id: str = field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:6]}")
    action: Any = "PUSH"
    variable: str = "color"
    operator: str = "=="
    value: Any = "red"
    consequence: str = "MOVES"
    confidence: float = 0.5
    interventions_tested: int = 0
    falsified: bool = False
    confirmed: bool = False
    supporting_episodes: list[str] = field(default_factory=list)
    counterexamples: list[str] = field(default_factory=list)

    def describe(self) -> str:
        act_val = self.action.value if hasattr(self.action, "value") else str(self.action)
        return (
            f"{act_val}(x) ∧ ({self.variable} {self.operator} {self.value}) => {self.consequence}"
        )


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
    def predict_hypothesis(h: CausalHypothesis, features: dict[str, Any]) -> bool:
        """Evaluate hypothesis prediction against observed features."""
        val = features.get(h.variable)
        if val is None:
            return False
        if h.operator == "==":
            return str(val) == str(h.value)
        elif h.operator == "!=":
            return str(val) != str(h.value)
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
    def _predict_hypothesis(h: CausalHypothesis, features: dict[str, Any]) -> bool:
        """Backward-compatible private alias."""
        return BaseCausalDiscoveryEngine.predict_hypothesis(h, features)

    def rank_interventional_candidates(
        self,
        candidate_ids: Sequence[str],
        active_hypotheses: Sequence[CausalHypothesis],
        feature_map: dict[str, dict[str, Any]],
        exploration_penalty: float = 0.5,
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

            predictions = [self.predict_hypothesis(h, percept) for h in active_hyps]

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
    ) -> list[BeliefTransitionEvent]:
        """Update posterior confidences and falsify invalidated hypotheses on counterexample."""
        did_move = bool(probe_result.get("did_move", False))
        target_id = str(probe_result.get("target_id", ""))
        events: list[BeliefTransitionEvent] = []

        for h in hypotheses:
            if h.falsified:
                continue

            prior_conf = h.confidence
            predicted_move = self.predict_hypothesis(h, probe_result)
            h.interventions_tested += 1

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
                        condition=f"{h.variable} {h.operator} {h.value}",
                        prior_confidence=prior_conf,
                        posterior_confidence=0.0,
                        is_falsified=True,
                        evidence={"counterexample": probe_result},
                    )
                )
                logger.info(
                    "Hypothesis falsified: %s (variable=%s)",
                    h.hypothesis_id,
                    h.variable,
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
                        condition=f"{h.variable} {h.operator} {h.value}",
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
                            condition=f"{h.variable} {h.operator} {h.value}",
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
            if hasattr(confirmed_hyp.action, "value")
            else str(confirmed_hyp.action)
        )

        for existing in rule_store:
            if (
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
                    condition=f"{confirmed_hyp.variable} {confirmed_hyp.operator} {confirmed_hyp.value}",
                    prior_confidence=confirmed_hyp.confidence,
                    posterior_confidence=existing["confidence"],
                    is_falsified=False,
                    evidence={"rule": existing},
                )
                return existing, event

        rule: dict[str, Any] = {
            "rule_id": f"causal_rule_{len(rule_store) + 1}",
            "action": action_val,
            "precondition": {
                "property": confirmed_hyp.variable,
                "operator": confirmed_hyp.operator,
                "value": confirmed_hyp.value,
            },
            "consequence": confirmed_hyp.consequence,
            "confidence": confirmed_hyp.confidence,
            "empirical_support_count": len(confirmed_hyp.supporting_episodes),
        }
        rule_store.append(rule)

        target_graph = causal_graph or self.causal_graph
        if target_graph is not None:
            source_node = f"{confirmed_hyp.variable}_{confirmed_hyp.operator}_{confirmed_hyp.value}"
            target_node = str(confirmed_hyp.consequence)
            target_graph.add_causal_relation(
                source_id=source_node,
                target_id=target_node,
                relationship=CausalEdgeType.CAUSES,
                weight=confirmed_hyp.confidence,
            )

        event = BeliefTransitionEvent(
            event_type=BeliefTransitionType.RULE_GENERALIZED,
            step_index=step_index,
            hypothesis_id=confirmed_hyp.hypothesis_id,
            variable=confirmed_hyp.variable,
            condition=f"{confirmed_hyp.variable} {confirmed_hyp.operator} {confirmed_hyp.value}",
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
        ]
        rule = phys_rules[0] if phys_rules else confirmed_rules[0]
        precond = rule["precondition"]
        prop_key = precond["property"]
        threshold = float(precond["value"])
        operator = precond["operator"]

        eval_records = []
        correct = 0

        for obj_info in test_objects:
            actual_mass = float(obj_info.get("mass", 5.0))
            actual_friction = float(obj_info.get("surface_friction", 1.0))
            actual_static = float(obj_info.get("static_threshold", 0.0))
            actual_clearance = float(obj_info.get("clearance_diameter", 0.4))

            if prop_key == "surface_friction":
                feat_val = actual_friction
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
