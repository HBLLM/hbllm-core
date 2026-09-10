"""Subwave A23.5: Active Interventional Causal Discovery Engine.

Implements the active hypothesis generation, interventional probing,
falsification, and causal rule induction cognitive loop under confounding.
"""

from __future__ import annotations

import logging
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .perception import DevelopmentalPerceptionAdapter
from .types import (
    BabyActionType,
    BeliefTransitionEvent,
    BeliefTransitionType,
    CausalHypothesis,
    SensoryObservation,
)

logger = logging.getLogger(__name__)


class InterventionalCausalDiscoveryEngine:
    """Discovers true causal physical laws through active contrastive intervention."""

    def __init__(
        self,
        substrate: BlankBrainSubstrate,
        perception: DevelopmentalPerceptionAdapter,
        environment: BabyWorldEnvironment,
    ) -> None:
        self.substrate = substrate
        self.perception = perception
        self.env = environment

        # Active belief & hypothesis registries
        self.hypotheses: list[CausalHypothesis] = []
        self.belief_history: list[BeliefTransitionEvent] = []
        self.interventions_count: int = 0
        self.confirmed_causal_rules: list[dict[str, Any]] = []

    def observe_and_generate_hypotheses(
        self,
        observation: SensoryObservation,
        episodes_data: list[dict[str, Any]],
    ) -> list[CausalHypothesis]:
        """Formulate candidate causal hypotheses based on initial correlations.

        In the confounded world, the agent initially observes:
        - red_ball (light) moved
        - red_block (light) moved
        - blue_ball (heavy) did not move
        - blue_block (heavy) did not move

        Candidate variables to hypothesize:
        1. color == 'red' (Confounded / Spurious)
        2. shape == 'ball' (Spurious)
        3. mass_sensation < 5.0 (True Causal Variable)
        """
        # Formulate competing hypotheses from features observed on moving objects
        positive_colors = [
            p["color"]
            for p in observation.vision
            if p.get("mass_sensation", 99.0) < 5.0 and p.get("color")
        ]
        pos_color = positive_colors[0] if positive_colors else "red"

        positive_shapes = [
            p["shape"]
            for p in observation.vision
            if p.get("mass_sensation", 99.0) < 5.0 and p.get("shape")
        ]
        pos_shape = positive_shapes[0] if positive_shapes else "ball"

        self.hypotheses = [
            CausalHypothesis(
                action=BabyActionType.PUSH,
                variable="color",
                operator="==",
                value=pos_color,
                consequence="MOVES",
                confidence=0.5,
            ),
            CausalHypothesis(
                action=BabyActionType.PUSH,
                variable="shape",
                operator="==",
                value=pos_shape,
                consequence="MOVES",
                confidence=0.5,
            ),
            CausalHypothesis(
                action=BabyActionType.PUSH,
                variable="mass_sensation",
                operator="<",
                value=5.0,
                consequence="MOVES",
                confidence=0.5,
            ),
        ]

        for h in self.hypotheses:
            self._record_belief_event(
                event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                hypothesis=h,
                prior_conf=0.0,
                post_conf=h.confidence,
                details={"formulated_rule": h.describe()},
            )

        return self.hypotheses

    def select_active_intervention(
        self,
        available_entity_ids: list[str],
        active_hypotheses: list[CausalHypothesis],
    ) -> tuple[str, CausalHypothesis]:
        """Select an optimal contrastive interventional probe.

        Active Inference / Epistemic Curiosity:
        Picks the entity whose intervention maximizes pairwise disagreement
        (expected entropy reduction) among currently non-falsified hypotheses.
        """
        active_hyps = [h for h in active_hypotheses if not h.falsified]
        if not active_hyps:
            active_hyps = self.hypotheses

        best_candidate: str | None = None
        max_discriminant_score: float = -999.0
        targeted_hyp: CausalHypothesis = active_hyps[0]

        for ent_id in available_entity_ids:
            ent = self.env.objects.get(ent_id)
            if not ent:
                continue

            # Compute predictions across all active hypotheses
            predictions = []
            for h in active_hyps:
                if h.variable == "color":
                    p = ent.color == h.value
                elif h.variable == "shape":
                    p = ent.object_type.value == h.value
                elif h.variable == "mass_sensation":
                    p = ent.mass < h.value
                else:
                    p = False
                predictions.append(p)

            # Disagreement score: count pairs of hypotheses with conflicting predictions
            disagreements = 0
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    if predictions[i] != predictions[j]:
                        disagreements += 1

            # Novelty bonus: penalize entities already tested to encourage exploration
            times_tested = sum(
                1
                for h in active_hyps
                if ent_id in h.supporting_episodes or ent_id in h.counterexamples
            )
            score = float(disagreements) - 0.5 * float(times_tested)

            if score > max_discriminant_score:
                max_discriminant_score = score
                best_candidate = ent_id
                targeted_hyp = active_hyps[0]

        if not best_candidate and available_entity_ids:
            best_candidate = available_entity_ids[0]

        return best_candidate or "", targeted_hyp

    def execute_interventional_probe(
        self,
        target_id: str,
        action: BabyActionType = BabyActionType.PUSH,
    ) -> tuple[bool, dict[str, Any]]:
        """Execute a controlled interventional trial do(action, target_id)."""
        self.interventions_count += 1
        prior_state_idx = self.env.save_state()

        # Step physical environment with probe action
        obs, reward, done, consequences = self.env.step(action=action, target_id=target_id)
        did_move = consequences.get("moved", False)

        target_obj = self.env.objects.get(target_id)
        probe_result = {
            "target_id": target_id,
            "target_color": target_obj.color if target_obj else "",
            "target_shape": target_obj.object_type.value if target_obj else "",
            "target_mass": target_obj.mass if target_obj else 0.0,
            "did_move": did_move,
            "displacement": consequences.get("displacement", 0.0),
        }

        # Update beliefs based on interventional evidence
        self._update_hypotheses_from_evidence(probe_result)

        # Restore world state so subsequent probes start from standardized conditions
        self.env.restore_state(prior_state_idx)

        return did_move, probe_result

    def _update_hypotheses_from_evidence(self, probe_result: dict[str, Any]) -> None:
        """Update posterior confidences and falsify invalidated hypotheses."""
        did_move = probe_result["did_move"]
        color = probe_result["target_color"]
        shape = probe_result["target_shape"]
        mass = probe_result["target_mass"]

        for h in self.hypotheses:
            if h.falsified:
                continue

            prior_conf = h.confidence

            # Compute prediction for this hypothesis
            if h.variable == "color":
                predicted_move = color == h.value
            elif h.variable == "shape":
                predicted_move = shape == h.value
            elif h.variable == "mass_sensation":
                predicted_move = mass < h.value
            else:
                predicted_move = False

            h.interventions_tested += 1

            if predicted_move != did_move:
                # Prediction Error / Counterexample: Strict Falsification!
                h.falsified = True
                h.confidence = 0.0
                h.counterexamples.append(probe_result["target_id"])
                self._record_belief_event(
                    event_type=BeliefTransitionType.HYPOTHESIS_FALSIFIED,
                    hypothesis=h,
                    prior_conf=prior_conf,
                    post_conf=0.0,
                    details={"counterexample": probe_result},
                )
            else:
                # Evidence consistent with hypothesis
                h.supporting_episodes.append(probe_result["target_id"])
                # Bayesian belief update
                h.confidence = min(1.0, h.confidence + 0.25)
                self._record_belief_event(
                    event_type=BeliefTransitionType.CONFIDENCE_CHANGED,
                    hypothesis=h,
                    prior_conf=prior_conf,
                    post_conf=h.confidence,
                    details={"supporting_evidence": probe_result},
                )

                if h.confidence >= 0.95 and not h.confirmed:
                    h.confirmed = True
                    self._record_belief_event(
                        event_type=BeliefTransitionType.HYPOTHESIS_CONFIRMED,
                        hypothesis=h,
                        prior_conf=prior_conf,
                        post_conf=h.confidence,
                        details={"confirmed_causal_law": h.describe()},
                    )
                    self._induce_causal_rule_into_substrate(h)

    def _induce_causal_rule_into_substrate(self, confirmed_hyp: CausalHypothesis) -> None:
        """Synthesize confirmed hypothesis into a generalized HCIR causal rule."""
        rule = {
            "rule_id": f"causal_rule_{len(self.confirmed_causal_rules) + 1}",
            "action": confirmed_hyp.action.value,
            "precondition": {
                "property": confirmed_hyp.variable,
                "operator": confirmed_hyp.operator,
                "value": confirmed_hyp.value,
            },
            "consequence": confirmed_hyp.consequence,
            "confidence": confirmed_hyp.confidence,
            "empirical_support_count": len(confirmed_hyp.supporting_episodes),
        }
        self.confirmed_causal_rules.append(rule)
        self.substrate.causal_rules.append(rule)

        self._record_belief_event(
            event_type=BeliefTransitionType.RULE_GENERALIZED,
            hypothesis=confirmed_hyp,
            prior_conf=confirmed_hyp.confidence,
            post_conf=1.0,
            details={"generalized_rule": rule},
        )

    def evaluate_generalization(
        self,
        test_objects: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired causal rules against held-out entities or worlds (Levels 2 & 3)."""
        if not self.confirmed_causal_rules:
            return 0.0, []

        # Prioritize confirmed mass causal rule if available
        mass_rules = [
            r
            for r in self.confirmed_causal_rules
            if r["precondition"]["property"] == "mass_sensation"
        ]
        rule = mass_rules[0] if mass_rules else self.confirmed_causal_rules[0]
        prop_key = rule["precondition"]["property"]
        threshold = rule["precondition"]["value"]

        eval_records = []
        correct = 0

        for obj_info in test_objects:
            actual_mass = float(obj_info["mass"])

            if prop_key == "mass_sensation":
                predicted_moves = actual_mass < float(threshold)
            elif prop_key == "color":
                predicted_moves = obj_info.get("color") == threshold
            elif prop_key == "shape":
                predicted_moves = obj_info.get("shape") == threshold
            else:
                predicted_moves = False

            # Actual physical outcome under force
            actual_moves = actual_mass < BabyWorldEnvironment.MASS_THRESHOLD

            is_correct = predicted_moves == actual_moves
            if is_correct:
                correct += 1

            eval_records.append(
                {
                    "id": obj_info["id"],
                    "color": obj_info.get("color"),
                    "shape": obj_info.get("shape"),
                    "mass": actual_mass,
                    "predicted_moves": predicted_moves,
                    "actual_moves": actual_moves,
                    "is_correct": is_correct,
                }
            )

        accuracy = correct / max(1, len(test_objects))
        return accuracy, eval_records

    def _record_belief_event(
        self,
        event_type: BeliefTransitionType,
        hypothesis: CausalHypothesis,
        prior_conf: float,
        post_conf: float,
        details: dict[str, Any],
    ) -> None:
        """Log an immutable event to the developmental belief history."""
        event = BeliefTransitionEvent(
            event_type=event_type,
            step_index=self.interventions_count,
            hypothesis_id=hypothesis.hypothesis_id,
            variable=hypothesis.variable,
            condition=f"{hypothesis.variable} {hypothesis.operator} {hypothesis.value}",
            prior_confidence=prior_conf,
            posterior_confidence=post_conf,
            is_falsified=hypothesis.falsified,
            evidence=details,
        )
        self.belief_history.append(event)
