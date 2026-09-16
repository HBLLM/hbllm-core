"""Subwave A23.5: Active Interventional Causal Discovery Engine.

Implements the active hypothesis generation, interventional probing,
falsification, and causal rule induction cognitive loop under confounding.
"""

import contextlib
import logging
import math
from typing import Any

from hbllm.hcir.world.causal_discovery import BaseCausalDiscoveryEngine

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .metrics import DevelopmentalTelemetryEmitter
from .perception import DevelopmentalPerceptionAdapter
from .types import (
    BabyActionType,
    BeliefTransitionEvent,
    BeliefTransitionType,
    CausalHypothesis,
    SensoryObservation,
    Vector2D,
)

try:
    from hbllm.observability import trace_span
except Exception:

    @contextlib.contextmanager  # type: ignore[no-redef]
    def trace_span(name: str, attributes: dict[str, Any] | None = None, **kwargs: Any):
        yield None


logger = logging.getLogger(__name__)


class InterventionalCausalDiscoveryEngine(BaseCausalDiscoveryEngine):
    """Discovers true causal physical laws through active contrastive intervention."""

    def __init__(
        self,
        substrate: BlankBrainSubstrate,
        perception: DevelopmentalPerceptionAdapter,
        environment: BabyWorldEnvironment,
    ) -> None:
        super().__init__()
        self.substrate = substrate
        self.perception = perception
        self.env = environment

    def _compute_hypothesis_entropy(self) -> float:
        """Compute Shannon entropy across active non-falsified hypotheses."""
        active_confs = [
            h.confidence for h in self.hypotheses if not h.falsified and h.confidence > 0.0
        ]
        total = sum(active_confs)
        if not active_confs or total <= 0.0:
            return 0.0
        probs = [c / total for c in active_confs]
        return -sum(p * math.log2(p) for p in probs if p > 0.0)

    def observe_and_generate_hypotheses(
        self,
        observation: SensoryObservation,
        episodes_data: list[dict[str, Any]] | None = None,
    ) -> list[CausalHypothesis]:
        """Formulate candidate causal hypotheses based on initial observational correlations.

        Processes observational demonstration episodes to identify which features
        (color, shape, mass_sensation) correlate with movement.
        """
        # If no observational episodes are provided, query environment demonstrations
        if not episodes_data:
            if hasattr(self.env, "generate_observational_demonstrations"):
                episodes_data = self.env.generate_observational_demonstrations()
            else:
                episodes_data = []

        positive_episodes = [
            ep for ep in episodes_data if ep.get("moved") or ep.get("outcome") == "MOVES"
        ]
        negative_episodes = [
            ep for ep in episodes_data if not (ep.get("moved") or ep.get("outcome") == "MOVES")
        ]

        if positive_episodes:
            from collections import Counter

            # Collect candidate variables from observational demonstration features
            candidate_features = set()
            for ep in episodes_data:
                candidate_features.update(ep.get("features", {}).keys())

            candidate_features.discard("size_extent")
            candidate_features.discard("spatial_coordinates")

            preferred_order = ["color", "shape", "surface_friction", "mass_sensation"]
            ordered_features = [f for f in preferred_order if f in candidate_features] + [
                f for f in sorted(candidate_features) if f not in preferred_order
            ]

            self.hypotheses = []
            for feat in ordered_features:
                sample_val = next(
                    (
                        ep["features"][feat]
                        for ep in episodes_data
                        if feat in ep.get("features", {})
                    ),
                    None,
                )
                if isinstance(sample_val, str):
                    # Categorical variable (e.g. color, shape)
                    pos_vals = Counter(
                        ep["features"][feat]
                        for ep in positive_episodes
                        if feat in ep.get("features", {})
                    )
                    top_val = pos_vals.most_common(1)[0][0] if pos_vals else sample_val
                    self.hypotheses.append(
                        CausalHypothesis(
                            action=BabyActionType.PUSH,
                            variable=feat,
                            operator="==",
                            value=top_val,
                            consequence="MOVES",
                            confidence=0.5,
                        )
                    )
                elif isinstance(sample_val, (int, float)):
                    # Continuous numeric variable (e.g. mass_sensation, surface_friction)
                    import math

                    pos_nums = []
                    for ep in positive_episodes:
                        val = ep.get("features", {}).get(feat)
                        if val is not None:
                            try:
                                fval = float(val)
                                if math.isfinite(fval):
                                    pos_nums.append(fval)
                            except (ValueError, TypeError):
                                pass

                    neg_nums = []
                    for ep in negative_episodes:
                        val = ep.get("features", {}).get(feat)
                        if val is not None:
                            try:
                                fval = float(val)
                                if math.isfinite(fval):
                                    neg_nums.append(fval)
                            except (ValueError, TypeError):
                                pass

                    if pos_nums and neg_nums:
                        mean_pos = sum(pos_nums) / len(pos_nums)
                        mean_neg = sum(neg_nums) / len(neg_nums)
                        if abs(mean_pos - mean_neg) < 0.05:
                            # Feature exhibits zero observational variance across outcomes;
                            # cannot explain why some moved and others remained stationary.
                            continue
                        elif mean_pos < mean_neg:
                            operator = "<"
                            threshold = round((max(pos_nums) + min(neg_nums)) / 2.0, 2)
                        else:
                            operator = ">"
                            threshold = round((min(pos_nums) + max(neg_nums)) / 2.0, 2)
                    elif pos_nums:
                        operator = "<"
                        threshold = round(max(pos_nums) * 1.5, 2)
                    else:
                        operator = "<"
                        threshold = 5.0

                    self.hypotheses.append(
                        CausalHypothesis(
                            action=BabyActionType.PUSH,
                            variable=feat,
                            operator=operator,
                            value=threshold,
                            consequence="MOVES",
                            confidence=0.5,
                        )
                    )
        else:
            # Fallback when no demonstrations are present: sample from visual percepts
            pos_color = observation.vision[0]["color"] if observation.vision else "red"
            pos_shape = observation.vision[0]["shape"] if observation.vision else "ball"
            import statistics

            masses = [
                p.get("mass_sensation", 5.0) for p in observation.vision if "mass_sensation" in p
            ]
            mass_threshold = round(statistics.median(masses), 1) if masses else 5.0

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
                    value=mass_threshold,
                    consequence="MOVES",
                    confidence=0.5,
                ),
            ]

        emitter = DevelopmentalTelemetryEmitter.get_instance()
        for h in self.hypotheses:
            emitter.record_hypothesis_event("generated")
            self._record_belief_event(
                event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                hypothesis=h,
                prior_conf=0.0,
                post_conf=h.confidence,
                details={"formulated_rule": h.describe()},
            )

        entropy = self._compute_hypothesis_entropy()
        emitter.record_entropy("causal_beliefs", entropy)
        logger.info(
            "Causal hypotheses formulated: %d generated (entropy=%.4f)",
            len(self.hypotheses),
            entropy,
            extra={"hypotheses_count": len(self.hypotheses), "entropy": entropy},
        )

        return self.hypotheses

    @staticmethod
    def _predict_hypothesis(h: CausalHypothesis, features: dict[str, Any]) -> bool:
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

    def select_active_intervention(
        self,
        available_entity_ids: list[str],
        active_hypotheses: list[CausalHypothesis],
    ) -> tuple[str, CausalHypothesis]:
        """Select an optimal contrastive interventional probe.

        Active Inference / Epistemic Curiosity:
        Picks the entity whose intervention maximizes pairwise disagreement
        (expected entropy reduction) among currently non-falsified hypotheses.
        Features are evaluated strictly through perceptual observations rather than private simulation state.
        """
        active_hyps = [h for h in active_hypotheses if not h.falsified]
        if not active_hyps:
            active_hyps = [h for h in self.hypotheses if not h.falsified] or self.hypotheses

        if not active_hyps:
            default_hyp = CausalHypothesis(
                action=BabyActionType.PUSH,
                variable="none",
                operator="==",
                value="none",
                consequence="NONE",
            )
            return available_entity_ids[0] if available_entity_ids else "", default_hyp

        # Query perceptual observation map
        obs = self.env.get_sensory_observation()
        percept_map = {p["percept_id"]: p for p in obs.vision}

        ranked = self.rank_interventional_candidates(
            candidate_ids=available_entity_ids,
            active_hypotheses=active_hyps,
            feature_map=percept_map,
        )

        best_candidate = (
            ranked[0][0] if ranked else (available_entity_ids[0] if available_entity_ids else "")
        )

        return best_candidate or "", active_hyps[0]

    def execute_interventional_probe(
        self,
        target_id: str,
        action: BabyActionType = BabyActionType.PUSH,
    ) -> tuple[bool, dict[str, Any]]:
        """Execute a controlled interventional trial do(action, target_id)."""
        action_name = (action.value if hasattr(action, "value") else str(action)).lower()
        emitter = DevelopmentalTelemetryEmitter.get_instance()

        with trace_span(
            "developmental.causal_discovery.probe",
            attributes={"target_id": target_id, "action": action_name},
        ):
            with emitter.measure_latency("intervention"):
                self.interventions_count += 1
                prior_state_idx = self.env.save_state()

                # Position agent adjacent to target object for physical interaction
                pre_obs = self.env.get_sensory_observation()
                target_percept = next(
                    (p for p in pre_obs.vision if p["percept_id"] == target_id), None
                )
                if target_percept:
                    pos_x, pos_y = target_percept["spatial_coordinates"]
                    self.env.agent_position = Vector2D(pos_x - 0.2, pos_y)

                # Step physical environment with probe action
                obs, reward, done, consequences = self.env.step(action=action, target_id=target_id)
                did_move = consequences.get("moved", False)

                probe_result = {
                    "target_id": target_id,
                    "target_color": target_percept["color"] if target_percept else "",
                    "target_shape": target_percept["shape"] if target_percept else "",
                    "color": target_percept["color"] if target_percept else "",
                    "shape": target_percept["shape"] if target_percept else "",
                    "texture": target_percept.get("texture", "smooth")
                    if target_percept
                    else "smooth",
                    "mass_sensation": float(target_percept["mass_sensation"])
                    if target_percept
                    else 0.0,
                    "surface_friction": float(target_percept.get("surface_friction", 1.0))
                    if target_percept
                    else 1.0,
                    "static_threshold": float(target_percept.get("static_threshold", 0.0))
                    if target_percept
                    else 0.0,
                    "clearance_diameter": float(target_percept.get("clearance_diameter", 0.4))
                    if target_percept
                    else 0.4,
                    "did_move": did_move,
                    "displacement": consequences.get("displacement", 0.0),
                }

                # Update beliefs based on interventional evidence
                self._update_hypotheses_from_evidence(probe_result)

                # Restore world state so subsequent probes start from standardized conditions
                self.env.restore_state(prior_state_idx)

            res_tag = "moved" if did_move else "static"
            emitter.record_intervention(action_type=action_name, result=res_tag)
            logger.info(
                "Intervention probe executed on '%s' (action=%s => %s)",
                target_id,
                action_name,
                res_tag,
                extra={"target_id": target_id, "action": action_name, "result": res_tag},
            )

        return did_move, probe_result

    def _update_hypotheses_from_evidence(self, probe_result: dict[str, Any]) -> None:
        """Update posterior confidences and falsify invalidated hypotheses."""
        events = self.update_hypotheses_from_evidence(
            hypotheses=self.hypotheses,
            probe_result=probe_result,
            step_index=self.interventions_count,
        )
        emitter = DevelopmentalTelemetryEmitter.get_instance()
        for ev in events:
            self.belief_history.append(ev)
            if ev.event_type == BeliefTransitionType.HYPOTHESIS_FALSIFIED:
                emitter.record_hypothesis_event("falsified")
            elif ev.event_type == BeliefTransitionType.HYPOTHESIS_CONFIRMED:
                hyp = next(
                    (h for h in self.hypotheses if h.hypothesis_id == ev.hypothesis_id), None
                )
                if hyp is not None:
                    self._induce_causal_rule_into_substrate(hyp)

        entropy = self._compute_hypothesis_entropy()
        emitter.record_entropy("causal_beliefs", entropy)

    def _induce_causal_rule_into_substrate(self, confirmed_hyp: CausalHypothesis) -> None:
        """Synthesize confirmed hypothesis into a generalized HCIR causal rule."""
        rule, ev = self.induce_causal_rule(
            confirmed_hyp=confirmed_hyp,
            rule_store=self.substrate.causal_rules,
            step_index=self.interventions_count,
        )
        if not any(r.get("rule_id") == rule.get("rule_id") for r in self.confirmed_causal_rules):
            self.confirmed_causal_rules.append(rule)
        else:
            for cr in self.confirmed_causal_rules:
                if cr.get("rule_id") == rule.get("rule_id"):
                    cr["empirical_support_count"] = rule["empirical_support_count"]
                    cr["confidence"] = rule["confidence"]
                    break

        self.belief_history.append(ev)
        emitter = DevelopmentalTelemetryEmitter.get_instance()
        emitter.record_hypothesis_event("confirmed")
        emitter.record_concept_acquired("causal_rule")
        logger.info(
            "Causal rule induced into substrate: %s %s %s => %s",
            confirmed_hyp.variable,
            confirmed_hyp.operator,
            confirmed_hyp.value,
            confirmed_hyp.consequence,
            extra={"rule": rule},
        )

    def evaluate_generalization(
        self,
        test_objects: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired causal rules against held-out entities or worlds (Levels 2 & 3)."""
        return self.test_generalization(
            test_objects=test_objects,
            confirmed_rules=self.confirmed_causal_rules,
        )

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
