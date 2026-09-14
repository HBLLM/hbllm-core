"""Stage D4: Functional Affordance Discovery Engine.

Enables a blank-brain HCIR agent to discover object affordances
(ROLLABLE, SLIDABLE, GRASPABLE, CONTAINABLE) from sensorimotor
interaction without pre-programmed semantic labels.
"""

from __future__ import annotations

import logging
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .perception import DevelopmentalPerceptionAdapter
from .types import (
    AffordanceHypothesis,
    BabyActionType,
    BeliefTransitionEvent,
    BeliefTransitionType,
)

logger = logging.getLogger(__name__)


class AffordanceDiscoveryEngine:
    """Discovers grounded functional affordances through active sensorimotor probing."""

    def __init__(
        self,
        substrate: BlankBrainSubstrate,
        perception: DevelopmentalPerceptionAdapter,
        env: BabyWorldEnvironment,
    ) -> None:
        self.substrate = substrate
        self.perception = perception
        self.env = env

        self.hypotheses: list[AffordanceHypothesis] = []
        self.confirmed_affordances: dict[str, list[str]] = {}  # shape -> list of affordances
        self.belief_history: list[BeliefTransitionEvent] = []
        self.interventions_count: int = 0

    def observe_and_generate_hypotheses(self) -> list[AffordanceHypothesis]:
        """Formulate initial candidate affordance hypotheses across observed shapes.

        Scientific Invariant: Generates hypotheses across all actions and observed shapes
        without knowing in advance which shapes afford which actions.
        """
        obs = self.env.get_sensory_observation()
        shapes = sorted(list({p["shape"] for p in obs.vision}))

        actions_to_test = [
            (BabyActionType.ROLL, "ROLLABLE"),
            (BabyActionType.PUSH, "SLIDABLE"),
            (BabyActionType.GRASP, "GRASPABLE"),
        ]

        self.hypotheses = []
        for shape in shapes:
            for action, label in actions_to_test:
                hyp = AffordanceHypothesis(
                    action=action,
                    entity_shape=shape,
                    affordance_label=label,
                    confidence=0.5,
                )
                self.hypotheses.append(hyp)
                self._record_event(
                    event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                    hyp=hyp,
                    prior_conf=0.0,
                    post_conf=0.5,
                )

        return self.hypotheses

    def discover_affordances(
        self,
        max_interventions: int = 25,
    ) -> dict[str, list[str]]:
        """Run active interventional loop to falsify or confirm candidate affordances."""
        if not self.hypotheses:
            self.observe_and_generate_hypotheses()

        obs = self.env.get_sensory_observation()
        shape_to_entities: dict[str, list[str]] = {}
        for p in obs.vision:
            shape_to_entities.setdefault(p["shape"], []).append(p["percept_id"])

        for hyp in list(self.hypotheses):
            if self.interventions_count >= max_interventions:
                break
            if hyp.falsified or hyp.confirmed:
                continue

            # Pick an entity of this shape to test
            candidate_ids = shape_to_entities.get(hyp.entity_shape, [])
            if not candidate_ids:
                continue

            target_id = candidate_ids[0]
            self.interventions_count += 1
            prior_state = self.env.save_state()

            # Execute test action
            _, _, _, consequences = self.env.step(hyp.action, target_id=target_id)
            hyp.interventions_tested += 1
            prior_conf = hyp.confidence

            success = False
            if hyp.action == BabyActionType.ROLL:
                success = consequences.get("rolled", False)
            elif hyp.action == BabyActionType.PUSH:
                success = consequences.get("moved", False)
            elif hyp.action == BabyActionType.GRASP:
                success = consequences.get("grasped", False)

            if success:
                hyp.confirmed = True
                hyp.confidence = 1.0
                hyp.supporting_episodes.append(target_id)
                self._record_event(
                    event_type=BeliefTransitionType.HYPOTHESIS_CONFIRMED,
                    hyp=hyp,
                    prior_conf=prior_conf,
                    post_conf=1.0,
                )
                # Register confirmed affordance in substrate
                self.confirmed_affordances.setdefault(hyp.entity_shape, []).append(
                    hyp.affordance_label
                )
                self.substrate.affordances.setdefault(hyp.entity_shape, []).append(
                    hyp.affordance_label
                )
            else:
                # Falsified by empirical counterexample
                hyp.falsified = True
                hyp.confidence = 0.0
                hyp.counterexamples.append(target_id)
                self._record_event(
                    event_type=BeliefTransitionType.HYPOTHESIS_FALSIFIED,
                    hyp=hyp,
                    prior_conf=prior_conf,
                    post_conf=0.0,
                )

            self.env.restore_state(prior_state)

        return self.confirmed_affordances

    def evaluate_novel_entity_transfer(
        self,
        held_out_entities: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired affordance transfer on held-out unseen shapes/objects."""
        if not self.confirmed_affordances:
            return 0.0, []

        correct = 0
        total = 0
        eval_records: list[dict[str, Any]] = []

        for entity in held_out_entities:
            shape = entity.get("shape", "")
            base_shape = entity.get("base_shape", shape)
            expected_affordances = set(entity.get("ground_truth_affordances", []))

            # Retrieve predicted affordances based on acquired shape schema
            predicted_affordances = set(self.confirmed_affordances.get(base_shape, []))

            # Check precision & recall match
            is_match = predicted_affordances == expected_affordances
            if is_match:
                correct += 1
            total += 1

            eval_records.append(
                {
                    "entity_id": entity.get("id"),
                    "shape": shape,
                    "predicted_affordances": sorted(list(predicted_affordances)),
                    "expected_affordances": sorted(list(expected_affordances)),
                    "correct": is_match,
                }
            )

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, eval_records

    def _record_event(
        self,
        event_type: BeliefTransitionType,
        hyp: AffordanceHypothesis,
        prior_conf: float,
        post_conf: float,
    ) -> None:
        self.belief_history.append(
            BeliefTransitionEvent(
                event_type=event_type,
                step_index=self.interventions_count,
                hypothesis_id=hyp.hypothesis_id,
                variable=hyp.entity_shape,
                condition=hyp.describe(),
                prior_confidence=prior_conf,
                posterior_confidence=post_conf,
                is_falsified=hyp.falsified,
            )
        )
