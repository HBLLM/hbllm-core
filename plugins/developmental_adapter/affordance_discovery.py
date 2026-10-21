"""Stage D4: Functional Affordance Discovery Engine.

Enables a blank-brain HCIR agent to discover object affordances
(ROLLABLE, SLIDABLE, GRASPABLE, CONTAINABLE) from sensorimotor
interaction without pre-programmed semantic labels.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.world.affordance_discovery import (
    AffordanceHypothesis,
    BaseAffordanceDiscoveryEngine,
)

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .perception import DevelopmentalPerceptionAdapter
from .types import (
    BabyActionType,
    BeliefTransitionType,
)

logger = logging.getLogger(__name__)


class AffordanceDiscoveryEngine(BaseAffordanceDiscoveryEngine):
    """Discovers grounded functional affordances through active sensorimotor probing."""

    def __init__(
        self,
        substrate: BlankBrainSubstrate,
        perception: DevelopmentalPerceptionAdapter,
        env: BabyWorldEnvironment,
    ) -> None:
        super().__init__()
        self.substrate = substrate
        self.perception = perception
        self.env = env

    def observe_and_generate_hypotheses(self) -> list[AffordanceHypothesis]:
        """Formulate candidate affordance hypotheses across observed shapes.

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

        return self.generate_hypotheses(shapes, actions_to_test)

    def discover_affordances(
        self,
        max_interventions: int = 25,
    ) -> dict[str, list[str]]:
        """Run active interventional loop to falsify or confirm candidate affordances."""
        self.observe_and_generate_hypotheses()

        obs = self.env.get_sensory_observation()
        shape_to_entities: dict[str, list[str]] = {}
        for p in obs.vision:
            shape_to_entities.setdefault(p["shape"], []).append(p["percept_id"])

        interventions_this_run = 0
        for hyp in list(self.hypotheses):
            if interventions_this_run >= max_interventions:
                break
            if hyp.falsified or hyp.confirmed:
                continue

            # Pick an entity of this shape to test
            candidate_ids = shape_to_entities.get(hyp.entity_shape, [])
            if not candidate_ids:
                continue

            target_id = candidate_ids[0]
            interventions_this_run += 1
            self.interventions_count += 1
            prior_state = self.env.save_state()

            # Execute test action
            _, _, _, consequences = self.env.step(hyp.action, target_id=target_id)

            success = False
            if hyp.action == BabyActionType.ROLL:
                success = consequences.get("rolled", False)
            elif hyp.action == BabyActionType.PUSH:
                success = consequences.get("moved", False)
            elif hyp.action == BabyActionType.GRASP:
                success = consequences.get("grasped", False)

            self.update_affordance_from_evidence(
                hypothesis=hyp,
                success=success,
                target_id=target_id,
                step_index=self.interventions_count,
                target_store=self.substrate.affordances,
            )

            self.env.restore_state(prior_state)

        return self.confirmed_affordances

    def evaluate_novel_entity_transfer(
        self,
        held_out_entities: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired affordance transfer on held-out unseen shapes/objects."""
        return self.test_novel_entity_transfer(
            held_out_entities=held_out_entities,
            confirmed_affordances=self.confirmed_affordances,
        )

    def _record_event(
        self,
        event_type: BeliefTransitionType,
        hyp: AffordanceHypothesis,
        prior_conf: float,
        post_conf: float,
    ) -> None:
        self._record_belief_event(
            event_type=event_type,
            hyp=hyp,
            prior_conf=prior_conf,
            post_conf=post_conf,
            step_index=self.interventions_count,
        )
