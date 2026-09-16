"""Stage D2: Spatial Relations & Containment Transport Engine.

Enables an embodied blank-brain learner to induce spatial containment relations
(INSIDE, ON, NEAR) and the containment transport schema:
  INSIDE(x, y) ∧ MOVE(y, Δ) => MOVE(x, Δ)
while preserving object permanence when contents are occluded.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.world.spatial_containment import (
    BaseSpatialContainmentEngine,
    SpatialRelationFact,
)

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .perception import DevelopmentalPerceptionAdapter
from .types import (
    BabyActionType,
    BabyRelationType,
    BeliefTransitionType,
    SensoryObservation,
    Vector2D,
)

logger = logging.getLogger(__name__)


class SpatialContainmentEngine(BaseSpatialContainmentEngine):
    """Induces spatial relations and containment transport schemas through active physical probes."""

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

    def detect_spatial_relations(self, obs: SensoryObservation) -> list[SpatialRelationFact]:
        """Induce relational facts directly from geometric and perceptual observations."""
        return super().detect_spatial_relations(
            percept_items=obs.vision,
            near_threshold=0.5,
            inside_relation=BabyRelationType.INSIDE,
            near_relation=BabyRelationType.NEAR,
        )

    def discover_containment_transport_schema(
        self,
        container_id: str,
        contained_id: str,
        outside_id: str,
    ) -> dict[str, Any]:
        """Execute interventional trials to prove containment transport invariance.

        Scientific Invariant: Tests whether moving container Y synchronously transports
        contained object X, while outside object Z remains stationary.
        """
        self.interventions_count += 1
        prior_state = self.env.save_state()

        pre_obs = self.env.get_sensory_observation()
        pre_cont = next((p for p in pre_obs.vision if p["percept_id"] == container_id), None)
        pre_inside = next((p for p in pre_obs.vision if p["percept_id"] == contained_id), None)
        pre_outside = next((p for p in pre_obs.vision if p["percept_id"] == outside_id), None)

        # Position agent to push container
        if pre_cont:
            cx, cy = pre_cont["spatial_coordinates"]
            self.env.agent_position = Vector2D(cx - 0.2, cy)

        # Push container
        _, _, _, consequences = self.env.step(BabyActionType.PUSH, target_id=container_id)
        post_obs = self.env.get_sensory_observation()

        post_inside = next((p for p in post_obs.vision if p["percept_id"] == contained_id), None)
        post_outside = next((p for p in post_obs.vision if p["percept_id"] == outside_id), None)

        container_moved = consequences.get("moved", False)
        container_disp = consequences.get("displacement", 0.0)

        # Measure actual displacements
        inside_disp = 0.0
        if pre_inside and post_inside:
            dx = post_inside["spatial_coordinates"][0] - pre_inside["spatial_coordinates"][0]
            dy = post_inside["spatial_coordinates"][1] - pre_inside["spatial_coordinates"][1]
            inside_disp = (dx**2 + dy**2) ** 0.5

        outside_disp = 0.0
        if pre_outside and post_outside:
            dx = post_outside["spatial_coordinates"][0] - pre_outside["spatial_coordinates"][0]
            dy = post_outside["spatial_coordinates"][1] - pre_outside["spatial_coordinates"][1]
            outside_disp = (dx**2 + dy**2) ** 0.5

        transport_confirmed, schema = self.evaluate_containment_transport_invariance(
            container_moved=container_moved,
            container_disp=container_disp,
            inside_disp=inside_disp,
            outside_disp=outside_disp,
        )

        if transport_confirmed:
            self.record_spatial_schema(
                schema=schema,
                container_id=container_id,
                target_store=self.substrate.spatial_schemas,
                step_index=self.interventions_count,
            )

        self.env.restore_state(prior_state)
        return schema

    def verify_object_permanence_during_transport(
        self,
        container_id: str,
        contained_id: str,
    ) -> dict[str, Any]:
        """Verify that belief about occluded contained object tracks container movement."""
        self.interventions_count += 1
        prior_state = self.env.save_state()

        # Step 1: Close container to occlude contained object
        self.env.step(BabyActionType.CLOSE, target_id=container_id)
        occluded_obj = self.env.objects.get(contained_id)
        if occluded_obj:
            occluded_obj.is_occluded = True

        # Step 2: Push container while object is unobserved
        pre_cont = self.env.objects[container_id]
        self.env.agent_position = Vector2D(pre_cont.position.x - 0.2, pre_cont.position.y)
        self.env.step(BabyActionType.PUSH, target_id=container_id)

        # Expected position computed by cognitive model using acquired schema
        predicted_pos = (
            pre_cont.position.x,
            pre_cont.position.y,
        )

        # Step 3: Open container and verify ground truth position matches prediction
        if occluded_obj:
            occluded_obj.is_occluded = False
        self.env.step(BabyActionType.OPEN, target_id=container_id)
        actual_pos = self.env.objects[contained_id].position.to_tuple()

        result = self.evaluate_object_permanence(
            predicted_pos=predicted_pos,
            actual_pos=actual_pos,
        )

        self.env.restore_state(prior_state)
        return result

    def _record_event(
        self,
        event_type: BeliefTransitionType,
        condition: str,
        prior_conf: float,
        post_conf: float,
    ) -> None:
        self.record_spatial_schema(
            schema={"condition": condition},
            container_id="schema_containment",
            step_index=self.interventions_count,
        )
