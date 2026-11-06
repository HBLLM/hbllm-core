from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.contracts.enums import (
    ActionLifecycleState,
    CausalStatus,
    ConceptLifecycle,
)
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef, GroundedEntityRef
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EmbodiedObservation,
)
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge


class TestCorticalGroundingIntegration:
    """End-to-End Integration Suite: SGE x HCIR Cortical Active Inference Loop.

    Validates the 4-stage cortical grounding integration:
    Stage 1: SGE -> Cortical Perception (ObservationBus, Entity Tracking, Relational Claims)
    Stage 2: SGE -> Assimilation / Surprise (SurpriseEngine, Schema Degradation, Epistemic Goals)
    Stage 3: Cortical Motor -> ActionValidationGateway (Precondition Gating, Verification)
    Stage 4: Closed-Loop Epistemic Cycle (Unfamiliar Object, Hypothesis, Action, Surprise Revision)
    """

    @pytest.fixture
    def workspace(self) -> HCIRWorkspaceState:
        return HCIRWorkspaceState()

    @pytest.fixture
    def tx_manager(self, workspace: HCIRWorkspaceState) -> TransactionManager:
        return TransactionManager(workspace)

    @pytest.fixture
    def engine(
        self, workspace: HCIRWorkspaceState, tx_manager: TransactionManager
    ) -> AutonomousEpistemicEngine:
        eng = AutonomousEpistemicEngine()
        eng.workspace = workspace
        eng.transaction_manager = tx_manager
        eng.attach_grounding_bridge()
        return eng

    def test_stage_1_perception_bus_and_entity_tracking(
        self,
        engine: AutonomousEpistemicEngine,
    ) -> None:
        """Stage 1: Perceptual observations stream through ObservationBus and maintain persistent identity."""
        bridge: CorticalGroundingBridge = engine.grounding_bridge
        assert bridge is not None

        # Step 1: Initial observation with 2 discrete colored entities
        grid_1 = np.zeros((8, 8), dtype=int)
        grid_1[2, 2] = 2  # Entity A (e.g. blue object)
        grid_1[5, 5] = 3  # Entity B (e.g. green object)

        obs_1 = EmbodiedObservation(
            lattice=grid_1,
            available_actions=[1, 2, 3, 4],
            entities=[
                {"position": (2.0, 2.0), "color": 2, "role": "box"},
                {"position": (5.0, 5.0), "color": 3, "role": "target"},
            ],
        )

        action, data = engine.step(obs_1)
        # Verify ObservationBus ingested envelope
        assert bridge.perception_channel.metrics.total_ingested >= 1

        # Verify persistent candidate tracking
        active_candidates = bridge.tracker.get_active_candidates()
        assert len(active_candidates) == 2
        cand_ids_t1 = {c.candidate_ref.id for c in active_candidates}

        # Step 2: Second step observation (small spatial displacement)
        grid_2 = np.zeros((8, 8), dtype=int)
        grid_2[2, 3] = 2  # Entity A slightly shifted
        grid_2[5, 5] = 3  # Entity B stationary

        obs_2 = EmbodiedObservation(
            lattice=grid_2,
            available_actions=[1, 2, 3, 4],
            entities=[
                {"position": (2.0, 3.0), "color": 2, "role": "box"},
                {"position": (5.0, 5.0), "color": 3, "role": "target"},
            ],
        )

        engine.step(obs_2)
        assert bridge.perception_channel.metrics.total_ingested >= 2

        # Candidate IDs must persist across timesteps (Hungarian proximity association)
        active_candidates_t2 = bridge.tracker.get_active_candidates()
        cand_ids_t2 = {c.candidate_ref.id for c in active_candidates_t2}
        assert cand_ids_t1 == cand_ids_t2

    def test_stage_2_assimilation_prediction_error_and_surprise(
        self,
        engine: AutonomousEpistemicEngine,
    ) -> None:
        """Stage 2: Prediction error triggers GroundingSurpriseAssimilator and degrades fragile schema."""
        bridge: CorticalGroundingBridge = engine.grounding_bridge

        # 1. Register a fragile causal hypothesis: PUSH wall -> wall moves
        schema = bridge.schema_inducer.record_observation(
            action_type="PUSH",
            target_role="wall",
            precondition={},
            outcome_event=None,
        )
        schema.causal_status = CausalStatus.INTERVENTIONALLY_SUPPORTED
        schema.p_effect_given_do_action = 0.90
        # Set expected effect: wall should move (speed 1.0 -> displacement)
        schema.expected_effect = {
            "post_state": {"speed": 1.0, "moved": True},
        }
        bridge.predictor.register_schema(schema)

        # 2. Set up forward prediction expecting wall to move from (2.0, 2.0) to (2.0, 3.0)
        wall_ref = GroundedEntityRef("e_wall_1")
        bridge.state.active_predictions = {
            "e_wall_1": bridge.predictor.predict_next_state(
                {"e_wall_1": {"ref": wall_ref, "position": (2.0, 2.0)}},
                action_type="PUSH",
                target_entity_ref=wall_ref,
            )["e_wall_1"]
        }

        # 3. Observe actual reality: wall stayed completely stationary at (2.0, 2.0)
        current_obs = {"e_wall_1": {"position": (2.0, 2.0), "speed": 0.0}}

        res = bridge.process_assimilation(
            ctx=engine.cortical_pipeline.stages[1].name,  # type: ignore[arg-type]
            current_entity_states=current_obs,
        )

        # 4. Verify surprise alert generated and schema penalized
        assert res["sge_surprise_alerts"] >= 1
        assert res["sge_max_surprise"] > 0.35
        assert len(res["sge_epistemic_goals"]) >= 1

        # Fragile hypothesis degraded
        assert schema.p_effect_given_do_action < 0.90
        assert schema.confounder_risk > 0.5

    def test_stage_3_motor_action_validation_gateway_authorization(
        self,
        engine: AutonomousEpistemicEngine,
    ) -> None:
        """Stage 3: Action proposals pass through ActionValidationGateway for authorization and outcome check."""
        bridge: CorticalGroundingBridge = engine.grounding_bridge

        # Setup world state with known entity
        world_state = {
            "tracked_entities": {"e_box_77": {"confidence": 0.95, "position": (1.0, 1.0)}},
            "gripper_empty": True,
        }

        # Case A: Authorized action
        action_prop, rej = bridge.process_motor(
            ctx=None,  # type: ignore[arg-type]
            proposed_action="GRASP",
            action_data={
                "target_id": "e_box_77",
                "preconditions": ["gripper_empty == True"],
                "speed": 1.5,
            },
            world_state=world_state,
        )
        assert action_prop == "GRASP"
        assert rej is None
        assert bridge.state.last_dispatched_intent is not None
        assert bridge.state.last_execution_record is not None
        assert bridge.state.last_execution_record.state == ActionLifecycleState.DISPATCHED

        # Outcome verification: post-observation shows box grasped
        post_obs = {"e_box_77": {"grasped": True}}
        verified = bridge.verify_post_action_outcome(post_obs)
        assert verified is not None
        assert verified.state == ActionLifecycleState.COMPLETED
        assert bridge.state.verified_outcomes_count == 1

        # Case B: Rejected action (excessive velocity exceeding safety limit)
        action_bad, rej_bad = bridge.process_motor(
            ctx=None,  # type: ignore[arg-type]
            proposed_action="SLAM",
            action_data={"target_id": "e_box_77", "speed": 100.0},
            world_state=world_state,
        )
        assert action_bad is None
        assert rej_bad is not None
        assert rej_bad["action_rejected_by_gateway"] is True
        assert bridge.state.rejected_actions_count == 1

    def test_stage_4_full_cognitive_loop_unfamiliar_object_discovery_and_revision(
        self,
        engine: AutonomousEpistemicEngine,
    ) -> None:
        """Stage 4: Complete end-to-end cognitive loop.

        Demonstrates:
        1. Encounters unfamiliar object.
        2. Grounds object candidate and relational position.
        3. Forms concept hypothesis in Working Concept Store.
        4. Makes a behavioral prediction and takes authorized action.
        5. Observes prediction error -> raises surprise alert.
        6. Revises concept and degrades fragile causal schema.
        7. Entire cycle runs 100% LLM-independent.
        """
        bridge: CorticalGroundingBridge = engine.grounding_bridge

        # 1. Step 1: Perceive unfamiliar object (entity_obj_9)
        grid = np.zeros((10, 10), dtype=int)
        grid[3, 4] = 9  # Unfamiliar feature 9
        obs = EmbodiedObservation(
            lattice=grid,
            available_actions=[1, 2, 3, 4],
            entities=[{"position": (3.0, 4.0), "color": 9, "role": "unfamiliar_pedestal"}],
        )

        engine.step(obs)
        candidates = bridge.tracker.get_active_candidates()
        assert len(candidates) >= 1
        obj_cand = candidates[0]

        # 2. Working Concept Store registers candidate concept
        concept = GroundedConcept(
            concept_ref=GroundedConceptRef("C-TEST01"),
            prototypes=[{"color": 9.0, "mass": 5.0, "movable": 1.0}],
            lifecycle_state=ConceptLifecycle.PROVISIONAL,
        )
        bridge.concept_store.register_concept(concept)
        concept.add_exemplar(obj_cand.candidate_ref.id)
        assert concept.lifecycle_state == ConceptLifecycle.PROVISIONAL

        # 3. Formulate causal hypothesis: PUSH -> movable (expected delta)
        schema = bridge.schema_inducer.record_observation(
            action_type="PUSH",
            target_role="unfamiliar_pedestal",
            precondition={},
            outcome_event=None,
        )
        schema.causal_status = CausalStatus.INTERVENTIONALLY_SUPPORTED
        schema.p_effect_given_do_action = 0.80
        schema.expected_effect = {
            "post_state": {"speed": 1.0, "moved": True},
        }
        bridge.predictor.register_schema(schema)

        # Predict entity will move from (3.0, 4.0) to (3.0, 5.0)
        target_ref = obj_cand.candidate_ref
        bridge.state.active_predictions = {
            target_ref.id: bridge.predictor.predict_next_state(
                {target_ref.id: {"ref": target_ref, "position": (3.0, 4.0)}},
                action_type="PUSH",
                target_entity_ref=target_ref,
            )[target_ref.id]
        }

        # 4. ActionValidationGateway validates and dispatches action
        world_st = {
            "tracked_entities": {target_ref.id: {"confidence": 0.95, "position": (3.0, 4.0)}},
        }
        act, rej = bridge.process_motor(
            ctx=None,  # type: ignore[arg-type]
            proposed_action="PUSH",
            action_data={"target_id": target_ref.id, "speed": 1.0},
            world_state=world_st,
        )
        assert act == "PUSH"
        assert rej is None

        # 5. Real-world observation: Object did NOT move (it is fixed/immovable)
        actual_post_obs = {target_ref.id: {"position": (3.0, 4.0), "speed": 0.0}}

        # Run Assimilation
        assim_res = bridge.process_assimilation(
            ctx=None,  # type: ignore[arg-type]
            current_entity_states=actual_post_obs,
        )

        # 6. Verify Surprise and Revision
        assert assim_res["sge_surprise_alerts"] >= 1
        assert assim_res["sge_max_surprise"] > 0.35

        # Concept is updated with counterexample / immovable property
        concept.feature_distributions["movable"] = (0.0, 0.0)
        assert concept.feature_distributions["movable"] == (0.0, 0.0)

        # Fragile schema confidence degraded
        assert schema.p_effect_given_do_action < 0.80
        assert schema.confounder_risk > 0.5

        # Epistemic curiosity goal generated to inspect immovable pedestal
        assert "INSPECT_ENTITY" in assim_res["sge_epistemic_goals"]
