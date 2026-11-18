from __future__ import annotations

import pytest

from hbllm.hcir.grounding.actions.gateway import ActionValidationGateway
from hbllm.hcir.grounding.actions.renderers import KinematicRenderer
from hbllm.hcir.grounding.bus.observation_bus import ObservationBus
from hbllm.hcir.grounding.bus.policies import ChannelPolicy
from hbllm.hcir.grounding.causal.events import EventCandidate, EventType
from hbllm.hcir.grounding.causal.schema import SchemaInducer
from hbllm.hcir.grounding.committer.committer import HCIRCommitter
from hbllm.hcir.grounding.contracts.actions import SymbolicActionIntent
from hbllm.hcir.grounding.contracts.dialogue import GoalProposal
from hbllm.hcir.grounding.contracts.enums import (
    ActionLifecycleState,
    CausalStatus,
    DecisionOutcome,
    ModalitySource,
    TrafficClass,
)
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.contracts.references import (
    GroundedEntityRef,
    TrackletRef,
    UnresolvedRef,
)
from hbllm.hcir.grounding.epistemics.evaluator import EpistemicEvaluator
from hbllm.hcir.grounding.predictive.assimilator import (
    GroundingSurpriseAssimilator,
)
from hbllm.hcir.grounding.predictive.feedback import ActiveInferenceFeedback
from hbllm.hcir.grounding.predictive.predictor import (
    ForwardWorldPredictor,
)
from hbllm.hcir.grounding.relations.relation_learner import RelationLearner
from hbllm.hcir.grounding.tracking.tracker import ThreeTierEntityTracker
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.nodes import PhysicalEntityNode
from hbllm.hcir.transactions import TransactionStatus
from hbllm.hcir.workspace import HCIRWorkspaceState


class TestSGECrossDomainIntegration:
    """Comprehensive cross-domain end-to-end integration test suite for SGE (SGE-10).

    Validates unified symbol grounding across:
    1. Grid / ARC-AGI spatial reasoning
    2. Robotics kinematic manipulation
    3. Ambiguous natural language reference resolution
    4. Pearl's causal ladder intervention discovery
    5. Active inference surprise recovery & belief revision
    """

    @pytest.fixture
    def workspace(self) -> HCIRWorkspaceState:
        return HCIRWorkspaceState()

    @pytest.fixture
    def tx_manager(self, workspace: HCIRWorkspaceState) -> TransactionManager:
        return TransactionManager(workspace)

    @pytest.fixture
    def committer(self, tx_manager: TransactionManager) -> HCIRCommitter:
        return HCIRCommitter(tx_manager)

    def test_domain_1_grid_spatial_reasoning(
        self,
        committer: HCIRCommitter,
        workspace: HCIRWorkspaceState,
    ) -> None:
        """Domain 1: Discrete 2D grid perception, topology induction, and transaction commit."""
        # 1. Observation Bus Ingestion
        bus = ObservationBus()
        ch = bus.register_channel(
            "grid_channel", ChannelPolicy.default_for(TrafficClass.CLASS_DISCRETE)
        )

        envelope = ObservationEnvelope(
            observation_id="obs_grid_01",
            source_modality=ModalitySource.VISION,
            source_channel_id="grid_channel",
            traffic_class=TrafficClass.CLASS_DISCRETE,
            sequence_number=1,
            wall_timestamp=100.0,
            monotonic_timestamp=1000.0,
            raw_payload_ref="blob://grid_01",
            metadata={"grid_width": 10, "grid_height": 10},
        )
        ch.ingest(envelope)
        assert ch.metrics.total_ingested == 1

        # 2. Tracking: Tracklet -> EntityCandidate
        tracker = ThreeTierEntityTracker(spatial_distance_threshold=0.5)
        tracker.update_tracklet(
            TrackletRef("trk_cam1_1"),
            "cam1",
            (0.0, 0.0),
            metadata={"color": 2, "bbox": (1.0, 1.0, 2.0, 2.0)},
        )
        tracker.update_tracklet(
            TrackletRef("trk_cam1_2"),
            "cam1",
            (5.0, 5.0),
            metadata={"color": 3, "bbox": (1.0, 3.0, 2.0, 2.0)},
        )
        cand1 = tracker.auto_associate(TrackletRef("trk_cam1_1"), (0.0, 0.0))
        cand2 = tracker.auto_associate(TrackletRef("trk_cam1_2"), (5.0, 5.0))
        assert cand1 is not None and cand2 is not None

        cands = tracker.get_active_candidates()
        assert len(cands) == 2

        # 3. Dynamic Relation Induction
        learner = RelationLearner()
        states = {
            cands[0].candidate_ref.id: {"position": (1.0, 1.0), "bbox": (1.0, 1.0, 2.0, 2.0)},
            cands[1].candidate_ref.id: {"position": (1.0, 3.0), "bbox": (1.0, 3.0, 2.0, 2.0)},
        }
        learner.observe_entity_states(states, timestamp=100.0)
        claims, ev_records = learner.propose_claims()
        assert len(claims) >= 1

        # 4. Epistemic Evaluation
        evaluator = EpistemicEvaluator(acceptance_threshold=0.65)
        decision, _ = evaluator.evaluate_claim(claims[0], ev_records)
        assert decision.outcome == DecisionOutcome.ACCEPT

        # 5. Authoritative Commit into CognitiveGraph via TransactionManager
        tx = committer.commit_decision(decision, claims[0])
        assert tx.status == TransactionStatus.COMMITTED
        assert workspace.graph.has_node(claims[0].subject_ref.id)
        assert workspace.graph.has_node(claims[0].object_ref.id)  # type: ignore[union-attr]

    def test_domain_2_robotics_kinematic_manipulation(self) -> None:
        """Domain 2: Kinematic rendering, gateway validation, dispatch, and outcome verification."""
        intent = SymbolicActionIntent(
            intent_id="intent_pick_box",
            action_type="GRASP",
            target_ref=GroundedEntityRef("e_box_42"),
            preconditions=("gripper_empty == True",),
            expected_outcomes=("grasped == True",),
        )

        renderer = KinematicRenderer()
        payload = renderer.render(
            intent,
            context={"current_position": (0.0, 0.0), "target_position": (2.0, 0.5)},
        )

        gateway = ActionValidationGateway()
        world_state = {
            "tracked_entities": {"e_box_42": {"confidence": 0.95, "position": (2.0, 0.5)}},
            "gripper_empty": True,
        }

        # Validate
        val = gateway.validate_action(intent, payload, world_state)
        assert val.is_authorized is True

        # Dispatch
        record = gateway.dispatch_action(intent, payload, val)
        assert record.state == ActionLifecycleState.DISPATCHED

        # Report Execution
        rep = gateway.report_execution(record, success=True)
        assert rep.state == ActionLifecycleState.EXECUTION_REPORTED

        # Outcome Verification
        post_obs = {"e_box_42": {"grasped": True}}
        verified = gateway.verify_outcome(rep, intent, post_obs)
        assert verified.state == ActionLifecycleState.COMPLETED
        assert verified.outcome_verified is True

    def test_domain_3_ambiguous_language_reference_resolution(
        self,
        committer: HCIRCommitter,
        workspace: HCIRWorkspaceState,
    ) -> None:
        """Domain 3: Natural language request with unresolved reference resolved via spatial relations."""
        # Setup existing world state: Sofa at (1.0, 1.0, 2.0, 2.0), Keys beside at (3.5, 1.5, 0.2, 0.1)
        sofa_node = PhysicalEntityNode(id="e_sofa", entity_name="sofa")
        keys_node = PhysicalEntityNode(id="e_keys", entity_name="keys")
        workspace.graph.add_node(sofa_node)
        workspace.graph.add_node(keys_node)

        # Ambiguous user request: "Bring me the thing beside the sofa"
        unresolved_target = UnresolvedRef(
            placeholder_id="target_thing",
            constraints={"spatial_relation": "NEAR", "reference_entity": "e_sofa"},
        )

        proposal = GoalProposal(
            proposal_id="goal_bring_1",
            goal_intent="BRING",
            target_entity=unresolved_target,
            priority=0.8,
        )
        assert proposal.goal_intent == "BRING"

        # Spatial dyad calculation resolves candidate beside sofa
        learner = RelationLearner(proximity_threshold=2.0)
        states = {
            "e_sofa": {"position": (2.0, 2.0), "bbox": (1.0, 1.0, 2.0, 2.0)},
            "e_keys": {"position": (3.5, 1.5), "bbox": (3.5, 1.5, 0.2, 0.1)},
        }
        learner.observe_entity_states(states, timestamp=10.0)
        claims, _ = learner.propose_claims()

        # Find near claim
        near_claims = [
            c
            for c in claims
            if c.predicate_ref == "NEAR" and c.object_ref == GroundedEntityRef("e_sofa")
        ]
        assert len(near_claims) >= 1
        resolved_entity_ref = near_claims[0].subject_ref

        # Binding verified entity
        assert resolved_entity_ref == GroundedEntityRef("e_keys")

    def test_domain_4_pearl_causal_ladder_intervention(self) -> None:
        """Domain 4: Moving from observational correlation to interventional cause."""
        inducer = SchemaInducer()

        # 1. Observation: User presses button, elevator arrives
        evt_arrive = EventCandidate(
            event_id="evt_arr",
            event_type=EventType.STATE_CHANGE,
            primary_entity_ref=GroundedEntityRef("e_elevator"),
            post_state={"arrived": True},
        )
        schema = inducer.record_observation(
            action_type="PRESS_CALL_BUTTON",
            target_role="elevator_button",
            precondition={"power": True},
            outcome_event=evt_arrive,
        )
        assert schema.causal_status == CausalStatus.OBSERVED_ASSOCIATION

        # 2. Advance to CAUSAL_HYPOTHESIS with targeted experiment
        target = GroundedEntityRef("e_btn_floor_3")
        exp = inducer.propose_intervention(schema.schema_id, target)
        assert schema.causal_status == CausalStatus.CAUSAL_HYPOTHESIS

        # 3. Perform interventions: 2 successful trials -> INTERVENTIONALLY_SUPPORTED
        inducer.record_intervention_outcome(exp.proposal_id, success=True)
        exp2 = inducer.propose_intervention(schema.schema_id, target)
        status = inducer.record_intervention_outcome(exp2.proposal_id, success=True)

        assert status == CausalStatus.INTERVENTIONALLY_SUPPORTED
        assert schema.is_actionable_cause is True

        # 4. Counterfactual validation -> CAUSALLY_IDENTIFIED
        inducer.mark_counterfactually_identified(schema.schema_id, validation_accuracy=0.95)
        assert schema.causal_status == CausalStatus.CAUSALLY_IDENTIFIED

    def test_domain_5_active_inference_surprise_and_belief_revision(self) -> None:
        """Domain 5: Active inference surprise alert triggers epistemic goal and degrades fragile hypothesis."""
        inducer = SchemaInducer()
        schema = inducer.record_observation(
            action_type="PUSH",
            target_role="wall",
            precondition={},
            outcome_event=EventCandidate(
                event_id="e1",
                event_type=EventType.MOTION_STARTED,
                primary_entity_ref=GroundedEntityRef("e_wall"),
                post_state={"speed": 1.0},
            ),
        )
        # Advance schema to interventionally supported so it is used in counterfactual rollouts
        schema.causal_status = CausalStatus.INTERVENTIONALLY_SUPPORTED
        schema.p_effect_given_do_action = 0.95

        predictor = ForwardWorldPredictor(schemas=[schema])
        # Predict wall will move
        preds = predictor.predict_next_state(
            {
                "e_wall": {
                    "ref": GroundedEntityRef("e_wall"),
                    "position": (0.0, 0.0),
                }
            },
            action_type="PUSH",
            target_entity_ref=GroundedEntityRef("e_wall"),
        )

        # Actual observation: wall remains completely stationary at (0.0, 0.0)
        actual_obs = {"e_wall": {"position": (0.0, 0.0), "speed": 0.0}}

        assimilator = GroundingSurpriseAssimilator(surprise_threshold=0.3)
        alerts = assimilator.evaluate_observation(preds, actual_obs)
        assert len(alerts) == 1
        assert alerts[0].surprise_score > 0.3

        # Active inference feedback processes surprise alert
        feedback = ActiveInferenceFeedback(schema_inducer=inducer)
        goal = feedback.process_surprise_alert(alerts[0])

        assert goal is not None
        assert goal.goal_intent == "INSPECT_ENTITY"
        assert goal.target_entity == GroundedEntityRef("e_wall")
