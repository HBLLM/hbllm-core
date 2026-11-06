from __future__ import annotations

from hbllm.hcir.grounding.causal.events import (
    EventCandidate,
    EventDetector,
    EventType,
)
from hbllm.hcir.grounding.causal.schema import (
    InterventionExperimentProposal,
    SchemaInducer,
)
from hbllm.hcir.grounding.contracts.enums import CausalStatus
from hbllm.hcir.grounding.contracts.references import GroundedEntityRef
from hbllm.hcir.grounding.relations.dyads import SpatialDyad


class TestEventDetector:
    def test_motion_start_and_stop(self) -> None:
        detector = EventDetector(speed_threshold=0.05)

        # Frame 1: entity at rest
        evts1 = detector.detect_events(
            {"e_cart": {"speed": 0.0, "position": (0.0, 0.0)}}, timestamp=10.0
        )
        assert len(evts1) == 0

        # Frame 2: entity begins moving
        evts2 = detector.detect_events(
            {"e_cart": {"speed": 0.2, "position": (0.2, 0.0)}}, timestamp=11.0
        )
        assert len(evts2) == 1
        assert evts2[0].event_type == EventType.MOTION_STARTED
        assert evts2[0].primary_entity_ref == GroundedEntityRef("e_cart")

        # Frame 3: entity stops
        evts3 = detector.detect_events(
            {"e_cart": {"speed": 0.0, "position": (0.5, 0.0)}}, timestamp=12.0
        )
        assert len(evts3) == 1
        assert evts3[0].event_type == EventType.MOTION_STOPPED

    def test_discrete_property_change(self) -> None:
        detector = EventDetector()

        detector.detect_events({"e_door": {"open": False}}, timestamp=10.0)
        evts = detector.detect_events({"e_door": {"open": True}}, timestamp=11.0)

        assert len(evts) == 1
        assert evts[0].event_type == EventType.STATE_CHANGE
        assert evts[0].pre_state == {"open": False}
        assert evts[0].post_state == {"open": True}

    def test_dyadic_contact_events(self) -> None:
        detector = EventDetector()

        dyad_no_contact = SpatialDyad(
            subject_id="e_avatar",
            object_id="e_button",
            distance=0.5,
            dx=0.5,
            dy=0.0,
            contact=False,
            timestamp=10.0,
        )
        detector.detect_events({}, [dyad_no_contact], timestamp=10.0)

        dyad_contact = SpatialDyad(
            subject_id="e_avatar",
            object_id="e_button",
            distance=0.0,
            dx=0.0,
            dy=0.0,
            contact=True,
            timestamp=11.0,
        )
        evts = detector.detect_events({}, [dyad_contact], timestamp=11.0)

        assert len(evts) == 1
        assert evts[0].event_type == EventType.CONTACT_INITIATED
        assert evts[0].primary_entity_ref == GroundedEntityRef("e_avatar")
        assert evts[0].secondary_entity_ref == GroundedEntityRef("e_button")


class TestSchemaInducerPearlHierarchy:
    def test_observational_to_interventional_lifecycle(self) -> None:
        inducer = SchemaInducer(
            min_association_prob=0.6,
            min_intervention_prob=0.75,
            min_intervention_trials=2,
        )

        outcome_event = EventCandidate(
            event_id="evt_motion_1",
            event_type=EventType.MOTION_STARTED,
            primary_entity_ref=GroundedEntityRef("e_box_1"),
            post_state={"speed": 0.5},
        )

        # 1. Rung 1: Observation
        schema = inducer.record_observation(
            action_type="PUSH",
            target_role="box",
            precondition={"contact": True},
            outcome_event=outcome_event,
        )
        assert schema.causal_status == CausalStatus.OBSERVED_ASSOCIATION
        assert schema.is_actionable_cause is False

        # Recurring observations elevate to PREDICTIVE_NON_CAUSAL
        inducer.record_observation("PUSH", "box", {"contact": True}, outcome_event=outcome_event)
        inducer.record_observation("PUSH", "box", {"contact": True}, outcome_event=outcome_event)
        assert schema.causal_status == CausalStatus.PREDICTIVE_NON_CAUSAL
        assert schema.p_effect_given_action == 1.0

        # 2. Formulate Intervention Proposal -> CAUSAL_HYPOTHESIS
        target_ref = GroundedEntityRef("e_box_2")
        proposal = inducer.propose_intervention(schema.schema_id, target_ref)
        assert isinstance(proposal, InterventionExperimentProposal)
        assert proposal.schema_id == schema.schema_id
        assert schema.causal_status == CausalStatus.CAUSAL_HYPOTHESIS

        # 3. Rung 2: Intervention measured
        # Trial 1: success
        status1 = inducer.record_intervention_outcome(proposal.proposal_id, success=True)
        assert status1 == CausalStatus.CAUSAL_HYPOTHESIS  # Needs 2 trials

        # Trial 2: success -> INTERVENTIONALLY_SUPPORTED
        proposal2 = inducer.propose_intervention(schema.schema_id, target_ref)
        status2 = inducer.record_intervention_outcome(proposal2.proposal_id, success=True)
        assert status2 == CausalStatus.INTERVENTIONALLY_SUPPORTED
        assert schema.is_actionable_cause is True
        assert schema.p_effect_given_do_action == 1.0

        # 4. Rung 3: Counterfactual Identification
        inducer.mark_counterfactually_identified(schema.schema_id, validation_accuracy=0.92)
        assert schema.causal_status == CausalStatus.CAUSALLY_IDENTIFIED

    def test_spurious_confounder_detection(self) -> None:
        inducer = SchemaInducer(min_intervention_trials=2)

        outcome_event = EventCandidate(
            event_id="evt_light_1",
            event_type=EventType.STATE_CHANGE,
            primary_entity_ref=GroundedEntityRef("e_light"),
            post_state={"illuminated": True},
        )

        # Co-occurrence in observation: Rooster crowing -> Light on (100% correlation)
        for _ in range(3):
            inducer.record_observation("ROOSTER_CROW", "sky", {}, outcome_event=outcome_event)

        schema = inducer.all_schemas()[0]
        assert schema.p_effect_given_action == 1.0
        assert schema.causal_status == CausalStatus.PREDICTIVE_NON_CAUSAL

        # Interventional test: do(ROOSTER_CROW) at night -> light does NOT turn on
        p1 = inducer.propose_intervention(schema.schema_id, GroundedEntityRef("e_rooster"))
        inducer.record_intervention_outcome(p1.proposal_id, success=False)

        p2 = inducer.propose_intervention(schema.schema_id, GroundedEntityRef("e_rooster"))
        status = inducer.record_intervention_outcome(p2.proposal_id, success=False)

        # High observational correlation, but zero interventional effect
        # => Detected as spurious confounder!
        assert status == CausalStatus.PREDICTIVE_NON_CAUSAL
        assert schema.confounder_risk >= 0.9
        assert schema.is_actionable_cause is False
