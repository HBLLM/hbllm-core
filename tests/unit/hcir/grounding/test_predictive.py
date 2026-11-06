from __future__ import annotations

import pytest

from hbllm.hcir.grounding.causal.schema import CausalSchema, SchemaInducer
from hbllm.hcir.grounding.contracts.dialogue import GoalProposal
from hbllm.hcir.grounding.contracts.enums import CausalStatus
from hbllm.hcir.grounding.contracts.references import GroundedEntityRef
from hbllm.hcir.grounding.predictive.assimilator import (
    GroundingSurpriseAlert,
    GroundingSurpriseAssimilator,
)
from hbllm.hcir.grounding.predictive.feedback import ActiveInferenceFeedback
from hbllm.hcir.grounding.predictive.predictor import (
    ForwardWorldPredictor,
    PredictedEntityState,
)


class TestForwardWorldPredictor:
    def test_kinematic_extrapolation(self) -> None:
        predictor = ForwardWorldPredictor()
        entities = {
            "e_ball": {
                "ref": GroundedEntityRef("e_ball"),
                "position": (0.0, 0.0),
                "velocity": (1.0, 2.0),
            }
        }

        preds = predictor.predict_next_state(entities, dt=2.0)
        assert "e_ball" in preds
        pred = preds["e_ball"]
        assert pred.predicted_position == pytest.approx((2.0, 4.0))
        assert pred.predicted_velocity == (1.0, 2.0)

    def test_causal_schema_application(self) -> None:
        schema = CausalSchema(
            schema_id="schema_door_open",
            action_type="OPEN",
            target_role="door",
            precondition={"locked": False},
            expected_effect={"post_state": {"open": True}},
            causal_status=CausalStatus.INTERVENTIONALLY_SUPPORTED,
            p_effect_given_do_action=0.95,
        )

        predictor = ForwardWorldPredictor(schemas=[schema])
        entities = {
            "e_door": {
                "ref": GroundedEntityRef("e_door"),
                "position": (5.0, 0.0),
                "locked": False,
                "open": False,
            }
        }

        preds = predictor.predict_next_state(
            entities,
            action_type="OPEN",
            target_entity_ref=GroundedEntityRef("e_door"),
        )

        assert "e_door" in preds
        pred = preds["e_door"]
        assert pred.applied_schema_id == "schema_door_open"
        assert pred.predicted_properties.get("open") is True
        assert pred.confidence == 0.95


class TestGroundingSurpriseAssimilator:
    def test_matching_observation_low_surprise(self) -> None:
        assimilator = GroundingSurpriseAssimilator(surprise_threshold=0.35)

        preds = {
            "e_box": PredictedEntityState(
                entity_ref=GroundedEntityRef("e_box"),
                predicted_position=(2.0, 0.0),
                predicted_properties={"color": "red"},
            )
        }

        obs = {"e_box": {"position": (2.02, 0.01), "color": "red"}}

        alerts = assimilator.evaluate_observation(preds, obs)
        assert len(alerts) == 0

    def test_position_error_surprise_alert(self) -> None:
        assimilator = GroundingSurpriseAssimilator(surprise_threshold=0.35)

        preds = {
            "e_box": PredictedEntityState(
                entity_ref=GroundedEntityRef("e_box"),
                predicted_position=(2.0, 0.0),
            )
        }

        # Observed 5 meters away
        obs = {"e_box": {"position": (7.0, 0.0)}}

        alerts = assimilator.evaluate_observation(preds, obs)
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.entity_ref == GroundedEntityRef("e_box")
        assert alert.surprise_score > 0.8
        assert alert.position_error == pytest.approx(5.0)

    def test_unexpected_disappearance(self) -> None:
        assimilator = GroundingSurpriseAssimilator()

        preds = {
            "e_cup": PredictedEntityState(
                entity_ref=GroundedEntityRef("e_cup"),
                predicted_position=(1.0, 1.0),
            )
        }

        alerts = assimilator.evaluate_observation(preds, {})
        assert len(alerts) == 1
        assert alerts[0].surprise_score >= 0.9
        assert "presence" in alerts[0].property_mismatches


class TestActiveInferenceFeedback:
    def test_feedback_emits_goal_and_penalizes_schema(self) -> None:
        inducer = SchemaInducer()
        schema = CausalSchema(
            schema_id="schema_flawed",
            action_type="KICK",
            target_role="rock",
            causal_status=CausalStatus.INTERVENTIONALLY_SUPPORTED,
            p_effect_given_do_action=0.9,
            confounder_risk=0.1,
        )
        inducer._schemas["KICK_rock"] = schema

        feedback = ActiveInferenceFeedback(schema_inducer=inducer)

        alert = GroundingSurpriseAlert(
            alert_id="alert_1",
            entity_ref=GroundedEntityRef("e_rock"),
            surprise_score=0.85,
            position_error=4.0,
            property_mismatches={"cracked": (True, False)},
            applied_schema_id="schema_flawed",
        )

        goal = feedback.process_surprise_alert(alert)

        # Verify GoalProposal formulated for active information gathering
        assert goal is not None
        assert isinstance(goal, GoalProposal)
        assert goal.goal_intent == "INSPECT_ENTITY"
        assert goal.target_entity == GroundedEntityRef("e_rock")
        assert goal.priority == 0.85

        # Verify flawed schema was penalized
        assert schema.confounder_risk > 0.1
        assert schema.p_effect_given_do_action < 0.9
