from __future__ import annotations

import pytest

from hbllm.hcir.grounding.actions.gateway import ActionValidationGateway
from hbllm.hcir.grounding.actions.renderers import (
    DialogueRenderer,
    GuiRenderer,
    KinematicRenderer,
)
from hbllm.hcir.grounding.contracts.actions import (
    ActionExecutionRecord,
    ActionValidationResult,
    RenderedActionPayload,
    SymbolicActionIntent,
)
from hbllm.hcir.grounding.contracts.enums import ActionLifecycleState
from hbllm.hcir.grounding.contracts.references import GroundedEntityRef


class TestActionRenderers:
    def test_kinematic_renderer(self) -> None:
        renderer = KinematicRenderer()
        intent = SymbolicActionIntent(
            intent_id="intent_nav_1",
            action_type="NAVIGATE_TO",
            target_ref=GroundedEntityRef("e_table"),
            parameters={"speed": 2.0},
        )
        assert renderer.can_render(intent) is True

        payload = renderer.render(
            intent,
            context={"current_position": (0.0, 0.0), "target_position": (4.0, 0.0)},
        )
        assert isinstance(payload, RenderedActionPayload)
        assert payload.intent_id == "intent_nav_1"
        assert payload.execution_channel == "kinematic_actuator"
        assert len(payload.payload_data["trajectory_waypoints"]) >= 2
        assert payload.estimated_duration_s == pytest.approx(2.0)

    def test_gui_renderer(self) -> None:
        renderer = GuiRenderer()
        intent = SymbolicActionIntent(
            intent_id="intent_click_1",
            action_type="CLICK",
            target_ref=GroundedEntityRef("e_submit_btn"),
        )
        assert renderer.can_render(intent) is True

        payload = renderer.render(
            intent,
            context={"element_bbox": (100.0, 200.0, 50.0, 20.0)},
        )
        assert payload.payload_data["ui_action"] == "click"
        assert payload.payload_data["coordinates"] == (125.0, 210.0)

    def test_dialogue_renderer(self) -> None:
        renderer = DialogueRenderer()
        intent = SymbolicActionIntent(
            intent_id="intent_dlg_1",
            action_type="INFORM",
            target_ref=GroundedEntityRef("e_user"),
            parameters={"template": "Hello {user_name}, task completed."},
        )
        assert renderer.can_render(intent) is True

        payload = renderer.render(intent, context={"user_name": "Alice"})
        assert payload.payload_data["dialogue_act"] == "inform"
        assert payload.payload_data["text"] == "Hello Alice, task completed."


class TestActionValidationGateway:
    def test_successful_action_validation_and_lifecycle(self) -> None:
        gateway = ActionValidationGateway()

        intent = SymbolicActionIntent(
            intent_id="intent_grasp_1",
            action_type="GRASP",
            target_ref=GroundedEntityRef("e_cup"),
            preconditions=("gripper_empty == True",),
            expected_outcomes=("grasped == True",),
        )

        renderer = KinematicRenderer()
        payload = renderer.render(intent, context={"target_position": (1.0, 1.0)})

        world_state = {
            "tracked_entities": {"e_cup": {"confidence": 0.95, "position": (1.0, 1.0)}},
            "gripper_empty": True,
        }

        # 1. Validation
        val_res = gateway.validate_action(intent, payload, world_state)
        assert isinstance(val_res, ActionValidationResult)
        assert val_res.is_authorized is True
        assert len(val_res.rejection_reasons) == 0

        # 2. Dispatch
        record = gateway.dispatch_action(intent, payload, val_res)
        assert isinstance(record, ActionExecutionRecord)
        assert record.state == ActionLifecycleState.DISPATCHED

        # 3. Execution Reported
        reported_rec = gateway.report_execution(record, success=True)
        assert reported_rec.state == ActionLifecycleState.EXECUTION_REPORTED

        # 4. Outcome Verification Success
        post_obs = {"e_cup": {"grasped": True}}
        verified_rec = gateway.verify_outcome(reported_rec, intent, post_obs)
        assert verified_rec.state == ActionLifecycleState.COMPLETED
        assert verified_rec.outcome_verified is True
        assert verified_rec.surprise_detected is False

    def test_validation_rejections(self) -> None:
        gateway = ActionValidationGateway(min_visibility_confidence=0.6, max_velocity=2.0)

        intent = SymbolicActionIntent(
            intent_id="intent_fast_1",
            action_type="MOVE",
            target_ref=GroundedEntityRef("e_hidden_target"),
            parameters={"speed": 5.0},  # Exceeds max_velocity
            preconditions=("has_battery == True",),
        )

        renderer = KinematicRenderer()
        payload = renderer.render(intent)

        # Missing target & failed precondition
        world_state = {
            "tracked_entities": {},
            "has_battery": False,
        }

        val_res = gateway.validate_action(intent, payload, world_state)
        assert val_res.is_authorized is False
        assert len(val_res.rejection_reasons) == 3  # not tracked, speed limit, precondition

        # Dispatching rejected action fails immediately
        record = gateway.dispatch_action(intent, payload, val_res)
        assert record.state == ActionLifecycleState.FAILED
        assert "Validation rejected" in record.error_message

    def test_outcome_verification_failure_detects_surprise(self) -> None:
        gateway = ActionValidationGateway()

        intent = SymbolicActionIntent(
            intent_id="intent_push_1",
            action_type="PUSH",
            target_ref=GroundedEntityRef("e_box"),
            expected_outcomes=("moved == True",),
        )

        renderer = KinematicRenderer()
        payload = renderer.render(intent)
        world_state = {"tracked_entities": {"e_box": {"confidence": 0.9}}}

        val = gateway.validate_action(intent, payload, world_state)
        rec = gateway.dispatch_action(intent, payload, val)
        rep = gateway.report_execution(rec, success=True)

        # Post observation: box did NOT move (e.g. wall in the way)
        post_obs = {"e_box": {"moved": False}}
        ver = gateway.verify_outcome(rep, intent, post_obs)

        assert ver.state == ActionLifecycleState.FAILED
        assert ver.outcome_verified is False
        assert ver.surprise_detected is True
