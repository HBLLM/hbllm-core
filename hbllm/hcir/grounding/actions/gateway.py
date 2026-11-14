from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.grounding.contracts.actions import (
    ActionExecutionRecord,
    ActionValidationResult,
    RenderedActionPayload,
    SymbolicActionIntent,
)
from hbllm.hcir.grounding.contracts.enums import ActionLifecycleState

logger = logging.getLogger(__name__)


class ActionValidationGateway:
    """The authoritative pre-execution validation and post-execution outcome verification gateway.

    Upholds Invariant 4 (Constitutional Governance) & SGE Section 13:
    Pure renderers convert intent to payload, but ONLY the ActionValidationGateway
    authorizes actuator dispatch and certifies verified real-world outcomes.
    """

    def __init__(
        self,
        min_visibility_confidence: float = 0.5,
        max_velocity: float = 5.0,
    ) -> None:
        self._min_visibility_conf = min_visibility_confidence
        self._max_velocity = max_velocity

    def validate_action(
        self,
        intent: SymbolicActionIntent,
        payload: RenderedActionPayload,
        world_state: dict[str, Any],
    ) -> ActionValidationResult:
        """Executes the 4-stage validation pipeline on proposed action."""
        rejection_reasons: list[str] = []
        requires_human_confirmation = False
        reversibility_verified = True

        # Check 1: Target Identity & Visibility Check
        tracked_entities = world_state.get("tracked_entities", {})
        target_id = intent.target_ref.id
        target_info = tracked_entities.get(target_id)

        if target_info is None:
            rejection_reasons.append(
                f"Target entity {target_id} is not tracked or visible in current world state"
            )
        else:
            conf = float(target_info.get("confidence", 1.0))
            if conf < self._min_visibility_conf:
                rejection_reasons.append(
                    f"Target entity {target_id} tracking confidence {conf:.2f} is below threshold {self._min_visibility_conf}"
                )

        # Check 2: Precondition and Affordance Checks
        for pre in intent.preconditions:
            if not self._check_precondition(pre, target_info or {}, world_state):
                rejection_reasons.append(f"Unsatisfied precondition: {pre}")

        # Check 3: Safety and Physical Feasibility
        speed = float(payload.payload_data.get("speed", 1.0))
        if speed > self._max_velocity:
            rejection_reasons.append(
                f"Requested speed {speed:.2f} exceeds maximum safety limit {self._max_velocity}"
            )

        forbidden_regions = world_state.get("forbidden_regions", [])
        target_coords = payload.payload_data.get("target_coordinates")
        if target_coords and self._in_forbidden_region(target_coords, forbidden_regions):
            rejection_reasons.append(
                f"Target coordinates {target_coords} are in a restricted/forbidden safety region"
            )

        # Check 4: Constitutional / Irreversibility Check
        if intent.action_type.upper() in ("DELETE", "DESTROY", "PURGE", "OVERWRITE_ALL"):
            requires_human_confirmation = True
            reversibility_verified = False

        is_authorized = len(rejection_reasons) == 0

        return ActionValidationResult(
            payload_id=payload.payload_id,
            is_authorized=is_authorized,
            rejection_reasons=tuple(rejection_reasons),
            safety_margin=0.9 if is_authorized else 0.0,
            requires_human_confirmation=requires_human_confirmation,
            reversibility_verified=reversibility_verified,
            validated_at=time.time(),
        )

    def dispatch_action(
        self,
        intent: SymbolicActionIntent,
        payload: RenderedActionPayload,
        validation: ActionValidationResult,
    ) -> ActionExecutionRecord:
        """Transitions action to DISPATCHED state if authorized."""
        now = time.time()
        if not validation.is_authorized:
            return ActionExecutionRecord(
                intent_id=intent.intent_id,
                payload_id=payload.payload_id,
                state=ActionLifecycleState.FAILED,
                dispatched_at=now,
                error_message=f"Validation rejected: {'; '.join(validation.rejection_reasons)}",
            )

        return ActionExecutionRecord(
            intent_id=intent.intent_id,
            payload_id=payload.payload_id,
            state=ActionLifecycleState.DISPATCHED,
            dispatched_at=now,
        )

    def report_execution(
        self,
        record: ActionExecutionRecord,
        success: bool,
        actuator_message: str = "",
    ) -> ActionExecutionRecord:
        """Transitions action from DISPATCHED to EXECUTION_REPORTED or FAILED."""
        if record.state != ActionLifecycleState.DISPATCHED:
            raise ValueError(
                f"Cannot report execution from non-dispatched state {record.state.value}"
            )

        now = time.time()
        if not success:
            return ActionExecutionRecord(
                intent_id=record.intent_id,
                payload_id=record.payload_id,
                state=ActionLifecycleState.FAILED,
                dispatched_at=record.dispatched_at,
                reported_at=now,
                error_message=actuator_message or "Actuator reported execution failure",
            )

        return ActionExecutionRecord(
            intent_id=record.intent_id,
            payload_id=record.payload_id,
            state=ActionLifecycleState.EXECUTION_REPORTED,
            dispatched_at=record.dispatched_at,
            reported_at=now,
        )

    def verify_outcome(
        self,
        record: ActionExecutionRecord,
        intent: SymbolicActionIntent,
        post_observation: dict[str, Any],
    ) -> ActionExecutionRecord:
        """Post-execution verification comparing expected outcomes with sensory reality."""
        if record.state != ActionLifecycleState.EXECUTION_REPORTED:
            raise ValueError(f"Cannot verify outcome from non-reported state {record.state.value}")

        now = time.time()
        target_id = intent.target_ref.id
        target_obs = post_observation.get(target_id, {})

        mismatches: list[str] = []
        for expected in intent.expected_outcomes:
            if not self._check_expected_outcome(expected, target_obs, post_observation):
                mismatches.append(f"Outcome not verified: {expected}")

        if not mismatches:
            return ActionExecutionRecord(
                intent_id=record.intent_id,
                payload_id=record.payload_id,
                state=ActionLifecycleState.COMPLETED,
                dispatched_at=record.dispatched_at,
                reported_at=record.reported_at,
                verified_at=now,
                outcome_verified=True,
                surprise_detected=False,
            )
        else:
            return ActionExecutionRecord(
                intent_id=record.intent_id,
                payload_id=record.payload_id,
                state=ActionLifecycleState.FAILED,
                dispatched_at=record.dispatched_at,
                reported_at=record.reported_at,
                verified_at=now,
                outcome_verified=False,
                surprise_detected=True,
                error_message=f"Outcome verification failed: {'; '.join(mismatches)}",
            )

    @staticmethod
    def _check_precondition(
        precondition: str, target_info: dict[str, Any], world_state: dict[str, Any]
    ) -> bool:
        """Parses and checks symbolic preconditions (e.g. 'open == False', 'gripper_empty')."""
        if "==" in precondition:
            parts = [p.strip() for p in precondition.split("==")]
            prop, val_str = parts[0], parts[1]
            val: Any = val_str
            if val_str.lower() == "true":
                val = True
            elif val_str.lower() == "false":
                val = False
            return target_info.get(prop) == val or world_state.get(prop) == val
        elif precondition in target_info:
            return bool(target_info[precondition])
        elif precondition in world_state:
            return bool(world_state[precondition])
        return True

    @staticmethod
    def _check_expected_outcome(
        outcome: str, target_obs: dict[str, Any], post_observation: dict[str, Any]
    ) -> bool:
        """Parses and verifies expected outcomes (e.g. 'open == True', 'contact == True')."""
        if "==" in outcome:
            parts = [p.strip() for p in outcome.split("==")]
            prop, val_str = parts[0], parts[1]
            val: Any = val_str
            if val_str.lower() == "true":
                val = True
            elif val_str.lower() == "false":
                val = False
            return target_obs.get(prop) == val or post_observation.get(prop) == val
        elif outcome in target_obs:
            return bool(target_obs[outcome])
        elif outcome in post_observation:
            return bool(post_observation[outcome])
        return False

    @staticmethod
    def _in_forbidden_region(
        coords: tuple[float, ...], forbidden_regions: list[dict[str, Any]]
    ) -> bool:
        for reg in forbidden_regions:
            min_x, max_x = reg.get("x", (-float("inf"), float("inf")))
            min_y, max_y = reg.get("y", (-float("inf"), float("inf")))
            if min_x <= coords[0] <= max_x and min_y <= coords[1] <= max_y:
                return True
        return False


@dataclass(frozen=True)
class CommunicationValidationResult:
    """Validation result for communicative / linguistic text output."""

    utterance_id: str
    is_authorized: bool
    filtered_text: str
    rejection_reasons: tuple[str, ...] = ()
    moderation_passed: bool = True


class CommunicationValidationGateway:
    """Validates natural language and speech outputs against conversational safety policies.

    Distinguishes communicative validation from physical robot/GUI actuation.
    Enforces privacy, politeness, and disclosure rules without blocking real-time speech.
    """

    def __init__(
        self,
        forbidden_phrases: set[str] | None = None,
        max_utterance_length: int = 1000,
    ) -> None:
        self._forbidden_phrases = forbidden_phrases or {"[CONFIDENTIAL]", "SYSTEM_PASSWORD"}
        self._max_len = max_utterance_length

    def validate_communication(
        self,
        intent: SymbolicActionIntent,
        payload: RenderedActionPayload,
    ) -> CommunicationValidationResult:
        reasons: list[str] = []
        text = str(payload.payload_data.get("utterance_text", ""))

        if len(text) > self._max_len:
            reasons.append(f"Utterance length {len(text)} exceeds maximum limit {self._max_len}")

        for phrase in self._forbidden_phrases:
            if phrase in text:
                reasons.append(f"Utterance contains forbidden token: {phrase}")

        is_auth = len(reasons) == 0
        return CommunicationValidationResult(
            utterance_id=payload.payload_id,
            is_authorized=is_auth,
            filtered_text=text if is_auth else "",
            rejection_reasons=tuple(reasons),
            moderation_passed=is_auth,
        )
