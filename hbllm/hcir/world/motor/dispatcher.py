"""Parietal Coordinate Transform & Egocentric Motor Dispatcher."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.parietal_coordinate_transform import HeadingOrientation

logger = logging.getLogger(__name__)


class EgocentricMotorIntent(StrEnum):
    """Viewpoint-relative embodied motor action intention."""

    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    TURN_AROUND = "TURN_AROUND"


@dataclass
class EgocentricDispatchResult:
    """Outcome of converting an egocentric motor intent into physical allocentric action."""

    intent: EgocentricMotorIntent | str
    chosen_action_id: Any | None
    resulting_heading: HeadingOrientation
    target_displacement: tuple[int, int]
    confidence: float = 1.0

    def to_causal_predicate(self) -> CausalPredicate:
        """Export egocentric dispatch as a formal declarative causal predicate."""
        return CausalPredicate(
            variable=f"ego_dispatch:{self.intent}",
            operator="==",
            value=str(self.chosen_action_id),
            relation=f"heading:{self.resulting_heading.value}",
            target_entity="cortex_motor",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording egocentric dispatch."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"ego_dispatch_{self.intent}_{self.chosen_action_id}",
                node_data={
                    "intent": str(self.intent),
                    "action_id": self.chosen_action_id,
                    "heading": str(self.resulting_heading),
                    "displacement": list(self.target_displacement),
                    "confidence": self.confidence,
                },
            )
        ]


class ParietalMotorDispatcher:
    """Coordinates egocentric action intentions with allocentric primitive motor effectors."""

    _HEADING_ROTATIONS_CW = {
        HeadingOrientation.NORTH: HeadingOrientation.EAST,
        HeadingOrientation.EAST: HeadingOrientation.SOUTH,
        HeadingOrientation.SOUTH: HeadingOrientation.WEST,
        HeadingOrientation.WEST: HeadingOrientation.NORTH,
    }

    _HEADING_ROTATIONS_CCW = {
        HeadingOrientation.NORTH: HeadingOrientation.WEST,
        HeadingOrientation.WEST: HeadingOrientation.SOUTH,
        HeadingOrientation.SOUTH: HeadingOrientation.EAST,
        HeadingOrientation.EAST: HeadingOrientation.NORTH,
    }

    @classmethod
    def dispatch_egocentric_intent(
        cls,
        intent: EgocentricMotorIntent | str,
        current_heading: HeadingOrientation | str,
        action_deltas: dict[Any, tuple[int, int]],
        available_actions: Sequence[Any] | None = None,
    ) -> EgocentricDispatchResult:
        """Translate egocentric intention into primitive action ID and updated heading."""
        if isinstance(intent, str):
            intent = EgocentricMotorIntent(intent.upper())
        if isinstance(current_heading, str):
            current_heading = HeadingOrientation(current_heading.upper())

        available = (
            set(available_actions) if available_actions is not None else set(action_deltas.keys())
        )

        # 1. Rotational intents
        if intent == EgocentricMotorIntent.TURN_LEFT:
            new_hdg = cls._HEADING_ROTATIONS_CCW[current_heading]
            return EgocentricDispatchResult(
                intent=intent,
                chosen_action_id=None,
                resulting_heading=new_hdg,
                target_displacement=(0, 0),
            )
        elif intent == EgocentricMotorIntent.TURN_RIGHT:
            new_hdg = cls._HEADING_ROTATIONS_CW[current_heading]
            return EgocentricDispatchResult(
                intent=intent,
                chosen_action_id=None,
                resulting_heading=new_hdg,
                target_displacement=(0, 0),
            )
        elif intent == EgocentricMotorIntent.TURN_AROUND:
            new_hdg = cls._HEADING_ROTATIONS_CW[cls._HEADING_ROTATIONS_CW[current_heading]]
            return EgocentricDispatchResult(
                intent=intent,
                chosen_action_id=None,
                resulting_heading=new_hdg,
                target_displacement=(0, 0),
            )

        # 2. Translational intents
        fwd_r, fwd_c = current_heading.unit_vector
        lat_r, lat_c = current_heading.right_vector

        if intent == EgocentricMotorIntent.MOVE_FORWARD:
            target_disp = (fwd_r, fwd_c)
        elif intent == EgocentricMotorIntent.MOVE_BACKWARD:
            target_disp = (-fwd_r, -fwd_c)
        elif intent == EgocentricMotorIntent.MOVE_RIGHT:
            target_disp = (lat_r, lat_c)
        elif intent == EgocentricMotorIntent.MOVE_LEFT:
            target_disp = (-lat_r, -lat_c)
        else:
            target_disp = (0, 0)

        matched_act = None
        for act, delta in action_deltas.items():
            if act in available and delta == target_disp:
                matched_act = act
                break

        return EgocentricDispatchResult(
            intent=intent,
            chosen_action_id=matched_act,
            resulting_heading=current_heading,
            target_displacement=target_disp,
            confidence=1.0 if matched_act is not None else 0.0,
        )

    @classmethod
    def resolve_tool_effector_targets(
        cls,
        avatar_pos: tuple[int, int],
        current_heading: HeadingOrientation | str,
        body_schema: Any | None = None,
    ) -> set[tuple[int, int]]:
        """Compute the spatial extent of the avatar and incorporated tools in the heading direction."""
        if isinstance(current_heading, str):
            current_heading = HeadingOrientation(current_heading.upper())
        targets: set[tuple[int, int]] = {avatar_pos}
        fwd_r, fwd_c = current_heading.unit_vector
        targets.add((avatar_pos[0] + fwd_r, avatar_pos[1] + fwd_c))
        if body_schema is not None and hasattr(body_schema, "held_tools"):
            for t in body_schema.held_tools:
                rel = getattr(t, "relative_offset", (0, 0))
                if rel != (0, 0):
                    targets.add((avatar_pos[0] + rel[0], avatar_pos[1] + rel[1]))
                else:
                    # Default tool reach extends 1 cell further along forward heading
                    targets.add((avatar_pos[0] + 2 * fwd_r, avatar_pos[1] + 2 * fwd_c))
        return targets
