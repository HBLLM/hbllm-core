"""
AI2-THOR Action Adapter and 3D Causal Manipulation Planner.

Decomposes high-level physical manipulation missions:
3D Locomotion -> Camera Alignment -> Receptacle Open -> Object Pickup ->
Transfer Locomotion -> Destination Receptacle Open -> Object Placement.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .types import (
    AI2ThorActionType,
    AI2ThorGoal,
    AI2ThorObservation,
)

logger = logging.getLogger(__name__)


class AI2ThorActionAdapter:
    """
    HCIR 3D Embodied Manipulation and Navigation Planner for AI2-THOR.
    """

    def __init__(self) -> None:
        self.reach_distance = 1.6
        self._last_action: Any = None
        self._stuck_count: int = 0

    def plan_next_action(self, obs: AI2ThorObservation, goal: AI2ThorGoal) -> dict[str, Any] | str:
        """Select next 3D discrete action or manipulation primitive."""
        act = self._plan_internal(obs, goal)
        self._last_action = act
        return act

    def _align_or_act(
        self,
        obs: AI2ThorObservation,
        target_pos: Any,
        action_payload: dict[str, Any],
    ) -> dict[str, Any] | str:
        """Ensure yaw and pitch alignment before performing physical manipulation."""
        ax = obs.agent_pose.position.x
        az = obs.agent_pose.position.z
        curr_rot = obs.agent_pose.rotation

        desired_yaw_rad = math.atan2(target_pos.x - ax, target_pos.z - az)
        desired_yaw_deg = (math.degrees(desired_yaw_rad) + 360.0) % 360.0
        yaw_diff = (desired_yaw_deg - curr_rot + 180.0) % 360.0 - 180.0

        if yaw_diff > 45.0:
            return AI2ThorActionType.ROTATE_RIGHT
        if yaw_diff < -45.0:
            return AI2ThorActionType.ROTATE_LEFT

        if obs.agent_pose.horizon < 30.0 and target_pos.y < obs.agent_pose.position.y - 0.2:
            return AI2ThorActionType.LOOK_DOWN

        return action_payload

    def _plan_internal(self, obs: AI2ThorObservation, goal: AI2ThorGoal) -> dict[str, Any] | str:
        held = obs.held_object_id

        # Case 1: Pure Receptacle Open / State Toggle (Tier 2)
        if goal.target_receptacle_id and not goal.target_object_id:
            rec = next((o for o in obs.objects if o.objectId == goal.target_receptacle_id), None)
            if not rec:
                return AI2ThorActionType.MOVE_AHEAD
            if rec.distance > self.reach_distance and not (
                not obs.last_action_success and rec.distance <= 2.2
            ):
                return self._navigate_toward(obs, rec.position.x, rec.position.z)
            if rec.isOpenable and not rec.isOpened:
                return self._align_or_act(
                    obs,
                    rec.position,
                    {
                        "action": AI2ThorActionType.OPEN_OBJECT,
                        "objectId": goal.target_receptacle_id,
                    },
                )
            if rec.isToggleable and not rec.isToggled:
                return self._align_or_act(
                    obs,
                    rec.position,
                    {
                        "action": AI2ThorActionType.TOGGLE_OBJECT_ON,
                        "objectId": goal.target_receptacle_id,
                    },
                )
            return AI2ThorActionType.MOVE_AHEAD

        target_obj = next((o for o in obs.objects if o.objectId == goal.target_object_id), None)
        if not target_obj:
            return AI2ThorActionType.MOVE_AHEAD

        # Phase 1: Acquire Target Object (Tiers 1, 3, 4)
        if held != goal.target_object_id:
            if target_obj.distance > self.reach_distance and not (
                not obs.last_action_success and target_obj.distance <= 2.2
            ):
                return self._navigate_toward(obs, target_obj.position.x, target_obj.position.z)

            # If inside closed receptacle, open it first
            if target_obj.parentReceptacles:
                parent_id = target_obj.parentReceptacles[0]
                parent = next((o for o in obs.objects if o.objectId == parent_id), None)
                if parent and parent.isOpenable and not parent.isOpened:
                    return self._align_or_act(
                        obs,
                        parent.position,
                        {"action": AI2ThorActionType.OPEN_OBJECT, "objectId": parent_id},
                    )

            return self._align_or_act(
                obs,
                target_obj.position,
                {"action": AI2ThorActionType.PICKUP_OBJECT, "objectId": goal.target_object_id},
            )

        # Case 2: Object Retrieval only (Tier 1) - already holding it!
        if not goal.target_receptacle_id:
            return AI2ThorActionType.MOVE_AHEAD

        # Phase 2: Deliver to Target Receptacle (Tiers 3, 4)
        target_rec = next((o for o in obs.objects if o.objectId == goal.target_receptacle_id), None)
        if not target_rec:
            return AI2ThorActionType.MOVE_AHEAD

        if target_rec.distance > self.reach_distance and not (
            not obs.last_action_success and target_rec.distance <= 2.2
        ):
            return self._navigate_toward(obs, target_rec.position.x, target_rec.position.z)

        # Open destination receptacle if needed (e.g. Microwave in Tier 4)
        if target_rec.isOpenable and not target_rec.isOpened:
            return self._align_or_act(
                obs,
                target_rec.position,
                {"action": AI2ThorActionType.OPEN_OBJECT, "objectId": goal.target_receptacle_id},
            )

        # Place object into receptacle
        return self._align_or_act(
            obs,
            target_rec.position,
            {
                "action": AI2ThorActionType.PUT_OBJECT,
                "objectId": goal.target_object_id,
                "receptacleObjectId": goal.target_receptacle_id,
            },
        )

    def _navigate_toward(self, obs: AI2ThorObservation, target_x: float, target_z: float) -> str:
        """Align yaw rotation toward target and move forward."""
        if not obs.last_action_success and self._last_action == AI2ThorActionType.MOVE_AHEAD:
            self._stuck_count += 1
            if self._stuck_count % 2 == 1:
                return AI2ThorActionType.MOVE_RIGHT
            return AI2ThorActionType.ROTATE_RIGHT
        self._stuck_count = 0

        ax = obs.agent_pose.position.x
        az = obs.agent_pose.position.z
        curr_rot = obs.agent_pose.rotation

        # AI2-THOR yaw: 0 = +z, 90 = +x, 180 = -z, 270 = -x
        desired_yaw_rad = math.atan2(target_x - ax, target_z - az)
        desired_yaw_deg = (math.degrees(desired_yaw_rad) + 360.0) % 360.0

        yaw_diff = (desired_yaw_deg - curr_rot + 180.0) % 360.0 - 180.0

        if yaw_diff > 45.0:
            return AI2ThorActionType.ROTATE_RIGHT
        elif yaw_diff < -45.0:
            return AI2ThorActionType.ROTATE_LEFT
        else:
            return AI2ThorActionType.MOVE_AHEAD
