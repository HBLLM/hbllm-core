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

    def plan_next_action(self, obs: AI2ThorObservation, goal: AI2ThorGoal) -> dict[str, Any] | str:
        """Select next 3D discrete action or manipulation primitive."""
        held = obs.held_object_id
        target_obj = next((o for o in obs.objects if o.objectId == goal.target_object_id), None)
        target_rec = next((o for o in obs.objects if o.objectId == goal.target_receptacle_id), None)

        if not target_obj or not target_rec:
            return AI2ThorActionType.MOVE_AHEAD

        # Phase 1: Acquire Target Object
        if held != goal.target_object_id:
            # If too far from object, navigate toward it
            if target_obj.distance > self.reach_distance:
                return self._navigate_toward(obs, target_obj.position.x, target_obj.position.z)

            # If object is inside a closed receptacle, open receptacle first
            if target_obj.parentReceptacles:
                parent_id = target_obj.parentReceptacles[0]
                parent = next((o for o in obs.objects if o.objectId == parent_id), None)
                if parent and parent.isOpenable and not parent.isOpened:
                    return {"action": AI2ThorActionType.OPEN_OBJECT, "objectId": parent_id}

            # Pickup target object
            return {"action": AI2ThorActionType.PICKUP_OBJECT, "objectId": goal.target_object_id}

        # Phase 2: Deliver to Target Receptacle
        if target_rec.distance > self.reach_distance:
            return self._navigate_toward(obs, target_rec.position.x, target_rec.position.z)

        # Open destination receptacle if closed
        if target_rec.isOpenable and not target_rec.isOpened:
            return {"action": AI2ThorActionType.OPEN_OBJECT, "objectId": goal.target_receptacle_id}

        # Place object into receptacle
        return {
            "action": AI2ThorActionType.PUT_OBJECT,
            "objectId": goal.target_object_id,
            "receptacleObjectId": goal.target_receptacle_id,
        }

    def _navigate_toward(self, obs: AI2ThorObservation, target_x: float, target_z: float) -> str:
        """Align yaw rotation toward target and move forward."""
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
