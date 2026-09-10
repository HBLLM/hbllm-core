"""
AI2-THOR Actuator and Affordance Bridge (Device Driver).

Transforms the AI2-THOR adapter into a pure hardware/simulator driver:
1. Affordance Enumerator: Generates candidate declarative ActionNodes from physical state.
2. Low-Level Actuator: Translates brain-selected declarative actions into 3D motor commands
   (yaw rotation, camera pitch alignment, and Unity RPC payloads).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph

from .perception import AI2ThorPerceptionAdapter
from .types import (
    AI2ThorActionType,
    AI2ThorGoal,
    AI2ThorObservation,
)

logger = logging.getLogger(__name__)


class AI2ThorActionAdapter:
    """
    HCIR 3D Embodied Actuator Driver and Affordance Bridge for AI2-THOR.

    Does NOT plan or make procedural decisions. It declares what actions are
    physically possible (affordances) and executes the actions selected by
    the Core Brain's UnifiedReasoningRuntime.
    """

    def __init__(self, reach_distance: float = 1.6) -> None:
        self.reach_distance = reach_distance
        self._last_action: Any = None
        self._stuck_count: int = 0
        # Internal runtime fallback for standalone execution
        self._perception = AI2ThorPerceptionAdapter()
        self._runtime = UnifiedReasoningRuntime(create_default_operator_registry())

    # ── Affordance Enumeration (Sensory → Declarative Actions) ────────

    def enumerate_affordances(
        self, obs: AI2ThorObservation, graph: CognitiveGraph | None = None
    ) -> list[ActionNode]:
        """Declare all physically possible candidate actions from current observation."""
        affordances: list[ActionNode] = []
        held = obs.held_object_id

        # 1. Navigation Affordances for all visible entities
        for obj in obs.objects:
            affordances.append(
                ActionNode(
                    id=f"act_nav_{obj.objectId}",
                    intent="navigate_toward",
                    properties={"targetId": obj.objectId, "target_pos": obj.position},
                    requirements=[],
                    produces=[f"near({obj.objectId})"],
                )
            )

        # 2. Manipulation Affordances
        for obj in obs.objects:
            # Pickup affordance
            if obj.isPickupable and held != obj.objectId:
                affordances.append(
                    ActionNode(
                        id=f"act_pickup_{obj.objectId}",
                        intent="pickup",
                        properties={"objectId": obj.objectId, "position": obj.position},
                        requirements=[
                            f"near({obj.objectId})",
                            f"not_contained_in_closed({obj.objectId})",
                        ],
                        produces=[f"holds({obj.objectId})"],
                    )
                )

            # Open affordance
            if obj.isOpenable and not obj.isOpened:
                affordances.append(
                    ActionNode(
                        id=f"act_open_{obj.objectId}",
                        intent="open",
                        properties={"objectId": obj.objectId, "position": obj.position},
                        requirements=[f"near({obj.objectId})"],
                        produces=[f"is_opened({obj.objectId})"],
                    )
                )

            # Toggle affordance
            if obj.isToggleable and not obj.isToggled:
                affordances.append(
                    ActionNode(
                        id=f"act_toggle_{obj.objectId}",
                        intent="toggle",
                        properties={"objectId": obj.objectId, "position": obj.position},
                        requirements=[f"near({obj.objectId})"],
                        produces=[f"is_toggled({obj.objectId})"],
                    )
                )

        # 3. Put Affordances for all pickupable objects (or currently held object) into receptacles
        receptacles = [o for o in obs.objects if o.isReceptacle]
        target_obj_ids = [o.objectId for o in obs.objects if o.isPickupable]
        if held and held not in target_obj_ids:
            target_obj_ids.append(held)

        for obj_id in target_obj_ids:
            for rec in receptacles:
                reqs = [f"holds({obj_id})", f"near({rec.objectId})"]
                if rec.isOpenable:
                    reqs.append(f"is_opened({rec.objectId})")
                affordances.append(
                    ActionNode(
                        id=f"act_put_{obj_id}_{rec.objectId}",
                        intent="put",
                        properties={
                            "objectId": obj_id,
                            "receptacleObjectId": rec.objectId,
                            "position": rec.position,
                        },
                        requirements=reqs,
                        produces=[f"inside({obj_id}, {rec.objectId})"],
                    )
                )

        # 3. Default idle / motor fallback
        affordances.append(
            ActionNode(
                id="act_move_ahead_default",
                intent="move_ahead",
                properties={},
                requirements=[],
                produces=[],
            )
        )

        return affordances

    # ── Actuator Dispatch (Declarative Action → Simulator Command) ───

    def execute_action(
        self, action: ActionNode | dict[str, Any] | str, obs: AI2ThorObservation
    ) -> dict[str, Any] | str:
        """Translate declarative action from reasoning core into low-level simulator motor command."""
        # Handle string or dict passed directly
        if isinstance(action, str):
            self._last_action = action
            return action
        if isinstance(action, dict) and "action" in action and "intent" not in action:
            self._last_action = action
            return action

        intent = (
            action.intent if isinstance(action, ActionNode) else action.get("intent", "move_ahead")
        )
        props = (
            action.properties
            if isinstance(action, ActionNode)
            else action.get("properties", action.get("params", {}))
        )

        act_result: dict[str, Any] | str

        if intent == "navigate_toward":
            target_pos = props.get("target_pos")
            if target_pos is None and "targetId" in props:
                target_obj = next((o for o in obs.objects if o.objectId == props["targetId"]), None)
                if target_obj:
                    target_pos = target_obj.position
            if target_pos:
                act_result = self._navigate_toward(obs, target_pos.x, target_pos.z)
            else:
                act_result = AI2ThorActionType.MOVE_AHEAD

        elif intent == "pickup":
            obj_id = props.get("objectId", "")
            target_pos = props.get("position")
            if target_pos is None:
                target_obj = next((o for o in obs.objects if o.objectId == obj_id), None)
                if target_obj:
                    target_pos = target_obj.position
            payload = {"action": AI2ThorActionType.PICKUP_OBJECT, "objectId": obj_id}
            act_result = self._align_or_act(obs, target_pos, payload) if target_pos else payload

        elif intent == "open":
            obj_id = props.get("objectId", "")
            target_pos = props.get("position")
            if target_pos is None:
                target_obj = next((o for o in obs.objects if o.objectId == obj_id), None)
                if target_obj:
                    target_pos = target_obj.position
            payload = {"action": AI2ThorActionType.OPEN_OBJECT, "objectId": obj_id}
            act_result = self._align_or_act(obs, target_pos, payload) if target_pos else payload

        elif intent == "toggle":
            obj_id = props.get("objectId", "")
            target_pos = props.get("position")
            if target_pos is None:
                target_obj = next((o for o in obs.objects if o.objectId == obj_id), None)
                if target_obj:
                    target_pos = target_obj.position
            payload = {"action": AI2ThorActionType.TOGGLE_OBJECT_ON, "objectId": obj_id}
            act_result = self._align_or_act(obs, target_pos, payload) if target_pos else payload

        elif intent == "put":
            obj_id = props.get("objectId", "")
            rec_id = props.get("receptacleObjectId", "")
            target_pos = props.get("position")
            if target_pos is None:
                target_rec = next((o for o in obs.objects if o.objectId == rec_id), None)
                if target_rec:
                    target_pos = target_rec.position
            payload = {
                "action": AI2ThorActionType.PUT_OBJECT,
                "objectId": obj_id,
                "receptacleObjectId": rec_id,
            }
            act_result = self._align_or_act(obs, target_pos, payload) if target_pos else payload

        else:  # move_ahead, no_op, idle
            act_result = AI2ThorActionType.MOVE_AHEAD

        self._last_action = act_result
        return act_result

    # ── High-Level Entry Point (Routes Through Core Unified Runtime) ──

    def plan_next_action(self, obs: AI2ThorObservation, goal: AI2ThorGoal) -> dict[str, Any] | str:
        """Route perception and affordances through UnifiedReasoningRuntime."""
        # 1. Perception Ingestion
        graph = self._perception.ingest_observation(obs, goal)

        # 2. Enumerate Affordances
        affordances = self.enumerate_affordances(obs, graph)
        for aff in affordances:
            if not graph.has_node(aff.id):
                graph.add_node(aff)
            else:
                existing = graph.get_node(aff.id)
                if isinstance(existing, ActionNode):
                    existing.requirements = aff.requirements
                    existing.produces = aff.produces
                    existing.properties = aff.properties

        # 3. Formulate ReasoningProblem for Core Brain
        problem = ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=("goal_active",),
            description="Plan next action to achieve goal in AI2-THOR",
        )

        # 4. Reason via UnifiedReasoningRuntime
        trace = self._runtime.reason(graph=graph, problem=problem)

        # 5. Extract Chosen Action
        action_id = trace.final_result.conclusions.get("action_id", "")
        chosen_node = graph.get_node(action_id) if action_id else None

        if isinstance(chosen_node, ActionNode):
            return self.execute_action(chosen_node, obs)

        # Fallback if no specific action node was selected
        best_intent = trace.final_result.conclusions.get("best_action", "move_ahead")
        return self.execute_action({"intent": best_intent}, obs)

    # ── Low-Level Motor Helpers ──────────────────────────────────────

    def _align_or_act(
        self,
        obs: AI2ThorObservation,
        target_pos: Any,
        action_payload: dict[str, Any],
    ) -> dict[str, Any] | str:
        """Ensure yaw and pitch alignment before performing physical manipulation."""
        if target_pos is None:
            return action_payload

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

        desired_yaw_rad = math.atan2(target_x - ax, target_z - az)
        desired_yaw_deg = (math.degrees(desired_yaw_rad) + 360.0) % 360.0

        yaw_diff = (desired_yaw_deg - curr_rot + 180.0) % 360.0 - 180.0

        if yaw_diff > 45.0:
            return AI2ThorActionType.ROTATE_RIGHT
        elif yaw_diff < -45.0:
            return AI2ThorActionType.ROTATE_LEFT
        else:
            return AI2ThorActionType.MOVE_AHEAD
