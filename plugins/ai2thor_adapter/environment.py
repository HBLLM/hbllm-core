"""
AI2-THOR Environment Wrapper.

Provides dual-mode execution:
1. Native `ai2thor.controller.Controller` if installed with Unity 3D binary.
2. High-fidelity `StandaloneAI2ThorEnv` implementing 3D coordinates, discrete locomotion,
   receptacle containment hierarchy, and physical affordance manipulations with zero dependencies.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

from .types import (
    AI2ThorActionType,
    AI2ThorAgentPose,
    AI2ThorGoal,
    AI2ThorObjectMetadata,
    AI2ThorObservation,
    AI2ThorVector3,
)

logger = logging.getLogger(__name__)


class StandaloneAI2ThorEnv:
    """
    High-fidelity, zero-dependency symbolic 3D AI2-THOR simulation engine.
    Simulates 3D poses, discrete motion grid, camera horizon/rotation,
    receptacle containment trees, and physical affordance state updates.
    """

    def __init__(self, seed: int | None = None, tier: int = 4) -> None:
        self.tier = tier
        self.rng = random.Random(seed)
        self.step_count = 0
        self.max_steps = 100

        self.agent_pose = AI2ThorAgentPose(
            position=AI2ThorVector3(0.0, 0.9, 0.0), rotation=0.0, horizon=0.0
        )
        self.objects: dict[str, AI2ThorObjectMetadata] = {}
        self.held_object_id: str | None = None
        self.goal: AI2ThorGoal | None = None
        self.last_action_success = True
        self.last_action_error = ""

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[AI2ThorObservation, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)

        self.step_count = 0
        self.held_object_id = None
        self.agent_pose = AI2ThorAgentPose(
            position=AI2ThorVector3(0.0, 0.9, 0.0), rotation=0.0, horizon=0.0
        )
        self.last_action_success = True
        self.last_action_error = ""

        self._build_scene()
        return self._get_obs(), {"goal": self.goal}

    def _build_scene(self) -> None:
        """Instantiate 3D kitchen scene objects and goal according to tier."""
        self.objects = {
            "CounterTop_1": AI2ThorObjectMetadata(
                objectId="CounterTop_1",
                objectType="CounterTop",
                position=AI2ThorVector3(-1.5, 0.8, 1.5),
                isReceptacle=True,
                isInteractable=True,
            ),
            "Microwave_1": AI2ThorObjectMetadata(
                objectId="Microwave_1",
                objectType="Microwave",
                position=AI2ThorVector3(1.5, 0.9, 1.5),
                isReceptacle=True,
                isOpenable=True,
                isOpened=False,
                isInteractable=True,
            ),
            "DiningTable_1": AI2ThorObjectMetadata(
                objectId="DiningTable_1",
                objectType="DiningTable",
                position=AI2ThorVector3(0.0, 0.75, 2.5),
                isReceptacle=True,
                isInteractable=True,
            ),
            "Mug_1": AI2ThorObjectMetadata(
                objectId="Mug_1",
                objectType="Mug",
                position=AI2ThorVector3(-1.4, 0.85, 1.4),
                isPickupable=True,
                isInteractable=True,
                parentReceptacles=["CounterTop_1"],
            ),
            "Apple_1": AI2ThorObjectMetadata(
                objectId="Apple_1",
                objectType="Apple",
                position=AI2ThorVector3(0.1, 0.8, 2.4),
                isPickupable=True,
                isInteractable=True,
                parentReceptacles=["DiningTable_1"],
            ),
        }

        # Update receptacleObjectIds
        self.objects["CounterTop_1"].receptacleObjectIds = ["Mug_1"]
        self.objects["DiningTable_1"].receptacleObjectIds = ["Apple_1"]

        if self.tier == 1:
            # Tier 1: Object Retrieval ("Pickup Mug_1")
            self.goal = AI2ThorGoal(
                target_object_id="Mug_1",
                target_receptacle_id=None,
                raw_instruction="Pick up Mug_1",
            )
        elif self.tier == 2:
            # Tier 2: State Toggling / Opening ("Open Microwave_1")
            self.goal = AI2ThorGoal(
                target_object_id=None,
                target_receptacle_id="Microwave_1",
                raw_instruction="Open Microwave_1",
            )
        elif self.tier == 3:
            # Tier 3: Surface Relocation ("Put Apple_1 on CounterTop_1")
            self.goal = AI2ThorGoal(
                target_object_id="Apple_1",
                target_receptacle_id="CounterTop_1",
                raw_instruction="Put Apple_1 on CounterTop_1",
            )
        else:
            # Tier 4: Container Transfer ("Put Mug_1 in Microwave_1")
            self.goal = AI2ThorGoal(
                target_object_id="Mug_1",
                target_receptacle_id="Microwave_1",
                raw_instruction="Put Mug_1 in Microwave_1",
            )

    def _dist_to_agent(self, pos: AI2ThorVector3) -> float:
        ap = self.agent_pose.position
        return math.hypot(ap.x - pos.x, ap.z - pos.z)

    def step(
        self, action: str | dict[str, Any]
    ) -> tuple[AI2ThorObservation, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        reward = 0.0
        self.last_action_success = True
        self.last_action_error = ""

        act_name = action["action"] if isinstance(action, dict) else action
        move_step = 0.25

        # 3D Locomotion
        if act_name == AI2ThorActionType.MOVE_AHEAD:
            rad = math.radians(self.agent_pose.rotation)
            self.agent_pose.position.x += move_step * math.sin(rad)
            self.agent_pose.position.z += move_step * math.cos(rad)

        elif act_name == AI2ThorActionType.MOVE_BACK:
            rad = math.radians(self.agent_pose.rotation)
            self.agent_pose.position.x -= move_step * math.sin(rad)
            self.agent_pose.position.z -= move_step * math.cos(rad)

        elif act_name == AI2ThorActionType.MOVE_LEFT:
            rad = math.radians((self.agent_pose.rotation - 90) % 360)
            self.agent_pose.position.x += move_step * math.sin(rad)
            self.agent_pose.position.z += move_step * math.cos(rad)

        elif act_name == AI2ThorActionType.MOVE_RIGHT:
            rad = math.radians((self.agent_pose.rotation + 90) % 360)
            self.agent_pose.position.x += move_step * math.sin(rad)
            self.agent_pose.position.z += move_step * math.cos(rad)

        elif act_name == AI2ThorActionType.ROTATE_RIGHT:
            self.agent_pose.rotation = (self.agent_pose.rotation + 90.0) % 360.0

        elif act_name == AI2ThorActionType.ROTATE_LEFT:
            self.agent_pose.rotation = (self.agent_pose.rotation - 90.0) % 360.0

        elif act_name == AI2ThorActionType.LOOK_UP:
            self.agent_pose.horizon = max(-30.0, self.agent_pose.horizon - 30.0)

        elif act_name == AI2ThorActionType.LOOK_DOWN:
            self.agent_pose.horizon = min(60.0, self.agent_pose.horizon + 30.0)

        # Manipulation
        elif act_name == AI2ThorActionType.PICKUP_OBJECT:
            oid = action.get("objectId") if isinstance(action, dict) else None
            if oid and oid in self.objects:
                obj = self.objects[oid]
                if self._dist_to_agent(obj.position) < 2.0 and obj.isPickupable:
                    if self.held_object_id is None:
                        self.held_object_id = oid
                        # Remove from parent receptacle
                        for pid in obj.parentReceptacles:
                            if pid in self.objects and oid in self.objects[pid].receptacleObjectIds:
                                self.objects[pid].receptacleObjectIds.remove(oid)
                        obj.parentReceptacles = []
                    else:
                        self.last_action_success = False
                        self.last_action_error = "Hands are full."
                else:
                    self.last_action_success = False
                    self.last_action_error = "Object too far or not pickupable."
            else:
                self.last_action_success = False
                self.last_action_error = "Invalid objectId."

        elif act_name == AI2ThorActionType.OPEN_OBJECT:
            oid = action.get("objectId") if isinstance(action, dict) else None
            if oid and oid in self.objects:
                obj = self.objects[oid]
                if self._dist_to_agent(obj.position) < 2.0 and obj.isOpenable:
                    obj.isOpened = True
                else:
                    self.last_action_success = False
                    self.last_action_error = "Object too far or not openable."

        elif act_name == AI2ThorActionType.CLOSE_OBJECT:
            oid = action.get("objectId") if isinstance(action, dict) else None
            if oid and oid in self.objects:
                obj = self.objects[oid]
                if self._dist_to_agent(obj.position) < 2.0 and obj.isOpenable:
                    obj.isOpened = False

        elif act_name == AI2ThorActionType.PUT_OBJECT:
            oid = action.get("objectId") if isinstance(action, dict) else self.held_object_id
            rec_id = action.get("receptacleObjectId") if isinstance(action, dict) else None

            if oid == self.held_object_id and rec_id and rec_id in self.objects:
                rec = self.objects[rec_id]
                if self._dist_to_agent(rec.position) < 2.0:
                    if not rec.isOpenable or rec.isOpened:
                        self.held_object_id = None
                        self.objects[oid].parentReceptacles = [rec_id]
                        rec.receptacleObjectIds.append(oid)
                    else:
                        self.last_action_success = False
                        self.last_action_error = f"{rec_id} is closed."
                else:
                    self.last_action_success = False
                    self.last_action_error = f"{rec_id} too far."
            else:
                self.last_action_success = False
                self.last_action_error = "Not holding object or invalid receptacle."

        # Verify Goal Satisfaction
        terminated = False
        if self.goal:
            t_obj = self.goal.target_object_id
            t_rec = self.goal.target_receptacle_id
            if t_obj and t_rec:
                if t_obj in self.objects and t_rec in self.objects:
                    if t_rec in self.objects[t_obj].parentReceptacles:
                        terminated = True
                        reward = 1.0
            elif t_obj and not t_rec:
                # Retrieval goal: holding target object
                if self.held_object_id == t_obj:
                    terminated = True
                    reward = 1.0
            elif t_rec and not t_obj:
                # State toggling/opening goal: receptacle opened
                if self.objects[t_rec].isOpened or self.objects[t_rec].isToggled:
                    terminated = True
                    reward = 1.0

        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"success": terminated}

    def _get_obs(self) -> AI2ThorObservation:
        # Update distances
        obj_list: list[AI2ThorObjectMetadata] = []
        for obj in self.objects.values():
            dist = self._dist_to_agent(obj.position)
            meta = AI2ThorObjectMetadata(
                objectId=obj.objectId,
                objectType=obj.objectType,
                position=AI2ThorVector3(obj.position.x, obj.position.y, obj.position.z),
                distance=round(dist, 3),
                isInteractable=obj.isInteractable,
                isPickupable=obj.isPickupable,
                isReceptacle=obj.isReceptacle,
                isOpenable=obj.isOpenable,
                isOpened=obj.isOpened,
                isToggleable=obj.isToggleable,
                isToggled=obj.isToggled,
                parentReceptacles=list(obj.parentReceptacles),
                receptacleObjectIds=list(obj.receptacleObjectIds),
            )
            obj_list.append(meta)

        return AI2ThorObservation(
            agent_pose=AI2ThorAgentPose(
                position=AI2ThorVector3(
                    self.agent_pose.position.x,
                    self.agent_pose.position.y,
                    self.agent_pose.position.z,
                ),
                rotation=self.agent_pose.rotation,
                horizon=self.agent_pose.horizon,
            ),
            objects=obj_list,
            held_object_id=self.held_object_id,
            last_action_success=self.last_action_success,
            last_action_error=self.last_action_error,
            step_count=self.step_count,
        )


class NativeAI2ThorWrapper:
    """
    Dual-mode wrapper wrapping authentic upstream ai2thor.controller.Controller.

    PERCEPTION CHANNEL SPECIFICATION:
    AI2-THOR offers two perceptual modalities:
    1. Symbolic 3D Scene Graph Channel: event.metadata['objects'] containing ground-truth 3D
       coordinates, receptacle containment trees, and physical affordance states.
    2. Visual Camera Channel: event.frame (RGB uint8 numpy array) from the simulated agent camera.

    This adapter implements the Ground-Truth 3D Scene Graph Channel by default, projecting
    event.metadata['objects'] into strongly-typed AI2ThorObjectMetadata instances, while
    attaching the raw RGB camera frame to AI2ThorObservation.raw_obs for multimodal / visual reasoning.
    """

    is_native: bool = True

    SCENE_MAP = {
        1: "FloorPlan1",  # Kitchen: Open navigation / inspect
        2: "FloorPlan2",  # Kitchen: Static object interaction
        3: "FloorPlan3",  # Kitchen: Receptacle containment
        4: "FloorPlan4",  # Kitchen: Multi-step affordance manipulation
        5: "FloorPlan5",  # Kitchen: Complex multi-room/chained task
    }

    def __init__(
        self,
        scene: str = "FloorPlan1",
        seed: int | None = None,
        tier: int = 4,
        grid_size: float = 0.25,
        render_depth_image: bool = False,
    ) -> None:
        try:
            from ai2thor.controller import Controller  # type: ignore
        except ImportError as err:
            raise ImportError(
                "ai2thor is required for NativeAI2ThorWrapper. "
                "Install via 'pip install ai2thor' or use StandaloneAI2ThorEnv."
            ) from err

        self._controller_cls = Controller
        self.tier = tier
        self.seed = seed
        self.grid_size = grid_size
        self.scene = self.SCENE_MAP.get(tier, scene)
        self.step_count = 0
        self.max_steps = 100
        from pathlib import Path

        releases_dir = Path.home() / ".ai2thor" / "releases"
        if not releases_dir.exists() or not any(releases_dir.iterdir()):
            raise RuntimeError(
                "AI2-THOR Unity standalone build not downloaded. "
                "Use StandaloneAI2ThorEnv or download the Unity build."
            )

        self.controller = self._controller_cls(
            scene=self.scene,
            gridSize=self.grid_size,
            renderDepthImage=render_depth_image,
            server_start_timeout=5.0,
            server_timeout=5.0,
        )
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[AI2ThorObservation, dict[str, Any]]:
        """Reset native AI2-THOR controller."""
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        event = self.controller.reset(scene=self.scene)
        obs = self._build_obs_from_event(event)
        return obs, {"event": event}

    def step(
        self, action: AI2ThorActionType | str, **action_kwargs: Any
    ) -> tuple[AI2ThorObservation, float, bool, bool, dict[str, Any]]:
        """Execute primitive action on native AI2-THOR controller."""
        self.step_count += 1
        action_name = action.value if isinstance(action, AI2ThorActionType) else str(action)
        event = self.controller.step(action=action_name, **action_kwargs)

        obs = self._build_obs_from_event(event)
        terminated = False
        truncated = self.step_count >= self.max_steps
        reward = 1.0 if obs.last_action_success else 0.0

        return obs, reward, terminated, truncated, {"event": event}

    def _build_obs_from_event(self, event: Any) -> AI2ThorObservation:
        """Project native AI2-THOR event metadata and camera vision into typed observation."""
        meta = event.metadata
        agent_meta = meta.get("agent", {})
        pos_dict = agent_meta.get("position", {})
        rot_dict = agent_meta.get("rotation", {})

        agent_pose = AI2ThorAgentPose(
            position=AI2ThorVector3(
                x=float(pos_dict.get("x", 0.0)),
                y=float(pos_dict.get("y", 0.9)),
                z=float(pos_dict.get("z", 0.0)),
            ),
            rotation=float(rot_dict.get("y", 0.0)),
            horizon=float(agent_meta.get("cameraHorizon", 0.0)),
        )

        obj_list: list[AI2ThorObjectMetadata] = []
        for obj in meta.get("objects", []):
            o_pos = obj.get("position", {})
            o_rot = obj.get("rotation", {})
            obj_list.append(
                AI2ThorObjectMetadata(
                    objectId=str(obj.get("objectId", "")),
                    objectType=str(obj.get("objectType", "")),
                    position=AI2ThorVector3(
                        x=float(o_pos.get("x", 0.0)),
                        y=float(o_pos.get("y", 0.0)),
                        z=float(o_pos.get("z", 0.0)),
                    ),
                    rotation=AI2ThorVector3(
                        x=float(o_rot.get("x", 0.0)),
                        y=float(o_rot.get("y", 0.0)),
                        z=float(o_rot.get("z", 0.0)),
                    ),
                    distance=float(obj.get("distance", 0.0)),
                    isInteractable=bool(obj.get("isInteractable", True)),
                    isPickupable=bool(obj.get("isPickupable", False)),
                    isReceptacle=bool(obj.get("isReceptacle", False)),
                    isOpenable=bool(obj.get("isOpenable", False)),
                    isOpened=bool(obj.get("isOpened", False)),
                    isToggleable=bool(obj.get("isToggleable", False)),
                    isToggled=bool(obj.get("isToggled", False)),
                    parentReceptacles=list(obj.get("parentReceptacles") or []),
                    receptacleObjectIds=list(obj.get("receptacleObjectIds") or []),
                )
            )

        held_ids = [o["objectId"] for o in meta.get("objects", []) if o.get("isPickedUp", False)]
        held_id = held_ids[0] if held_ids else None
        frame = getattr(event, "frame", None)

        return AI2ThorObservation(
            agent_pose=agent_pose,
            objects=obj_list,
            held_object_id=held_id,
            last_action_success=bool(meta.get("lastActionSuccess", True)),
            last_action_error=str(meta.get("errorMessage", "")),
            step_count=self.step_count,
            raw_obs={"frame": frame, "metadata": meta},
        )


def make_ai2thor_env(
    seed: int | None = None,
    tier: int = 4,
    prefer_native: bool = False,
    require_native: bool = False,
) -> StandaloneAI2ThorEnv | NativeAI2ThorWrapper:
    """Instantiate AI2-THOR environment with dual-mode native/standalone selection."""
    if prefer_native or require_native:
        try:
            return NativeAI2ThorWrapper(seed=seed, tier=tier)
        except Exception as e:
            if require_native:
                raise RuntimeError(
                    f"Native 'ai2thor' package is strictly required; standalone fallback is disabled. Cause: {e}"
                ) from e
            logger.warning(
                "Native ai2thor controller unavailable (%s), falling back to StandaloneAI2ThorEnv",
                e,
            )
    return StandaloneAI2ThorEnv(seed=seed, tier=tier)
