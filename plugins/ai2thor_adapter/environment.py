"""
AI2-THOR Native Environment Wrapper.

Provides native integration with upstream `ai2thor.controller.Controller`
projecting 3D scene-graph metadata and camera frames into typed AI2ThorObservation.
"""

from __future__ import annotations

import logging
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


class NativeAI2ThorWrapper:
    """Wrapper around authentic upstream ai2thor.controller.Controller."""

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
                "ai2thor is required for NativeAI2ThorWrapper. Install via 'pip install ai2thor'."
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
                "Download the Unity build via Controller(download_only=True)."
            )

        self.controller = self._controller_cls(
            scene=self.scene,
            gridSize=self.grid_size,
            renderDepthImage=render_depth_image,
            server_start_timeout=60.0,
            server_timeout=60.0,
        )
        self.goal: AI2ThorGoal | None = None
        self._last_obs: AI2ThorObservation | None = None
        self.reset(seed=seed)

    def close(self) -> None:
        """Stop native AI2-THOR controller."""
        if hasattr(self, "controller") and self.controller is not None:
            try:
                self.controller.stop()
            except Exception:
                pass

    def _build_tier_goal(self, obs: AI2ThorObservation) -> AI2ThorGoal:
        pickupable = [o for o in obs.objects if o.isPickupable]
        receptacles = [o for o in obs.objects if o.isReceptacle]
        openable = [o for o in obs.objects if o.isOpenable]

        seed_offset = self.seed or 0

        target_obj_meta = pickupable[seed_offset % len(pickupable)] if pickupable else None
        target_obj = target_obj_meta.objectId if target_obj_meta else "Apple"

        if self.tier == 1:
            target_obj = (
                pickupable[seed_offset % len(pickupable)].objectId if pickupable else "Apple"
            )
            return AI2ThorGoal(
                target_object_id=target_obj,
                target_receptacle_id=None,
                raw_instruction=f"Pick up {target_obj}",
            )
        elif self.tier == 2:
            rec_id = openable[seed_offset % len(openable)].objectId if openable else "Cabinet"
            return AI2ThorGoal(
                target_object_id=None,
                target_receptacle_id=rec_id,
                raw_instruction=f"Open {rec_id}",
            )
        elif self.tier == 3:
            # Tier 3: Surface Relocation onto valid receptacle
            valid_receptacles = [r for r in receptacles if r.objectId != target_obj]
            rec_id = (
                valid_receptacles[(seed_offset + 1) % len(valid_receptacles)].objectId
                if valid_receptacles
                else "CounterTop"
            )
            return AI2ThorGoal(
                target_object_id=target_obj,
                target_receptacle_id=rec_id,
                raw_instruction=f"Put {target_obj} on {rec_id}",
            )
        else:
            # Tier 4: Container Transfer into enclosed openable containers (Cabinet, Fridge, Microwave, Drawer)
            containers = [r for r in receptacles if r.isOpenable]
            container = containers[seed_offset % len(containers)] if containers else None
            rec_id = container.objectId if container else "Cabinet"
            c_type = container.objectType if container else "Cabinet"
            if c_type == "Drawer":
                valid_objs = [o for o in pickupable if o.objectType not in ("Pan", "Pot", "Kettle")]
            elif c_type == "Microwave":
                valid_objs = [
                    o
                    for o in pickupable
                    if o.objectType
                    in ("Bowl", "Plate", "Mug", "Cup", "Egg", "Potato", "Apple", "Bread")
                ]
            else:
                valid_objs = pickupable
            target_obj = (
                valid_objs[seed_offset % len(valid_objs)].objectId
                if valid_objs
                else (pickupable[0].objectId if pickupable else "Apple")
            )
            return AI2ThorGoal(
                target_object_id=target_obj,
                target_receptacle_id=rec_id,
                raw_instruction=f"Put {target_obj} in {rec_id}",
            )

    def reset(self, seed: int | None = None) -> tuple[AI2ThorObservation, dict[str, Any]]:
        """Reset native AI2-THOR controller."""
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        event = self.controller.reset(scene=self.scene)
        obs = self._build_obs_from_event(event)
        self._last_obs = obs
        self.goal = self._build_tier_goal(obs)
        return obs, {"goal": self.goal, "event": event}

    def step(
        self, action: AI2ThorActionType | str | dict[str, Any], **action_kwargs: Any
    ) -> tuple[AI2ThorObservation, float, bool, bool, dict[str, Any]]:
        """Execute primitive action on native AI2-THOR controller."""
        self.step_count += 1
        if isinstance(action, dict):
            act_dict = dict(action)
            raw_act = act_dict.pop("action", "Pass")
            action_name = raw_act.value if hasattr(raw_act, "value") else str(raw_act)
            action_kwargs = {**act_dict, **action_kwargs}
        elif hasattr(action, "value"):
            action_name = str(action.value)
        else:
            action_name = str(action)

        # In AI2-THOR native, PutObject expects `objectId` to be the receptacle object ID
        if action_name == "PutObject" and "receptacleObjectId" in action_kwargs:
            action_kwargs["objectId"] = action_kwargs.pop("receptacleObjectId")

        # If within reach (< 2.5m), enable forceAction=True to avoid camera tilt occlusion false-negatives
        if action_name in (
            "PickupObject",
            "PutObject",
            "OpenObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
        ):
            target_id = action_kwargs.get("objectId")
            if target_id and "forceAction" not in action_kwargs:
                if self._last_obs:
                    target_meta = next(
                        (o for o in self._last_obs.objects if o.objectId == target_id), None
                    )
                    if target_meta and target_meta.distance <= 2.5:
                        action_kwargs["forceAction"] = True
                else:
                    action_kwargs["forceAction"] = True

        event = self.controller.step(action=action_name, **action_kwargs)

        obs = self._build_obs_from_event(event)
        self._last_obs = obs
        terminated = False
        reward = 0.0

        if self.goal:
            t_obj = self.goal.target_object_id
            t_rec = self.goal.target_receptacle_id
            obj_map = {o.objectId: o for o in obs.objects}

            if t_obj and t_rec:
                # Placement goal: object is inside or on target receptacle
                if t_obj in obj_map:
                    obj = obj_map[t_obj]
                    if t_rec in obj.parentReceptacles:
                        terminated = True
                        reward = 1.0
            elif t_obj and not t_rec:
                # Retrieval goal: holding target object
                if obs.held_object_id == t_obj:
                    terminated = True
                    reward = 1.0
            elif t_rec and not t_obj:
                # Receptacle open or toggle goal
                if t_rec in obj_map:
                    rec = obj_map[t_rec]
                    if rec.isOpened or rec.isToggled:
                        terminated = True
                        reward = 1.0

        truncated = self.step_count >= self.max_steps

        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                "success": terminated,
                "goal": self.goal,
                "event": event,
            },
        )

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
                    isPickupable=bool(obj.get("pickupable", obj.get("isPickupable", False))),
                    isReceptacle=bool(obj.get("receptacle", obj.get("isReceptacle", False))),
                    isOpenable=bool(obj.get("openable", obj.get("isOpenable", False))),
                    isOpened=bool(obj.get("isOpen", obj.get("isOpened", False))),
                    isToggleable=bool(obj.get("toggleable", obj.get("isToggleable", False))),
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
) -> NativeAI2ThorWrapper:
    """Instantiate AI2-THOR environment binding strictly to native controller."""
    try:
        wrapper = NativeAI2ThorWrapper(seed=seed, tier=tier)
        logger.info("Successfully bound to native AI2-THOR controller (tier=%d)", tier)
        return wrapper
    except Exception as e:
        raise RuntimeError(f"Native 'ai2thor' upstream package is required but failed: {e}") from e
