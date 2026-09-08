"""
AI2-THOR Adapter Types.

Defines strongly-typed representations of AI2-THOR 3D poses, objects,
scene metadata, actions, and embodied manipulation goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AI2ThorActionType(StrEnum):
    """Canonical AI2-THOR navigation and manipulation action strings."""

    MOVE_AHEAD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    MOVE_LEFT = "MoveLeft"
    MOVE_RIGHT = "MoveRight"
    ROTATE_RIGHT = "RotateRight"
    ROTATE_LEFT = "RotateLeft"
    LOOK_UP = "LookUp"
    LOOK_DOWN = "LookDown"
    PICKUP_OBJECT = "PickupObject"
    PUT_OBJECT = "PutObject"
    OPEN_OBJECT = "OpenObject"
    CLOSE_OBJECT = "CloseObject"
    TOGGLE_OBJECT_ON = "ToggleObjectOn"
    TOGGLE_OBJECT_OFF = "ToggleObjectOff"


@dataclass
class AI2ThorVector3:
    """3D Cartesian vector."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class AI2ThorObjectMetadata:
    """Metadata describing an interactive 3D scene object."""

    objectId: str  # noqa: N815
    objectType: str  # noqa: N815
    position: AI2ThorVector3
    rotation: AI2ThorVector3 = field(default_factory=AI2ThorVector3)
    distance: float = 0.0
    isInteractable: bool = True  # noqa: N815
    isPickupable: bool = False  # noqa: N815
    isReceptacle: bool = False  # noqa: N815
    isOpenable: bool = False  # noqa: N815
    isOpened: bool = False  # noqa: N815
    isToggleable: bool = False  # noqa: N815
    isToggled: bool = False  # noqa: N815
    parentReceptacles: list[str] = field(default_factory=list)  # noqa: N815
    receptacleObjectIds: list[str] = field(default_factory=list)  # noqa: N815


@dataclass
class AI2ThorAgentPose:
    """Agent 3D position and camera orientation."""

    position: AI2ThorVector3
    rotation: float = 0.0  # yaw angle in degrees (0, 90, 180, 270)
    horizon: float = 0.0  # camera pitch degrees (-30, 0, 30, 60)


@dataclass
class AI2ThorObservation:
    """Observation frame containing 3D object metadata and agent pose."""

    agent_pose: AI2ThorAgentPose
    objects: list[AI2ThorObjectMetadata]
    held_object_id: str | None = None
    last_action_success: bool = True
    last_action_error: str = ""
    step_count: int = 0
    raw_obs: Any = None


@dataclass
class AI2ThorGoal:
    """Task target specification."""

    target_object_id: str
    target_receptacle_id: str
    raw_instruction: str = ""
