"""Macro Action Models, Specifications, and Base Classes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialActionIntent

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


def get_sub_attr(target: Any, attr: str, default: Any = None) -> Any:
    return getattr(target, attr, default) if target is not None else default


import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class MacroActionStatus(StrEnum):
    """Execution status of an active macro action."""

    PENDING = "pending"
    RUNNING = "running"
    SATISFIED = "satisfied"
    FAILED = "failed"
    BLOCKED = "blocked"


def _spatial_effector_payload(
    engine: Any, eff: Any, target_coord: tuple[int, int]
) -> dict[str, Any]:
    """Build a point-actuation payload for ``target_coord`` (row, col).

    When the effector's affordance declares a discrete ``valid_targets``
    lattice (adapter-enumerated reachable actuation coordinates), the raw
    intent coordinate is snapped to the nearest declared target so the
    emitted probe stays on the reachable actuation manifold; coordinates off
    the lattice are physical no-ops.

    Args:
        engine: Epistemic engine (or journaled view) exposing
            ``action_affordances``.
        eff: The chosen spatial effector action.
        target_coord: Preferred (row, col) actuation coordinate.

    Returns:
        The action-data payload for the effector command.
    """
    data: dict[str, Any] = {"x": int(target_coord[1]), "y": int(target_coord[0])}
    affordances = getattr(engine, "action_affordances", None) or {}
    aff = affordances.get(eff) if hasattr(affordances, "get") else None
    valid_targets = (
        aff.metadata.get("valid_targets")
        if (aff is not None and isinstance(getattr(aff, "metadata", None), dict))
        else None
    )
    if isinstance(valid_targets, (list, tuple)) and valid_targets:
        from hbllm.hcir.world.cortex_motor import MotorCortexEffector

        snapped = MotorCortexEffector.snap_effector_to_valid_targets(
            engine, eff, valid_targets, target_coord[0], target_coord[1]
        )
        if snapped is not None:
            data = snapped
    return data


@dataclass
class ObjectSpec:
    """Formal specification of an entity and its properties passed to the cognitive core."""

    id: str
    pos: tuple[int, int]
    feature_id: int = 0
    role: str = "unknown"  # "goal", "cargo", "key", "receptacle", "switch", "barrier", "hazard", "manipulandum"
    properties: dict[str, Any] = field(default_factory=dict)
    macro_tasks: list[MacroAction] = field(default_factory=list)

    @property
    def grid_pos(self) -> tuple[int, int]:
        return self.pos

    @grid_pos.setter
    def grid_pos(self, val: tuple[int, int]) -> None:
        self.pos = val

    @property
    def area(self) -> int:
        return int(self.properties.get("area", 1))


@dataclass
class MicroActionSpec:
    """Formal specification of an environment primitive micro-action and its physical dynamics."""

    action_id: Any
    name: str = ""
    delta_r: int = 0
    delta_c: int = 0
    is_spatial_effector: bool = False
    is_focus_switch: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)


class MacroAction(ABC):
    """Abstract base operator for discrete neuro-symbolic macro-actions."""

    def __init__(self, name: str, intent: SpatialActionIntent | SubgoalType) -> None:
        self.name = name
        self.intent = intent
        self.status: MacroActionStatus = MacroActionStatus.PENDING
        self.consecutive_steps: int = 0
        self.max_steps: int = 150

    @abstractmethod
    def is_satisfied(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> bool:
        """Evaluate if postconditions have been achieved."""
        ...

    @abstractmethod
    def is_feasible(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> bool:
        """Evaluate if preconditions and affordances are met."""
        ...

    @abstractmethod
    def step(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        """Produce the next concrete environment action."""
        ...

    def reset(self) -> None:
        """Reset execution counters."""
        self.status = MacroActionStatus.PENDING
        self.consecutive_steps = 0
