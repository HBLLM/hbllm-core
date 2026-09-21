"""Arcade Environment Driver for ARC-AGI-3 benchmark games."""

from __future__ import annotations

from typing import Any

from arcengine import GameAction

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
)
from hbllm.hcir.spatial_planner import SpatialEntity
from plugins.arc_agi_adapter.arc_agi_3_runner import ARCGrid, GridTopologyExtractor
from plugins.arc_agi_adapter.arc_spatial_agent import ARCPerceptualLifter


class ArcadeDriver(BaseDriver):
    """Driver connecting HBLLM to ARC-AGI-3 environments via arcade interface."""

    def __init__(self) -> None:
        super().__init__(
            name="arcade_driver",
            capabilities={
                DriverCapability.DISCRETE_ACTIONS,
                DriverCapability.SPATIAL_2D,
                DriverCapability.STEP_BASED_EXECUTION,
            },
        )
        self._env: Any = None
        self._last_frame_data: Any = None
        self._current_frame: Any = None
        self._prev_frame: Any = None

    def connect(self, target: Any) -> bool:
        """Connect to an Arcade environment instance."""
        self._env = target
        self._target = target
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from environment."""
        self._env = None
        self._target = None
        self.is_connected = False
        self._last_frame_data = None
        self._current_frame = None
        self._prev_frame = None

    def set_initial_state(self, frame_data: Any) -> None:
        """Set initial frame data after env.reset()."""
        self._last_frame_data = frame_data
        if hasattr(frame_data, "frame") and len(frame_data.frame) > 0:
            self._current_frame = frame_data.frame[0]
            self._prev_frame = self._current_frame

    def get_inputs(self) -> DriverInput:
        """Retrieve current visual frame and environment state."""
        fd = self._last_frame_data
        curr_grid = self._current_frame
        metadata = {
            "levels_completed": getattr(fd, "levels_completed", 0) if fd else 0,
            "available_actions": getattr(fd, "available_actions", [1, 2, 3, 4, 5])
            if fd
            else [1, 2, 3, 4, 5],
            "score": getattr(fd, "score", 0) if fd else 0,
        }
        return DriverInput(
            raw_data=curr_grid,
            metadata=metadata,
        )

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        """Query currently valid actions (1..5)."""
        avail = inputs.metadata.get("available_actions", [1, 2, 3, 4, 5])
        action_names = {
            1: "UP",
            2: "DOWN",
            3: "LEFT",
            4: "RIGHT",
            5: "ACTION / INTERACT",
        }
        return [
            DriverAction(
                action_id=a,
                semantic_intent=action_names.get(a, f"ACTION_{a}"),
            )
            for a in avail
        ]

    def send_output(self, action: DriverAction) -> Any:
        """Dispatch concrete GameAction to ARC environment."""
        if not self._env:
            raise RuntimeError("ArcadeDriver not connected to an environment")

        act_enum = getattr(GameAction, f"ACTION{action.action_id}")
        raw_result = self._env.step(act_enum)
        self._last_frame_data = raw_result
        self._prev_frame = self._current_frame
        if hasattr(raw_result, "frame") and len(raw_result.frame) > 0:
            self._current_frame = raw_result.frame[0]
        return raw_result

    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        """Normalize raw Arcade step result into standardized DriverFeedback."""
        fd = raw_result
        terminated = not (hasattr(fd, "frame") and len(fd.frame) > 0)
        levels_completed = getattr(fd, "levels_completed", 0) if fd else 0
        reward = float(getattr(fd, "score", 0)) if fd else 0.0

        return DriverFeedback(
            success=True,
            reward=reward,
            terminated=terminated,
            causal_delta=self._current_frame,
            info={
                "levels_completed": levels_completed,
                "prev_frame": self._prev_frame,
                "curr_frame": self._current_frame,
            },
            raw_response=raw_result,
        )

    def lift_to_hcir(self, inputs: DriverInput) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
        """Lift raw pixel grid into abstract HCIR SpatialEntity objects and raw barriers."""
        grid = inputs.raw_data
        if grid is None:
            return [], set()

        arc_grid = ARCGrid.from_list(grid.tolist())
        objs = GridTopologyExtractor.extract_objects(arc_grid)

        # Lift through perceptual lifter (domain adapter)
        entities, raw_barriers = ARCPerceptualLifter.lift(
            grid=grid,
            raw_objects=objs,
            step_size=inputs.metadata.get("step_size", 1),
        )
        return entities, raw_barriers
