"""ARC-AGI Driver — Thin I/O adapter for ARC-AGI game environments.

This driver has ZERO knowledge of HCIR internals. It only knows how to:
1. Connect to an ARC-AGI game environment
2. Read the current grid state
3. List available actions
4. Send actions and receive feedback

All cognition, planning, perception interpretation, and learning happens
inside the CognitiveBlackbox (core). This driver is a dumb pipe.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
    DriverModality,
)

logger = logging.getLogger(__name__)

# ── ARC-AGI SDK imports (graceful fallback) ───────────────────────────────
try:
    from arcle import Arcade  # type: ignore
    from arcle.envs import ARCGameAction, ARCGameState  # type: ignore
except ImportError:
    try:
        from arc_agi import Arcade  # type: ignore
        from arc_agi.types import ARCGameAction, ARCGameState  # type: ignore
    except ImportError:
        Arcade = None  # type: ignore

        class _FallbackARCGameAction(Enum):
            ACTION1 = 1
            ACTION2 = 2
            ACTION3 = 3
            ACTION4 = 4
            ACTION5 = 5
            ACTION6 = 6

        class _FallbackARCGameState(Enum):
            PLAYING = "playing"
            WIN = "win"
            GAME_OVER = "game_over"

        ARCGameAction = _FallbackARCGameAction  # type: ignore
        ARCGameState = _FallbackARCGameState  # type: ignore


class ARC3Driver(BaseDriver):
    """Thin I/O adapter for ARC-AGI-3 game environments.

    This driver provides raw grid observations and accepts discrete actions.
    It has NO knowledge of spatial entities, planning, or HCIR internals.
    All interpretation and learning happens inside the CognitiveBlackbox.
    """

    def __init__(self, name: str = "arcade_driver") -> None:
        super().__init__(
            name=name,
            capabilities={
                DriverCapability.DISCRETE_ACTIONS,
                DriverCapability.SPATIAL_2D,
                DriverCapability.STEP_BASED_EXECUTION,
            },
        )
        self._env: Any = None
        self._frame_data: Any = None
        self._tags: list[str] = []
        self._game_id: str = ""
        self._prev_grid: np.ndarray | None = None

    def connect(self, target: Any) -> bool:
        """Connect to ARC-AGI game instance.

        Args:
            target: Tuple of (arcade_client, game_id) or just game_id string.
        """
        if isinstance(target, tuple) and len(target) == 2:
            arcade_client, game_id = target
        elif isinstance(target, str):
            if Arcade is None:
                logger.error("ARC-AGI SDK not installed")
                return False
            arcade_client = Arcade()
            game_id = target
        else:
            logger.error("Invalid target: expected (arcade_client, game_id) or game_id string")
            return False

        try:
            self._env = arcade_client.make(game_id, render_mode=None)
            self._frame_data = self._env.reset()
            self._game_id = game_id
            self._tags = []
            if (
                hasattr(self._env, "info")
                and hasattr(self._env.info, "tags")
                and self._env.info.tags
            ):
                self._tags = list(self._env.info.tags)
            self.is_connected = True
            self._target = target
            logger.info("Connected to ARC-AGI game: %s", game_id)
            return True
        except Exception as e:
            logger.error("Failed to connect to ARC-AGI game '%s': %s", game_id, e)
            return False

    def disconnect(self) -> None:
        """Disconnect from the ARC-AGI environment."""
        self._env = None
        self._frame_data = None
        self.is_connected = False
        self._target = None

    def get_inputs(self) -> DriverInput:
        """Return current grid as raw DriverInput."""
        grid = np.zeros((16, 16), dtype=int)
        if self._frame_data and hasattr(self._frame_data, "frame") and self._frame_data.frame:
            grid = self._frame_data.frame[-1]

        return DriverInput(
            raw_data=grid,
            modality=DriverModality.GRID_2D,
            source_id=self.name,
            metadata={
                "tags": self._tags,
                "grid_shape": tuple(grid.shape),
                "game_id": self._game_id,
                "levels_completed": getattr(self._frame_data, "levels_completed", 0),
                "win_levels": getattr(self._frame_data, "win_levels", 1),
            },
        )

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        """Return available ARC actions as DriverActions."""
        avail = getattr(self._frame_data, "available_actions", [1, 2, 3, 4])
        if not avail:
            avail = [1, 2, 3, 4]
        return [DriverAction(action_id=a, semantic_intent=f"action_{a}") for a in avail]

    def resolve_action(
        self,
        intent: Any,
        available_actions: list[DriverAction],
        context: dict[str, Any] | None = None,
    ) -> DriverAction | None:
        """Dynamically resolve an ARC action for the requested cognitive intent."""
        avail_dict = {a.action_id: a for a in available_actions}
        norm = str(intent).upper()
        # In ARC-AGI games, action 5 is the primary interaction / manipulation button
        if norm in ("INTERACT", "PICKUP", "DROP", "ACTIVATE", "ACTUATE"):
            if 5 in avail_dict:
                return avail_dict[5]
            if 6 in avail_dict:
                return avail_dict[6]
        return super().resolve_action(intent, available_actions, context=context)

    def send_output(self, action: DriverAction) -> Any:
        """Send action to ARC environment and return raw frame data."""
        self._prev_grid = self.get_inputs().raw_data.copy()

        game_act = getattr(ARCGameAction, f"ACTION{action.action_id}", ARCGameAction.ACTION1)
        try:
            if action.parameters:
                self._frame_data = self._env.step(game_act, data=action.parameters)
            else:
                self._frame_data = self._env.step(game_act)
        except TypeError:
            self._frame_data = self._env.step(game_act)

        return self._frame_data

    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        """Normalize ARC game response to DriverFeedback."""
        fd = raw_result
        if fd is None:
            return DriverFeedback(success=False, terminated=True)

        state = getattr(fd, "state", None)
        completed = state == ARCGameState.WIN
        game_over = state == ARCGameState.GAME_OVER

        # causal_delta carries (prev_grid, curr_grid) for the core to learn from
        curr_grid = fd.frame[-1] if fd and hasattr(fd, "frame") and fd.frame else None
        causal_delta = None
        if self._prev_grid is not None and curr_grid is not None:
            causal_delta = {
                "prev_grid": self._prev_grid,
                "curr_grid": curr_grid,
            }

        return DriverFeedback(
            success=completed,
            reward=1.0 if completed else 0.0,
            terminated=completed or game_over,
            causal_delta=causal_delta,
            info={
                "levels_completed": getattr(fd, "levels_completed", 0),
                "state": str(state) if state else "unknown",
            },
        )

    def get_perception_data(self, inputs: DriverInput) -> dict[str, Any]:
        """Return raw structured perception data for the blackbox.

        The driver provides the raw grid and metadata. The core blackbox
        does all interpretation (avatar detection, barrier learning, etc.)
        through experience, not hardcoded rules.
        """
        return {
            "grid": inputs.raw_data,
            "grid_shape": inputs.metadata.get("grid_shape", (16, 16)),
            "tags": inputs.metadata.get("tags", []),
            "game_id": inputs.metadata.get("game_id", ""),
        }


# Backward-compatibility alias
ArcadeDriver = ARC3Driver
