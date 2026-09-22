"""ARC-3 Spatial Cognitive Agent — Thin adapter delegating to CognitiveBlackbox.

The agent is a dumb I/O adapter. All cognition, planning, learning, and
reasoning lives inside the core CognitiveBlackbox. This module only:
  1. Wraps raw grids into DriverInput / perception data.
  2. Forwards actions/feedback to the blackbox.
  3. Maintains backward-compatible public API surface.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np

from hbllm.drivers.base import DriverAction, DriverFeedback, DriverInput, DriverModality
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.hcir.world.morphology import MorphologicalConcept, ShapeArchetype
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel, StateMutationModel

from plugins.arc_agi_adapter.arc_memory import AgentPhase, HCIRCrossGameMemory
from plugins.arc_agi_adapter.arc_perception import ARCPerceptualLifter

logger = logging.getLogger(__name__)

# Graceful import of official Arcade & arcengine
try:
    from arc_agi import Arcade
except ImportError:
    Arcade = None

try:
    from arcengine import GameAction as ARCGameAction
    from arcengine import GameState as ARCGameState
except ImportError:

    class _FallbackARCGameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    class _FallbackARCGameState(Enum):
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    ARCGameAction = _FallbackARCGameAction  # type: ignore
    ARCGameState = _FallbackARCGameState  # type: ignore


class ARC3SpatialCognitiveAgent:
    """Thin adapter for ARC-AGI-3 interactive games.

    All cognition is delegated to CognitiveBlackbox (core).
    This class exists only for backward-compatible public API.
    """

    # Shared global memory across game instances in the process
    global_memory: HCIRCrossGameMemory = HCIRCrossGameMemory()

    def __init__(
        self,
        step_size: int = 1,
        enable_soft_restart: bool = False,
        shared_memory: HCIRCrossGameMemory | None = None,
    ) -> None:
        self.step_size: int = step_size
        self.enable_soft_restart: bool = enable_soft_restart

        # Core cognitive engine — all state and learning lives here
        self.blackbox = CognitiveBlackbox()

        # Cross-game persistent memory
        self.cross_game_memory: HCIRCrossGameMemory = (
            shared_memory if shared_memory is not None else ARC3SpatialCognitiveAgent.global_memory
        )

        # ── Backward-compat stub state ────────────────────────────────
        # These exist ONLY so downstream consumers (inductive_learner, arc_memory)
        # that read/write agent attributes directly don't crash.
        # Over time these consumers should be refactored to use the blackbox.
        self.avatar_color: int | None = None
        self.avatar_centroid: tuple[float, float] | None = None
        self.action_models: dict[int, Any] = {}
        self.learned_barrier_colors: set[int] = set()
        self.learned_walkable_colors: set[int] = set()
        self.learned_item_colors: set[int] | dict[int, Any] = set()
        self.learned_receptacle_colors: set[int] = set()
        self.learned_receptacle_bounds: tuple[int, int, int, int] | None = None
        self.known_barriers: np.ndarray | None = None
        self.workspace = self.blackbox.workspace
        self.spatial_planner = self.blackbox.spatial_planner
        self.action_5_affordance: str = "UNKNOWN"
        self.goal_centroid: tuple[float, float] | None = None
        self.primary_goal_node: Any = None
        self.last_action_data: dict[str, int] | None = None
        self.shape_concepts: dict[tuple[tuple[int, int], ...], MorphologicalConcept] = {}
        self.phase: AgentPhase = AgentPhase.EPISTEMIC_LEARNING
        self.control_context: Any = None
        self.decomposer: Any = self.blackbox.goal_decomposer
        self.state_mutations: list[StateMutationModel] = []
        self.pushable_colors: set[int] = set()
        self.holding_item: bool = False
        self.carried_offset: tuple[float, float] = (0.0, 0.0)
        self.current_plan: list[Any] = []
        self.optimal_task_plan: list[Any] = []
        self.visited_positions: list[tuple[int, int]] = []
        self.blocked_actions: set[int] = set()
        self.stuck_counter: int = 0
        self.last_action: int | None = None
        self.should_soft_restart: bool = False
        self.target_zone_bounds: tuple[int, int, int, int] | None = None
        self.target_zones: set[tuple[int, int]] = set()
        self.is_first_level_learning: bool = True
        self.level_step_counter: int = 0
        self.probe_step_counter: int = 0
        self.available_actions: list[int] = []
        self.current_facing: tuple[int, int] = (0, 0)
        self.delivered_positions: set[tuple[int, int]] = set()
        self.active_goal_node: Any = None
        self.start_pos: tuple[int, int] | None = None
        self.raw_avatar_centroid: tuple[float, float] | None = None
        self.raw_start_pos: tuple[float, float] | None = None
        self.initial_grid: np.ndarray | None = None
        self.action_queue: list[int] = []
        self.visited_cells: set[tuple[int, int]] = set()
        self.learned_walkable_cells: set[tuple[int, int]] = set()
        self.actuator_to_portals: dict[str, set[str]] = {}
        self.latching_portals: set[str] = set()
        self.target_zone_base_colors: set[int] = set()
        self.is_cooperative_handoff: bool = False
        self.interruption_stack: list[Any] = []
        self.gate_target_cell: tuple[int, int] | None = None
        self.gate_approach_cell: tuple[int, int] | None = None
        self.attempted_pickup_item_id: str | None = None
        self.picked_up_source_position: tuple[int, int] | None = None
        self.probe_reset_done: bool = False
        self.lattice_offset: tuple[int, int] = (0, 0)
        self.walkable_colors: set[int] = self.learned_walkable_colors

        # Seed from persistent cross-game memory
        self.cross_game_memory.transfer_to_agent(self)

    @classmethod
    def is_spatial_candidate(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment possesses 2D movement actions."""
        return any(a in available_actions for a in [1, 2, 3, 4])

    @classmethod
    def is_cooperative_candidate(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Backward-compatible alias for is_spatial_candidate."""
        return cls.is_spatial_candidate(grid, available_actions)

    def reset_episode(self, retain_dynamics: bool = False, is_retry: bool = False) -> None:
        """Reset agent state for a new level/episode."""
        self.blackbox.reset(retain_memory=retain_dynamics)

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        tags: list[str] | None = None,
        allow_soft_restart: bool = False,
    ) -> tuple[int, float]:
        """Plan next action by delegating to CognitiveBlackbox."""
        driver_input = DriverInput(
            raw_data=curr_grid,
            modality=DriverModality.GRID_2D,
            source_id="arc_agi",
            metadata={
                "grid_shape": tuple(curr_grid.shape),
                "step_size": self.step_size,
                "tags": tags or [],
            },
        )

        perception_data = {
            "grid": curr_grid,
            "grid_shape": tuple(curr_grid.shape),
            "tags": tags or [],
        }

        self.blackbox.observe(driver_input, perception_data)

        driver_actions = [DriverAction(action_id=a) for a in available_actions]
        action = self.blackbox.decide(driver_actions, source_id="arc_agi")

        return action.action_id, 0.5

    def update_causal_dynamics(
        self,
        action_id: int,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
    ) -> None:
        """Update blackbox with action-outcome feedback for trial-and-error learning."""
        if prev_grid.shape != curr_grid.shape:
            return

        # Compute observed delta from grid diff for the blackbox to learn from
        info: dict[str, Any] = {}
        diff_mask = prev_grid != curr_grid
        if np.any(diff_mask):
            info["grid_changed"] = True

        action = DriverAction(action_id=action_id)
        feedback = DriverFeedback(
            success=False,
            reward=0.0,
            terminated=False,
            info=info,
        )
        self.blackbox.update(action, feedback, source_id="arc_agi")

    def record_episode_outcome(self, completed: bool, reason: str = "") -> None:
        """Record trial outcome in cross-game memory."""
        if completed:
            self.cross_game_memory.record_success(
                session_id="episode",
                score=1.0,
            )

    def export_knowledge(self) -> dict[str, Any]:
        """Export serialized cross-game knowledge dictionary."""
        return self.cross_game_memory.export_dict()

    def import_knowledge(self, data: dict[str, Any]) -> None:
        """Import cross-game knowledge."""
        self.cross_game_memory.import_dict(data)

    async def plan_next_action_counterfactual(
        self, grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        """Backward-compatible async counterfactual planner wrapper."""
        return self.plan_next_action(grid, available_actions)


# ── Backward-compatible aliases ──────────────────────────────────────────────
ARC3InteractiveAgent = ARC3SpatialCognitiveAgent

# Re-export benchmark classes from canonical location for backward compat
from plugins.arc_agi_adapter.arc_agi_3_runner import (  # noqa: E402, F401
    ARC3BenchmarkReport,
    ARC3BenchmarkRunner,
    ARC3EnvironmentResult,
    ARC3LevelResult,
)
