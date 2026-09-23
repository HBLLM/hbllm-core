"""ARC-3 Spatial Cognitive Agent — Thin adapter delegating to CognitiveBlackbox.

The agent is a dumb I/O adapter. All cognition, planning, learning, and
reasoning lives inside the core CognitiveBlackbox. This module only:
  1. Wraps raw grids into DriverInput / perception data.
  2. Forwards actions/feedback to the blackbox.
  3. Maintains backward-compatible public API surface.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from hbllm.drivers.base import DriverAction, DriverFeedback, DriverInput, DriverModality
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.world.morphology import MorphologicalConcept
from hbllm.hcir.world.motor_calibration import StateMutationModel
from plugins.arc_agi_adapter.arc_driver import ARC3Driver
from plugins.arc_agi_adapter.arc_memory import AgentPhase, HCIRCrossGameMemory

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

        from plugins.arc_agi_adapter.arc_perception import arc_perception_lifter

        self.blackbox = CognitiveBlackbox()
        self.driver = ARC3Driver(name="arc_agi")
        self.blackbox.register_driver(self.driver)
        self.blackbox.register_lifter("arc_agi", arc_perception_lifter)

        # Feed domain-specific instructions to CognitiveBlackbox
        self.blackbox.feed_instructions(
            "arc_agi",
            {
                "interaction_actions": [5, 6, 7],
                "action_capabilities": {"interact": True, "pickup_drop": True, "undo": True},
            },
        )

        # Cross-game persistent memory
        self.cross_game_memory: HCIRCrossGameMemory = (
            shared_memory if shared_memory is not None else ARC3SpatialCognitiveAgent.global_memory
        )

        # ── Backward-compat stub state ────────────────────────────────
        # These exist ONLY so downstream consumers (inductive_learner, arc_memory)
        # that read/write agent attributes directly don't crash.
        # Over time these consumers should be refactored to use the blackbox.
        self.avatar_centroid: tuple[float, float] | None = None
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

        # Seed from persistent cross-game memory
        self.cross_game_memory.transfer_to_agent(self)

    @property
    def avatar_color(self) -> int | None:
        """Proxy to blackbox avatar feature for backward compatibility."""
        return self.blackbox.get_state("arc_agi").avatar_feature

    @avatar_color.setter
    def avatar_color(self, value: int | None) -> None:
        self.blackbox.get_state("arc_agi").avatar_feature = value

    @property
    def current_plan(self) -> list[Any]:
        """Proxy to blackbox current plan for backward compatibility."""
        return self.blackbox.get_state("arc_agi").current_plan

    @current_plan.setter
    def current_plan(self, value: list[Any]) -> None:
        self.blackbox.get_state("arc_agi").current_plan = list(value)

    @property
    def action_models(self) -> dict[int, Any]:
        """Proxy to blackbox action models for backward compatibility."""
        return self.blackbox.get_state("arc_agi").action_models

    @action_models.setter
    def action_models(self, value: dict[int, Any]) -> None:
        self.blackbox.get_state("arc_agi").action_models = dict(value)

    @property
    def learned_barrier_colors(self) -> set[int]:
        """Proxy to blackbox learned obstacle features."""
        return self.blackbox.get_state("arc_agi").learned_obstacle_features

    @learned_barrier_colors.setter
    def learned_barrier_colors(self, value: set[int]) -> None:
        self.blackbox.get_state("arc_agi").learned_obstacle_features = set(value)

    @property
    def learned_walkable_colors(self) -> set[int]:
        """Proxy to blackbox learned traversable features."""
        return self.blackbox.get_state("arc_agi").learned_traversable_features

    @learned_walkable_colors.setter
    def learned_walkable_colors(self, value: set[int]) -> None:
        self.blackbox.get_state("arc_agi").learned_traversable_features = set(value)

    @property
    def learned_item_colors(self) -> set[int]:
        """Proxy to blackbox learned target features."""
        return self.blackbox.get_state("arc_agi").learned_target_features

    @learned_item_colors.setter
    def learned_item_colors(self, value: set[int] | dict[int, Any]) -> None:
        if isinstance(value, dict):
            self.blackbox.get_state("arc_agi").learned_target_features = set(value.keys())
        else:
            self.blackbox.get_state("arc_agi").learned_target_features = set(value)

    @property
    def walkable_colors(self) -> set[int]:
        """Backward-compatible alias for learned_walkable_colors."""
        return self.learned_walkable_colors

    @walkable_colors.setter
    def walkable_colors(self, value: set[int]) -> None:
        self.learned_walkable_colors = value

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
        if not retain_dynamics:
            self.step_size = 1
            self.avatar_color = None
        self.blackbox.reset(source_id="arc_agi", retain_memory=retain_dynamics)
        self.blackbox.reset(source_id="default", retain_memory=retain_dynamics)

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
            "step_size": self.step_size,
            "tags": tags or [],
            "avatar_color": self.avatar_color,
            "target_zone_bounds": getattr(self, "target_zone_bounds", None),
        }
        self.blackbox.observe(driver_input, perception_data)

        blackbox_state = self.blackbox.get_state("arc_agi")

        # Prioritize exploratory probing of untested actions to calibrate motor models
        untested = [
            a
            for a in available_actions
            if a not in blackbox_state.action_models
            or getattr(blackbox_state.action_models[a], "probes_tested", 0) == 0
        ]
        if untested:
            return untested[0], 0.5

        driver_actions = [
            DriverAction(
                action_id=a,
                semantic_intent=(
                    SpatialActionIntent.INTERACT if a in (5, 6) else SpatialActionIntent.NAVIGATE
                ),
            )
            for a in available_actions
        ]
        action = self.blackbox.decide(driver_actions, source_id="arc_agi")

        model = blackbox_state.action_models.get(action.action_id)
        conf = getattr(model, "confidence", 0.5) if model is not None else 0.5
        if blackbox_state.current_plan:
            conf = max(conf, 0.85)

        return action.action_id, conf

    def update_causal_dynamics(
        self,
        action_id: int,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        won: bool = False,
        lost: bool = False,
    ) -> None:
        """Update blackbox with action-outcome feedback for trial-and-error learning."""
        if prev_grid.shape != curr_grid.shape:
            return

        diff_mask = prev_grid != curr_grid
        changed = bool(np.any(diff_mask))
        info: dict[str, Any] = {"grid_changed": changed}

        # Delegate to blackbox state for learned action models
        blackbox_state = self.blackbox.get_state("arc_agi")
        action_models = blackbox_state.action_models

        # Auto-detect moving entity if avatar_color is not yet known
        avatar_col = self.avatar_color
        if avatar_col is None and changed:
            cand_colors = [int(c) for c in np.unique(curr_grid[diff_mask])]
            bg = getattr(self, "background_color", None)
            if bg is None and prev_grid.size > 0:
                bg = int(np.bincount(prev_grid.flatten()).argmax())

            cands_with_size: list[tuple[int, int]] = []
            for c in cand_colors:
                if c == bg or c == 0:
                    continue
                prev_c = np.argwhere(prev_grid == c)
                curr_c = np.argwhere(curr_grid == c)
                if (
                    len(prev_c) > 0
                    and len(prev_c) == len(curr_c)
                    and len(prev_c) < (prev_grid.size * 0.25)
                ):
                    pr, pc = prev_c.mean(axis=0)
                    cr, cc = curr_c.mean(axis=0)
                    dist = math.hypot(cr - pr, cc - pc)
                    if dist >= 0.5:
                        cands_with_size.append((len(prev_c), c))
            if cands_with_size:
                # Select the most compact translating entity (controllable avatar sprite)
                cands_with_size.sort(key=lambda x: x[0])
                avatar_col = cands_with_size[0][1]
                self.avatar_color = avatar_col

        prev_pts = (
            np.argwhere(prev_grid == avatar_col)
            if avatar_col is not None
            else np.empty((0, 2), dtype=int)
        )
        curr_pts = (
            np.argwhere(curr_grid == avatar_col)
            if avatar_col is not None
            else np.empty((0, 2), dtype=int)
        )
        prev_set = {tuple(p) for p in prev_pts}
        curr_set = {tuple(p) for p in curr_pts}

        is_interaction = action_id in (5, 6, 7)
        is_terminal = won or lost
        diff_count = int(np.sum(diff_mask))
        is_global_transition = diff_count > int(prev_grid.size * 0.40)

        if avatar_col is not None and len(prev_pts) > 0 and len(curr_pts) > 0:
            pr, pc = prev_pts.mean(axis=0)
            cr, cc = curr_pts.mean(axis=0)
            dr, dc = int(round(cr - pr)), int(round(cc - pc))

            size_ratio = max(len(prev_pts), len(curr_pts)) / max(
                1, min(len(prev_pts), len(curr_pts))
            )
            is_valid_move = (
                not is_interaction
                and not is_terminal
                and not is_global_transition
                and size_ratio <= 1.35
                and (
                    self.step_size <= 1
                    or (abs(dr) <= 2 * self.step_size and abs(dc) <= 2 * self.step_size)
                )
                and (abs(dr) <= 8 and abs(dc) <= 8)
            )

            if is_valid_move and (dr != 0 or dc != 0):
                info["observed_delta"] = [dr, dc]
                stride = (
                    math.gcd(abs(dr), abs(dc)) if (dr != 0 and dc != 0) else max(abs(dr), abs(dc))
                )
                if stride > 0:
                    if self.step_size <= 1:
                        self.step_size = stride
                    else:
                        self.step_size = math.gcd(self.step_size, stride)
                    info["step_size"] = self.step_size

                # Align step_size with calibrated orthogonal motor dynamics
                calibrated_deltas = [
                    max(abs(m.delta_r), abs(m.delta_c))
                    for m in action_models.values()
                    if (m.delta_r != 0 or m.delta_c != 0) and getattr(m, "confidence", 0) >= 0.5
                ]
                if calibrated_deltas:
                    self.step_size = min(calibrated_deltas)
                    info["step_size"] = self.step_size
                # Sample from prev_grid at newly entered cells to record the traversed floor feature
                newly_entered = curr_set - prev_set
                cand_feat = (
                    int(prev_grid[next(iter(newly_entered))])
                    if newly_entered
                    else int(prev_grid[int(round(cr)), int(round(cc))])
                )
                if (
                    cand_feat not in blackbox_state.learned_obstacle_features
                    and cand_feat != avatar_col
                ):
                    info["traversed_feature"] = cand_feat

            elif not is_interaction and not is_terminal and not is_global_transition:
                # Movement was blocked by an obstacle
                info["observed_delta"] = [0, 0]
                model = action_models.get(action_id)
                expected_dr = getattr(model, "delta_r", 0) if model else 0
                expected_dc = getattr(model, "delta_c", 0) if model else 0
                if expected_dr or expected_dc:
                    h, w = prev_grid.shape
                    # Check exterior boundary points in direction of expected movement
                    cand_targets = {
                        (r + expected_dr, c + expected_dc) for r, c in prev_set
                    } - prev_set
                    bg = getattr(self, "background_color", None)
                    if bg is None and prev_grid.size > 0:
                        bg = int(np.bincount(prev_grid.flatten()).argmax())
                    for check_r, check_c in cand_targets:
                        if 0 <= check_r < h and 0 <= check_c < w:
                            feat = int(prev_grid[check_r, check_c])
                            if feat != avatar_col and feat != bg:
                                info["collision_feature"] = feat
                                break

        # Payload pickup/drop affordance detection (Action 5)
        if action_id == 5 and changed:
            if not blackbox_state.carrying.holding:
                diff_cells = np.argwhere(prev_grid != curr_grid)
                for dr_c in diff_cells:
                    r, c = int(dr_c[0]), int(dr_c[1])
                    if avatar_col is not None and prev_grid[r, c] != avatar_col:
                        if len(prev_pts) > 0:
                            pr, pc = prev_pts.mean(axis=0)
                            if math.hypot(r - pr, c - pc) <= 2.0:
                                blackbox_state.carrying.holding = True
                                blackbox_state.carrying.entity_id = f"item_{prev_grid[r, c]}"
                                blackbox_state.carrying.offset = (float(r - pr), float(c - pc))
                                break
            else:
                blackbox_state.carrying.holding = False
                blackbox_state.carrying.entity_id = ""
                blackbox_state.carrying.offset = (0.0, 0.0)

        # Empirical Goal / Hazard Attribution
        if won:
            # The cell entered resulted in victory
            newly_entered = curr_set - prev_set
            if newly_entered:
                sr, sc = next(iter(newly_entered))
                info["reached_feature"] = int(prev_grid[sr, sc])
            elif len(curr_pts) > 0:
                cr, cc = curr_pts.mean(axis=0)
                info["reached_feature"] = int(prev_grid[int(round(cr)), int(round(cc))])
            elif avatar_col is not None and len(prev_pts) > 0:
                model = action_models.get(action_id)
                edr = getattr(model, "delta_r", 0) if model else 0
                edc = getattr(model, "delta_c", 0) if model else 0
                pr, pc = prev_pts.mean(axis=0)
                tr, tc = int(round(pr + edr)), int(round(pc + edc))
                if 0 <= tr < prev_grid.shape[0] and 0 <= tc < prev_grid.shape[1]:
                    info["reached_feature"] = int(prev_grid[tr, tc])

        elif lost:
            # Terminated without winning — hazard stepped into
            newly_entered = curr_set - prev_set
            if newly_entered:
                sr, sc = next(iter(newly_entered))
                info["hazard_feature"] = int(prev_grid[sr, sc])
            elif avatar_col is not None and len(prev_pts) > 0:
                model = action_models.get(action_id)
                edr = getattr(model, "delta_r", 0) if model else 0
                edc = getattr(model, "delta_c", 0) if model else 0
                pr, pc = prev_pts.mean(axis=0)
                tr, tc = int(round(pr + edr)), int(round(pc + edc))
                if 0 <= tr < prev_grid.shape[0] and 0 <= tc < prev_grid.shape[1]:
                    info["hazard_feature"] = int(prev_grid[tr, tc])

        action = DriverAction(
            action_id=action_id,
            semantic_intent=(
                SpatialActionIntent.INTERACT
                if action_id in (5, 6, 7)
                else SpatialActionIntent.NAVIGATE
            ),
        )
        feedback = DriverFeedback(
            success=won,
            reward=1.0 if won else 0.0,
            terminated=(won or lost),
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

    def save_knowledge(self, path_or_dir: str | Path, game_id: str = "arc_agi") -> Path:
        """Persist learned KnowledgeGraph to disk via CognitiveBlackbox."""
        return self.blackbox.save_knowledge(path_or_dir, source_id=game_id)

    def load_knowledge(self, path_or_dir: str | Path, game_id: str = "arc_agi") -> bool:
        """Load persistent KnowledgeGraph from disk into CognitiveBlackbox."""
        loaded = self.blackbox.load_knowledge(path_or_dir, source_id=game_id)
        if loaded:
            st = self.blackbox.get_state(game_id)
            if hasattr(self, "step_size"):
                self.step_size = st.step_size
        return loaded

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
