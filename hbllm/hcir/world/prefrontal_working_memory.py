"""Prefrontal Working Memory & Hierarchical Subgoal Schema System.

Modeled on the primate dorsolateral prefrontal cortex (dlPFC):
1. Latent State Retention (Object Permanence): Retains representations of items/tools acquired
   by the agent even when they vanish from the sensory grid upon contact/pickup.
2. Hierarchical Subgoal Decomposition: Formulates and manages multi-stage behavioral schemas:
   [Acquire Tool / Key] -> [Transport & Actuate Barrier] -> [Reach Primary Goal].
3. Tool-Affordance Resonance: Learns and associates tool features with barrier features they unlock.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HeldItem:
    """An entity retained in latent working memory as carried / possessed by the agent."""

    feature_id: int
    role: str
    acquired_step: int
    original_position: tuple[int, int]
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgoalStage:
    """A discrete stage in a hierarchical prefrontal behavioral schema."""

    stage_id: str
    target_role: str
    target_positions: set[tuple[int, int]]
    required_feature: int | None = None
    action_on_arrival: int | None = None  # None = CONTACT, or explicit action like 5
    is_completed: bool = False


@dataclass
class SubgoalSchema:
    """An active hierarchical plan composed of sequential stages."""

    schema_name: str
    stages: list[SubgoalStage]
    current_stage_idx: int = 0

    @property
    def current_stage(self) -> SubgoalStage | None:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    def advance(self) -> SubgoalStage | None:
        if self.current_stage:
            self.current_stage.is_completed = True
        self.current_stage_idx += 1
        return self.current_stage

    @property
    def is_finished(self) -> bool:
        return self.current_stage_idx >= len(self.stages)


from hbllm.hcir.world.symbolic_constraints import SymbolicConstraintSolver


class PrefrontalWorkingMemory:
    """Biologically-modeled Prefrontal Working Memory (dlPFC) for latent state and subgoal reasoning."""

    def __init__(self) -> None:
        self.held_items: list[HeldItem] = []
        self.active_schema: SubgoalSchema | None = None
        self.unlocked_barriers: set[tuple[int, int]] = set()
        self.tool_barrier_affinities: dict[int, set[int]] = {}  # tool_feat -> set(barrier_feat)
        self.activated_triggers: set[tuple[int, int]] = set()
        self.topological_subgoals: list[tuple[int, int]] = []
        self.active_macro_goal: str | None = None
        self.focal_attention_point: tuple[int, int] | None = None
        self.constraint_solver: SymbolicConstraintSolver = SymbolicConstraintSolver()
        self.executive_directives: list[str] = []

    def load_instructions(self, instructions: Sequence[str] | str | None) -> None:
        """Store executive directives in working memory to guide cognitive policies."""
        if not instructions:
            return
        if isinstance(instructions, str):
            lines = [l.strip() for l in instructions.strip().split("\n") if l.strip()]
        else:
            lines = [str(l).strip() for l in instructions if str(l).strip()]
        self.executive_directives = lines
        logger.info("PrefrontalWorkingMemory: Loaded %d executive directives.", len(lines))

    def reset_episode(self, retain_long_term: bool = True) -> None:
        """Reset transient working memory for a new trial while optionally retaining cross-trial tool affinities."""
        self.held_items.clear()
        self.active_schema = None
        self.unlocked_barriers.clear()
        self.activated_triggers.clear()
        self.topological_subgoals.clear()
        self.active_macro_goal = None
        self.focal_attention_point = None
        self.constraint_solver.reset_episode()
        if not retain_long_term:
            self.tool_barrier_affinities.clear()

    def register_cardinality_constraint(
        self,
        center: tuple[int, int],
        count: int,
        grid_shape: tuple[int, int],
        radius: int = 1,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Register a local numerical count constraint and run unit propagation."""
        self.constraint_solver.register_observation(center, count, grid_shape, radius)
        return self.constraint_solver.propagate_constraints()

    def get_unrevealed_safe_cells(self) -> set[tuple[int, int]]:
        """Query cells deduced by prefrontal constraint solver to be guaranteed safe."""
        return self.constraint_solver.get_unrevealed_safe_cells()

    def register_topological_doorway(self, door_coord: tuple[int, int], target_room: int) -> None:
        """Register a doorway transition as an active macro-subgoal in prefrontal memory."""
        self.topological_subgoals.append(door_coord)
        self.active_macro_goal = f"doorway_to_room_{target_room}"

    def orient_attention(self, coord: tuple[int, int]) -> None:
        """Orient prefrontal attentional focus to surprising or salient coordinates."""
        self.focal_attention_point = coord

    def acquire_item(
        self,
        feature_id: int,
        role: str,
        step: int,
        position: tuple[int, int],
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Store an acquired entity into prefrontal working memory (latent object permanence)."""
        # Avoid duplicate entries for same item
        if not any(item.feature_id == feature_id for item in self.held_items):
            self.held_items.append(
                HeldItem(
                    feature_id=feature_id,
                    role=role,
                    acquired_step=step,
                    original_position=position,
                    properties=properties or {},
                )
            )
            logger.info(
                "WorkingMemory: Acquired item [feature=%d, role=%s] into latent store at step %d.",
                feature_id,
                role,
                step,
            )

    def expend_item(self, feature_id: int) -> bool:
        """Remove a consumable item from working memory when applied to a barrier."""
        for i, item in enumerate(self.held_items):
            if item.feature_id == feature_id:
                self.held_items.pop(i)
                logger.info("WorkingMemory: Expended held item [feature=%d].", feature_id)
                return True
        return False

    def is_holding(self, feature_id: int) -> bool:
        """Check if an item of specified feature is currently held in working memory."""
        return any(item.feature_id == feature_id for item in self.held_items)

    def register_tool_unlock(self, tool_feature: int, barrier_feature: int) -> None:
        """Learn causal resonance: tool_feature unlocks barriers of barrier_feature."""
        self.tool_barrier_affinities.setdefault(tool_feature, set()).add(barrier_feature)
        logger.info(
            "WorkingMemory: Learned tool-barrier resonance: Tool %d unlocks Barrier %d.",
            tool_feature,
            barrier_feature,
        )

    def formulate_tool_use_schema(
        self,
        avatar_pos: tuple[int, int],
        tools: Sequence[tuple[int, int, int]],  # (r, c, feature_id)
        barriers: Sequence[tuple[int, int, int]],  # (r, c, feature_id)
        goals: Sequence[tuple[int, int]],
    ) -> SubgoalSchema | None:
        """Formulate a 3-stage prefrontal behavioral schema:

        Stage 1: Acquire Tool/Key.
        Stage 2: Transport & Unlock Barrier.
        Stage 3: Navigate to Primary Goal.
        """
        if not goals:
            return None

        # If already holding a tool that matches an obstructing barrier:
        held_feats = {item.feature_id for item in self.held_items}

        stages: list[SubgoalStage] = []

        # Find barrier that obstructs goals
        obstructing_barriers = [
            b for b in barriers if any(abs(b[0] - g[0]) + abs(b[1] - g[1]) <= 3 for g in goals)
        ]
        target_barrier_cells = {(b[0], b[1]) for b in obstructing_barriers} or {
            (b[0], b[1]) for b in barriers
        }

        if not held_feats and tools:
            # Stage 1: Acquire nearest tool
            best_tool = min(
                tools, key=lambda t: abs(t[0] - avatar_pos[0]) + abs(t[1] - avatar_pos[1])
            )
            stages.append(
                SubgoalStage(
                    stage_id="ACQUIRE_TOOL",
                    target_role="tool",
                    target_positions={(best_tool[0], best_tool[1])},
                    required_feature=best_tool[2],
                )
            )

        if target_barrier_cells:
            # Stage 2: Apply to Barrier
            stages.append(
                SubgoalStage(
                    stage_id="UNLOCK_BARRIER",
                    target_role="barrier",
                    target_positions=target_barrier_cells,
                )
            )

        # Stage 3: Reach Final Goal
        stages.append(
            SubgoalStage(
                stage_id="REACH_GOAL",
                target_role="goal",
                target_positions=set(goals),
            )
        )

        schema = SubgoalSchema(
            schema_name="TOOL_ASSISTED_NAVIGATION",
            stages=stages,
        )
        self.active_schema = schema
        return schema
