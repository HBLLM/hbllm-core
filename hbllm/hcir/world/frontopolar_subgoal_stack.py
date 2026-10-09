"""Anterior Prefrontal Cortex (aPFC / Brodmann Area 10) Frontopolar Subgoal Stack.

Modeled on human frontopolar branching (Koechlin & Hyafil, 2007; Ramnani & Owen, 2004):
1. Cognitive Branching & Temporal Stacking: Maintains a suspended primary goal while executing
   an immediate secondary obstacle-clearing subgoal, seamlessly popping and resuming the master goal.
2. Safe Holding Bay Allocation: Discovers non-deadlock parking alcoves where movable obstacles
   or passive entities can be temporarily stashed without causing permanent corner or line-freeze deadlocks.
3. Bottleneck Relief Scheduling: When two entities contend for a narrow transit corridor,
   schedules sequential right-of-way routing: park entity A in holding bay -> move entity B through ->
   retrieve entity A and deliver to destination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector

logger = logging.getLogger(__name__)


class SubgoalType(StrEnum):
    """Categorical frontopolar executive subgoal intention."""

    PARK_IN_HOLDING_BAY = "park_in_holding_bay"
    TRANSIT_CORRIDOR = "transit_corridor"
    DELIVER_TO_GOAL = "deliver_to_goal"
    UNLOCK_REMOTE_MECHANISM = "unlock_remote_mechanism"
    ALIGN_OPTICAL_REFLECTOR = "align_optical_reflector"


@dataclass
class FrontopolarSubgoal:
    """A single executive subgoal on the frontopolar cognitive stack."""

    subgoal_id: str
    subgoal_type: SubgoalType
    target_entity_pos: tuple[int, int]
    target_destination: tuple[int, int]
    required_feature: int | None = None
    priority: float = 1.0
    is_terminal: bool = False
    prerequisite_subgoal_ids: list[str] = field(default_factory=list)
    prerequisite_features: set[int] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def are_prerequisites_met(
        self,
        completed_ids: set[str],
        held_features: set[int] | None = None,
    ) -> bool:
        """Verify whether all causal prerequisite subgoals and affordance features are fulfilled."""
        if any(req_id not in completed_ids for req_id in self.prerequisite_subgoal_ids):
            return False
        if held_features is not None:
            if any(feat not in held_features for feat in self.prerequisite_features):
                return False
        return True


class FrontopolarSubgoalStack:
    """Apex prefrontal executive subgoal hierarchy manager (Brodmann Area 10)."""

    def __init__(self, max_depth: int = 8) -> None:
        self.max_depth = max_depth
        self.stack: list[FrontopolarSubgoal] = []
        self.completed_subgoals: list[str] = []
        self.allocated_holding_bays: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        """Reset transient subgoal stack for fresh trial."""
        self.stack.clear()
        self.completed_subgoals.clear()
        self.allocated_holding_bays.clear()

    @property
    def current_subgoal(self) -> FrontopolarSubgoal | None:
        """Return the active top-of-stack subgoal being executed whose prerequisites are satisfied."""
        if not self.stack:
            return None
        completed_set = set(self.completed_subgoals)
        for sg in reversed(self.stack):
            if sg.are_prerequisites_met(completed_set):
                return sg
        return self.stack[-1]

    @property
    def has_pending_goals(self) -> bool:
        """True if cognitive stack has subgoals to execute."""
        return bool(self.stack)

    def push_subgoal(self, subgoal: FrontopolarSubgoal) -> None:
        """Push a newly formulated subgoal to the top of the stack (cognitive branching)."""
        if len(self.stack) >= self.max_depth:
            logger.warning(
                "FrontopolarSubgoalStack: Maximum stack depth reached, dropping bottom goal."
            )
            self.stack.pop(0)
        self.stack.append(subgoal)
        logger.info(
            "FrontopolarSubgoalStack: Pushed subgoal [%s: %s -> %s] (depth=%d)",
            subgoal.subgoal_type,
            subgoal.target_entity_pos,
            subgoal.target_destination,
            len(self.stack),
        )

    def pop_subgoal(self) -> FrontopolarSubgoal | None:
        """Pop and complete the current active subgoal, resuming the underlying suspended plan."""
        if not self.stack:
            return None
        completed = self.stack.pop()
        self.completed_subgoals.append(completed.subgoal_id)
        logger.info(
            "FrontopolarSubgoalStack: Completed subgoal [%s]. Stack depth now %d.",
            completed.subgoal_id,
            len(self.stack),
        )
        return completed

    def is_subgoal_satisfied(
        self,
        subgoal: FrontopolarSubgoal,
        current_entity_positions: set[tuple[int, int]],
        avatar_pos: tuple[int, int] | None = None,
    ) -> bool:
        """Evaluate if the current subgoal's postconditions are satisfied."""
        dest = subgoal.target_destination
        if subgoal.subgoal_type in (SubgoalType.PARK_IN_HOLDING_BAY, SubgoalType.DELIVER_TO_GOAL):
            return dest in current_entity_positions
        if subgoal.subgoal_type in (
            SubgoalType.TRANSIT_CORRIDOR,
            SubgoalType.UNLOCK_REMOTE_MECHANISM,
            SubgoalType.ALIGN_OPTICAL_REFLECTOR,
        ):
            return avatar_pos == dest
        return False

    def spawn_causal_unlock_subgoal(
        self,
        barrier_pos: tuple[int, int],
        barrier_feat: int,
        trigger_pos: tuple[int, int],
        trigger_feat: int,
    ) -> FrontopolarSubgoal:
        """Spawn a high-priority frontopolar unlock subgoal to clear a causal prerequisite barrier."""
        unlock_id = f"unlock_{barrier_feat}_via_{trigger_feat}_{trigger_pos[0]}_{trigger_pos[1]}"
        subgoal = FrontopolarSubgoal(
            subgoal_id=unlock_id,
            subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
            target_entity_pos=trigger_pos,
            target_destination=trigger_pos,
            required_feature=trigger_feat,
            priority=2.0,
        )
        self.push_subgoal(subgoal)
        return subgoal

    @staticmethod
    def find_safe_holding_bay(
        entity_pos: tuple[int, int],
        destination_goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        dynamic_obstacles: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        transit_corridor: set[tuple[int, int]] | None = None,
        max_search_radius: int = 6,
    ) -> tuple[int, int] | None:
        """Find a safe holding bay cell to park an obstruction without causing deadlocks.

        A valid holding bay:
        1. Is passable (not in barriers or existing obstacles).
        2. Is NOT in the critical transit corridor needed by other entities.
        3. Is NOT an irreversible deadlock location according to the OFC Deadlock Detector.
        4. Minimizes distance to original position to economize motor budget.
        """
        H, W = grid_shape
        er, ec = entity_pos
        corridor = transit_corridor or set()

        candidates: list[tuple[int, int, float]] = []

        for dr in range(-max_search_radius, max_search_radius + 1):
            for dc in range(-max_search_radius, max_search_radius + 1):
                r = er + dr
                c = ec + dc
                if not (0 <= r < H and 0 <= c < W):
                    continue
                cand = (r, c)
                if cand in static_barriers or cand in dynamic_obstacles or cand in corridor:
                    continue

                # OFC Deadlock Check: parking entity at cand must not be a deadlock!
                simulated_obstacles = set(dynamic_obstacles)
                simulated_obstacles.discard(entity_pos)
                simulated_obstacles.add(cand)

                dl_eval = CounterfactualDeadlockDetector.evaluate_deadlock(
                    cand,
                    goals=destination_goals,
                    static_barriers=static_barriers,
                    all_blocks=simulated_obstacles,
                    grid_shape=grid_shape,
                )
                if dl_eval.is_deadlock:
                    continue

                dist = abs(dr) + abs(dc)
                candidates.append((r, c, float(dist)))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[2])
        best_r, best_c, _ = candidates[0]
        return (best_r, best_c)
