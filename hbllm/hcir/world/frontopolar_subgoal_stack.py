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
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector

logger = logging.getLogger(__name__)


class SubgoalType(StrEnum):
    """Categorical frontopolar executive subgoal intention."""

    PARK_IN_HOLDING_BAY = "park_in_holding_bay"
    TRANSIT_CORRIDOR = "transit_corridor"
    DELIVER_TO_GOAL = "deliver_to_goal"
    UNLOCK_REMOTE_MECHANISM = "unlock_remote_mechanism"
    ALIGN_OPTICAL_REFLECTOR = "align_optical_reflector"
    ACQUIRE_TOOL = "acquire_tool"
    COLLECT_ITEM = "collect_item"
    CLEAR_OBSTACLE = "clear_obstacle"
    NAVIGATE_TO = "navigate_to"
    BILATERAL_COORDINATION = "bilateral_coordination"
    SWITCH_FOCUS = "switch_focus"


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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export frontopolar subgoal as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"subgoal:{self.subgoal_type.value}",
            operator="==",
            value=f"({self.target_destination[0]},{self.target_destination[1]})",
            relation=f"prio:{self.priority:.1f}",
            target_entity=self.subgoal_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording frontopolar subgoal."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"subgoal_{self.subgoal_id}",
                node_data={
                    "subgoal_id": self.subgoal_id,
                    "subgoal_type": self.subgoal_type.value,
                    "target_entity_pos": self.target_entity_pos,
                    "target_destination": self.target_destination,
                    "priority": self.priority,
                    "is_terminal": self.is_terminal,
                    "prerequisites": self.prerequisite_subgoal_ids,
                },
            )
        ]

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

    def clear_subgoals(self) -> None:
        """Clear all active and pending subgoals."""
        self.reset_episode()

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

    def peek_active_subgoal(self) -> FrontopolarSubgoal | None:
        """Inspect the current active subgoal on top of stack."""
        return self.current_subgoal

    @property
    def has_pending_goals(self) -> bool:
        """True if cognitive stack has subgoals to execute."""
        return bool(self.stack)

    @property
    def depth(self) -> int:
        """Current stack depth."""
        return len(self.stack)

    def is_empty(self) -> bool:
        """Check if frontopolar subgoal stack is empty."""
        return len(self.stack) == 0

    def prune_completed_subgoals(self, completed_ids: set[str]) -> None:
        """W119: Prune satisfied subgoals from the active planning stack."""
        self.stack = [sg for sg in self.stack if sg.subgoal_id not in completed_ids]

    def push(self, subgoal: FrontopolarSubgoal) -> None:
        """Push a newly formulated subgoal to the top of the stack."""
        self.push_subgoal(subgoal)

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

    def pop_active_subgoal(self) -> FrontopolarSubgoal | None:
        """Alias for pop_subgoal to complete and pop the active subgoal."""
        return self.pop_subgoal()

    def is_subgoal_satisfied(
        self,
        subgoal: FrontopolarSubgoal,
        current_entity_positions: set[tuple[int, int]],
        avatar_pos: tuple[int, int] | None = None,
        held_features: set[int] | None = None,
        grid: np.ndarray | None = None,
        background_feature: int | None = None,
    ) -> bool:
        """Evaluate if the current subgoal's postconditions are satisfied."""
        dest = subgoal.target_destination
        if subgoal.subgoal_type in (SubgoalType.PARK_IN_HOLDING_BAY, SubgoalType.DELIVER_TO_GOAL):
            return dest in current_entity_positions or avatar_pos == dest
        if subgoal.subgoal_type in (SubgoalType.ACQUIRE_TOOL, SubgoalType.COLLECT_ITEM):
            if held_features is not None and subgoal.required_feature is not None:
                if subgoal.required_feature in held_features:
                    return True
            return avatar_pos == dest or (
                avatar_pos is not None
                and abs(avatar_pos[0] - dest[0]) + abs(avatar_pos[1] - dest[1]) <= 1
            )
        if subgoal.subgoal_type in (
            SubgoalType.CLEAR_OBSTACLE,
            SubgoalType.UNLOCK_REMOTE_MECHANISM,
        ):
            if grid is not None:
                r, c = dest
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    bg = background_feature if background_feature is not None else 0
                    obs_feat = subgoal.metadata.get("obstacle_feature")
                    if grid[r, c] == bg or (obs_feat is not None and grid[r, c] != obs_feat):
                        return True
            return False
        if subgoal.subgoal_type in (
            SubgoalType.TRANSIT_CORRIDOR,
            SubgoalType.ALIGN_OPTICAL_REFLECTOR,
            SubgoalType.BILATERAL_COORDINATION,
            SubgoalType.SWITCH_FOCUS,
            SubgoalType.NAVIGATE_TO,
        ):
            return avatar_pos == dest
        return False

    def decompose_target_goal(
        self,
        target_goal: tuple[int, int],
        grid: np.ndarray,
        avatar_pos: tuple[int, int],
        static_barriers: set[tuple[int, int]] | None = None,
        tools: Sequence[Any] | None = None,
        receptacles_or_gates: Sequence[Any] | None = None,
        switches_or_triggers: Sequence[Any] | None = None,
        held_features: set[int] | None = None,
        theory: Any | None = None,
    ) -> list[FrontopolarSubgoal]:
        """Brodmann Area 10 Cognitive Branching: Decompose obstructed master goal into nested subgoals.

        Pushes subgoals onto the stack such that the immediate prerequisite
        is at the top of the stack (TOS), and returns the sequence in execution order (TOS first).
        """
        master_goal = FrontopolarSubgoal(
            subgoal_id=f"terminal_goal_{target_goal[0]}_{target_goal[1]}",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=target_goal,
            target_destination=target_goal,
            priority=1.0,
            is_terminal=True,
        )

        subgoals_to_push: list[FrontopolarSubgoal] = [master_goal]
        held = held_features or set()

        gate_entities = list(receptacles_or_gates or [])
        tool_entities = list(tools or [])
        switch_entities = list(switches_or_triggers or [])

        for gate in gate_entities:
            gate_pos = gate.grid_pos if hasattr(gate, "grid_pos") else getattr(gate, "pos", None)
            if gate_pos is None and isinstance(gate, tuple):
                gate_pos = gate
            gate_feat = getattr(gate, "feature_id", None)
            if gate_feat is None and gate_pos is not None:
                r, c = gate_pos
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    gate_feat = int(grid[r, c])

            matching_tool = None
            for tool in tool_entities:
                tool_feat = getattr(tool, "feature_id", None)
                if tool_feat is not None:
                    if theory is not None and gate_feat is not None:
                        if not theory.is_tool_compatible(tool_feat, gate_feat):
                            continue
                    matching_tool = tool
                    break

            if gate_pos is not None:
                clear_sg = FrontopolarSubgoal(
                    subgoal_id=f"clear_gate_{gate_pos[0]}_{gate_pos[1]}",
                    subgoal_type=SubgoalType.CLEAR_OBSTACLE,
                    target_entity_pos=gate_pos,
                    target_destination=gate_pos,
                    required_feature=getattr(matching_tool, "feature_id", None)
                    if matching_tool
                    else None,
                    priority=2.0,
                    prerequisite_features={getattr(matching_tool, "feature_id", 0)}
                    if matching_tool and getattr(matching_tool, "feature_id", 0) not in held
                    else set(),
                    metadata={"obstacle_feature": gate_feat},
                )
                subgoals_to_push.append(clear_sg)

            if matching_tool is not None:
                tool_pos = (
                    matching_tool.grid_pos
                    if hasattr(matching_tool, "grid_pos")
                    else getattr(matching_tool, "pos", None)
                )
                tool_feat = getattr(matching_tool, "feature_id", None)
                if tool_pos is not None and (tool_feat is None or tool_feat not in held):
                    tool_sg = FrontopolarSubgoal(
                        subgoal_id=f"acquire_tool_{tool_pos[0]}_{tool_pos[1]}",
                        subgoal_type=SubgoalType.ACQUIRE_TOOL,
                        target_entity_pos=tool_pos,
                        target_destination=tool_pos,
                        required_feature=tool_feat,
                        priority=3.0,
                    )
                    subgoals_to_push.append(tool_sg)

        if not tool_entities and switch_entities:
            for sw in switch_entities:
                sw_pos = sw.grid_pos if hasattr(sw, "grid_pos") else getattr(sw, "pos", None)
                if sw_pos is None and isinstance(sw, tuple):
                    sw_pos = sw
                sw_feat = getattr(sw, "feature_id", None)
                if sw_pos is not None:
                    switch_sg = FrontopolarSubgoal(
                        subgoal_id=f"actuate_switch_{sw_pos[0]}_{sw_pos[1]}",
                        subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
                        target_entity_pos=sw_pos,
                        target_destination=sw_pos,
                        required_feature=sw_feat,
                        priority=2.5,
                    )
                    subgoals_to_push.append(switch_sg)

        self.clear_subgoals()
        for sg in subgoals_to_push:
            self.push_subgoal(sg)

        logger.info(
            "FrontopolarSubgoalStack: Decomposed goal %s into %d subgoals (TOS=%s)",
            target_goal,
            len(subgoals_to_push),
            self.current_subgoal.subgoal_type if self.current_subgoal else None,
        )
        return list(reversed(subgoals_to_push))

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

    def spawn_bilateral_switch_subgoal(
        self,
        partner_pos: tuple[int, int],
        partner_feat: int,
        target_destination: tuple[int, int],
        switch_action_id: Any = 5,
    ) -> FrontopolarSubgoal:
        """Spawn a frontopolar bilateral switch subgoal when the active agent's path is obstructed by a remote gate."""
        switch_id = f"switch_to_partner_{partner_feat}_{partner_pos[0]}_{partner_pos[1]}"
        subgoal = FrontopolarSubgoal(
            subgoal_id=switch_id,
            subgoal_type=SubgoalType.BILATERAL_COORDINATION,
            target_entity_pos=partner_pos,
            target_destination=target_destination,
            required_feature=partner_feat,
            priority=2.5,
            metadata={"switch_action": switch_action_id},
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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export frontopolar stack state as a declarative causal predicate."""
        top_id = self.current_subgoal.subgoal_id if self.current_subgoal else "empty"
        return CausalPredicate(
            variable="frontopolar_subgoal_stack",
            operator="==",
            value=top_id,
            relation=f"depth:{self.depth}/{self.max_depth}",
            target_entity="apfc_ba10",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export stack summary and all active subgoals as declarative causal predicates."""
        preds = [self.to_causal_predicate()]
        for sg in self.stack:
            preds.append(sg.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting frontopolar subgoal stack state."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="frontopolar_stack_summary",
                node_data={
                    "depth": self.depth,
                    "max_depth": self.max_depth,
                    "completed_count": len(self.completed_subgoals),
                },
            )
        ]
        for sg in self.stack:
            ops.extend(sg.to_transaction_operations())
        return ops


@dataclass
class ComputationBudget:
    """W120: Dynamic computation-budget allocation across executive planning horizons."""

    total_tokens: int = 1000
    search_depth_limit: int = 5
    beam_width: int = 4
    allocated_subgoal_budgets: dict[str, int] = field(default_factory=dict)


class ComputationBudgetAllocator:
    """W120: Allocates and dynamically scales computation budget based on cognitive complexity."""

    @staticmethod
    def allocate_budget(
        subgoals: list[FrontopolarSubgoal],
        total_budget: int = 1000,
        min_budget_per_subgoal: int = 50,
    ) -> ComputationBudget:
        """Allocates compute proportionally to subgoal priority and prerequisite depth."""
        if not subgoals:
            return ComputationBudget(total_tokens=total_budget)

        total_priority = sum(sg.priority for sg in subgoals) or 1.0
        allocations: dict[str, int] = {}

        for sg in subgoals:
            share = int(round((sg.priority / total_priority) * total_budget))
            alloc = max(min_budget_per_subgoal, share)
            allocations[sg.subgoal_id] = alloc

        return ComputationBudget(
            total_tokens=total_budget,
            search_depth_limit=min(10, max(3, len(subgoals) * 2)),
            beam_width=max(2, min(8, total_budget // 200)),
            allocated_subgoal_budgets=allocations,
        )
