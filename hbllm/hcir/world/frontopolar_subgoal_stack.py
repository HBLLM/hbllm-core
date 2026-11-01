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

import numpy as np

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
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
    causal_preconditions: list[CausalPredicate] = field(default_factory=list)
    bound_skill_name: str | None = None
    bound_skill: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def bind_skill(self, skill: Any) -> None:
        """Bind a declarative neuro-symbolic skill directly to this executive subgoal."""
        self.bound_skill = skill
        self.bound_skill_name = getattr(skill, "skill_name", str(skill))
        self.metadata["bound_skill"] = self.bound_skill_name

    def bind_causal_precondition(self, predicate: CausalPredicate) -> None:
        """Bind a verified causal prerequisite predicate that must be satisfied."""
        self.causal_preconditions.append(predicate)
        self.metadata.setdefault("causal_preconditions", []).append(str(predicate))

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
                    "id": f"subgoal_{self.subgoal_id}",
                    "node_type": "goal",
                    "category": "directives",
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

    def compile_to_instructions(self, author: str = "apfc_ba10") -> list[Instruction]:
        """Compile frontopolar executive subgoal into primitive HCIR bytecode instructions."""
        instructions: list[Instruction] = []
        for pre in self.causal_preconditions:
            instructions.append(
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={
                        "type": "precondition",
                        "variable": pre.variable,
                        "operator": pre.operator,
                        "value": str(pre.value),
                        "subgoal_id": self.subgoal_id,
                        "author": author,
                    },
                    cost_estimate=2,
                )
            )

        instructions.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "type": "subgoal_target",
                    "subgoal_id": self.subgoal_id,
                    "subgoal_type": self.subgoal_type.value,
                    "target_pos": list(self.target_entity_pos),
                    "destination": list(self.target_destination),
                    "priority": self.priority,
                    "author": author,
                },
                cost_estimate=5,
            )
        )

        capability_name = f"navigation.{self.subgoal_type.value.lower()}"
        instructions.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": capability_name,
                    "params": {
                        "subgoal_id": self.subgoal_id,
                        "target_pos": list(self.target_entity_pos),
                        "destination": list(self.target_destination),
                    },
                },
                cost_estimate=10,
            )
        )
        return instructions


class FrontopolarSubgoalStack:
    """Apex prefrontal executive subgoal hierarchy manager (Brodmann Area 10)."""

    def __init__(self, max_depth: int = 8) -> None:
        self.max_depth = max_depth
        self.stack: list[FrontopolarSubgoal] = []
        self.completed_subgoals: list[str] = []
        self.allocated_holding_bays: set[tuple[int, int]] = set()

    def on_boundary(self, boundary) -> None:
        """Episode lifecycle boundary handler (WP0.5). Delegates to the
        deprecated reset_episode for backward-compatible semantics."""
        self.reset_episode()

    @property
    def boundary_retention(self):
        """Declared retention policy: default level-local semantics."""
        from hbllm.hcir.world.percept import RetentionPolicy

        return RetentionPolicy.LEVEL_LOCAL

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

    def schedule_bottleneck_relief_subgoal(
        self,
        bottleneck_cell: tuple[int, int],
        approach_cell: tuple[int, int],
        priority: float = 1.5,
    ) -> FrontopolarSubgoal:
        """Schedule high-priority corridor bottleneck navigation subgoal (BA10 cognitive branching)."""
        sg_id = f"bottleneck_relief_{bottleneck_cell[0]}_{bottleneck_cell[1]}"
        subgoal = FrontopolarSubgoal(
            subgoal_id=sg_id,
            subgoal_type=SubgoalType.TRANSIT_CORRIDOR,
            target_entity_pos=approach_cell,
            target_destination=bottleneck_cell,
            priority=priority,
            metadata={"bottleneck_cell": bottleneck_cell, "approach_cell": approach_cell},
        )
        self.push_subgoal(subgoal)
        return subgoal

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

    def decompose_subgoal_with_skill(
        self,
        subgoal_id: str,
        skill: Any,
    ) -> list[FrontopolarSubgoal]:
        """Decompose a high-level frontopolar subgoal into primitive subgoals derived from skill program."""
        target_sg = None
        target_idx = -1
        for i, sg in enumerate(self.stack):
            if sg.subgoal_id == subgoal_id:
                target_sg = sg
                target_idx = i
                break

        if target_sg is None or not hasattr(skill, "program") or not skill.program:
            return []

        decomposed: list[FrontopolarSubgoal] = []
        for idx, sub_item in enumerate(skill.program.subgoals):
            t_query = sub_item.target_query or {}
            role = t_query.get("role", "waypoint")
            sub_id = f"{subgoal_id}_decomp_{idx}"

            sg_type = SubgoalType.NAVIGATE_TO
            if role == "goal":
                sg_type = SubgoalType.DELIVER_TO_GOAL
            elif role in ("barrier", "obstacle"):
                sg_type = SubgoalType.CLEAR_OBSTACLE
            elif role in ("key", "tool"):
                sg_type = SubgoalType.ACQUIRE_TOOL
            elif role in ("switch", "mechanism"):
                sg_type = SubgoalType.UNLOCK_REMOTE_MECHANISM

            pos = t_query.get("pos") or target_sg.target_entity_pos
            dest = t_query.get("pos") or target_sg.target_destination

            child_sg = FrontopolarSubgoal(
                subgoal_id=sub_id,
                subgoal_type=sg_type,
                target_entity_pos=pos,
                target_destination=dest,
                priority=target_sg.priority + 0.1 * (len(skill.program.subgoals) - idx),
                prerequisite_subgoal_ids=[decomposed[-1].subgoal_id]
                if decomposed
                else list(target_sg.prerequisite_subgoal_ids),
                metadata={
                    "parent_subgoal_id": subgoal_id,
                    "skill_name": getattr(skill, "skill_name", ""),
                },
            )
            decomposed.append(child_sg)

        if decomposed:
            self.stack.pop(target_idx)
            for child in reversed(decomposed):
                self.stack.insert(target_idx, child)

        return decomposed

    def decompose_causal_chain(
        self,
        target_subgoal: FrontopolarSubgoal,
        causal_prerequisites: list[tuple[str, tuple[int, int], SubgoalType, int | None]],
    ) -> list[FrontopolarSubgoal]:
        """Decompose a high-level target goal into a causal prerequisite chain.

        Args:
            target_subgoal: The primary objective subgoal (e.g. deliver to terminal goal).
            causal_prerequisites: List of tuples (prereq_id, prereq_coord, prereq_type, required_feat)
                                  ordered from root dependency to immediate prerequisite.

        Returns:
            list[FrontopolarSubgoal]: The ordered sequence of subgoals pushed to the stack.
        """
        chain: list[FrontopolarSubgoal] = []
        prior_id: str | None = None

        for idx, (p_id, p_coord, p_type, req_feat) in enumerate(causal_prerequisites):
            prereq_ids = [prior_id] if prior_id else []
            sg = FrontopolarSubgoal(
                subgoal_id=p_id,
                subgoal_type=p_type,
                target_entity_pos=p_coord,
                target_destination=p_coord,
                required_feature=req_feat,
                priority=target_subgoal.priority + 0.2 * (len(causal_prerequisites) - idx),
                prerequisite_subgoal_ids=prereq_ids,
                metadata={"causal_chain_target": target_subgoal.subgoal_id, "chain_index": idx},
            )
            chain.append(sg)
            prior_id = p_id

        # Target subgoal now depends on the final prerequisite in the chain
        if prior_id:
            target_subgoal.prerequisite_subgoal_ids.append(prior_id)

        # Push target subgoal first (bottom of stack), followed by prerequisites in reverse order (LIFO top)
        self.push_subgoal(target_subgoal)
        for sg in reversed(chain):
            self.push_subgoal(sg)

        return chain

    def decompose_and_push_prerequisite(
        self,
        target_subgoal_id: str,
        prerequisite_subgoal: FrontopolarSubgoal,
    ) -> bool:
        """Temporarily suspend an active subgoal and push a prerequisite to top of stack."""
        for sg in self.stack:
            if sg.subgoal_id == target_subgoal_id:
                if prerequisite_subgoal.subgoal_id not in sg.prerequisite_subgoal_ids:
                    sg.prerequisite_subgoal_ids.append(prerequisite_subgoal.subgoal_id)
                self.push_subgoal(prerequisite_subgoal)
                return True
        # If target not found in stack, push prerequisite anyway
        self.push_subgoal(prerequisite_subgoal)
        return False

    def synthesize_declarative_program(
        self,
        completed_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Export executed subgoals as a structured declarative program with causal preconditions."""
        steps: list[dict[str, Any]] = []
        source_subgoals = (
            self.completed_subgoals if completed_only else [sg.subgoal_id for sg in self.stack]
        )

        for sg_id in source_subgoals:
            steps.append(
                {
                    "subgoal_id": sg_id,
                    "completed": True,
                }
            )
        return steps

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
            return dest in current_entity_positions
        if subgoal.subgoal_type in (
            SubgoalType.TRANSIT_CORRIDOR,
            SubgoalType.UNLOCK_REMOTE_MECHANISM,
            SubgoalType.ALIGN_OPTICAL_REFLECTOR,
        ):
            if avatar_pos is not None and avatar_pos == dest:
                return True
            if grid is not None and background_feature is not None and dest is not None:
                dr, dc = dest
                if 0 <= dr < grid.shape[0] and 0 <= dc < grid.shape[1]:
                    if int(grid[dr, dc]) == background_feature:
                        return True
            return False
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

    def decompose_target_goal(
        self,
        target_goal: tuple[int, int],
        grid: np.ndarray,
        avatar_pos: tuple[int, int],
        static_barriers: set[tuple[int, int]],
        tools: list[Any],
        receptacles_or_gates: list[tuple[int, int]],
        held_features: set[int] | None = None,
        theory: Any = None,
    ) -> list[FrontopolarSubgoal]:
        """Decompose a high-level target goal into prerequisite frontopolar subgoals.

        If transit to target_goal is obstructed by gates/receptacles requiring tools/keys:
        Schedules: [ACQUIRE_TOOL / COLLECT_ITEM] -> [CLEAR_OBSTACLE] -> [DELIVER_TO_GOAL].
        """
        held = held_features or set()
        subgoals: list[FrontopolarSubgoal] = []

        if tools and not held:
            closest_tool = min(
                tools,
                key=lambda t: (
                    abs(t.grid_pos[0] - avatar_pos[0]) + abs(t.grid_pos[1] - avatar_pos[1])
                ),
            )
            subgoals.append(
                FrontopolarSubgoal(
                    subgoal_id=f"acquire_tool_{closest_tool.grid_pos[0]}_{closest_tool.grid_pos[1]}",
                    subgoal_type=SubgoalType.ACQUIRE_TOOL,
                    target_entity_pos=closest_tool.grid_pos,
                    target_destination=closest_tool.grid_pos,
                    required_feature=getattr(closest_tool, "feature_id", None),
                    priority=2.0,
                )
            )

        if receptacles_or_gates:
            closest_gate = min(
                receptacles_or_gates,
                key=lambda g: abs(g[0] - avatar_pos[0]) + abs(g[1] - avatar_pos[1]),
            )
            subgoals.append(
                FrontopolarSubgoal(
                    subgoal_id=f"clear_gate_{closest_gate[0]}_{closest_gate[1]}",
                    subgoal_type=SubgoalType.CLEAR_OBSTACLE,
                    target_entity_pos=closest_gate,
                    target_destination=closest_gate,
                    priority=1.5,
                )
            )

        subgoals.append(
            FrontopolarSubgoal(
                subgoal_id=f"reach_goal_{target_goal[0]}_{target_goal[1]}",
                subgoal_type=SubgoalType.DELIVER_TO_GOAL,
                target_entity_pos=target_goal,
                target_destination=target_goal,
                priority=1.0,
            )
        )
        return subgoals

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
                    "id": "frontopolar_stack_summary",
                    "node_type": "event",
                    "category": "directives",
                    "depth": self.depth,
                    "max_depth": self.max_depth,
                    "completed_count": len(self.completed_subgoals),
                },
            )
        ]
        for sg in self.stack:
            ops.extend(sg.to_transaction_operations())
        return ops

    def compile_to_instruction_stream(self, author: str = "apfc_ba10") -> InstructionStream:
        """Compile active frontopolar subgoals into an executable HCIR InstructionStream."""
        stream = InstructionStream(
            author=author,
            description=f"Frontopolar Subgoal Execution Plan (depth={self.depth})",
        )
        for sg in self.stack:
            for ins in sg.compile_to_instructions(author=author):
                stream.append(ins)
        return stream


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
