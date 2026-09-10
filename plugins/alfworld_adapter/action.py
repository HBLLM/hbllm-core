"""
ALFWorld Action Adapter and Causal Affordance Planner.

Decomposes complex household instructions into causal subgoals:
Search -> Open -> Acquire -> Transform (Clean/Heat/Cool/Examine) -> Deliver -> Place.
"""

from __future__ import annotations

import logging
import re

from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph

from .perception import ALFWorldPerceptionAdapter
from .types import (
    ALFWorldGoal,
    ALFWorldObservation,
    ALFWorldTaskType,
)

logger = logging.getLogger(__name__)


class ALFWorldActionAdapter:
    """
    HCIR Causal Task and Affordance Planner for ALFWorld.
    Selects valid admissible commands to progress toward goal completion with 0 LLM tokens.
    """

    def __init__(self) -> None:
        self.explored_receptacles: set[str] = set()
        self.known_locations: dict[str, str] = {}  # obj_id -> receptacle_name
        self.held_transformed: bool = False
        self.perception = ALFWorldPerceptionAdapter()
        self.runtime = UnifiedReasoningRuntime(create_default_operator_registry())

    def reset(self) -> None:
        """Reset internal exploration history and perception."""
        self.explored_receptacles.clear()
        self.known_locations.clear()
        self.held_transformed = False
        self.perception = ALFWorldPerceptionAdapter()

    def enumerate_affordances(
        self, obs: ALFWorldObservation, graph: CognitiveGraph
    ) -> list[ActionNode]:
        """Declare candidate ActionNodes matching current admissible text commands."""
        affordances: list[ActionNode] = []
        stale = [n.id for n in graph.all_nodes() if n.id.startswith("act_")]
        for sid in stale:
            graph.remove_node(sid)

        for cmd in obs.admissible_commands:
            cmd_norm = cmd.replace(" ", "_")
            if cmd.startswith("go to"):
                target = cmd.replace("go to ", "").strip()
                affordances.append(
                    ActionNode(
                        id=f"act_{cmd_norm}",
                        intent=cmd,
                        requirements=[],
                        produces=[f"near({target})"],
                    )
                )
            elif cmd.startswith("take "):
                target = cmd.replace("take ", "").split(" from ")[0].strip()
                affordances.append(
                    ActionNode(
                        id=f"act_{cmd_norm}",
                        intent=cmd,
                        requirements=[f"near({target})"],
                        produces=[f"holds({target})"],
                    )
                )
            elif cmd.startswith("put "):
                affordances.append(
                    ActionNode(
                        id=f"act_{cmd_norm}",
                        intent=cmd,
                        requirements=["holds(object)"],
                        produces=["goal_achieved"],
                    )
                )
            else:
                affordances.append(
                    ActionNode(
                        id=f"act_{cmd_norm}",
                        intent=cmd,
                        requirements=[],
                        produces=["state_changed"],
                    )
                )

        for act in affordances:
            graph.add_node(act)

        return affordances

    def plan_next_action(self, obs: ALFWorldObservation, goal: ALFWorldGoal) -> str:
        """Generate next admissible text action through HCIR causal subgoaling."""
        # 1. Update graph and declare affordances
        self.perception.ingest_observation(obs)
        goal_node = self.perception.ingest_goal(goal)
        self.enumerate_affordances(obs, self.perception.graph)

        # 2. Query UnifiedReasoningRuntime
        problem = ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=(goal_node.id,),
            description=f"ALFWorld task: {goal.task_type.value}",
        )
        self.runtime.reason(graph=self.perception.graph, problem=problem)
        admissible = set(obs.admissible_commands)
        current_loc = obs.current_location

        # 1. Update known object locations and explored receptacles
        if current_loc and "is closed" not in obs.text_obs:
            self.explored_receptacles.add(current_loc)

        m = re.search(r"On/in the (.+?), you see: (.+?)\.", obs.text_obs)
        if m:
            rec = m.group(1).strip()
            self.explored_receptacles.add(rec)
            for oid in m.group(2).split(","):
                self.known_locations[oid.strip()] = rec

        # Check if held item is already transformed
        if (
            "You clean " in obs.text_obs
            or "You heat " in obs.text_obs
            or "You cool " in obs.text_obs
            or "You turn on " in obs.text_obs
        ):
            self.held_transformed = True

        # Phase 1: Determine target object to acquire
        target_type = goal.target_object_type
        held_target = next((oid for oid in obs.inventory if target_type in oid), None)

        # If not holding target object, we must find and acquire it
        if held_target is None:
            self.held_transformed = False

            # Check if target object is visible in current location (and not already at target destination)
            if current_loc != goal.target_receptacle_type:
                take_cmd = next(
                    (c for c in admissible if c.startswith("take ") and target_type in c), None
                )
                if take_cmd:
                    return take_cmd

            # If current location is closed, open it if possible
            open_cmd = next((c for c in admissible if c.startswith("open ")), None)
            if open_cmd:
                return open_cmd

            # Check if target object is known to be in a specific receptacle (outside destination)
            known_rec = next(
                (
                    rec
                    for oid, rec in self.known_locations.items()
                    if target_type in oid and rec != goal.target_receptacle_type
                ),
                None,
            )
            if known_rec and current_loc != known_rec:
                go_cmd = f"go to {known_rec}"
                if go_cmd in admissible:
                    return go_cmd

            # Otherwise, explore an unexplored receptacle
            for cmd in admissible:
                if cmd.startswith("go to "):
                    rec = cmd[len("go to ") :].strip()
                    if rec not in self.explored_receptacles and rec != current_loc:
                        return cmd

            # Fallback to any admissible go to other than current
            go_cmd = next(
                (c for c in admissible if c.startswith("go to ") and c != f"go to {current_loc}"),
                None,
            )
            if go_cmd:
                return go_cmd

            return "look"

        # Phase 2: Transformation (Clean / Heat / Cool / Examine)
        needs_transform = goal.task_type in (
            ALFWorldTaskType.CLEAN_AND_PLACE,
            ALFWorldTaskType.HEAT_AND_PLACE,
            ALFWorldTaskType.COOL_AND_PLACE,
            ALFWorldTaskType.EXAMINE_IN_LIGHT,
        )

        if needs_transform and not self.held_transformed:
            apparatus = goal.apparatus_receptacle_type
            if apparatus and current_loc != apparatus:
                go_cmd = f"go to {apparatus}"
                if go_cmd in admissible:
                    return go_cmd

            # Open apparatus if needed
            open_cmd = next(
                (c for c in admissible if c.startswith("open ") and apparatus in c), None
            )
            if open_cmd:
                return open_cmd

            # Perform transformation command
            if goal.task_type == ALFWorldTaskType.CLEAN_AND_PLACE:
                clean_cmd = next((c for c in admissible if c.startswith("clean ")), None)
                if clean_cmd:
                    return clean_cmd
            elif goal.task_type == ALFWorldTaskType.HEAT_AND_PLACE:
                heat_cmd = next((c for c in admissible if c.startswith("heat ")), None)
                if heat_cmd:
                    return heat_cmd
            elif goal.task_type == ALFWorldTaskType.COOL_AND_PLACE:
                cool_cmd = next((c for c in admissible if c.startswith("cool ")), None)
                if cool_cmd:
                    return cool_cmd
            elif goal.task_type == ALFWorldTaskType.EXAMINE_IN_LIGHT:
                use_cmd = next((c for c in admissible if c.startswith("use ")), None)
                if use_cmd:
                    return use_cmd

        # Phase 3: Delivery to Target Receptacle
        target_rec = goal.target_receptacle_type
        if target_rec:
            if current_loc != target_rec:
                go_cmd = f"go to {target_rec}"
                if go_cmd in admissible:
                    return go_cmd

            # Open target receptacle if closed
            open_cmd = next(
                (c for c in admissible if c.startswith("open ") and target_rec in c), None
            )
            if open_cmd:
                return open_cmd

            # Put target object in/on receptacle
            put_cmd = next(
                (c for c in admissible if c.startswith("put ") and held_target in c), None
            )
            if put_cmd:
                return put_cmd

        # If already done or examine completed
        return "look"
