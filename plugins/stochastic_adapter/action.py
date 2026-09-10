"""
Stochastic Action Adapter.

Implements closed-loop, surprise-resilient pathfinding with HCIR UnifiedReasoningRuntime
and automatic re-planning when sensory or actuator discrepancies occur.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph

from .perception import StochasticPerceptionAdapter
from .predicates import register_stochastic_predicates
from .types import StochasticAction, StochasticObservation

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = [
    (StochasticAction.UP, -1, 0),
    (StochasticAction.DOWN, 1, 0),
    (StochasticAction.LEFT, 0, -1),
    (StochasticAction.RIGHT, 0, 1),
]


class StochasticActionAdapter:
    """Surprise-reactive closed-loop navigation planner and device driver."""

    def __init__(self) -> None:
        self.planned_actions: list[StochasticAction] = []
        self.runtime = UnifiedReasoningRuntime(create_default_operator_registry())
        register_stochastic_predicates()

    def reset(self) -> None:
        """Clear action buffer."""
        self.planned_actions.clear()

    def enumerate_affordances(
        self, obs: StochasticObservation, graph: CognitiveGraph
    ) -> list[ActionNode]:
        """Declare candidate directional movement ActionNodes for HCIR reasoning."""
        affordances: list[ActionNode] = []
        stale = [n.id for n in graph.all_nodes() if n.id.startswith("act_")]
        for sid in stale:
            graph.remove_node(sid)

        for act, dr, dc in DIRECTION_DELTAS:
            act_name = act.name.lower()
            nr, nc = obs.player_pos[0] + dr, obs.player_pos[1] + dc
            affordances.append(
                ActionNode(
                    id=f"act_move_{act_name}",
                    intent=f"move_{act_name}",
                    requirements=[],
                    produces=[f"at_pos({nr},{nc})"],
                    properties={"action": act, "target_pos": (nr, nc)},
                )
            )

        affordances.append(
            ActionNode(
                id="act_noop",
                intent="noop",
                requirements=[],
                produces=[],
                properties={"action": StochasticAction.NOOP},
            )
        )

        for aff in affordances:
            graph.add_node(aff)

        return affordances

    def select_action(
        self,
        obs: StochasticObservation,
        perception: StochasticPerceptionAdapter,
        perception_data: dict[str, Any] | None = None,
    ) -> StochasticAction:
        """Select action, triggering instant re-plan if surprise detected or path invalidated."""
        if perception_data is None:
            perception_data = perception.process_observation(obs)
        else:
            perception.ingest_observation(obs)

        graph = perception.graph
        self.enumerate_affordances(obs, graph)
        goal_node = perception.ingest_goal()

        # Query UnifiedReasoningRuntime for high-level epistemic verification
        try:
            problem = ReasoningProblem(
                problem_type=ProblemType.PLANNING,
                goal_node_ids=(goal_node.id,),
                description="Stochastic navigation goal verification",
            )
            self.runtime.reason(graph=graph, problem=problem)
        except Exception as exc:
            logger.debug("UnifiedReasoningRuntime resolution fallback: %s", exc)

        surprise = perception_data.get("surprise_detected", False)
        target_pos = perception_data.get("target_pos")

        # Re-plan if surprise detected (actuator slip or sudden drift) or plan empty
        if surprise or not self.planned_actions:
            self._plan_path(obs.player_pos, target_pos, perception_data.get("obstacles", set()))

        action = self.planned_actions.pop(0) if self.planned_actions else StochasticAction.NOOP

        # Compute expected transition and register with perception
        pr, pc = obs.player_pos
        dr, dc = (0, 0)
        for act, r_delta, c_delta in DIRECTION_DELTAS:
            if act == action:
                dr, dc = r_delta, c_delta
                break

        expected_pos = (pr + dr, pc + dc)
        perception.register_expected_transition(expected_pos)

        return action

    def _plan_path(
        self,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int] | None,
        obstacles: set[tuple[int, int]],
    ) -> None:
        """Find shortest BFS path from current pos to believed target pos."""
        self.planned_actions.clear()
        if target_pos is None or start_pos == target_pos:
            return

        queue: deque[tuple[tuple[int, int], list[StochasticAction]]] = deque([(start_pos, [])])
        visited = {start_pos}

        while queue:
            curr_pos, path = queue.popleft()
            if curr_pos == target_pos:
                self.planned_actions = path
                return

            cr, cc = curr_pos
            for act, dr, dc in DIRECTION_DELTAS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    if (nr, nc) not in obstacles and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))
