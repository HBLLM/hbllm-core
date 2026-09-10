"""
Overcooked Action Adapter.

Device driver for Overcooked kitchen environments.
Cognitive recipe scheduling and goal resolution are delegated entirely to UnifiedReasoningRuntime
via EmbodiedCausalOperator backward chaining. This adapter handles affordance declaration,
low-level BFS motor pathfinding, and physical collision avoidance.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph

from .perception import OvercookedPerceptionAdapter
from .predicates import register_overcooked_predicates
from .types import (
    CulinaryItem,
    KitchenTile,
    OvercookedAction,
    OvercookedObservation,
    PotStatus,
)

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = [
    (OvercookedAction.UP, -1, 0),
    (OvercookedAction.DOWN, 1, 0),
    (OvercookedAction.LEFT, 0, -1),
    (OvercookedAction.RIGHT, 0, 1),
]


class OvercookedActionAdapter:
    """
    Pure Device Driver for Overcooked environments.

    Decoupled into:
    1. enumerate_affordances: Declares ActionNodes with causal preconditions and recipe outputs.
    2. execute_action / dispatch: Low-level motor actuation (BFS navigation, orientation, interact).
    3. select_action: Routes cognitive planning through UnifiedReasoningRuntime.
    """

    def __init__(self) -> None:
        register_overcooked_predicates()
        self.planned_actions: list[OvercookedAction] = []
        self.expected_pos: tuple[int, int] | None = None
        self.perception = OvercookedPerceptionAdapter()
        self.runtime = UnifiedReasoningRuntime(create_default_operator_registry())

    def reset(self) -> None:
        """Reset internal plan queue, position tracker, and perception cache."""
        self.planned_actions.clear()
        self.expected_pos = None
        self.perception.reset()

    def enumerate_affordances(
        self, obs: OvercookedObservation, graph: CognitiveGraph
    ) -> list[ActionNode]:
        """Declare candidate ActionNodes with recipe causal preconditions and outputs."""
        all_nodes = list(graph.all_nodes())
        has_onion_dispenser = any(
            getattr(n, "entity_name", None) == "onion_dispenser" for n in all_nodes
        )
        has_dish_dispenser = any(
            getattr(n, "entity_name", None) == "dish_dispenser" for n in all_nodes
        )
        has_serving_station = any(
            getattr(n, "entity_name", None) == "serving_station" for n in all_nodes
        )
        has_pot = any(getattr(n, "entity_name", None) == "pot" for n in all_nodes)
        has_shared_counter = any(
            getattr(n, "entity_name", None) == "shared_counter" for n in all_nodes
        )

        affordances: list[ActionNode] = []

        # 1. Culinary Serving & Plating Pipeline (Chef role)
        if has_pot:
            if has_serving_station:
                affordances.append(
                    ActionNode(
                        id="act_serve_soup",
                        intent="serve_soup",
                        requirements=["has(soup)", "near(serving_station)"],
                        produces=["soup_served"],
                    )
                )
            affordances.extend(
                [
                    ActionNode(
                        id="act_plate_soup",
                        intent="plate_soup",
                        requirements=["is_cooked(pot)", "has(dish)", "near(pot)"],
                        produces=["has(soup)"],
                    ),
                    ActionNode(
                        id="act_cook_pot",
                        intent="cook_pot",
                        requirements=["pot_has_items(pot, onion, 3)"],
                        produces=["is_cooked(pot)"],
                    ),
                    ActionNode(
                        id="act_put_onion_in_pot",
                        intent="put_onion_in_pot",
                        requirements=["has(onion)", "near(pot)"],
                        produces=["pot_has_items(pot, onion, 3)"],
                    ),
                    ActionNode(
                        id="act_wait_cooking",
                        intent="wait",
                        requirements=["is_cooking(pot)"],
                        produces=["is_cooked(pot)"],
                    ),
                    ActionNode(
                        id="act_approach_pot",
                        intent="approach_pot",
                        requirements=[],
                        produces=["near(pot)"],
                    ),
                ]
            )

        # 2. Dispensers and Stations
        if has_onion_dispenser:
            affordances.extend(
                [
                    ActionNode(
                        id="act_fetch_onion",
                        intent="fetch_onion",
                        requirements=["has(none)", "near(onion_dispenser)"],
                        produces=["has(onion)"],
                    ),
                    ActionNode(
                        id="act_approach_onion_dispenser",
                        intent="approach_onion_dispenser",
                        requirements=[],
                        produces=["near(onion_dispenser)"],
                    ),
                ]
            )

        if has_dish_dispenser:
            affordances.extend(
                [
                    ActionNode(
                        id="act_fetch_dish",
                        intent="fetch_dish",
                        requirements=["has(none)", "near(dish_dispenser)"],
                        produces=["has(dish)"],
                    ),
                    ActionNode(
                        id="act_approach_dish_dispenser",
                        intent="approach_dish_dispenser",
                        requirements=[],
                        produces=["near(dish_dispenser)"],
                    ),
                ]
            )

        if has_serving_station:
            affordances.append(
                ActionNode(
                    id="act_approach_serving_station",
                    intent="approach_serving_station",
                    requirements=[],
                    produces=["near(serving_station)"],
                )
            )

        # 3. Partitioned Multi-Agent Coordination (Shared Counters)
        if has_shared_counter:
            affordances.extend(
                [
                    ActionNode(
                        id="act_pass_onion_to_counter",
                        intent="deposit_on_counter",
                        requirements=["has(onion)", "near(shared_counter)"],
                        produces=["onion_on_counter"],
                    ),
                    ActionNode(
                        id="act_fetch_onion_from_counter",
                        intent="fetch_from_counter",
                        requirements=["has(none)", "near(shared_counter)", "onion_on_counter"],
                        produces=["has(onion)"],
                    ),
                    ActionNode(
                        id="act_pass_soup_to_counter",
                        intent="deposit_on_counter",
                        requirements=["has(soup)", "near(shared_counter)"],
                        produces=["soup_on_counter"],
                    ),
                    ActionNode(
                        id="act_fetch_soup_from_counter",
                        intent="fetch_from_counter",
                        requirements=["has(none)", "near(shared_counter)", "soup_on_counter"],
                        produces=["has(soup)"],
                    ),
                    ActionNode(
                        id="act_pass_dish_to_counter",
                        intent="deposit_dish_on_counter",
                        requirements=["has(dish)", "near(shared_counter)"],
                        produces=["dish_on_counter"],
                    ),
                    ActionNode(
                        id="act_fetch_dish_from_counter",
                        intent="fetch_dish_from_counter",
                        requirements=["has(none)", "near(shared_counter)", "dish_on_counter"],
                        produces=["has(dish)"],
                    ),
                    ActionNode(
                        id="act_approach_shared_counter",
                        intent="approach_shared_counter",
                        requirements=[],
                        produces=["near(shared_counter)"],
                    ),
                    ActionNode(
                        id="act_deposit_excess",
                        intent="deposit_on_counter",
                        requirements=["near(shared_counter)"],
                        produces=["has(none)"],
                    ),
                ]
            )

        # 4. Default Idle
        affordances.append(
            ActionNode(
                id="act_stay",
                intent="stay",
                requirements=[],
                produces=["idle_done"],
            )
        )

        for act in affordances:
            if graph.has_node(act.id):
                graph.remove_node(act.id)
            graph.add_node(act)

        return affordances

    def select_action(
        self,
        obs: OvercookedObservation,
        perception_data: dict[str, Any],
    ) -> OvercookedAction:
        """Select next kitchen action via UnifiedReasoningRuntime with dynamic collision avoidance."""
        # 1. Collision and unexpected bump detection
        if self.expected_pos is not None and obs.agent.pos != self.expected_pos:
            self.planned_actions.clear()
            if obs.agent.agent_id == 1:
                self.expected_pos = obs.agent.pos
                return OvercookedAction.STAY

        # 2. Dynamic partner trajectory conflict check
        if self.planned_actions and obs.partner is not None:
            first_act = self.planned_actions[0]
            for act, dr, dc in DIRECTION_DELTAS:
                if act == first_act:
                    dest = (obs.agent.pos[0] + dr, obs.agent.pos[1] + dc)
                    if dest == obs.partner.pos:
                        self.planned_actions.clear()
                        if obs.agent.agent_id == 1:
                            for evasive_act, er, ec in DIRECTION_DELTAS:
                                cand = (obs.agent.pos[0] + er, obs.agent.pos[1] + ec)
                                if (
                                    0 <= cand[0] < len(obs.grid)
                                    and 0 <= cand[1] < len(obs.grid[0])
                                    and obs.grid[cand[0]][cand[1]] == int(KitchenTile.FLOOR)
                                    and cand != obs.partner.pos
                                    and cand != dest
                                ):
                                    self.expected_pos = cand
                                    return evasive_act
                        self.expected_pos = obs.agent.pos
                        return OvercookedAction.STAY

        # 3. If plan queue is empty, query UnifiedReasoningRuntime
        if not self.planned_actions:
            self._plan_cognitive_step(obs, perception_data)

        # 4. Pop next motor action
        if self.planned_actions:
            action = self.planned_actions.pop(0)
            pr, pc = obs.agent.pos
            dr, dc = (0, 0)
            for act, r_delta, c_delta in DIRECTION_DELTAS:
                if act == action:
                    dr, dc = r_delta, c_delta
                    break
            nr, nc = pr + dr, pc + dc
            if (
                0 <= nr < len(obs.grid)
                and 0 <= nc < len(obs.grid[0])
                and obs.grid[nr][nc] == int(KitchenTile.FLOOR)
            ):
                self.expected_pos = (nr, nc)
            else:
                self.expected_pos = (pr, pc)
            return action

        self.expected_pos = obs.agent.pos
        return OvercookedAction.STAY

    def _plan_cognitive_step(
        self,
        obs: OvercookedObservation,
        perception_data: dict[str, Any],
    ) -> None:
        """Query UnifiedReasoningRuntime with EmbodiedCausalOperator and dispatch chosen intent."""
        # --- Held-item fast-path: bypass causal backward-chaining when the agent
        # already holds an item and the next action is unambiguous.  This prevents
        # the re-plan deadlock where the backward chainer resolves deposit_excess
        # (to satisfy has(none) for fetch_from_counter) even though the agent
        # already has the ingredient it needs.
        held = obs.agent.held_item
        pots = obs.pots
        ready_pots = [p for p in pots if p.status == PotStatus.READY]
        filling_pots = [
            p
            for p in pots
            if p.onions_in_pot < p.required_onions
            and p.status in (PotStatus.EMPTY, PotStatus.FILLING)
        ]

        if held == CulinaryItem.ONION:
            # If any reachable filling/empty pot exists → put onion in it.
            target_pots = filling_pots or pots
            reachable = [p for p in target_pots if self._is_reachable(obs, p.pos)]
            if reachable:
                self._dispatch_intent("put_onion_in_pot", obs, perception_data)
                return
            # No reachable pot: deposit on shared counter for partner.
            self._dispatch_intent("deposit_on_counter", obs, perception_data)
            return

        if held == CulinaryItem.DISH:
            # If a ready pot is reachable → plate the soup.
            reachable_ready = [p for p in ready_pots if self._is_reachable(obs, p.pos)]
            if reachable_ready:
                self._dispatch_intent("plate_soup", obs, perception_data)
                return
            # Otherwise deposit dish so partner can use it.
            self._dispatch_intent("deposit_on_counter", obs, perception_data)
            return

        if held == CulinaryItem.SOUP:
            serving_stations = (
                perception_data.get("serving_stations") or self.perception.serving_stations
            )
            reachable_serving = [s for s in serving_stations if self._is_reachable(obs, s)]
            if reachable_serving:
                self._dispatch_intent("serve_soup", obs, perception_data)
                return
            # Can't reach serving station: deposit soup on shared counter for partner.
            self._dispatch_intent("deposit_on_counter", obs, perception_data)
            return

        # --- General path: delegate to UnifiedReasoningRuntime ---
        graph = self.perception.to_cognitive_graph(obs, perception_data)
        self.enumerate_affordances(obs, graph)

        problem = ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=("goal_active",),
            description="Overcooked recipe causal resolution",
        )
        trace = self.runtime.reason(graph=graph, problem=problem)

        chosen_intent = "stay"
        if (
            trace
            and trace.final_result
            and trace.final_result.conclusions
            and "best_action" in trace.final_result.conclusions
        ):
            chosen_intent = trace.final_result.conclusions["best_action"]

        self._dispatch_intent(chosen_intent, obs, perception_data)

    def _dispatch_intent(
        self,
        intent: str,
        obs: OvercookedObservation,
        perception_data: dict[str, Any],
    ) -> None:
        """Actuator dispatch: translates high-level causal intent into motor trajectory."""
        onion_dispensers = (
            perception_data.get("onion_dispensers") or self.perception.onion_dispensers
        )
        dish_dispensers = perception_data.get("dish_dispensers") or self.perception.dish_dispensers
        serving_stations = (
            perception_data.get("serving_stations") or self.perception.serving_stations
        )
        pots = obs.pots
        ready_pots = [p for p in pots if p.status == PotStatus.READY]
        filling_pots = [
            p
            for p in pots
            if p.onions_in_pot < p.required_onions
            and p.status in (PotStatus.EMPTY, PotStatus.FILLING)
        ]
        cooking_pots = [p for p in pots if p.status == PotStatus.COOKING]

        # 1. Delivery
        if intent in ("serve_soup", "approach_serving_station"):
            reachable_serving = [s for s in serving_stations if self._is_reachable(obs, s)]
            if reachable_serving:
                target = self._nearest_appliance(obs.agent.pos, reachable_serving)
                self._plan_interact_with(obs, target)
                return

        # 2. Plating soup
        if intent == "plate_soup":
            target_pots = ready_pots or pots
            reachable_pots = [p for p in target_pots if self._is_reachable(obs, p.pos)]
            if reachable_pots:
                target_p = min(
                    reachable_pots,
                    key=lambda p: (
                        abs(obs.agent.pos[0] - p.pos[0]) + abs(obs.agent.pos[1] - p.pos[1])
                    ),
                )
                self._plan_interact_with(obs, target_p.pos)
                return

        # 3. Fetch dish
        if intent in ("fetch_dish", "approach_dish_dispenser"):
            reachable_dishes = [d for d in dish_dispensers if self._is_reachable(obs, d)]
            if reachable_dishes:
                target = self._nearest_appliance(obs.agent.pos, reachable_dishes)
                self._plan_interact_with(obs, target)
                return

        # 4. Cook pot
        if intent == "cook_pot":
            full_pots = [
                p
                for p in pots
                if p.onions_in_pot >= p.required_onions and self._is_reachable(obs, p.pos)
            ]
            if full_pots:
                self._plan_interact_with(obs, full_pots[0].pos)
                return

        # 5. Put onion in pot / Approach pot
        if intent in ("put_onion_in_pot", "approach_pot"):
            target_pots = filling_pots or ready_pots or cooking_pots or pots
            reachable_pots = [p for p in target_pots if self._is_reachable(obs, p.pos)]
            if reachable_pots:
                target_p = min(
                    reachable_pots,
                    key=lambda p: (
                        abs(obs.agent.pos[0] - p.pos[0]) + abs(obs.agent.pos[1] - p.pos[1])
                    ),
                )
                self._plan_interact_with(obs, target_p.pos)
                return

        # 6. Fetch onion
        if intent in ("fetch_onion", "approach_onion_dispenser"):
            shared_counters = self._get_shared_counters(obs)
            shared_onions = [
                pos
                for pos in shared_counters
                if obs.counter_items.get(pos) == CulinaryItem.ONION and self._is_reachable(obs, pos)
            ]
            reachable_onions = [
                o for o in onion_dispensers if self._is_reachable(obs, o, ignore_partner=True)
            ]
            if shared_onions and not reachable_onions:
                target = self._nearest_appliance(obs.agent.pos, shared_onions)
                self._plan_interact_with(obs, target)
                return

            if reachable_onions:
                target = self._nearest_appliance(obs.agent.pos, reachable_onions)
                self._plan_interact_with(obs, target)
                return

        # 6b. Fetch dish from shared counter
        if intent == "fetch_dish_from_counter":
            shared_counters = self._get_shared_counters(obs)
            dish_counters = [
                pos
                for pos in shared_counters
                if obs.counter_items.get(pos) == CulinaryItem.DISH and self._is_reachable(obs, pos)
            ]
            if dish_counters:
                target = self._nearest_appliance(obs.agent.pos, dish_counters)
                self._plan_interact_with(obs, target)
                return

        # 6c. Deposit dish on shared counter
        if intent == "deposit_dish_on_counter":
            counter = self._find_shared_counter(obs, empty_only=True) or self._find_empty_counter(
                obs
            )
            if counter is not None:
                self._plan_interact_with(obs, counter)
                return

        # 7. Shared counter interaction
        if intent in ("deposit_on_counter", "approach_shared_counter"):
            counter = self._find_shared_counter(obs, empty_only=True) or self._find_empty_counter(
                obs
            )
            if counter is not None:
                self._plan_interact_with(obs, counter)
                return

        if intent == "fetch_from_counter":
            shared_counters = self._get_shared_counters(obs)
            counter_items = [
                pos
                for pos in shared_counters
                if pos in obs.counter_items and self._is_reachable(obs, pos)
            ]
            if counter_items:
                target = min(
                    counter_items,
                    key=lambda p: (
                        0 if obs.counter_items[p] == CulinaryItem.SOUP else 1,
                        abs(obs.agent.pos[0] - p[0]) + abs(obs.agent.pos[1] - p[1]),
                    ),
                )
                self._plan_interact_with(obs, target)
                return

        if intent == "wait":
            self.planned_actions = [OvercookedAction.STAY]
            return

        # Courtesy Yield: If empty-handed and standing adjacent to pot/station partner needs
        if (
            obs.agent.held_item == CulinaryItem.NONE
            and obs.partner is not None
            and obs.partner.held_item != CulinaryItem.NONE
        ):
            blocking = any(
                abs(obs.agent.pos[0] - p.pos[0]) + abs(obs.agent.pos[1] - p.pos[1]) == 1
                for p in obs.pots
            )
            if blocking:
                for act, dr, dc in DIRECTION_DELTAS:
                    nr, nc = obs.agent.pos[0] + dr, obs.agent.pos[1] + dc
                    if (
                        0 <= nr < len(obs.grid)
                        and 0 <= nc < len(obs.grid[0])
                        and obs.grid[nr][nc] == int(KitchenTile.FLOOR)
                    ):
                        if (nr, nc) != obs.partner.pos and not any(
                            abs(nr - p.pos[0]) + abs(nc - p.pos[1]) == 1 for p in obs.pots
                        ):
                            self.planned_actions = [act]
                            return

        self.planned_actions = [OvercookedAction.STAY]

    def _nearest_appliance(
        self, agent_pos: tuple[int, int], locations: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """Return closest location by Manhattan distance."""
        return min(locations, key=lambda p: abs(agent_pos[0] - p[0]) + abs(agent_pos[1] - p[1]))

    def _get_shared_counters(self, obs: OvercookedObservation) -> list[tuple[int, int]]:
        """Get all counter locations reachable by both agents (or reachable by agent if solo)."""
        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0
        shared = []
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(KitchenTile.COUNTER):
                    if self._is_reachable(
                        obs, (r, c), ignore_partner=True, from_partner=False
                    ) and (
                        obs.partner is None
                        or self._is_reachable(obs, (r, c), ignore_partner=True, from_partner=True)
                    ):
                        shared.append((r, c))
        return shared

    def _find_shared_counter(
        self, obs: OvercookedObservation, empty_only: bool = True
    ) -> tuple[int, int] | None:
        """Find nearest shared counter accessible by both agent and partner."""
        shared = self._get_shared_counters(obs)
        ar, ac = obs.agent.pos
        best_counter = None
        best_dist = float("inf")
        for pos in shared:
            if empty_only and pos in obs.counter_items:
                continue
            if self._is_reachable(obs, pos):
                dist = abs(ar - pos[0]) + abs(ac - pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_counter = pos
        return best_counter

    def _find_empty_counter(self, obs: OvercookedObservation) -> tuple[int, int] | None:
        """Find nearest accessible counter tile not currently holding an item."""
        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0
        best_counter = None
        best_dist = float("inf")
        ar, ac = obs.agent.pos
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(KitchenTile.COUNTER) and (r, c) not in obs.counter_items:
                    if self._is_reachable(obs, (r, c)):
                        dist = abs(ar - r) + abs(ac - c)
                        if dist < best_dist:
                            best_dist = dist
                            best_counter = (r, c)
        return best_counter

    def _is_reachable(
        self,
        obs: OvercookedObservation,
        appliance_pos: tuple[int, int],
        ignore_partner: bool = False,
        from_partner: bool = False,
    ) -> bool:
        """Check if any adjacent interaction cell is reachable by agent."""
        start_pos = obs.partner.pos if from_partner and obs.partner is not None else obs.agent.pos
        other_pos = (
            None
            if ignore_partner
            else (obs.agent.pos if from_partner else (obs.partner.pos if obs.partner else None))
        )
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (appliance_pos[0] + dr, appliance_pos[1] + dc)
            if (
                self._bfs_path(
                    obs,
                    start_pos,
                    adj,
                    ignore_partner=ignore_partner,
                    other_pos=other_pos,
                )
                is not None
            ):
                return True
        return False

    def _plan_interact_with(
        self, obs: OvercookedObservation, appliance_pos: tuple[int, int]
    ) -> None:
        """Find path to adjacent interaction cell and face appliance to interact."""
        best_path: list[OvercookedAction] | None = None
        best_interact_turn: OvercookedAction | None = None

        for act, dr, dc in DIRECTION_DELTAS:
            adj = (appliance_pos[0] - dr, appliance_pos[1] - dc)
            path = self._bfs_path(obs, obs.agent.pos, adj)
            if path is not None:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                    best_interact_turn = act

        if best_path is not None and best_interact_turn is not None:
            self.planned_actions = list(best_path) + [
                best_interact_turn,
                OvercookedAction.INTERACT,
            ]
            return

        # If primary interaction cell is occupied by partner, queue adjacent to that cell
        if obs.partner is not None:
            for act, dr, dc in DIRECTION_DELTAS:
                adj = (appliance_pos[0] - dr, appliance_pos[1] - dc)
                if obs.partner.pos == adj:
                    for _, ndr, ndc in DIRECTION_DELTAS:
                        nadj = (adj[0] + ndr, adj[1] + ndc)
                        path = self._bfs_path(obs, obs.agent.pos, nadj)
                        if path is not None and len(path) > 0:
                            self.planned_actions = list(path)
                            return

        self.planned_actions = [OvercookedAction.STAY]

    def _bfs_path(
        self,
        obs: OvercookedObservation,
        start_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        ignore_partner: bool = False,
        other_pos: tuple[int, int] | None = None,
    ) -> list[OvercookedAction] | None:
        """BFS navigation avoiding partner collision and counter obstacles."""
        if start_pos == goal_pos:
            return []

        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0

        gr, gc = goal_pos
        if not (0 <= gr < height and 0 <= gc < width) or obs.grid[gr][gc] != int(KitchenTile.FLOOR):
            return None

        blocked_pos = (
            None
            if ignore_partner
            else (
                other_pos if other_pos is not None else (obs.partner.pos if obs.partner else None)
            )
        )

        queue: deque[tuple[tuple[int, int], list[OvercookedAction]]] = deque([(start_pos, [])])
        visited = {start_pos}

        while queue:
            curr_pos, path = queue.popleft()
            if curr_pos == goal_pos:
                return path

            cr, cc = curr_pos
            for act, dr, dc in DIRECTION_DELTAS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if obs.grid[nr][nc] == int(KitchenTile.FLOOR) and (nr, nc) != blocked_pos:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [act]))

        return None
