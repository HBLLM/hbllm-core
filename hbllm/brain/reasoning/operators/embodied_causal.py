"""
Embodied Causal Reasoning Operator — domain-agnostic causal goal resolution over HCIR.

Resolves embodied goals by performing backward-chaining causal dependency analysis
across declarative ActionNodes and physical entity state:

    Goal State (e.g. holds(apple) or inside(apple, microwave))
        ↓
    Match Action that produces goal condition (e.g. PutObject)
        ↓
    Evaluate Action preconditions against current FrozenGraphView
        ├── If all preconditions met → Execute immediately
        └── If precondition missing (e.g. near(microwave), holds(apple))
                ↓
            Spawn causal sub-goal recursively
                ↓
            Find primitive action satisfying sub-goal (e.g. Navigate, Open, Pickup)

Independence Level: L1 (100% deterministic, 0 LLM tokens).
"""

from __future__ import annotations

import logging
import re
import time

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    FrozenGraphView,
    ProblemType,
    ProvenanceChain,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.hcir.graph import (
    ActionNode,
    GoalNode,
    HCIREdgeType,
    HCIRNodeType,
)

logger = logging.getLogger(__name__)

# Regex pattern for condition strings: predicate(arg1, arg2, ...)
_COND_PATTERN = re.compile(r"^([a-zA-Z0-9_]+)(?:\((.*)\))?$")


def parse_condition(cond: str) -> tuple[str, list[str]]:
    """Parse condition string 'predicate(arg1, arg2)' into ('predicate', ['arg1', 'arg2'])."""
    cleaned = cond.strip()
    match = _COND_PATTERN.match(cleaned)
    if not match:
        return cleaned, []
    pred = match.group(1)
    args_str = match.group(2)
    if not args_str:
        return pred, []
    args = [a.strip() for a in args_str.split(",") if a.strip()]
    return pred, args


class EmbodiedCausalOperator:
    """Domain-agnostic causal goal resolution and precondition backward-chaining operator.

    Evaluates active goals against the current physical state in FrozenGraphView
    and selects the causal primitive action that advances toward goal satisfaction.
    """

    def __init__(self, default_reach_distance: float = 1.6, max_recursion_depth: int = 10) -> None:
        self.default_reach_distance = default_reach_distance
        self.max_recursion_depth = max_recursion_depth

    @property
    def operator_id(self) -> str:
        return "embodied_causal"

    @property
    def operator_name(self) -> str:
        return "Embodied Causal Goal Resolution Operator"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(self, problem: ReasoningProblem, context: CognitiveContext) -> float:
        """Score applicability for planning and embodied manipulation problems."""
        view = context.graph_view
        n_actions = len(view.nodes_by_type(HCIRNodeType.ACTION))
        n_entities = len(view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY))

        if problem.problem_type == ProblemType.PLANNING:
            if n_actions > 0 and n_entities > 0:
                return 0.98
            elif n_actions > 0:
                return 0.90
            return 0.40
        elif problem.problem_type == ProblemType.CAUSAL:
            return 0.65
        return 0.05

    def estimated_cost(self, problem: ReasoningProblem, context: CognitiveContext) -> ResourceCost:
        n_actions = len(context.graph_view.nodes_by_type(HCIRNodeType.ACTION))
        n_entities = len(context.graph_view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY))
        return ResourceCost(
            wall_clock_ms=max(1.0, (n_actions + n_entities) * 0.1),
            nodes_read=n_actions + n_entities,
            edges_read=context.graph_view.edge_count,
        )

    def execute(self, problem: ReasoningProblem, context: CognitiveContext) -> CognitiveResult:
        """Execute backward-chaining causal resolution to select the next action."""
        start_time = time.time()
        view = context.graph_view

        # 1. Extract Goal Conditions
        goal_conditions = self._extract_goal_conditions(problem, view)
        if not goal_conditions:
            elapsed_ms = (time.time() - start_time) * 1000
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "No goal conditions found in problem or graph"},
                resource_cost=ResourceCost(wall_clock_ms=elapsed_ms, nodes_read=view.node_count),
            )

        # 2. Check if all goal conditions are already satisfied
        unsatisfied_initial = [
            c for c in goal_conditions if not self.is_condition_satisfied(c, view)
        ]
        if not unsatisfied_initial:
            elapsed_ms = (time.time() - start_time) * 1000
            return CognitiveResult(
                status=ResultStatus.SUCCESS,
                conclusions={
                    "status": "goal_satisfied",
                    "best_action": "no_op",
                    "action_id": "",
                    "action_properties": {},
                    "unsatisfied_conditions": [],
                },
                confidence=1.0,
                evidence_refs=[],
                provenance_chains=[
                    ProvenanceChain(
                        conclusion="All goal conditions are already satisfied in current state",
                        evidence_node_ids=[],
                        operator_id=self.operator_id,
                        reasoning_steps=["Goal state verified against active graph view"],
                        confidence=1.0,
                    )
                ],
                operator_id=self.operator_id,
                resource_cost=ResourceCost(
                    wall_clock_ms=elapsed_ms,
                    nodes_read=view.node_count,
                    edges_read=view.edge_count,
                ),
            )

        # 3. Extract Candidate Actions
        candidate_actions = [
            n for n in view.nodes_by_type(HCIRNodeType.ACTION) if isinstance(n, ActionNode)
        ]
        if not candidate_actions:
            elapsed_ms = (time.time() - start_time) * 1000
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "No ActionNodes available in graph view"},
                resource_cost=ResourceCost(wall_clock_ms=elapsed_ms, nodes_read=view.node_count),
            )

        # 4. Resolve Goal via Causal Backward Chaining
        provenance_steps: list[str] = []
        selected_action, unsatisfied_conditions = self._resolve_backward(
            goal_conditions=goal_conditions,
            candidate_actions=candidate_actions,
            view=view,
            depth=0,
            visited_conditions=set(),
            provenance_steps=provenance_steps,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Check if already satisfied (from backward chaining)
        if not unsatisfied_conditions:
            return CognitiveResult(
                status=ResultStatus.SUCCESS,
                conclusions={
                    "status": "goal_satisfied",
                    "best_action": "no_op",
                    "action_id": "",
                    "action_properties": {},
                    "unsatisfied_conditions": [],
                },
                confidence=1.0,
                evidence_refs=[],
                provenance_chains=[
                    ProvenanceChain(
                        conclusion="All goal conditions are already satisfied in current state",
                        evidence_node_ids=[],
                        operator_id=self.operator_id,
                        reasoning_steps=["Goal state verified against active graph view"],
                        confidence=1.0,
                    )
                ],
                operator_id=self.operator_id,
                resource_cost=ResourceCost(
                    wall_clock_ms=elapsed_ms,
                    nodes_read=view.node_count,
                    edges_read=view.edge_count,
                ),
            )

        if selected_action is None:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                conclusions={
                    "status": "unresolvable",
                    "best_action": "",
                    "action_id": "",
                    "unsatisfied_conditions": unsatisfied_conditions,
                },
                confidence=0.0,
                evidence_refs=[],
                provenance_chains=[
                    ProvenanceChain(
                        conclusion="Could not resolve causal chain to satisfy preconditions",
                        evidence_node_ids=[],
                        operator_id=self.operator_id,
                        reasoning_steps=provenance_steps,
                        confidence=0.0,
                    )
                ],
                operator_id=self.operator_id,
                resource_cost=ResourceCost(
                    wall_clock_ms=elapsed_ms,
                    nodes_read=view.node_count,
                    edges_read=view.edge_count,
                ),
            )

        # Successfully selected next primitive action
        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "status": "in_progress",
                "best_action": selected_action.intent,
                "action_id": selected_action.id,
                "action_properties": selected_action.properties,
                "requirements": selected_action.requirements,
                "produces": selected_action.produces,
                "unsatisfied_conditions": unsatisfied_conditions,
            },
            confidence=0.95,
            evidence_refs=[selected_action.id],
            provenance_chains=[
                ProvenanceChain(
                    conclusion=f"Selected action '{selected_action.intent}' to advance toward goal",
                    evidence_node_ids=[selected_action.id],
                    operator_id=self.operator_id,
                    reasoning_steps=provenance_steps,
                    confidence=0.95,
                )
            ],
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )

    # ── Causal Backward Chaining ─────────────────────────────────────

    def _resolve_backward(
        self,
        goal_conditions: list[str],
        candidate_actions: list[ActionNode],
        view: FrozenGraphView,
        depth: int,
        visited_conditions: set[str],
        provenance_steps: list[str],
    ) -> tuple[ActionNode | None, list[str]]:
        """Recursively resolve preconditions until an immediately executable action is found."""
        if depth > self.max_recursion_depth:
            provenance_steps.append(
                f"Depth limit ({self.max_recursion_depth}) exceeded in causal recursion"
            )
            return None, goal_conditions

        # 1. Identify which goal conditions are currently unsatisfied
        unsatisfied: list[str] = []
        for cond in goal_conditions:
            if not self.is_condition_satisfied(cond, view):
                unsatisfied.append(cond)
            else:
                provenance_steps.append(f"Condition '{cond}' is SATISFIED in current state")

        if not unsatisfied:
            return None, []

        # 2. Pick the primary unsatisfied condition to resolve
        target_cond = unsatisfied[0]
        if target_cond in visited_conditions:
            provenance_steps.append(f"Circular dependency detected for condition '{target_cond}'")
            return None, unsatisfied
        visited_conditions.add(target_cond)

        provenance_steps.append(f"[Depth {depth}] Resolving unsatisfied condition: '{target_cond}'")

        # 3. Find candidate action(s) that produce target_cond
        producing_actions = [
            act for act in candidate_actions if any(p.strip() == target_cond for p in act.produces)
        ]

        if not producing_actions:
            # Special case: check if target_cond is 'not_contained_in_closed(X)'
            # which requires opening the closed parent container
            pred, args = parse_condition(target_cond)
            if pred == "not_contained_in_closed" and args:
                target_obj_id = args[0]
                target_node = view.get_node(target_obj_id)
                if target_node and hasattr(target_node, "properties"):
                    parents = target_node.properties.get("parent_receptacles", [])
                    for parent_id in parents:
                        p_node = view.get_node(parent_id)
                        if p_node and hasattr(p_node, "properties"):
                            if p_node.properties.get("is_openable") and not p_node.properties.get(
                                "is_opened"
                            ):
                                open_cond = f"is_opened({parent_id})"
                                provenance_steps.append(
                                    f"Target '{target_obj_id}' is enclosed in closed '{parent_id}' -> Sub-goal: '{open_cond}'"
                                )
                                return self._resolve_backward(
                                    goal_conditions=[open_cond],
                                    candidate_actions=candidate_actions,
                                    view=view,
                                    depth=depth + 1,
                                    visited_conditions=visited_conditions,
                                    provenance_steps=provenance_steps,
                                )

            provenance_steps.append(f"No candidate action produces condition '{target_cond}'")
            return None, unsatisfied

        # 4. Evaluate producing action requirements
        best_candidate: ActionNode | None = None
        for act in producing_actions:
            provenance_steps.append(
                f"[Depth {depth}] Evaluating candidate action '{act.intent}' (requires {act.requirements})"
            )
            act_unsatisfied_reqs = [
                req for req in act.requirements if not self.is_condition_satisfied(req, view)
            ]
            # Prioritize container unblocking preconditions before direct approach
            act_unsatisfied_reqs.sort(
                key=lambda r: 0 if r.startswith("not_contained_in_closed") else 1
            )

            if not act_unsatisfied_reqs:
                # All requirements are satisfied! This action is immediately executable!
                provenance_steps.append(
                    f"[Depth {depth}] All requirements satisfied for '{act.intent}' -> SELECTING ACTION"
                )
                return act, unsatisfied

            # Action has unsatisfied requirements -> Recurse on requirements
            provenance_steps.append(
                f"[Depth {depth}] Action '{act.intent}' has unsatisfied requirements: {act_unsatisfied_reqs}"
            )
            sub_action, _ = self._resolve_backward(
                goal_conditions=act_unsatisfied_reqs,
                candidate_actions=candidate_actions,
                view=view,
                depth=depth + 1,
                visited_conditions=visited_conditions,
                provenance_steps=provenance_steps,
            )
            if sub_action is not None:
                return sub_action, unsatisfied

        return best_candidate, unsatisfied

    # ── Condition Evaluation Against HCIR Graph ───────────────────────

    def is_condition_satisfied(self, cond: str, view: FrozenGraphView) -> bool:
        """Evaluate whether a semantic predicate is satisfied in the current FrozenGraphView."""
        pred, args = parse_condition(cond)
        agent = view.get_node("agent")
        agent_props = agent.properties if agent and hasattr(agent, "properties") else {}

        reach = float(agent_props.get("reach_distance", self.default_reach_distance))

        if pred == "holds":
            if not args:
                return False
            target_id = args[0]
            # Check agent held property
            if agent_props.get("held_object_id") == target_id:
                return True
            # Check graph edge from agent to target
            for edge in view.edges_from("agent"):
                if (
                    edge
                    and target_id in edge.targets
                    and edge.edge_type
                    in (
                        HCIREdgeType.DEPENDS_ON,
                        HCIREdgeType.REQUIRES,
                    )
                ):
                    return True
            return False

        elif pred == "near":
            if not args:
                return False
            target_id = args[0]
            obj = view.get_node(target_id)
            if not obj or not hasattr(obj, "properties"):
                return False
            dist = float(obj.properties.get("distance", float("inf")))
            return dist <= reach

        elif pred == "is_opened":
            if not args:
                return False
            target_id = args[0]
            obj = view.get_node(target_id)
            if not obj or not hasattr(obj, "properties"):
                return False
            return bool(obj.properties.get("is_opened", False))

        elif pred == "is_toggled":
            if not args:
                return False
            target_id = args[0]
            obj = view.get_node(target_id)
            if not obj or not hasattr(obj, "properties"):
                return False
            return bool(obj.properties.get("is_toggled", False))

        elif pred == "inside":
            if len(args) < 2:
                return False
            obj_id, rec_id = args[0], args[1]
            obj = view.get_node(obj_id)
            if not obj or not hasattr(obj, "properties"):
                return False
            parent_recs = obj.properties.get("parent_receptacles", [])
            if rec_id in parent_recs:
                return True
            # Check containment edges
            for edge in view.edges_from(obj_id):
                if (
                    edge
                    and rec_id in edge.targets
                    and edge.edge_type in (HCIREdgeType.PART_OF, HCIREdgeType.DEPENDS_ON)
                ):
                    return True
            return False

        elif pred == "not_contained_in_closed":
            if not args:
                return True
            target_id = args[0]
            obj = view.get_node(target_id)
            if not obj or not hasattr(obj, "properties"):
                return True
            parent_recs = obj.properties.get("parent_receptacles", [])
            for pid in parent_recs:
                p_node = view.get_node(pid)
                if p_node and hasattr(p_node, "properties"):
                    is_openable = p_node.properties.get("is_openable", False)
                    is_opened = p_node.properties.get("is_opened", False)
                    if is_openable and not is_opened:
                        return False
            return True

        # Fallback: check custom properties if condition is 'prop_name(node_id)'
        if len(args) == 1:
            target_id = args[0]
            node = view.get_node(target_id)
            if node and hasattr(node, "properties"):
                val = node.properties.get(pred)
                if isinstance(val, bool):
                    return val
                return val is not None

        return False

    # ── Goal Extraction Helper ───────────────────────────────────────

    def _extract_goal_conditions(
        self, problem: ReasoningProblem, view: FrozenGraphView
    ) -> list[str]:
        """Extract target goal condition strings from problem definition or GoalNodes."""
        conditions: list[str] = []

        # 1. From problem parameters
        params = problem.param_dict
        if "target_conditions" in params:
            conds = params["target_conditions"]
            if isinstance(conds, (list, tuple)):
                conditions.extend(str(c) for c in conds)
            elif isinstance(conds, str):
                conditions.append(conds)

        if conditions:
            return conditions

        # 2. From GoalNodes referenced in problem.goal_node_ids
        for gid in problem.goal_node_ids:
            node = view.get_node(gid)
            if isinstance(node, GoalNode):
                conds = self._parse_goal_node_conditions(node)
                conditions.extend(conds)

        if conditions:
            return conditions

        # 3. Fallback: Search any GoalNode in the view
        for node in view.nodes_by_type(HCIRNodeType.GOAL):
            if isinstance(node, GoalNode) and not node.resolved:
                conds = self._parse_goal_node_conditions(node)
                conditions.extend(conds)
                if conditions:
                    break

        return conditions

    @staticmethod
    def _parse_goal_node_conditions(goal: GoalNode) -> list[str]:
        """Extract condition strings from GoalNode properties/criteria."""
        conditions: list[str] = []
        props = goal.properties if hasattr(goal, "properties") else {}

        # Check explicit target_conditions list
        if "target_conditions" in props:
            tc = props["target_conditions"]
            if isinstance(tc, (list, tuple)):
                return [str(c) for c in tc]
            elif isinstance(tc, str):
                return [tc]

        # Check criteria dictionary
        criteria = props.get("criteria", {})
        if isinstance(criteria, dict):
            for key, val in criteria.items():
                if key == "holds" and val:
                    conditions.append(f"holds({val})")
                elif key == "is_opened" and val:
                    conditions.append(f"is_opened({val})")
                elif key == "is_toggled" and val:
                    conditions.append(f"is_toggled({val})")
                elif key == "inside" and isinstance(val, (tuple, list)) and len(val) >= 2:
                    conditions.append(f"inside({val[0]}, {val[1]})")

        # Check target_object_id / target_receptacle_id attributes
        target_obj = props.get("target_object_id")
        target_rec = props.get("target_receptacle_id")

        if target_rec and target_obj:
            conditions.append(f"inside({target_obj}, {target_rec})")
        elif target_rec and not target_obj:
            # Check if receptacle is openable or toggleable
            action_type = props.get("goal_type", "")
            if "toggle" in action_type:
                conditions.append(f"is_toggled({target_rec})")
            else:
                conditions.append(f"is_opened({target_rec})")
        elif target_obj and not target_rec:
            conditions.append(f"holds({target_obj})")

        # Check description if matches a predicate pattern
        if goal.description and "(" in goal.description and goal.description.endswith(")"):
            conditions.append(goal.description.strip())

        # Check tags
        for tag in getattr(goal, "tags", []):
            if "(" in tag and tag.endswith(")"):
                conditions.append(tag.strip())

        return conditions
