"""
Unit tests for EmbodiedCausalOperator and UnifiedReasoningRuntime planning.

Verifies:
1. Immediate action execution when preconditions are satisfied.
2. 1-step precondition backward chaining (approach before pickup).
3. Hierarchical container unblocking (approach receptacle -> open receptacle -> pickup object).
4. Complex object relocation (pickup -> approach destination -> open destination -> put).
5. Goal termination when all conditions are satisfied (no-op).
6. Integration with UnifiedReasoningRuntime and OperatorRegistry.
"""

from __future__ import annotations

from typing import Any

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    FrozenGraphView,
    ProblemType,
    ReasoningProblem,
    ResultStatus,
)
from hbllm.brain.reasoning.operators.embodied_causal import (
    EmbodiedCausalOperator,
    parse_condition,
)
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import (
    ActionNode,
    CognitiveGraph,
    GoalNode,
    PhysicalEntityNode,
)


def test_parse_condition_syntax() -> None:
    """Verify condition string parser handles unary, binary, and unparameterized predicates."""
    pred, args = parse_condition("holds(apple_1)")
    assert pred == "holds"
    assert args == ["apple_1"]

    pred, args = parse_condition("inside(apple_1, microwave_2)")
    assert pred == "inside"
    assert args == ["apple_1", "microwave_2"]

    pred, args = parse_condition("not_contained_in_closed(apple_1)")
    assert pred == "not_contained_in_closed"
    assert args == ["apple_1"]

    pred, args = parse_condition("system_healthy")
    assert pred == "system_healthy"
    assert args == []


def test_direct_action_selection() -> None:
    """When all preconditions for pickup are met, pickup is chosen immediately."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_type="agent",
            properties={"held_object_id": None, "reach_distance": 1.6},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="apple_1",
            entity_type="object",
            properties={"distance": 1.2, "is_pickupable": True, "parent_receptacles": []},
        )
    )
    graph.add_node(
        ActionNode(
            id="act_pickup_apple",
            intent="pickup",
            properties={"objectId": "apple_1"},
            requirements=["near(apple_1)", "not_contained_in_closed(apple_1)"],
            produces=["holds(apple_1)"],
        )
    )
    graph.add_node(
        ActionNode(
            id="act_nav_apple",
            intent="navigate_toward",
            properties={"targetId": "apple_1"},
            requirements=[],
            produces=["near(apple_1)"],
        )
    )
    goal = GoalNode(id="goal_retrieve", properties={"target_conditions": ["holds(apple_1)"]})
    graph.add_node(goal)

    frozen = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(
        graph_view=frozen,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING, goal_node_ids=("goal_retrieve",)
        ),
    )

    op = EmbodiedCausalOperator()
    result = op.execute(context.problem, context)

    assert result.status == ResultStatus.SUCCESS
    assert result.conclusions["best_action"] == "pickup"
    assert result.conclusions["action_id"] == "act_pickup_apple"


def test_precondition_backward_chaining() -> None:
    """When agent is far from target, operator selects navigate_toward."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_type="agent",
            properties={"held_object_id": None, "reach_distance": 1.6},
        )
    )
    # Apple is 3.5m away (beyond 1.6m reach)
    graph.add_node(
        PhysicalEntityNode(
            id="apple_1",
            entity_type="object",
            properties={"distance": 3.5, "is_pickupable": True, "parent_receptacles": []},
        )
    )
    graph.add_node(
        ActionNode(
            id="act_pickup_apple",
            intent="pickup",
            properties={"objectId": "apple_1"},
            requirements=["near(apple_1)", "not_contained_in_closed(apple_1)"],
            produces=["holds(apple_1)"],
        )
    )
    graph.add_node(
        ActionNode(
            id="act_nav_apple",
            intent="navigate_toward",
            properties={"targetId": "apple_1"},
            requirements=[],
            produces=["near(apple_1)"],
        )
    )
    goal = GoalNode(id="goal_retrieve", properties={"target_conditions": ["holds(apple_1)"]})
    graph.add_node(goal)

    frozen = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(
        graph_view=frozen,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING, goal_node_ids=("goal_retrieve",)
        ),
    )

    op = EmbodiedCausalOperator()
    result = op.execute(context.problem, context)

    assert result.status == ResultStatus.SUCCESS
    assert result.conclusions["best_action"] == "navigate_toward"
    assert result.conclusions["action_id"] == "act_nav_apple"
    assert any("near(apple_1)" in step for step in result.provenance_chains[0].reasoning_steps)


def test_hierarchical_container_unblocking() -> None:
    """When target object is enclosed in a closed fridge, operator chains approach/open on fridge."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_type="agent",
            properties={"held_object_id": None, "reach_distance": 1.6},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="fridge_1",
            entity_type="receptacle",
            properties={"distance": 2.8, "is_openable": True, "is_opened": False},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="apple_1",
            entity_type="object",
            properties={
                "distance": 2.8,
                "is_pickupable": True,
                "parent_receptacles": ["fridge_1"],
            },
        )
    )
    # Affordances
    graph.add_node(
        ActionNode(
            id="act_pickup_apple",
            intent="pickup",
            properties={"objectId": "apple_1"},
            requirements=["near(apple_1)", "not_contained_in_closed(apple_1)"],
            produces=["holds(apple_1)"],
        )
    )
    graph.add_node(
        ActionNode(
            id="act_open_fridge",
            intent="open",
            properties={"objectId": "fridge_1"},
            requirements=["near(fridge_1)"],
            produces=["is_opened(fridge_1)"],
        )
    )
    graph.add_node(
        ActionNode(
            id="act_nav_fridge",
            intent="navigate_toward",
            properties={"targetId": "fridge_1"},
            requirements=[],
            produces=["near(fridge_1)"],
        )
    )
    graph.add_node(
        ActionNode(
            id="act_nav_apple",
            intent="navigate_toward",
            properties={"targetId": "apple_1"},
            requirements=[],
            produces=["near(apple_1)"],
        )
    )

    goal = GoalNode(id="goal_retrieve", properties={"target_conditions": ["holds(apple_1)"]})
    graph.add_node(goal)

    frozen = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(
        graph_view=frozen,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING, goal_node_ids=("goal_retrieve",)
        ),
    )

    op = EmbodiedCausalOperator()
    # Initial state: fridge is far and closed -> navigate to fridge
    result = op.execute(context.problem, context)
    assert result.status == ResultStatus.SUCCESS
    assert result.conclusions["best_action"] == "navigate_toward"
    assert result.conclusions["action_id"] == "act_nav_fridge"

    # Next state: agent has arrived at fridge (distance=1.0m)
    fridge_node = graph.get_node("fridge_1")
    assert isinstance(fridge_node, PhysicalEntityNode)
    fridge_node.properties["distance"] = 1.0

    frozen2 = FrozenGraphView.from_graph(graph)
    context2 = CognitiveContext(
        graph_view=frozen2,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING, goal_node_ids=("goal_retrieve",)
        ),
    )
    result2 = op.execute(context2.problem, context2)
    assert result2.status == ResultStatus.SUCCESS
    assert result2.conclusions["best_action"] == "open"
    assert result2.conclusions["action_id"] == "act_open_fridge"

    # Next state: fridge is now open, apple is within reach
    fridge_node.properties["is_opened"] = True
    apple_node = graph.get_node("apple_1")
    assert isinstance(apple_node, PhysicalEntityNode)
    apple_node.properties["distance"] = 1.1

    frozen3 = FrozenGraphView.from_graph(graph)
    context3 = CognitiveContext(
        graph_view=frozen3,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING, goal_node_ids=("goal_retrieve",)
        ),
    )
    result3 = op.execute(context3.problem, context3)
    assert result3.status == ResultStatus.SUCCESS
    assert result3.conclusions["best_action"] == "pickup"
    assert result3.conclusions["action_id"] == "act_pickup_apple"


def test_goal_termination_when_satisfied() -> None:
    """When all goal criteria are already satisfied, operator outputs goal_satisfied with no-op."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_type="agent",
            properties={"held_object_id": "apple_1", "reach_distance": 1.6},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="apple_1",
            entity_type="object",
            properties={"distance": 0.5, "parent_receptacles": []},
        )
    )
    goal = GoalNode(id="goal_retrieve", properties={"target_conditions": ["holds(apple_1)"]})
    graph.add_node(goal)

    frozen = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(
        graph_view=frozen,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING, goal_node_ids=("goal_retrieve",)
        ),
    )

    op = EmbodiedCausalOperator()
    result = op.execute(context.problem, context)

    assert result.status == ResultStatus.SUCCESS
    assert result.conclusions["status"] == "goal_satisfied"
    assert result.conclusions["best_action"] == "no_op"


def test_unified_reasoning_runtime_integration() -> None:
    """Verify that UnifiedReasoningRuntime automatically selects and executes EmbodiedCausalOperator."""
    registry = create_default_operator_registry()
    assert "embodied_causal" in registry.operator_ids

    runtime = UnifiedReasoningRuntime(registry)

    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_type="agent",
            properties={"held_object_id": None, "reach_distance": 1.6},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="box_1",
            entity_type="receptacle",
            properties={"distance": 1.2, "is_openable": True, "is_opened": False},
        )
    )
    graph.add_node(
        ActionNode(
            id="act_open_box",
            intent="open",
            properties={"objectId": "box_1"},
            requirements=["near(box_1)"],
            produces=["is_opened(box_1)"],
        )
    )
    goal = GoalNode(id="goal_box", properties={"target_conditions": ["is_opened(box_1)"]})
    graph.add_node(goal)

    problem = ReasoningProblem(
        problem_type=ProblemType.PLANNING,
        goal_node_ids=("goal_box",),
        description="Open box_1 in physical environment",
    )

    trace = runtime.reason(graph=graph, problem=problem)

    assert trace is not None
    assert trace.final_result is not None
    assert trace.final_result.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)
    # EmbodiedCausalOperator should have been invoked and succeeded
    invocations = [inv for inv in trace.invocations if inv.operator_id == "embodied_causal"]
    assert len(invocations) >= 1
    assert invocations[0].result.status == ResultStatus.SUCCESS
    assert invocations[0].result.conclusions["best_action"] == "open"
    assert invocations[0].result.conclusions["action_id"] == "act_open_box"


def test_tech_tree_crafting_causal_resolution() -> None:
    """EmbodiedCausalOperator backward-chains recursive crafting tech-tree DAG:
    Goal: make_wood_pickaxe
    Requires: near(crafting_table), has(wood, 1)
    crafting_table requires: place_table (requires has(wood, 2))
    has(wood) requires: collect_wood (requires near(tree))
    Agent has no wood and is near tree -> selects collect_wood!
    """
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_type="agent",
            properties={"inventory": {"wood": 0}, "x": 10, "y": 10, "reach_distance": 1.5},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="tree_1",
            entity_type="resource",
            properties={"x": 10, "y": 10, "distance": 0.0},
        )
    )
    # Affordances
    graph.add_node(
        ActionNode(
            id="act_make_pickaxe",
            intent="make_wood_pickaxe",
            requirements=["near(crafting_table)", "has(wood, 1)"],
            produces=["has(wood_pickaxe)"],
        )
    )
    graph.add_node(
        ActionNode(
            id="act_place_table",
            intent="place_table",
            requirements=["has(wood, 2)"],
            produces=["near(crafting_table)", "has(table)"],
        )
    )
    graph.add_node(
        ActionNode(
            id="act_collect_wood",
            intent="collect_wood",
            requirements=["near(tree_1)"],
            produces=["has(wood, 1)", "has(wood, 2)"],
        )
    )
    goal = GoalNode(id="goal_craft", properties={"target_conditions": ["has(wood_pickaxe)"]})
    graph.add_node(goal)

    frozen = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(
        graph_view=frozen,
        problem=ReasoningProblem(problem_type=ProblemType.PLANNING, goal_node_ids=("goal_craft",)),
    )

    op = EmbodiedCausalOperator()
    result = op.execute(context.problem, context)

    assert result.status == ResultStatus.SUCCESS
    assert result.conclusions["best_action"] == "collect_wood"
    assert result.conclusions["action_id"] == "act_collect_wood"


def test_grid_predicates_and_custom_registry() -> None:
    """Verify standing_on, adjacency, and custom predicate extension registry."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_type="agent",
            properties={"drink": 3.0, "x": 5, "y": 5, "coords": (5, 5)},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="water_1",
            entity_type="resource",
            properties={"coords": (5, 6), "x": 5, "y": 6},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="stairs_down",
            entity_type="stairs",
            properties={"coords": (5, 5), "x": 5, "y": 5},
        )
    )

    view = FrozenGraphView.from_graph(graph)
    op = EmbodiedCausalOperator()

    # Core universal primitives
    assert op.is_condition_satisfied("standing_on(stairs_down)", view)
    assert op.is_condition_satisfied("adjacent(water_1)", view)

    # Unregistered predicate returns False
    assert not op.is_condition_satisfied("custom_dummy_vital(drink, 2)", view)

    # Register custom predicate via Extension Registry
    def eval_vital(args: list[str], v: FrozenGraphView, props: dict[str, Any]) -> bool:
        vital_name = args[0] if args else "health"
        min_val = float(args[1]) if len(args) > 1 else 4.0
        return float(props.get(vital_name, 10.0)) >= min_val

    EmbodiedCausalOperator.register_predicate("custom_dummy_vital", eval_vital)
    try:
        assert not op.is_condition_satisfied("custom_dummy_vital(drink, 4)", view)
        assert op.is_condition_satisfied("custom_dummy_vital(drink, 2)", view)
    finally:
        EmbodiedCausalOperator.unregister_predicate("custom_dummy_vital")
