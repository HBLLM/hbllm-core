"""Unit tests verifying HCIR complexity enhancements extended into the core reasoning substrate."""

from __future__ import annotations

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    FrozenGraphView,
    ProblemType,
    ReasoningProblem,
    ResultStatus,
)
from hbllm.brain.reasoning.operators.causal import CausalOperator
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import (
    ActionNode,
    BeliefNode,
    CognitiveGraph,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.types import UncertaintyVector


def test_frozen_graph_view_to_workspace() -> None:
    """Verify that FrozenGraphView lifts seamlessly into an active HCIRWorkspaceState."""
    graph = CognitiveGraph()
    agent = PhysicalEntityNode(id="agent", entity_name="agent", properties={"position": [0, 0]})
    box = PhysicalEntityNode(
        id="box_1", entity_name="box", properties={"position": [0, 1], "movable": True}
    )
    goal = GoalNode(id="g_1", description="Solve puzzle")
    graph.add_node(agent)
    graph.add_node(box)
    graph.add_node(goal)

    frozen = FrozenGraphView.from_graph(graph)
    ws = frozen.to_workspace()

    assert ws.get_node("agent") is not None
    assert ws.get_node("box_1") is not None
    assert ws.get_node("g_1") is not None
    assert ws.graph.node_count == 3

    # Verify workspace can fork simulation branches
    ws.fork_branch("test_branch")
    assert "test_branch" in ws._branches
    ws.drop_branch("test_branch")


def test_kernel_services_create_default() -> None:
    """Verify KernelServices.create_default constructs a complete dependency container."""
    graph = CognitiveGraph()
    frozen = FrozenGraphView.from_graph(graph)
    ws = frozen.to_workspace()
    services = KernelServices.create_default(ws)

    assert services.workspace is ws
    assert services.transaction_manager is not None
    assert services.capability_resolver is not None
    assert services.scheduler is not None


def test_counterfactual_planner_evaluate_and_select_sync() -> None:
    """Verify that evaluate_and_select_sync runs synchronously without deadlock or errors."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(id="agent", entity_name="agent", properties={"position": [0, 0]})
    )
    frozen = FrozenGraphView.from_graph(graph)
    ws = frozen.to_workspace()
    services = KernelServices.create_default(ws)

    planner = CounterfactualPlanner(ws, services)
    goal = GoalNode(id="g_test", description="Step forward")
    candidates = [
        ActionNode(id="act_1", intent="FAST_STEP", estimated_cost=0),
        ActionNode(id="act_2", intent="SLOW_STEP", estimated_cost=50),
    ]

    result = planner.evaluate_and_select_sync(
        goal=goal,
        candidate_actions=candidates,
        horizon=1,
    )
    assert result.candidate_id == "act_1"
    assert result.utility_score > 0.0


def test_embodied_causal_prunes_deadlock_actions_via_counterfactual_simulation() -> None:
    """Verify EmbodiedCausalOperator evaluates candidates and prunes deadlocks using simulation."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_name="agent",
            properties={"position": [2, 2], "reach_distance": 1.5, "is_avatar": True},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="box",
            entity_name="box",
            properties={"position": [2, 3], "movable": True, "passable": False},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="wall",
            entity_name="wall",
            properties={"position": [2, 4], "movable": False, "passable": False},
        )
    )

    # Two competing actions that both produce the goal condition 'box_moved'
    # but one pushes into a corner deadlock, and the other moves it into an open area.
    act_deadlock = ActionNode(
        id="act_push_corner",
        intent="PUSH_BOX_INTO_CORNER",
        requirements=["near(box)"],
        produces=["box_moved"],
        properties={"predicted_state": {"spatial_outcome": {"deadlock": True, "progress": -1.0}}},
    )
    act_safe = ActionNode(
        id="act_push_open",
        intent="PUSH_BOX_OPEN_PATH",
        requirements=["near(box)"],
        produces=["box_moved"],
        properties={"predicted_state": {"spatial_outcome": {"deadlock": False, "progress": 1.0}}},
    )

    graph.add_node(act_deadlock)
    graph.add_node(act_safe)

    goal = GoalNode(id="g_box", description="Move the box")
    graph.add_node(goal)

    frozen = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(
        graph_view=frozen,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=(goal.id,),
            description="box_moved",
        ),
    )

    op = EmbodiedCausalOperator(enable_counterfactual_simulation=True)
    result = op.execute(context.problem, context)

    assert result.status == ResultStatus.SUCCESS
    assert result.conclusions["best_action"] == "PUSH_BOX_OPEN_PATH"
    assert result.conclusions["action_id"] == "act_push_open"


def test_embodied_causal_composite_and_or_not_predicates() -> None:
    """Verify that EmbodiedCausalOperator correctly evaluates AND, OR, NOT and relational predicates."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_name="agent",
            properties={"position": [1, 1], "held_object_id": "key_silver", "reach_distance": 2.0},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="door",
            entity_name="door",
            properties={"position": [1, 2], "is_opened": False, "is_locked": True},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="key_gold",
            entity_name="key_gold",
            properties={"position": [5, 5]},
        )
    )

    frozen = FrozenGraphView.from_graph(graph)
    op = EmbodiedCausalOperator()

    # AND: holds(key_silver) is true, near(door) is true -> True
    assert op.is_condition_satisfied("AND(holds(key_silver), near(door))", frozen)

    # AND with one false condition -> False
    assert not op.is_condition_satisfied("AND(holds(key_silver), near(key_gold))", frozen)

    # OR: one true -> True
    assert op.is_condition_satisfied("OR(holds(key_gold), holds(key_silver))", frozen)
    # OR: both false -> False
    assert not op.is_condition_satisfied("OR(holds(key_gold), near(key_gold))", frozen)

    # NOT: is_opened(door) is False -> NOT(is_opened(door)) is True
    assert op.is_condition_satisfied("NOT(is_opened(door))", frozen)
    # NOT: holds(key_silver) is True -> NOT(holds(key_silver)) is False
    assert not op.is_condition_satisfied("NOT(holds(key_silver))", frozen)

    # Relational: adjacent_to(agent, door)
    assert op.is_condition_satisfied("adjacent_to(agent, door)", frozen)
    assert not op.is_condition_satisfied("adjacent_to(agent, key_gold)", frozen)


def test_embodied_causal_pomdp_latent_belief_awareness() -> None:
    """Verify that EmbodiedCausalOperator checks active POMDP latent beliefs."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(id="agent", entity_name="agent", properties={"position": [0, 0]})
    )
    # Latent belief node
    graph.add_node(
        BeliefNode(
            id="belief_latent_wind",
            claim="Latent wind direction is north",
            belief_type="causal",
            uncertainty=UncertaintyVector(confidence=0.9),
            properties={
                "subject": "wind_direction",
                "value": "north",
                "is_latent": True,
            },
        )
    )

    frozen = FrozenGraphView.from_graph(graph)
    op = EmbodiedCausalOperator()

    # Match subject and value
    assert op.is_condition_satisfied("latent_belief(wind_direction, north)", frozen)
    # Match subject with wrong value -> False
    assert not op.is_condition_satisfied("latent_belief(wind_direction, south)", frozen)
    # Match subject existence
    assert op.is_condition_satisfied("latent_belief(wind_direction)", frozen)
    # Nonexistent subject -> False
    assert not op.is_condition_satisfied("latent_belief(temperature)", frozen)


def test_causal_operator_with_composite_factors() -> None:
    """Verify that CausalOperator traverses edges and includes composite factor hyperedges."""
    graph = CognitiveGraph()
    graph.add_node(PhysicalEntityNode(id="switch_a", entity_name="switch"))
    graph.add_node(PhysicalEntityNode(id="relay_b", entity_name="relay"))
    graph.add_node(PhysicalEntityNode(id="light_c", entity_name="light"))

    # Edge switch_a -> relay_b with composite factor [voltage_high]
    edge_1 = HCIREdge(
        id="e1",
        edge_type=HCIREdgeType.CAUSES,
        sources=["switch_a"],
        targets=["relay_b"],
        weight=0.95,
        properties={"factors": ["voltage_high", "fuse_intact"]},
    )
    # Edge relay_b -> light_c with factor [circuit_closed]
    edge_2 = HCIREdge(
        id="e2",
        edge_type=HCIREdgeType.CAUSES,
        sources=["relay_b"],
        targets=["light_c"],
        weight=0.90,
        properties={"factors": ["circuit_closed"]},
    )

    graph.add_edge(edge_1)
    graph.add_edge(edge_2)

    frozen = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(
        graph_view=frozen,
        problem=ReasoningProblem(
            problem_type=ProblemType.CAUSAL,
            focus_node_ids=("switch_a",),
        ),
    )

    op = CausalOperator(max_depth=3)
    result = op.execute(context.problem, context)

    assert result.status == ResultStatus.SUCCESS
    chains = result.conclusions["top_chains"]
    # Should discover multi-hop chain switch_a -> relay_b -> light_c
    multi_hop = next((c for c in chains if c["target"] == "light_c"), None)
    assert multi_hop is not None
    assert "factors" in multi_hop
    assert "voltage_high" in multi_hop["factors"]
    assert "circuit_closed" in multi_hop["factors"]


def test_unified_reasoning_runtime_autonomously_executes_enhanced_embodied_causal() -> None:
    """Verify UnifiedReasoningRuntime seamlessly dispatches planning problem through enhanced EmbodiedCausalOperator."""
    graph = CognitiveGraph()
    graph.add_node(
        PhysicalEntityNode(
            id="agent",
            entity_name="agent",
            properties={"position": [0, 0], "reach_distance": 1.5},
        )
    )
    graph.add_node(
        PhysicalEntityNode(
            id="bread",
            entity_name="bread",
            properties={"position": [0, 1], "is_sliced": False},
        )
    )
    act_slice = ActionNode(
        id="act_slice",
        intent="SliceBread",
        requirements=["near(bread)"],
        produces=["is_sliced(bread)"],
    )
    graph.add_node(act_slice)

    goal = GoalNode(id="g_slice", description="is_sliced(bread)")
    graph.add_node(goal)

    runtime = UnifiedReasoningRuntime(create_default_operator_registry())
    trace = runtime.reason(
        graph=graph,
        problem=ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=(goal.id,),
            description="is_sliced(bread)",
        ),
    )

    assert len(trace.invocations) > 0
    embodied_inv = next(
        (inv for inv in trace.invocations if inv.operator_id == "embodied_causal"), None
    )
    assert embodied_inv is not None
    assert embodied_inv.result.status == ResultStatus.SUCCESS
    assert embodied_inv.result.conclusions["best_action"] == "SliceBread"
