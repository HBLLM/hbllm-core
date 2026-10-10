"""Unit tests for HCIR Relational Causal Schema Induction, Skill Macro-Chunking,
and Spatio-Temporal Hazard Dynamics.
"""

from __future__ import annotations

import pytest

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    FrozenGraphView,
    ProblemType,
    ReasoningProblem,
)
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator
from hbllm.brain.reasoning.operators.relational_schema import (
    FirstOrderPredicate,
    RelationalSchemaInducer,
)
from hbllm.brain.reasoning.operators.skill_chunker import (
    MacroActionNode,
    SkillMacroChunker,
)
from hbllm.brain.reasoning.operators.temporal_dynamics import TemporalVelocityModel
from hbllm.hcir.graph import ActionModality, ActionNode, GoalNode, PhysicalEntityNode


def test_first_order_predicate_unification() -> None:
    """Test variable binding, unification, and ground predicate matching."""
    # Pattern: holds(?agent, ?item)
    pattern = FirstOrderPredicate.from_string("holds(?agent, ?item)")
    assert not pattern.is_ground()

    # Ground predicate: holds(robot_1, key_gold)
    ground = FirstOrderPredicate.from_string("holds(robot_1, key_gold)")
    assert ground.is_ground()

    # Unify pattern against ground
    bindings = pattern.unify(ground)
    assert bindings is not None
    assert bindings == {"?agent": "robot_1", "?item": "key_gold"}

    # Substitute bindings back into pattern
    bound = pattern.bind(bindings)
    assert bound.is_ground()
    assert bound.to_string() == "holds(robot_1, key_gold)"

    # Mismatched predicate names
    mismatched = FirstOrderPredicate.from_string("inside(robot_1, key_gold)")
    assert pattern.unify(mismatched) is None

    # Variable conflict test: same variable ?x cannot bind to two different constants
    conflict_pattern = FirstOrderPredicate.from_string("same(?x, ?x)")
    conflict_ground = FirstOrderPredicate.from_string("same(a, b)")
    assert conflict_pattern.unify(conflict_ground) is None

    valid_ground = FirstOrderPredicate.from_string("same(a, a)")
    assert conflict_pattern.unify(valid_ground) == {"?x": "a"}


def test_relational_schema_induction_from_transitions() -> None:
    """Test online induction of causal schemas from concrete state transitions."""
    inducer = RelationalSchemaInducer(min_instances_to_validate=3, confidence_threshold=0.70)

    # Observation 1: agent picks up apple_1
    schema1 = inducer.observe_transition(
        action_name="pickup(apple_1)",
        pre_state_conditions={"adjacent_to(agent, apple_1)", "clear(apple_1)"},
        post_state_conditions={"holds(agent, apple_1)"},
        entity_types={"agent": "Agent", "apple_1": "Object"},
    )
    assert schema1 is not None
    assert schema1.success_count == 1
    # Posterior confidence = (1 + 1) / (1 + 1 + 1 + 0) = 2/3 ≈ 0.667
    assert pytest.approx(schema1.confidence, 0.01) == 0.667
    assert len(inducer.candidate_schemas) == 1
    assert len(inducer.validated_schemas) == 0

    # Observation 2: agent picks up book_2
    schema2 = inducer.observe_transition(
        action_name="pickup(book_2)",
        pre_state_conditions={"adjacent_to(agent, book_2)", "clear(book_2)"},
        post_state_conditions={"holds(agent, book_2)"},
        entity_types={"agent": "Agent", "book_2": "Object"},
    )
    assert schema2.schema_id == schema1.schema_id
    assert schema2.success_count == 2
    # Confidence = (1 + 2) / (1 + 1 + 2 + 0) = 3/4 = 0.75

    # Observation 3: agent picks up cup_3 -> should trigger validation
    schema3 = inducer.observe_transition(
        action_name="pickup(cup_3)",
        pre_state_conditions={"adjacent_to(agent, cup_3)", "clear(cup_3)"},
        post_state_conditions={"holds(agent, cup_3)"},
        entity_types={"agent": "Agent", "cup_3": "Object"},
    )
    assert schema3.success_count == 3
    # Confidence = (1 + 3) / (1 + 1 + 3 + 0) = 4/5 = 0.80 >= 0.70 threshold & instances >= 3
    assert len(inducer.validated_schemas) == 1
    assert len(inducer.candidate_schemas) == 0

    # Query schema for a new goal: holds(agent, ruby_9)
    matches = inducer.find_schemas_for_goal("holds(agent, ruby_9)", only_validated=True)
    assert len(matches) > 0
    matched_schema, bindings = matches[0]
    assert matched_schema.schema_id == schema3.schema_id
    # Check that variables were properly bound to goal arguments
    assert "agent" in bindings.values()
    assert "ruby_9" in bindings.values()


def test_skill_macro_chunking_and_composition() -> None:
    """Test analytical composition of ActionNodes into MacroActionNode."""
    chunker = SkillMacroChunker(min_trace_occurrences=2)

    # Action 1: navigate to key
    a1 = ActionNode(
        id="act_nav",
        intent="navigate_to(key_gold)",
        modality=ActionModality.LOCOMOTION,
        requirements=["visible(key_gold)"],
        produces=["adjacent_to(agent, key_gold)"],
        estimated_cost=2,
    )

    # Action 2: pick up key
    a2 = ActionNode(
        id="act_pick",
        intent="pickup(key_gold)",
        modality=ActionModality.MANIPULATION,
        requirements=["adjacent_to(agent, key_gold)"],
        produces=["holds(agent, key_gold)"],
        properties={"drops": "empty_box"},
        estimated_cost=1,
    )

    # Compile sequence
    macro = chunker.compile_sequence(
        actions=[a1, a2],
        macro_name="AcquireKey(key_gold)",
        goal_target="holds(agent, key_gold)",
    )

    assert isinstance(macro, MacroActionNode)
    assert macro.intent == "AcquireKey(key_gold)"
    # adjacent_to should be absorbed because a1 produces it
    assert "adjacent_to(agent, key_gold)" not in macro.requirements
    assert "visible(key_gold)" in macro.requirements
    assert "holds(agent, key_gold)" in macro.produces
    assert "holds(empty_box)" in macro.composite_negates
    assert int(macro.estimated_cost) == 3

    # Step execution pointer
    assert macro.step_pointer == 0
    assert not macro.is_completed
    step1 = macro.advance()
    assert step1 is not None and step1.intent == "navigate_to(key_gold)"
    assert macro.step_pointer == 1
    step2 = macro.advance()
    assert step2 is not None and step2.intent == "pickup(key_gold)"
    assert macro.is_completed
    assert macro.advance() is None

    # Trace discovery
    chunker.record_trace([a1, a2], goal_achieved="holds(agent, key_gold)", success=True)
    chunker.record_trace([a1, a2], goal_achieved="holds(agent, key_gold)", success=True)
    compiled = chunker.discover_and_compile_skills()
    assert len(compiled) == 1
    assert "Skill(navigate_to(key_gold)..pickup(key_gold))" in compiled[0].intent


def test_temporal_hazard_velocity_forecasting() -> None:
    """Test velocity tracking and periodic oscillation forecasting."""
    model = TemporalVelocityModel()

    # 1. Linear trajectory: moving right +2 units per step
    for step in range(5):
        model.record_observation(
            "sentinel_linear", position=(float(step * 2), 5.0), timestamp=float(step)
        )

    traj = model.trajectories["sentinel_linear"]
    assert pytest.approx(traj.velocity[0], 0.1) == 2.0
    assert pytest.approx(traj.velocity[1], 0.1) == 0.0

    # Predict position at future step +2 (t=6)
    pred_pos = model.predict_position("sentinel_linear", future_dt=2.0)
    assert pytest.approx(pred_pos[0], 0.2) == 12.0
    assert pytest.approx(pred_pos[1], 0.2) == 5.0

    # 2. Periodic trajectory: oscillating between x=1 and x=3
    osc_points = [
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (2.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
    ]
    for t, pt in enumerate(osc_points):
        model.record_observation("laser_barrier", position=pt, timestamp=float(t))

    traj_osc = model.trajectories["laser_barrier"]
    assert traj_osc.is_periodic
    assert traj_osc.detected_period > 0

    # Test collision imminent detection
    colliding = model.is_collision_imminent(agent_pos=(12.0, 5.0), future_dt=2.0, safe_margin=1.0)
    assert "sentinel_linear" in colliding

    # Test safe wait time search
    safe_step = model.find_safe_wait_time(
        target_pos=(1.0, 0.0), entity_id="laser_barrier", max_lookahead=5, safe_margin=0.5
    )
    assert safe_step is not None


def test_macro_action_embodied_causal_resolution() -> None:
    """Test that EmbodiedCausalOperator evaluates and selects a MacroActionNode."""
    agent = PhysicalEntityNode(
        id="agent",
        name="agent",
        properties={"position": (0, 0), "reach_distance": 1.5},
    )
    goal = GoalNode(
        id="goal_gem",
        description="Acquire gem",
        properties={"target_conditions": ["holds(agent, gem)"]},
    )

    a1 = ActionNode(
        id="a1",
        intent="walk_to(gem)",
        modality=ActionModality.LOCOMOTION,
        requirements=["visible(gem)"],
        produces=["adjacent(agent, gem)"],
    )
    a2 = ActionNode(
        id="a2",
        intent="grab(gem)",
        modality=ActionModality.MANIPULATION,
        requirements=["adjacent(agent, gem)"],
        produces=["holds(agent, gem)"],
    )

    chunker = SkillMacroChunker()
    macro = chunker.compile_sequence([a1, a2], macro_name="MacroAcquireGem")

    view = FrozenGraphView(
        nodes={"agent": agent, "goal_gem": goal, "macro_act": macro},
        edges={},
    )

    operator = EmbodiedCausalOperator()
    problem = ReasoningProblem(
        problem_type=ProblemType.PLANNING,
        description="Acquire gem",
        goal_node_ids=("goal_gem",),
    )
    context = CognitiveContext(problem=problem, graph_view=view)

    result = operator.execute(problem, context)
    assert result.status.name == "SUCCESS"
    assert result.conclusions["is_macro"] is True
    assert result.conclusions["best_action"] == "MacroAcquireGem"
    assert result.conclusions["macro_current_action"] == "walk_to(gem)"
