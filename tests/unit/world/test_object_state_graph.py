"""Unit tests for ObjectStateGraphPlanner and 5 Executive Cognitive Directives."""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel, StateMutationModel
from hbllm.hcir.world.object_state_graph import ObjectCategory, ObjectStateGraphPlanner


def test_object_extraction_and_categorization() -> None:
    """Verify salient objects are properly extracted, categorized, and given interaction stances."""
    planner = ObjectStateGraphPlanner()
    engine = AutonomousEpistemicEngine()

    grid = np.zeros((10, 10), dtype=int)
    # Walls
    grid[0, :] = 1
    grid[9, :] = 1
    grid[:, 0] = 1
    grid[:, 9] = 1
    engine.symbolic_theory.barrier_features.add(1)

    # Avatar at (2, 2)
    grid[2, 2] = 2
    engine.avatar_feature = 2
    engine.avatar_pos = (2, 2)

    # Exit at (8, 8)
    grid[8, 8] = 3
    engine.symbolic_theory.goal_features.add(3)

    # Key / interactive item at (5, 5)
    grid[5, 5] = 4

    objects = planner.extract_salient_objects(engine, grid, bg=0)
    assert len(objects) >= 2

    exit_objs = [o for o in objects if o.category == ObjectCategory.EXIT]
    assert len(exit_objs) == 1
    assert (8, 8) in exit_objs[0].cells

    key_objs = [o for o in objects if o.category == ObjectCategory.KEY]
    assert len(key_objs) == 1
    assert (5, 5) in key_objs[0].cells
    assert (5, 5) in key_objs[0].interaction_stances


def test_directive_1_and_2_exit_open() -> None:
    """Verify Directive 1 & 2: when exit is open, planner plans directly to exit."""
    engine = AutonomousEpistemicEngine()

    # Ground motor dynamics: actions 1, 2, 3, 4 = UP, DOWN, LEFT, RIGHT
    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=1.0
    )
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=1.0
    )

    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Avatar
    engine.avatar_feature = 2
    engine.avatar_pos = (2, 2)

    grid[2, 6] = 3  # Goal / Exit
    engine.symbolic_theory.goal_features.add(3)

    plan = engine.object_planner.plan_macro_option(engine, grid, [1, 2, 3, 4, 5])
    assert plan is not None
    assert len(plan) == 4  # 4 steps RIGHT: (2,3), (2,4), (2,5), (2,6)
    assert all(step.action == 4 for step in plan)
    assert plan[-1].predicted_avatar_pos == (2, 6)


def test_directive_3_and_4_causal_switch_unlocking() -> None:
    """Verify Directive 4: when exit is blocked, planner navigates to known unlocking switch."""
    engine = AutonomousEpistemicEngine()

    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=1.0
    )
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=1.0
    )

    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Avatar
    engine.avatar_feature = 2
    engine.avatar_pos = (2, 2)

    # Goal at (2, 8)
    grid[2, 8] = 3
    engine.symbolic_theory.goal_features.add(3)

    # Wall completely blocking goal column 6
    grid[:, 6] = 1
    engine.symbolic_theory.barrier_features.add(1)

    # Switch at (5, 2) known to toggle barrier
    grid[5, 2] = 5
    mutation = StateMutationModel(
        trigger_type="CONTACT",
        trigger_pos=(5, 2),
        trigger_feature=5,
        prior_value=1,
        posterior_value=0,
        confidence=1.0,
    )
    engine.state_mutations.append(mutation)

    # Directive 1 & 2 fails because goal (2,8) is behind wall column 6.
    # Directive 4 should trigger: path to switch at (5, 2)!
    plan = engine.object_planner.plan_macro_option(engine, grid, [1, 2, 3, 4, 5])
    assert plan is not None
    assert len(plan) == 3  # 3 steps DOWN: (3,2), (4,2), (5,2)
    assert all(step.action == 2 for step in plan)
    assert plan[-1].predicted_avatar_pos == (5, 2)


def test_directive_5_epistemic_probing() -> None:
    """Verify Directive 5: when exit is blocked and no known mutations, probe untouched objects."""
    engine = AutonomousEpistemicEngine()

    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=1.0
    )
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=1.0
    )

    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Avatar
    engine.avatar_feature = 2
    engine.avatar_pos = (2, 2)

    # Goal at (2, 8) behind wall column 6
    grid[2, 8] = 3
    engine.symbolic_theory.goal_features.add(3)
    grid[:, 6] = 1
    engine.symbolic_theory.barrier_features.add(1)

    # Untouched candidate object at (4, 2)
    grid[4, 2] = 7

    plan = engine.object_planner.plan_macro_option(engine, grid, [1, 2, 3, 4, 5])
    assert plan is not None
    assert len(plan) == 2  # 2 steps DOWN to (4, 2)
    assert plan[-1].predicted_avatar_pos == (4, 2)


def test_decide_macro_option_execution_and_transition() -> None:
    """Verify engine.decide() executes macro options seamlessly and transitions to exit."""
    engine = AutonomousEpistemicEngine()

    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=1.0
    )
    engine.action_dynamics[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=1.0
    )
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=1.0
    )

    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Avatar
    engine.avatar_feature = 2
    engine.avatar_pos = (2, 2)

    # Goal at (2, 5)
    grid[2, 5] = 3
    engine.symbolic_theory.goal_features.add(3)

    # Step 1: decide should trigger macro option to exit (3 steps RIGHT)
    act1, data1 = engine.decide(grid, [1, 2, 3, 4, 5])
    assert act1 == 4
    assert len(engine.mental_plan) == 2  # remaining steps in queue

    # Step 2: avatar advances to (2, 3) in environment
    grid2 = np.zeros((10, 10), dtype=int)
    grid2[2, 3] = 2
    grid2[2, 5] = 3
    act2, _ = engine.decide(grid2, [1, 2, 3, 4, 5])
    assert act2 == 4
    assert len(engine.mental_plan) == 1

    # Step 3: avatar advances to (2, 4) in environment
    grid3 = np.zeros((10, 10), dtype=int)
    grid3[2, 4] = 2
    grid3[2, 5] = 3
    act3, _ = engine.decide(grid3, [1, 2, 3, 4, 5])
    assert act3 == 4
    assert len(engine.mental_plan) == 0
