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


def test_spatial_effector_macro_option_click() -> None:
    """Verify ObjectStateGraphPlanner generates intentional click macro options for spatial effectors."""
    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [{"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}}]
    )

    grid = np.zeros((10, 10), dtype=int)
    # Exit object at (5, 5)
    grid[5, 5] = 3
    engine.symbolic_theory.goal_features.add(3)

    plan = engine.object_planner.plan_macro_option(engine, grid, [6])
    assert plan is not None
    assert len(plan) == 1
    assert plan[0].action == 6
    assert plan[0].action_data == {"x": 5, "y": 5}
    assert plan[0].expected_mutation == "CLICK_EXIT"


def test_object_causal_effect_tracking() -> None:
    """Verify ObjectStateLedger records causal mutations and unlocks barriers."""
    engine = AutonomousEpistemicEngine()

    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Avatar
    engine.avatar_feature = 2
    engine.avatar_pos = (2, 2)

    # Barrier at (2, 6)
    grid[2, 6] = 1
    engine.symbolic_theory.barrier_features.add(1)
    engine.learned_barriers.add((2, 6))

    # Switch object at (3, 2)
    grid[3, 2] = 4
    engine.object_planner.active_target_object_id = "switch_4"

    # Simulate interaction where barrier (2, 6) disappears
    engine.prev_grid = grid.copy()
    grid_post = grid.copy()
    grid_post[2, 6] = 0  # Barrier unlocked / removed

    engine.assimilate_feedback(
        grid_post,
        [1, 2, 3, 4, 5],
        action=2,
    )

    # Ledger must have recorded causal consequence
    assert "switch_4" in engine.object_planner.ledger.unlocked_by_object
    assert (2, 6) in engine.object_planner.ledger.unlocked_by_object["switch_4"]
    assert (2, 6) in engine.object_planner.ledger.get_unlocked_barriers()


def test_causal_actuator_contingency_pursuit() -> None:
    """Verify that mutating an object triggers PURSUIT_CAUSAL_ACTUATOR options on subsequent steps."""
    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [{"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}}]
    )

    grid = np.zeros((10, 10), dtype=int)
    grid[4, 4] = 5  # Actuator object

    # First interaction at (4, 4) mutates world
    engine.prev_grid = grid.copy()
    grid_post = grid.copy()
    grid_post[4, 4] = 7  # Color changed
    engine.assimilate_feedback(
        grid_post,
        [6],
        action=6,
        action_data={"x": 4, "y": 4},
    )

    assert engine.active_causal_actuator_coord == (4, 4)
    assert engine.consecutive_causal_probes == 1

    # Planner should now immediately pursue the active causal actuator
    plan = engine.object_planner.plan_macro_option(engine, grid_post, [6])
    assert plan is not None
    assert len(plan) == 1
    assert plan[0].action == 6
    assert plan[0].action_data == {"x": 4, "y": 4}
    assert plan[0].expected_mutation == "PURSUIT_CAUSAL_ACTUATOR"


def test_multi_cell_unprobed_selection_and_quiescence_avoidance() -> None:
    """Verify multi-cell objects cycle to unprobed cells and avoid quiescent coordinates."""
    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [{"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}}]
    )

    grid = np.zeros((10, 10), dtype=int)
    # 2-cell object at (3, 3) and (3, 4)
    grid[3, 3] = 8
    grid[3, 4] = 8

    # Mark anchor (3, 3) as quiescent
    engine.quiescent_click_targets.add((3, 3))

    plan = engine.object_planner.plan_macro_option(engine, grid, [6])
    assert plan is not None
    assert len(plan) == 1
    assert plan[0].action == 6
    # Must target the unprobed cell (3, 4), not the quiescent anchor (3, 3)
    assert plan[0].action_data == {"x": 4, "y": 3}

    # Now mark the second cell as quiescent as well
    engine.quiescent_click_targets.add((3, 4))
    plan2 = engine.object_planner.plan_macro_option(engine, grid, [6])
    # When all cells are quiescent and no other objects exist, it must return None instead of looping
    assert plan2 is None


def test_retry_decay_and_quiescence_preservation() -> None:
    """Verify same-level retry preserves quiescent targets and decays ledger counts,
    while cross-level reset clears them."""
    engine = AutonomousEpistemicEngine()
    engine.quiescent_click_targets.add((1, 1))
    engine.object_planner.ledger.interaction_counts["obj_A"] = 4

    # 1. Same-level retry (is_new_level=False, retain_dynamics=True)
    engine.reset_episode(retain_dynamics=True, is_new_level=False, level=0)
    assert (1, 1) in engine.quiescent_click_targets
    assert engine.object_planner.ledger.interaction_counts["obj_A"] == 2

    # 2. New-level transition (is_new_level=True)
    engine.reset_episode(retain_dynamics=True, is_new_level=True, level=1)
    assert (1, 1) not in engine.quiescent_click_targets
    assert "obj_A" not in engine.object_planner.ledger.interaction_counts


def test_peripheral_frame_avoidance() -> None:
    """Verify interior candidates take priority over peripheral border frame cells."""
    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [{"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}}]
    )

    grid = np.zeros((10, 10), dtype=int)
    # Border wall on row 0
    grid[0, :] = 1

    # Interior entity at (4, 4)
    grid[4, 4] = 8

    plan = engine.object_planner.plan_macro_option(engine, grid, [6])
    assert plan is not None
    # Must target interior entity at (4, 4), not border cells at row 0
    assert plan[0].action_data == {"x": 4, "y": 4}


def test_macro_option_valid_targets_snapping() -> None:
    """Verify that macro option clicks snap to discrete valid_targets and avoid quiescent ones."""
    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [
            {
                "action_id": 6,
                "name": "CLICK_CELL",
                "parameters": {"x": "int", "y": "int"},
                "metadata": {
                    "valid_targets": [
                        {"x": 0, "y": 10},
                        {"x": 0, "y": 14},
                    ]
                },
            }
        ]
    )

    grid = np.zeros((15, 15), dtype=int)
    grid[8, 1] = 8  # Salient object near (y=10, x=0)

    # 1. Probing snaps to closest valid target (y=10, x=0)
    plan1 = engine.object_planner.plan_macro_option(engine, grid, [6])
    assert plan1 is not None
    assert plan1[0].action == 6
    assert plan1[0].action_data == {"x": 0, "y": 10}

    # 2. Mark (10, 0) as quiescent: planner should snap to (14, 0)
    engine.quiescent_click_targets.add((10, 0))
    plan2 = engine.object_planner.plan_macro_option(engine, grid, [6])
    assert plan2 is not None
    assert plan2[0].action == 6
    assert plan2[0].action_data == {"x": 0, "y": 14}

    # 3. Mark all valid targets as quiescent: planner returns None instead of invalid clicks
    engine.quiescent_click_targets.add((14, 0))
    plan3 = engine.object_planner.plan_macro_option(engine, grid, [6])
    assert plan3 is None


def test_motor_effector_discrete_valid_targets_quiescence() -> None:
    """Verify MotorCortexEffector.ground_effector_action avoids quiescent valid_targets and inhibits when exhausted."""
    from hbllm.hcir.world.cortex_motor import MotorCortexEffector

    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [
            {
                "action_id": 6,
                "name": "CLICK_CELL",
                "parameters": {"x": "int", "y": "int"},
                "metadata": {
                    "valid_targets": [
                        {"x": 0, "y": 10},
                        {"x": 0, "y": 14},
                    ]
                },
            }
        ]
    )

    grid = np.zeros((15, 15), dtype=int)
    grid[8, 1] = 8

    # 1. When (10, 0) is quiescent, fallback should pick (14, 0)
    engine.quiescent_click_targets.add((10, 0))
    coords = MotorCortexEffector.ground_effector_action(engine, grid, 6)
    assert coords is not None
    assert coords["x"] == 0 and coords["y"] == 14

    # 2. When both are quiescent, fallback returns None and inhibits action 6
    engine.quiescent_click_targets.add((14, 0))
    coords2 = MotorCortexEffector.ground_effector_action(engine, grid, 6)
    assert coords2 is None
    assert engine.inhibited_actions.get(6, 0) > 0


def test_cross_level_causal_affinity_unlocking() -> None:
    """Verify that tool-barrier affinities in symbolic_theory immediately guide Directive 4 across levels."""
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

    # Barrier 1 blocking exit
    grid[:, 6] = 1
    engine.symbolic_theory.barrier_features.add(1)

    # Tool A (feature 6) at (2, 4) - closer (2 steps) but unlocks unrelated barrier 99
    grid[2, 4] = 6
    engine.symbolic_theory.record_tool_barrier_affinity(tool_feature=6, barrier_feature=99)

    # Tool B (feature 5) at (5, 2) - farther (3 steps) but unlocks barrier 1!
    grid[5, 2] = 5
    engine.symbolic_theory.record_tool_barrier_affinity(tool_feature=5, barrier_feature=1)

    # Working memory is fresh (simulating level reset)
    engine.working_memory.tool_barrier_affinities.clear()

    # Directive 4 should pick Tool B (feature 5) because of causal affinity with blocking barrier 1!
    plan = engine.object_planner.plan_macro_option(engine, grid, [1, 2, 3, 4, 5])
    assert plan is not None
    assert len(plan) == 3  # 3 steps DOWN towards (5, 2)
    assert all(step.action == 2 for step in plan)
    assert plan[-1].predicted_avatar_pos == (5, 2)
