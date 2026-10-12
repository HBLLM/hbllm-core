"""Targeted Bottleneck Scenarios — Comprehensive unit and integration test suite.

Tests end-to-end cognitive solutions for key bottlenecks identified in benchmarks:
1. Cargo corner and wall deadlock detection & pruning in continuous push kinematics.
2. Hierarchical subgoal goal-reachability invariant (rejecting dead-end approaches to locked doors).
3. Oriented creature physical footprint collision safety in Space-Time A*.
4. Allocentric click effector grounding onto salient entity centroids with inhibition-of-return.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.cortex_motor import MotorCortexEffector
from hbllm.hcir.world.cortex_planner import (
    EpistemicGoalScheduler,
    MentalSimulationPlanner,
)
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAffordance


def test_cargo_corner_and_wall_deadlock_detection() -> None:
    """Verify CounterfactualDeadlockDetector identifies corner and wall deadlocks for cargo."""
    grid_shape = (10, 10)
    static_barriers = {
        (1, 1),
        (1, 2),
        (1, 3),  # Horizontal wall
        (2, 1),
        (3, 1),  # Vertical wall forming corner at (2, 2)
    }
    goals = {(5, 5)}

    # Corner at (2, 2) bounded by barriers at (1, 2) and (2, 1)
    is_corner = CounterfactualDeadlockDetector.is_corner_deadlock(
        pos=(2, 2),
        goals=goals,
        static_barriers=static_barriers,
        grid_shape=grid_shape,
    )
    assert is_corner, (
        "Block at (2, 2) surrounded by orthogonal barriers must be detected as corner deadlock!"
    )

    # If (2, 2) IS a goal, it must NOT be a deadlock
    is_not_deadlock = CounterfactualDeadlockDetector.is_corner_deadlock(
        pos=(2, 2),
        goals={(2, 2)},
        static_barriers=static_barriers,
        grid_shape=grid_shape,
    )
    assert not is_not_deadlock, "Goal position must never be flagged as deadlock!"

    # Wall deadlock along grid boundary with no goals on boundary
    is_wall = CounterfactualDeadlockDetector.is_wall_deadlock(
        pos=(0, 4),
        goals=goals,
        static_barriers=set(),
        grid_shape=grid_shape,
    )
    assert is_wall, (
        "Cargo against perimeter wall with no goals on that wall must be a wall deadlock!"
    )


def test_sokoban_simulation_prunes_deadlock_branch() -> None:
    """Verify mental simulation avoids pushing cargo into a corner deadlock."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (3, 3)

    # 4 cardinal movements calibrated
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 7x7 grid:
    # Wall corner at (1, 1) formed by (0, 1) and (1, 0)
    # Cargo block (feature 4) at (2, 2)
    # Goal (feature 3) at (5, 5)
    # Avatar at (3, 3)
    grid = np.zeros((7, 7), dtype=int)
    grid[3, 3] = 1  # Avatar
    grid[2, 2] = 4  # Pushable cargo block
    grid[5, 5] = 3  # Goal

    # Perimeter walls
    grid[0, :] = 2
    grid[:, 0] = 2

    engine.learned_goal_features.add(3)
    engine.learned_cargo_features.add(4)
    engine.symbolic_theory.cargo_features.add(4)
    engine.symbolic_theory.barrier_features.add(2)
    engine.learned_barriers.update([(0, c) for c in range(7)])
    engine.learned_barriers.update([(r, 0) for r in range(7)])

    # Deadlock detector should reject candidate state that pushes block to (1, 1)
    eval_deadlock = CounterfactualDeadlockDetector.evaluate_deadlock(
        pos=(1, 1),
        goals={(5, 5)},
        static_barriers=engine.learned_barriers,
        all_blocks={(1, 1)},
        grid_shape=(7, 7),
    )
    assert eval_deadlock.is_deadlock, (
        "Pushing cargo into (1, 1) perimeter corner must be evaluated as deadlock!"
    )


def test_hierarchical_subgoal_unreachable_barrier_rejection() -> None:
    """Verify hierarchical subgoal planning returns None when goal is completely unreachable without switch."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (2, 1)

    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 5x7 grid:
    # Avatar at (2, 1)
    # Solid wall along column 3: (0, 3) through (4, 3)
    # Goal at (2, 5) behind the wall with NO switch
    grid = np.zeros((5, 7), dtype=int)
    grid[2, 1] = 1
    grid[2, 5] = 3
    grid[:, 3] = 2

    engine.learned_goal_features.add(3)
    engine.symbolic_theory.barrier_features.add(2)
    engine.learned_barriers.update([(r, 3) for r in range(5)])

    entities = engine.extract_entities(grid, bg=0)
    proposal = EpistemicGoalScheduler.schedule_goals(
        engine, grid, entities, bg=0, av_feats={1}, available_actions=[1, 2, 3, 4]
    )
    movable_actions = [
        (a, engine.action_dynamics[a].delta_r, engine.action_dynamics[a].delta_c)
        for a in [1, 2, 3, 4]
    ]

    plan = MentalSimulationPlanner._plan_hierarchical_subgoals(
        engine=engine,
        curr_grid=grid,
        available_actions=[1, 2, 3, 4],
        start_pos=(2, 1),
        goals=set(proposal.goals),
        static_barriers=engine.learned_barriers,
        mutation_triggers={},
        movable_actions=movable_actions,
    )

    assert plan is None, (
        "Hierarchical subgoal planner must return None when goal is unreachable and no trigger exists, rather than bumping the wall!"
    )


def test_hierarchical_subgoal_solves_with_reachable_trigger() -> None:
    """Verify hierarchical subgoal planning successfully synthesizes trigger sequence when switch opens barrier."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (2, 1)

    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 5x7 grid:
    # Avatar at (2, 1)
    # Solid wall along col 3: (0, 3) through (4, 3)
    # Switch at (4, 1) that opens barrier at (2, 3)
    # Goal at (2, 5)
    grid = np.zeros((5, 7), dtype=int)
    grid[2, 1] = 1
    grid[2, 5] = 3
    grid[:, 3] = 2
    grid[4, 1] = 6  # Switch

    engine.learned_goal_features.add(3)
    engine.symbolic_theory.barrier_features.add(2)
    engine.learned_barriers.update([(r, 3) for r in range(5)])

    aff = RemoteCausalAffordance(
        trigger_pos=(4, 1),
        trigger_feature=6,
        remote_pos=(2, 3),
        original_remote_feature=2,
        mutated_remote_feature=0,
        confidence=1.0,
    )
    engine.working_memory.remote_causal.register_affordance(aff)

    entities = engine.extract_entities(grid, bg=0)
    proposal = EpistemicGoalScheduler.schedule_goals(
        engine, grid, entities, bg=0, av_feats={1}, available_actions=[1, 2, 3, 4]
    )
    movable_actions = [
        (a, engine.action_dynamics[a].delta_r, engine.action_dynamics[a].delta_c)
        for a in [1, 2, 3, 4]
    ]
    mutation_triggers = {(4, 1): {(2, 3)}}

    plan = MentalSimulationPlanner._plan_hierarchical_subgoals(
        engine=engine,
        curr_grid=grid,
        available_actions=[1, 2, 3, 4],
        start_pos=(2, 1),
        goals=set(proposal.goals),
        static_barriers=engine.learned_barriers,
        mutation_triggers=mutation_triggers,
        movable_actions=movable_actions,
    )

    assert plan is not None, (
        "Hierarchical subgoal planner must synthesize multi-stage plan via remote trigger!"
    )
    assert any(step.predicted_avatar_pos == (4, 1) for step in plan), (
        "Plan must visit trigger switch at (4, 1)!"
    )
    assert plan[-1].predicted_avatar_pos == (2, 5), (
        "Final step of plan must reach terminal goal at (2, 5)!"
    )


def test_oriented_threat_body_collision_avoidance() -> None:
    """Verify Space-Time A* avoids stepping into oriented enemy body cell."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (2, 1)

    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 5x7 grid:
    # Avatar at (2, 1)
    # Enemy at (2, 3) facing DOWN (gaze at (3, 3))
    # Goal at (2, 5)
    grid = np.zeros((5, 7), dtype=int)
    grid[2, 1] = 1  # Avatar
    grid[2, 5] = 3  # Goal

    # 3x3 enemy centered at (2, 3): body feature 8, eye feature 15 facing down
    grid[1:4, 2:5] = 8
    grid[3, 3] = 15  # eye facing DOWN (positive delta_r)

    engine.learned_goal_features.add(3)
    engine.hazard_tracker.known_lethal_features.add(8)

    available_actions = [1, 2, 3, 4]
    plan = engine.simulate_in_mind(grid, available_actions)

    assert plan is not None
    # None of the plan steps should land on the enemy's center (2, 3)
    assert not any(step.predicted_avatar_pos == (2, 3) for step in plan), (
        "Plan must never path through enemy body cell!"
    )


def test_pure_click_salient_centroid_grounding_and_ior() -> None:
    """Verify MotorCortexEffector grounds click onto salient candidate centroids and applies IOR."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_pos = None
    engine.avatar_feature = None

    # 20x20 grid with 2 candidate clickable tiles on background 0
    grid = np.zeros((20, 20), dtype=int)
    # Tile 1: 3x3 at rows 4..6, cols 4..6 (feature 5, centroid at (5, 5))
    grid[4:7, 4:7] = 5
    # Tile 2: 3x3 at rows 14..16, cols 14..16 (feature 7, centroid at (15, 15))
    grid[14:17, 14:17] = 7

    # Ground action 6 (click)
    coords1 = MotorCortexEffector.ground_effector_action(engine, grid, action=6)
    assert coords1 is not None
    assert "x" in coords1 and "y" in coords1
    c1, r1 = coords1["x"], coords1["y"]

    # Must be grounded near one of the tile centroids
    is_tile1 = abs(r1 - 5) <= 1 and abs(c1 - 5) <= 1
    is_tile2 = abs(r1 - 15) <= 1 and abs(c1 - 15) <= 1
    assert is_tile1 or is_tile2, (
        f"Click 1 ({r1}, {c1}) must target one of the two candidate tile centroids!"
    )

    # Advance step counter to model elapsed action step
    engine.step_counter += 1

    # Second click must apply Inhibition of Return (IOR) and select the other tile
    coords2 = MotorCortexEffector.ground_effector_action(engine, grid, action=6)
    assert coords2 is not None
    c2, r2 = coords2["x"], coords2["y"]

    if is_tile1:
        assert abs(r2 - 15) <= 1 and abs(c2 - 15) <= 1, (
            "After clicking tile 1, IOR must steer next click to tile 2!"
        )
    else:
        assert abs(r2 - 5) <= 1 and abs(c2 - 5) <= 1, (
            "After clicking tile 2, IOR must steer next click to tile 1!"
        )


def test_stationary_sentry_gaze_is_never_relaxed_as_safe_path() -> None:
    """Verify that stationary sentry gaze/line of fire is never treated as a passable relaxed route."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (5, 1)

    # 4 cardinal movements calibrated
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 10x10 grid:
    # Avatar at (5, 1)
    # Goal at (5, 9)
    # Sentry 3x3 at rows 4..6, cols 5..7, eye at (5, 5) facing LEFT -> gaze at (5, 4)
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 1] = 1  # Avatar
    grid[5, 9] = 3  # Goal behind sentry

    grid[4:7, 5:8] = 8
    grid[5, 5] = 15  # eye facing LEFT (negative delta_c)

    # Corridor is row 5, walled at row 3 and row 7
    grid[3, :] = 2
    grid[7, :] = 2
    # Walls above and below corridor
    grid[4, :5] = 2
    grid[4, 8:] = 2
    grid[6, :5] = 2
    grid[6, 8:] = 2

    engine.learned_goal_features.add(3)
    engine.hazard_tracker.known_lethal_features.add(8)
    engine.symbolic_theory.barrier_features.add(2)

    available_actions = [1, 2, 3, 4]
    plan = engine.simulate_in_mind(grid, available_actions)

    # Since the only path to goal (5, 9) crosses the sentry's living gaze at (5, 4)
    # and the sentry cannot be flanked due to walls, the plan MUST be None.
    # It must NEVER relax the gaze constraint to walk through the lethal sightline!
    assert plan is None, (
        "Planner must never relax a stationary sentry gaze sightline into a valid plan!"
    )


def test_tactical_flanking_neutralizes_threat_in_mental_simulation() -> None:
    """Verify that flanking an oriented sentry from behind/side neutralizes it and opens the path."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (1, 4)  # Starting above the sentry

    # 4 cardinal movements calibrated
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 10x10 grid:
    # Avatar at (1, 4)
    # Sentry centered at (3, 4) facing DOWN with gaze at (5, 4)
    # Goal at (6, 4) beyond the sentry's gaze
    grid = np.zeros((10, 10), dtype=int)
    grid[1, 4] = 1  # Avatar
    grid[6, 4] = 3  # Goal

    # 3x3 enemy centered at (3, 4): rows 2..4, cols 3..5. Feature 8, eye at (4, 4) facing DOWN
    grid[2:5, 3:6] = 8
    grid[4, 4] = 15  # eye facing DOWN -> gaze at (4 + 1, 4) = (5, 4)

    # Walls block left and right detours, so avatar must walk DOWN through sentry: (1, 4) -> (2, 4) -> (3, 4)
    # Since avatar approaches from (1, 4) (NOT from gaze pos (5, 4)), this is a tactical flank!
    grid[:, 2] = 2
    grid[:, 6] = 2

    engine.learned_goal_features.add(3)
    engine.hazard_tracker.known_lethal_features.add(8)
    engine.symbolic_theory.barrier_features.add(2)

    available_actions = [1, 2, 3, 4]
    plan = engine.simulate_in_mind(grid, available_actions)

    assert plan is not None, "Planner should find a path by tactically flanking the sentry!"
    # The plan must reach the goal (6, 4)
    assert plan[-1].predicted_avatar_pos == (6, 4), "Plan must successfully reach the goal!"


def test_oriented_threat_observation_rejects_spurious_translations() -> None:
    """Verify that creature motion observation enforces entity count conservation and step size bounds."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (5, 5)
    engine.step_counter = 1  # Initial step
    engine.prev_grid = np.zeros((10, 10), dtype=int)

    grid1 = np.zeros((10, 10), dtype=int)
    grid1[5, 5] = 1  # Avatar
    # Sentry: 3x3 at rows 2..4, cols 2..4 (feature 8, eye at (4, 3) facing DOWN)
    grid1[2:5, 2:5] = 8
    grid1[4, 3] = 15

    available_actions = [1, 2, 3, 4]
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)

    # Step 1: should not mark feature 8 as mobile
    engine._observe_oriented_threat_motion(grid1, available_actions, av_feats={1})
    assert 8 not in engine.mobile_threat_features, (
        "On step_counter <= 1, threats must not be marked mobile!"
    )

    # Step 2 with spurious teleportation across grid (dist = 5 > step_size * 2)
    engine.step_counter = 2
    grid2_teleport = np.zeros((10, 10), dtype=int)
    grid2_teleport[5, 5] = 1
    # Teleported to rows 7..9, cols 2..4
    grid2_teleport[7:10, 2:5] = 8
    grid2_teleport[9, 3] = 15
    engine._observe_oriented_threat_motion(grid2_teleport, available_actions, av_feats={1})
    assert 8 not in engine.mobile_threat_features, (
        "Teleportation / reset must not be marked mobile!"
    )

    # Step 3 with entity death / count mismatch (1 vanished, 0 appeared)
    engine.step_counter = 3
    grid3_dead = np.zeros((10, 10), dtype=int)
    grid3_dead[5, 5] = 1
    engine._observe_oriented_threat_motion(grid3_dead, available_actions, av_feats={1})
    assert 8 not in engine.mobile_threat_features, (
        "Threat death/neutralization must not be marked mobile!"
    )


def test_click_effector_lattice_quantization() -> None:
    """Verify MotorCortexEffector quantizes fallback clicks to lattice quantum centroids when step_sz > 1."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_pos = None
    engine.avatar_feature = None
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=3, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=-3, delta_c=0, confidence=1.0)

    # 15x15 grid with a single 3x3 block of feature 7 at rows 6..8, cols 9..11
    # Background 0
    grid = np.zeros((15, 15), dtype=int)
    grid[6:9, 9:12] = 7

    # Ground click action (e.g., action 6)
    coords = MotorCortexEffector.ground_effector_action(engine, grid, action=6)
    assert coords is not None
    # Lattice cell with step_sz=3 for row 6..8 has centroid (6 // 3)*3 + 3//2 = 7
    # For col 9..11 has centroid (9 // 3)*3 + 3//2 = 10
    assert coords["y"] == 7 and coords["x"] == 10, (
        f"Grounded click ({coords['y']}, {coords['x']}) must match lattice centroid (7, 10)!"
    )
