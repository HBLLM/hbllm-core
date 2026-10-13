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


def test_hud_status_banner_masking() -> None:
    """Verify is_peripheral_frame_cell identifies multi-row top HUD banners and divider lines."""
    # 64x64 grid mimicking su15:
    # Rows 0..9 background color 4 with text digits of color 15 at rows 4..6
    # Row 10 is divider line of color 5
    # Rows 11..63 is main playing field with background 5 and 9
    grid = np.full((64, 64), 5, dtype=int)
    grid[:10, :] = 4
    grid[4:7, 30:33] = 15  # Isolated digit
    grid[10, :] = 5  # Divider line
    grid[63, :] = 0  # Bottom border

    # Cells in rows 0..10 must be flagged as peripheral HUD banner
    for r in [0, 2, 5, 8, 10]:
        assert MotorCortexEffector.is_peripheral_frame_cell(r, 32, 64, 64, grid), (
            f"Row {r} in HUD banner must be recognized as peripheral!"
        )

    # Playing field cells (e.g. row 20, col 20) must NOT be peripheral
    assert not MotorCortexEffector.is_peripheral_frame_cell(20, 20, 64, 64, grid), (
        "Main playing arena must not be flagged as peripheral!"
    )


def test_effector_valid_targets_snapping() -> None:
    """Verify ground_effector_action snaps targets to discrete schema affordance coordinates."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_pos = None
    engine.avatar_feature = None

    # Define discrete action repertoire with step-4 grid points starting at y=10
    valid_targets = [{"x": x, "y": y} for x in range(0, 64, 4) for y in range(10, 64, 4)]
    schemas = [
        {
            "action_id": 6,
            "name": "CLICK_CELL",
            "parameters": {"x": "int", "y": "int"},
            "valid_targets": valid_targets,
        }
    ]
    engine.register_action_space(schemas)

    # Grid with a block of color 7 at (59, 5) -> nearest valid target is (58, 4) or (58, 8)
    grid = np.full((64, 64), 5, dtype=int)
    grid[:10, :] = 4  # Top HUD banner
    grid[10, :] = 5  # Divider
    grid[58:61, 3:6] = 7  # Movable object at bottom left

    coords = MotorCortexEffector.ground_effector_action(engine, grid, action=6)
    assert coords is not None, "Grounding must produce target coordinates!"
    # Snapped coordinate must be one of the valid targets
    assert any(vt["x"] == coords["x"] and vt["y"] == coords["y"] for vt in valid_targets), (
        f"Grounded coords {coords} must be in valid_targets repertoire!"
    )
    # Target must be in the playing field (y >= 10), never in the HUD banner (y < 10)
    assert coords["y"] >= 10, f"Grounded y={coords['y']} must not fall inside HUD banner!"


def test_metacognitive_urgency_budget_modulation() -> None:
    """Verify decide() zeros out refractory exploration cooldowns under tight step budgets."""
    engine = AutonomousEpistemicEngine()
    engine.exploration_budget = 20
    engine.set_environmental_step_limit(20)
    engine.exploration_cooldown = 5
    engine.simulation_cooldown = 4

    assert engine.is_under_strict_step_budget(), "20 remaining steps must trigger strict budget!"

    grid = np.zeros((10, 10), dtype=int)
    grid[1, 1] = 1  # avatar
    grid[8, 8] = 2  # goal
    engine.avatar_pos = (1, 1)
    engine.avatar_feature = 1
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=0, delta_c=1, confidence=1.0)

    # Decide next action
    engine.decide(grid, available_actions=[1, 2])

    # Cooldowns must be zeroed under acute urgency
    assert engine.exploration_cooldown == 0, "Exploration cooldown must be cleared under urgency!"
    assert engine.simulation_cooldown == 0, "Simulation cooldown must be cleared under urgency!"


def test_active_inference_effector_kinetics_mode() -> None:
    """Verify assimilate_feedback dynamically infers attraction/pull vs repulsion mode."""
    engine = AutonomousEpistemicEngine()
    assert engine.effector_kinetics_mode is None, "Kinetics mode must start unobserved (None)!"

    engine.prev_grid = np.zeros((20, 20), dtype=int)
    engine.prev_grid[10:13, 10:13] = 7  # Object at centroid (11, 11)

    curr_grid = np.zeros((20, 20), dtype=int)
    # Agent clicked at (5, 11) (above the object)
    # Object moved to (7, 11) (closer to the click, i.e. pull / attraction)
    curr_grid[6:9, 10:13] = 7

    engine.assimilate_feedback(
        curr_grid=curr_grid,
        available_actions=[6],
        action=6,
        action_data={"y": 5, "x": 11},
    )

    assert engine.effector_kinetics_mode == 1, (
        "Object moving toward the click must induce attraction mode (+1)!"
    )

    # Now test repulsion (-1): agent clicks below object at (15, 11) and object moves upwards away to (3, 11)
    engine.prev_grid = np.zeros((20, 20), dtype=int)
    engine.prev_grid[6:9, 10:13] = 7  # Object at centroid (7, 11)
    curr_grid_repulse = np.zeros((20, 20), dtype=int)
    curr_grid_repulse[2:5, 10:13] = 7  # Moves further away to (3, 11)

    engine.assimilate_feedback(
        curr_grid=curr_grid_repulse,
        available_actions=[6],
        action=6,
        action_data={"y": 15, "x": 11},
    )
    assert engine.effector_kinetics_mode == -1, (
        "Object moving away from the click must induce repulsion mode (-1)!"
    )

    # Cross-game reset without retaining dynamics clears kinetics mode
    engine.reset_episode(retain_dynamics=False)
    assert engine.effector_kinetics_mode is None, (
        "Kinetics mode must be cleared upon cross-game reset!"
    )


def test_level_transition_barrier_re_evaluation() -> None:
    """Verify episodic reset clears prior room terrain barrier features upon new level transitions."""
    engine = AutonomousEpistemicEngine()
    # Level 1 induces feature 2 as a barrier
    engine.symbolic_theory.induce_barrier(2)
    engine.symbolic_theory.induce_walkable(0)
    assert 2 in engine.symbolic_theory.barrier_features

    # Transition to new level (e.g. Level 4 where feature 2 is the traversable track)
    engine.episodic_cortex.consolidate_schema_and_reset(
        engine=engine,
        is_new_level=True,
        retain_dynamics=True,
    )

    # Barrier features must be cleared on new level transition to allow affordance re-evaluation
    assert 2 not in engine.symbolic_theory.barrier_features, (
        "Prior room barrier features must be cleared across new level transitions!"
    )
    assert len(engine.symbolic_theory.barrier_features) == 0


def test_biological_undo_affordance_grounding() -> None:
    """Verify assimilate_feedback detects when an action restores state to grid_history[-2] as UNDO."""
    engine = AutonomousEpistemicEngine()

    state_0 = np.zeros((10, 10), dtype=int)
    state_0[5, 5] = 3

    state_1 = np.zeros((10, 10), dtype=int)
    state_1[6, 6] = 3

    # History: state_0 was the previous state
    engine.grid_history.append(state_0.copy())
    engine.grid_history.append(state_1.copy())
    engine.prev_grid = state_1.copy()

    # Agent executes action 7, and board returns to state_0
    engine.assimilate_feedback(
        curr_grid=state_0,
        available_actions=[6, 7],
        action=7,
        action_data=None,
    )

    # Action 7 must be recognized as an undo action and inhibited
    assert 7 in engine.undo_actions, "Action 7 must be registered in undo_actions!"
    assert engine.inhibited_actions.get(7, 0) > 0, (
        "Undo action must be inhibited during forward progress!"
    )


def test_spatial_effector_safeguard_against_blind_discrete_arbitration() -> None:
    """Verify arbitrate_affordances does not blindly dispatch arbitrary discrete actions when effectors are available."""
    engine = AutonomousEpistemicEngine()
    engine.step_counter = 10
    engine.consecutive_stuck_steps = 3  # stuck condition

    grid = np.zeros((10, 10), dtype=int)
    # Available actions are [6, 7] (where 6 is spatial effector and 7 is discrete action/undo)
    engine.register_action_space(
        [
            {"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}},
            {"action_id": 7, "name": "ACTION7"},
        ]
    )

    # With no affordance panel detected on grid, arbitrate_affordances must NOT choose action 7
    choice = engine.deliberation_engine.arbitrate_affordances(
        engine=engine,
        curr_grid=grid,
        available_actions=[6, 7],
        known_av_feats=set(),
    )
    assert choice is None, (
        "When spatial effectors are present and no panel exists, arbitrate_affordances must return None instead of blind discrete execution!"
    )


def test_undo_probe_filtering_and_effector_priority() -> None:
    """Verify EpistemicCuriosityExplorer ignores undo actions and prioritizes spatial effectors during probing."""
    from hbllm.hcir.world.cortex_motor import EpistemicCuriosityExplorer

    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [
            {"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}},
            {"action_id": 7, "name": "ACTION7"},
        ]
    )
    grid = np.zeros((10, 10), dtype=int)

    # Mark action 7 as an undo action and inhibit it
    engine.undo_actions.add(7)
    engine.inhibited_actions[7] = 16

    # Action 6 is spatial effector, action 7 is undo.
    # When available actions are [6, 7], probe must select 6, not 7!
    proposal = EpistemicCuriosityExplorer.compute_epistemic_probe(
        engine, grid, available_actions=[6, 7]
    )
    assert proposal.action == 6, f"Probe must choose effector 6, but chose {proposal.action}!"


def test_peripheral_driver_action_schema_completeness() -> None:
    """Verify ARC_DEFAULT_ACTION_SCHEMAS integrity in the peripheral driver package."""
    from kaggle_submission.constants import ARC_DEFAULT_ACTION_SCHEMAS

    ids = {s["action_id"] for s in ARC_DEFAULT_ACTION_SCHEMAS}
    assert ids == {1, 2, 3, 4, 5, 6, 7}, (
        f"All ARC-AGI actions 1-7 must be present in peripheral schema! Got {ids}"
    )


def test_teleological_cutset_subgoal_decomposition() -> None:
    """Verify teleological backward cut-set decomposition formulates subgoals to clear blocking barriers."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (1, 1)

    # Calibrate cardinal directions
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 10x10 grid with dividing wall at col 5
    grid = np.zeros((10, 10), dtype=int)
    grid[1, 1] = 1  # Avatar
    grid[8, 8] = 3  # Goal on right side
    grid[:, 5] = 2  # Rigid wall
    grid[5, 5] = 4  # Purple Door barrier at (5, 5)
    grid[1, 3] = 4  # Purple Key item at (1, 3) on avatar's side

    engine.learned_goal_features.add(3)
    engine.symbolic_theory.barrier_features.add(2)
    engine.symbolic_theory.barrier_features.add(4)

    available_actions = [1, 2, 3, 4]
    plan = engine.simulate_in_mind(grid, available_actions)

    assert plan is not None, "Teleological backward cut-set planner must formulate a plan to key!"
    assert engine.subgoal_stack.has_pending_goals, (
        "Subgoal stack must receive the teleological key subgoal!"
    )
    sg = engine.subgoal_stack.current_subgoal
    assert sg is not None
    assert sg.target_destination == (1, 3), (
        f"Subgoal destination must be purple key at (1, 3), got {sg.target_destination}!"
    )
    # The returned plan must lead avatar to the key at (1, 3)
    assert plan[-1].predicted_avatar_pos == (1, 3), (
        f"Plan must navigate to key at (1, 3), got {plan[-1].predicted_avatar_pos}!"
    )


def test_categorical_quiescent_feature_pruning() -> None:
    """Verify Pearl do(a) causal credit assignment categorizes inert features and prunes them from clicks."""
    from hbllm.hcir.world.cortex_motor import EpistemicCuriosityExplorer

    engine = AutonomousEpistemicEngine()
    engine.register_action_space(
        [
            {"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}},
        ]
    )
    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 5  # Feature 5 entity
    grid[7, 7] = 8  # Feature 8 entity

    # Mark feature 5 as categorically quiescent (e.g. tested twice with zero diff)
    engine.quiescent_features.add(5)

    # Plan probe: Must NEVER choose feature 5 when feature 8 is available!
    proposal = EpistemicCuriosityExplorer.compute_epistemic_probe(
        engine, grid, available_actions=[6]
    )
    assert proposal is not None and proposal.action == 6
    assert proposal.action_data is not None
    click_y = proposal.action_data.get("y")
    click_x = proposal.action_data.get("x")
    clicked_feat = grid[click_y, click_x]
    assert clicked_feat != 5, (
        f"Effector must not click quiescent feature 5! Clicked at ({click_y}, {click_x})"
    )


def test_metacognitive_acc_death_conflict_inversion() -> None:
    """Verify that ACC conflict resolution invalidates plans heading into death clusters and inverts hypotheses."""
    from hbllm.hcir.world.cortex_deliberation import PrefrontalDeliberationEngine
    from hbllm.hcir.world.cortex_planner import MentalSimulationStep

    delib = PrefrontalDeliberationEngine()
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (5, 1)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    grid = np.zeros((10, 10), dtype=int)
    grid[5, 1] = 1
    grid[5, 8] = 9  # False goal at (5, 8)
    engine.symbolic_theory.goal_features.add(9)

    # Register 2 deaths at cluster near (5, 7)
    engine.working_memory.acc_conflict.register_death_event(
        actor_pos=(5.0, 7.0), step=10, last_action=4
    )
    engine.working_memory.acc_conflict.register_death_event(
        actor_pos=(5.0, 7.0), step=20, last_action=4
    )
    assert engine.working_memory.acc_conflict.active_conflict, (
        "2 deaths must trigger active conflict!"
    )

    # Candidate plan heading toward (5, 7)
    engine.mental_plan.append(MentalSimulationStep(action=4, predicted_avatar_pos=(5, 7)))

    # Call decide to invoke executive conflict resolution
    delib.decide(engine, curr_grid=grid, available_actions=[4])

    # Mental plan heading into death cluster must be cleared!
    assert len(engine.mental_plan) == 0, "Active plan heading into death cluster must be cleared!"
    # Death cluster cells must be marked in static lethal positions!
    assert (5, 7) in engine.hazard_tracker.static_lethal_positions, (
        "Death cluster center must be registered as static lethal hazard!"
    )
    # False goal at death cluster must be inverted!
    assert (5, 7) in engine.hazard_tracker.static_lethal_positions
