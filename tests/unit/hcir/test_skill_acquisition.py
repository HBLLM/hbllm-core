"""Unit tests for the HCIR Multi-Paradigm Skill Acquisition System.

Tests all 6 skill acquisition architectures:
1. Spatiotemporal & Periodic Phase Acquisition (Space-time A*, hazard tracking)
2. Algebraic & Permutation Operator Inversion (GF(2) Gaussian elimination, Lights-Out)
3. Kinematics, Inertia & Momentum Models (Ice sliding raycasts, momentum detection)
4. Relational Object-to-Object Affordances (Sokoban pushing, tool-target rules)
5. Morphological Program Synthesis (Geometric reflection/rotation, macro painting)
6. Coupled Controllable Coordination (Mirrored multi-agent joint convergence)
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.skills import (
    AutonomousTrajectoryModel,
    CoupledControllableSkillAcquisition,
    GF2LinearSolver,
    KinematicMomentumSkillAcquisition,
    MorphologicalProgramSynthesis,
    PermutationAlgebraSkillAcquisition,
    RelationalAffordanceSkillAcquisition,
    SpatiotemporalSkillAcquisition,
)
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity

# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatiotemporal Dynamics & Periodic Phase
# ─────────────────────────────────────────────────────────────────────────────


def test_spatiotemporal_trajectory_and_periodicity() -> None:
    """Test periodic hazard oscillation detection and space-time A* pathing."""
    history = [(2, 2), (2, 3), (2, 4), (2, 3), (2, 2), (2, 3), (2, 4), (2, 3)]
    timestamps = list(range(len(history)))

    traj = AutonomousTrajectoryModel.fit(
        "hazard_1", feature_id=4, history=history, timestamps=timestamps
    )
    assert traj.is_periodic is True
    assert traj.period == 4
    # Predict future position at t=8 (should cycle back to (2, 2))
    assert traj.predict_position(8) == (2, 2)
    assert traj.predict_position(9) == (2, 3)

    acq = SpatiotemporalSkillAcquisition()
    acq.trajectories["hazard_1"] = traj
    acq.hazard_features.add(4)

    # Safe crossing check: (2, 4) is occupied at t=2, t=6, t=10
    assert acq.is_cell_safe_at_time(2, 4, 2) is False
    assert acq.is_cell_safe_at_time(2, 4, 0) is True

    # Plan space-time path crossing an oscillating hazard zone
    # Start at (1, 3), Goal at (3, 3). Hazard oscillates across row 2 at columns 2-4.
    path = acq.plan_space_time_path(
        start=(1, 3),
        goal=(3, 3),
        current_t=0,
        grid_shape=(5, 5),
        static_barriers=set(),
    )
    assert path is not None
    assert path[0] == (1, 3)
    assert path[-1] == (3, 3)
    # Check that at step 1 or 2, the agent didn't collide with hazard
    for step_idx, pos in enumerate(path):
        t = step_idx
        assert acq.is_cell_safe_at_time(pos[0], pos[1], t) is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Algebraic Permutations & Galois Field (GF2) Solver
# ─────────────────────────────────────────────────────────────────────────────


def test_gf2_solver_and_lights_out() -> None:
    """Test Gaussian elimination over GF(2) and Lights-Out puzzle solving."""
    # Test simple 2x2 system:
    # A = [[1, 1], [0, 1]], b = [0, 1]
    # x1 + x2 = 0 => x1 = 1, x2 = 1
    A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    b = np.array([0, 1], dtype=np.uint8)
    x = GF2LinearSolver.solve(A, b)
    assert x is not None
    assert np.array_equal((A @ x) % 2, b)

    # Test full 3x3 Lights-Out solver
    acq = PermutationAlgebraSkillAcquisition()
    # Click at (1, 1) toggles cross [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    prev_g = np.zeros((3, 3), dtype=np.uint8)
    curr_g = np.zeros((3, 3), dtype=np.uint8)
    for r, c in [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]:
        curr_g[r, c] = 1

    acq.observe_toggle(click_pos=(1, 1), prev_grid=prev_g, curr_grid=curr_g)
    assert acq.is_toggle_puzzle is True

    # Now solve a board where only center and its neighbors are lit
    solution = acq.solve_lights_out(current_grid=curr_g)
    assert solution is not None
    # Clicking (1, 1) should immediately turn everything off
    assert (1, 1) in solution


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kinematics, Momentum & Ice Sliding
# ─────────────────────────────────────────────────────────────────────────────


def test_kinematics_sliding_ice() -> None:
    """Test frictionless sliding simulation and multi-step inertial path planning."""
    acq = KinematicMomentumSkillAcquisition()
    barriers = {(0, 2), (2, 4), (4, 2), (3, 0)}
    grid_shape = (5, 5)

    # Slide RIGHT from (2, 0) -> hits barrier at (2, 4), rests at (2, 3)
    dest, path = acq.simulate_slide(
        start_pos=(2, 0),
        direction=(0, 1),
        barriers=barriers,
        grid_shape=grid_shape,
    )
    assert dest == (2, 3)
    assert path == [(2, 0), (2, 1), (2, 2), (2, 3)]

    # Observe displacement to detect ice mechanics
    acq.observe_displacement(
        intended_delta=(0, 1),
        start_pos=(2, 0),
        end_pos=(2, 3),
        barriers=barriers,
        grid_shape=grid_shape,
    )
    assert acq.model.is_sliding_environment is True
    assert acq.model.friction == 0.0

    # Plan sliding path to goal at (2, 3) from (0, 0)
    # Move DOWN to (2, 0) (blocked by (3, 0)), then RIGHT to (2, 3) (blocked by (2, 4))
    slide_path = acq.plan_sliding_path(
        start=(0, 0),
        goal=(2, 3),
        barriers=barriers,
        grid_shape=grid_shape,
    )
    assert slide_path is not None
    assert len(slide_path) > 0
    # Final destination of the last slide must be the goal
    assert slide_path[-1][1] == (2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Relational Affordances & Sokoban Pushing
# ─────────────────────────────────────────────────────────────────────────────


def test_relational_affordance_and_sokoban_push() -> None:
    """Test Sokoban pushability and higher-order relational affordance rules."""
    barriers = {(1, 4), (2, 4)}
    grid_shape = (5, 5)

    # Box at (2, 2), pushing RIGHT ((0, 1)) -> destination cell is (2, 3) (free)
    assert (
        RelationalAffordanceSkillAcquisition.is_pushable(
            box_pos=(2, 2),
            push_dir=(0, 1),
            barriers=barriers,
            other_entity_positions=set(),
            grid_shape=grid_shape,
        )
        is True
    )

    # Pushing box at (2, 3) RIGHT -> destination (2, 4) is a barrier!
    assert (
        RelationalAffordanceSkillAcquisition.is_pushable(
            box_pos=(2, 3),
            push_dir=(0, 1),
            barriers=barriers,
            other_entity_positions=set(),
            grid_shape=grid_shape,
        )
        is False
    )

    # Plan push step: agent at (2, 0), box at (2, 2), target at (2, 4)
    step = RelationalAffordanceSkillAcquisition.plan_sokoban_push_step(
        agent_pos=(2, 0),
        box_pos=(2, 2),
        target_pos=(2, 4),
        barriers=barriers,
        grid_shape=grid_shape,
    )
    assert step is not None
    stand_pos, push_dir = step
    assert stand_pos == (2, 1)  # Agent must stand behind box
    assert push_dir == (0, 1)  # Push rightward

    # Test relational contact observation
    acq = RelationalAffordanceSkillAcquisition()
    e_box = SpatialEntity(
        "box_1", EntityRole.MANIPULABLE, (2.0, 2.0), (2, 2), 4, (1, 2, 1, 2), feature_id=3
    )
    e_socket = SpatialEntity(
        "socket_1", EntityRole.GOAL, (2.0, 4.0), (2, 4), 1, (2, 2, 4, 4), feature_id=5
    )

    acq.observe_contact(e_box, e_socket, relation="PUSH_INTO", barriers_cleared=3)
    rule_key = (e_box.get_signature_key(), e_socket.get_signature_key(), "PUSH_INTO")
    assert rule_key in acq.rules
    assert acq.rules[rule_key].outcome_effect == "REMOVES_BARRIER"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Morphological Program Synthesis & Visual Analogies
# ─────────────────────────────────────────────────────────────────────────────


def test_morphological_program_synthesis() -> None:
    """Test geometric analogy detection and macro paint command synthesis."""
    source = np.array(
        [
            [1, 2],
            [1, 1],
        ],
        dtype=np.uint8,
    )

    # Horizontal reflection + color substitution (1->7, 2->8)
    target = np.array(
        [
            [8, 7],
            [7, 7],
        ],
        dtype=np.uint8,
    )

    trans = MorphologicalProgramSynthesis.detect_transformation(source, target)
    assert trans is not None
    assert trans.op_type == "REFLECT_H"
    assert trans.color_map[1] == 7
    assert trans.color_map[2] == 8

    # Also test with identical colors (0 mutations)
    source_same = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=np.uint8,
    )
    target_same = np.array(
        [
            [2, 1],
            [4, 3],
        ],
        dtype=np.uint8,
    )
    trans_same = MorphologicalProgramSynthesis.detect_transformation(source_same, target_same)
    assert trans_same is not None
    assert trans_same.op_type == "REFLECT_H"

    # Synthesize paint commands onto canvas at origin (5, 5)
    cmds = MorphologicalProgramSynthesis.synthesize_paint_commands(
        source_patch=source,
        canvas_origin=(5, 5),
        transform=trans,
        ignore_background=0,
    )
    assert len(cmds) == 4
    # Check top-left of canvas (5, 5) should get color 8 (from mirrored cell (0, 0))
    assert (5, 5, 8) in cmds
    assert (5, 6, 7) in cmds


# ─────────────────────────────────────────────────────────────────────────────
# 6. Coupled Controllable Coordination (Mirrored Agents)
# ─────────────────────────────────────────────────────────────────────────────


def test_coupled_controllables_mirrored() -> None:
    """Test joint state convergence planning with obstacle-assisted desynchronization."""
    acq = CoupledControllableSkillAcquisition()

    # Agent 1 starts at (1, 1), Agent 2 starts at (1, 3)
    # Goal 1 is at (3, 1), Goal 2 is at (3, 3)
    # Let's verify direct convergence
    plan = acq.plan_joint_convergence(
        start1=(1, 1),
        start2=(1, 3),
        goal1=(3, 1),
        goal2=(3, 3),
        barriers=set(),
        grid_shape=(5, 5),
    )
    assert plan is not None
    assert len(plan) > 0
