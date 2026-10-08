"""Unit Tests for Biological Cortical Faculties C, D, and E.

Verifies:
- Faculty C: Dorsal Visual Stream (Area MT/V5) Kinetic Figure-Ground Segregation.
- Faculty D: Bilateral Convergent Coordinate Frames (Corpus Callosum & SMA).
- Faculty E: Basal Ganglia & Cerebellar Predictive Phase Entrainment.
"""

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.bilateral_coordination import (
    BilateralCoordinateIntegrator,
    SymmetryAxis,
)
from hbllm.hcir.world.cerebellar_phase_clock import CerebellarPhaseClock
from hbllm.hcir.world.kinetic_stream import DorsalKineticStream
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel


def test_faculty_c_kinetic_figure_ground_segregation() -> None:
    """Test MT/V5 kinetic segregation on a camouflaged avatar and external threat."""
    stream = DorsalKineticStream()

    # Frame 1: 10x10 grid. Background = 0.
    # Camouflaged avatar at (5, 5) with feature 3.
    # Independent threat at (2, 2) with feature 8.
    prev_grid = np.zeros((10, 10), dtype=int)
    prev_grid[5, 5] = 3
    prev_grid[2, 2] = 8

    # Frame 2: Motor action moves UP: dr = -1, dc = 0
    # Avatar moves to (4, 5)
    # External threat wanders RIGHT: moves to (2, 3)
    curr_grid = np.zeros((10, 10), dtype=int)
    curr_grid[4, 5] = 3
    curr_grid[2, 3] = 8

    # Corollary discharge matches (-1, 0)
    res = stream.segregate_motion(
        prev_grid=prev_grid,
        curr_grid=curr_grid,
        commanded_delta=(-1, 0),
        background_feature=0,
    )

    assert res.self_avatar is not None
    assert res.self_avatar.is_self is True
    assert (4, 5) in res.self_avatar.cells
    assert res.self_avatar.velocity == (-1.0, 0.0)

    # Threat segregated as external agent
    assert len(res.external_agents) == 1
    threat = res.external_agents[0]
    assert threat.is_self is False
    assert (2, 3) in threat.cells
    assert threat.velocity == (0.0, 1.0)


def test_faculty_d_bilateral_coordination_and_wall_slip() -> None:
    """Test Bilateral Coordinate Integrator in planning convergence with wall slip."""
    integrator = BilateralCoordinateIntegrator()

    # 1. Detect bilateral pairing across vertical reflection axis
    # Agent 1 at (4, 2) moves Right -> delta=(0, 1)
    # Agent 2 at (4, 8) moves Left -> delta=(0, -1)
    moved_entities = [
        ((4, 2), (0, 1), 5),
        ((4, 8), (0, -1), 5),
    ]
    state = integrator.detect_bilateral_pairing(moved_entities, (10, 10))
    assert state is not None
    assert state.symmetry_axis == SymmetryAxis.VERTICAL

    # 2. Plan convergence sequence with wall slip
    # Grid: 1x7 row.
    # Agent 1 at (0, 1), Agent 2 at (0, 5)
    # Passable everywhere except (0, 0) is a wall.
    passable = np.ones((1, 7), dtype=bool)
    passable[0, 0] = False  # Wall on the left

    action_deltas = {
        3: (0, -1),  # LEFT for agent 1 (which means RIGHT for agent 2)
        4: (0, 1),  # RIGHT for agent 1 (which means LEFT for agent 2)
    }

    plan = integrator.plan_convergence_sequence(
        pos1=(0, 1),
        pos2=(0, 5),
        axis=SymmetryAxis.VERTICAL,
        passable_mask=passable,
        action_deltas=action_deltas,
    )

    assert plan is not None
    assert len(plan) > 0

    # Simulate plan execution
    p1, p2 = (0, 1), (0, 5)
    for act in plan:
        d1 = action_deltas[act]
        d2 = integrator.transform_action_for_agent2(d1, SymmetryAxis.VERTICAL)
        np1 = (p1[0] + d1[0], p1[1] + d1[1])
        np2 = (p2[0] + d2[0], p2[1] + d2[1])
        p1 = np1 if (0 <= np1[1] < 7 and passable[np1]) else p1
        p2 = np2 if (0 <= np2[1] < 7 and passable[np2]) else p2

    # Verify convergence achieved
    assert abs(p1[1] - p2[1]) <= 1


def test_faculty_e_cerebellar_phase_entrainment() -> None:
    """Test macro clock detection and basal ganglia motor gating."""
    clock = CerebellarPhaseClock()

    # Simulate periodic environmental resets every 64 steps (as in BP35)
    dummy_grid = np.zeros((8, 8), dtype=int)
    for step in [64, 128, 192]:
        clock.record_step(step, dummy_grid, is_reset=True)

    assert clock.detected_macro_period == 64

    # Test phase calculation
    assert clock.get_current_phase(60) == 60
    assert clock.get_epoch_runway(60) == 4  # 4 steps left before reset!

    # Test motor phase gating on oscillating bottleneck
    # Bottleneck period T = 3: cycle = [9, 0, 9] (9=lethal laser, 0=open)
    # Travel time = 2 steps.
    # Immediate arrival at step 10 -> (10 + 2 - 1) % 3 = 11 % 3 = 2 -> value 9 (lethal!)
    decision = clock.evaluate_phase_gate(
        current_step=10,
        travel_steps_to_hazard=2,
        hazard_period=3,
        hazard_cycle_values=[9, 0, 9],
        safe_values={0},
    )

    assert decision.should_wait is True
    # Waiting 2 steps -> arrival at 10 + 2 + 2 = 14 -> (14 - 1) % 3 = 13 % 3 = 1 -> value 0 (safe!)
    assert decision.wait_steps_recommended == 2


def test_deliberate_to_habitual_search_backoff() -> None:
    """Test Daw & Dayan deliberate-to-habitual search backoff when paths are blocked."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 4
    engine.avatar_features = {4}
    engine.avatar_pos = (1, 1)

    # Calibrate directional motor dynamics
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 7x7 grid with a goal at (5, 5) completely enclosed by wall barriers (feature 9)
    grid = np.zeros((7, 7), dtype=int)
    grid[1, 1] = 4  # avatar
    grid[5, 5] = 7  # goal
    grid[4, 4:7] = 9
    grid[6, 4:7] = 9
    grid[5, 4] = 9
    grid[5, 6] = 9
    for r in range(4, 7):
        for c in range(4, 7):
            if (r, c) != (5, 5):
                engine.learned_barriers.add((r, c))
    engine.learned_goal_positions.add((5, 5))

    available_actions = [1, 2, 3, 4]

    # Step 1: Initial state, simulation_cooldown == 0. Forward simulation runs, fails to find path,
    # sets consecutive_simulation_failures = 1 and simulation_cooldown = 2.
    act1, _ = engine.decide(grid, available_actions)
    assert engine.consecutive_simulation_failures == 1
    assert engine.simulation_cooldown == 2

    # Step 2: During cooldown, deliberate simulation is skipped (habitual exploration engaged).
    dr, dc = engine.action_dynamics[act1].get_displacement()
    grid[1, 1] = 0
    grid[1 + int(dr), 1 + int(dc)] = 4
    act2, _ = engine.decide(grid, available_actions)
    assert engine.simulation_cooldown == 1

    # Step 3: simulation_cooldown decrements to 0.
    grid[1 + int(dr), 1 + int(dc)] = 0
    dr2, dc2 = engine.action_dynamics[act2].get_displacement()
    nr, nc = 1 + int(dr) + int(dr2), 1 + int(dc) + int(dc2)
    grid[nr, nc] = 4
    engine.decide(grid, available_actions)
    assert engine.simulation_cooldown == 0

    # Step 4: Salience Awakening:
    # Set simulation_cooldown back to 4. Discovering a new barrier or goal immediately
    # resets cooldown to 0 to trigger fresh forward planning!
    engine.simulation_cooldown = 4
    engine.learned_goal_positions.add((2, 5))
    engine.decide(grid, available_actions)
    # The new goal was detected, resetting cooldown and attempting simulation
    assert engine.simulation_cooldown <= 2  # Salience triggered, simulation attempted and updated
