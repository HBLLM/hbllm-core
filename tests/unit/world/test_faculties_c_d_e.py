"""Unit Tests for Biological Cortical Faculties C, D, and E.

Verifies:
- Faculty C: Dorsal Visual Stream (Area MT/V5) Kinetic Figure-Ground Segregation.
- Faculty D: Bilateral Convergent Coordinate Frames (Corpus Callosum & SMA).
- Faculty E: Basal Ganglia & Cerebellar Predictive Phase Entrainment.
"""

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel


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


def test_peripheral_hud_filtering() -> None:
    """Test that peripheral HUD border lines are not confused with avatars."""
    engine = AutonomousEpistemicEngine()
    grid = np.zeros((30, 30), dtype=int)

    # 1-pixel-wide HUD energy line along column 29
    for r in range(30):
        grid[r, 29] = 2

    # True avatar inside room at (15, 15)
    grid[14:17, 14:17] = 2

    engine._update_avatar_position_from_grid(grid, known_av_feats={2})
    # Should localize to interior avatar (15, 15), not the HUD border (c=29)
    assert engine.avatar_pos is not None
    assert engine.avatar_pos == (15, 15)
