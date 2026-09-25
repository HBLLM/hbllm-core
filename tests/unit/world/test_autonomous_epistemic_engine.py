"""Unit tests for AutonomousEpistemicEngine — Motor Grounding, Curiosity, and Mental Simulation."""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EpistemicPhase,
)


def test_motor_grounding_avatar_self_identification() -> None:
    """Verify agent self-identifies avatar and motor dynamics from pixel diffs."""
    engine = AutonomousEpistemicEngine()

    # Step 0: Grid with background 0, an avatar entity (val 1) at (2, 2)
    grid_0 = np.zeros((10, 10), dtype=int)
    grid_0[2, 2] = 1  # Candidate avatar
    grid_0[8, 8] = 3  # Target goal

    available_actions = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT

    # First decision: untested actions probe
    act, data = engine.decide(grid_0, available_actions)
    assert act in available_actions

    # Step 1: Environment executes the chosen action (e.g. 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
    move_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    dr, dc = move_map[act]

    grid_1 = np.zeros((10, 10), dtype=int)
    grid_1[2 + dr, 2 + dc] = 1
    grid_1[8, 8] = 3

    # Engine decides next action and assimilates feedback
    act2, _ = engine.decide(grid_1, available_actions)

    # Avatar should now be identified
    assert engine.avatar_feature == 1
    assert engine.avatar_pos == (2 + dr, 2 + dc)
    assert act in engine.action_dynamics
    dyn = engine.action_dynamics[act]
    assert dyn.get_displacement() == (dr, dc)
    assert dyn.confidence >= 0.6


def test_forward_mental_simulation_and_exploitation() -> None:
    """Verify that once dynamics are grounded, the agent plans in imagination and executes."""
    engine = AutonomousEpistemicEngine()

    # Pre-ground motor model
    engine.avatar_feature = 1
    engine.avatar_pos = (2, 2)
    # Ground actions 1, 2, 3, 4
    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)
    engine.learned_goal_features.add(3)

    # Grid: Avatar at (2, 2), Goal at (2, 5) — 3 steps RIGHT (action 4)
    grid = np.zeros((8, 8), dtype=int)
    grid[2, 2] = 1
    grid[2, 5] = 3

    available_actions = [1, 2, 3, 4]

    # Mental simulation should find the 3-step path [4, 4, 4]
    simulated_steps = engine.simulate_in_mind(grid, available_actions)
    assert simulated_steps is not None
    assert len(simulated_steps) == 3
    assert [s.action for s in simulated_steps] == [4, 4, 4]

    # In decide(), it should switch to EXPLOITATION and return the first action
    chosen_act, _ = engine.decide(grid, available_actions)
    assert chosen_act == 4
    assert engine.phase == EpistemicPhase.EXPLOITATION


def test_causal_mutation_induction() -> None:
    """Verify that distant pixel mutations from actions induce StateMutationModel."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (5, 5)

    # Initial frame
    prev_grid = np.zeros((10, 10), dtype=int)
    prev_grid[5, 5] = 1
    prev_grid[1, 1] = 6  # Button/switch
    prev_grid[8, 8] = 2  # Locked door/barrier

    engine.prev_grid = prev_grid
    engine.last_action = 6
    engine.last_action_data = {"x": 1, "y": 1}

    # Next frame: Button clicked -> door at (8, 8) opened (changed to 0)
    curr_grid = prev_grid.copy()
    curr_grid[8, 8] = 0

    engine.assimilate_feedback(curr_grid, [6])

    assert len(engine.state_mutations) == 1
    mutation = engine.state_mutations[0]
    assert mutation.trigger_pos == (1, 1)
    assert mutation.trigger_feature == 6
    assert mutation.prior_value == 2
    assert mutation.posterior_value == 0
    assert mutation.confidence >= 0.8


def test_full_autonomous_loop_unseen_puzzle() -> None:
    """Verify end-to-end autonomous loop on an unseen maze puzzle:

    Self-identifies avatar -> explores unknown space -> mental simulation -> exploitation to WIN.
    """
    engine = AutonomousEpistemicEngine()

    # Environment Setup: 7x7 grid
    # 0 = floor, 9 = wall, 4 = avatar, 7 = goal
    H, W = 7, 7
    wall_map = np.zeros((H, W), dtype=bool)
    wall_map[0, :] = True
    wall_map[-1, :] = True
    wall_map[:, 0] = True
    wall_map[:, -1] = True
    # Partition barrier
    wall_map[3, 1:4] = True

    avatar_pos = [1, 1]
    goal_pos = (5, 5)
    avatar_color = 4
    goal_color = 7

    available_actions = [1, 2, 3, 4]  # 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT
    action_effects = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def render_grid() -> np.ndarray:
        g = np.zeros((H, W), dtype=int)
        g[wall_map] = 9
        g[goal_pos[0], goal_pos[1]] = goal_color
        g[avatar_pos[0], avatar_pos[1]] = avatar_color
        return g

    won = False
    for step in range(60):
        grid = render_grid()
        if tuple(avatar_pos) == goal_pos:
            won = True
            # Let engine assimilate the win state
            engine.decide(grid, available_actions, is_win=True)
            break

        act, data = engine.decide(grid, available_actions, is_win=False)
        dr, dc = action_effects[act]
        nr, nc = avatar_pos[0] + dr, avatar_pos[1] + dc

        # Check collision with wall
        if not wall_map[nr, nc]:
            avatar_pos[0] = nr
            avatar_pos[1] = nc

    assert won, "Agent failed to solve the unseen maze autonomously!"
    assert engine.avatar_feature == avatar_color
    assert len(engine.action_dynamics) >= 2
