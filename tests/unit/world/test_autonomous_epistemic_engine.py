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


def test_compound_mental_simulation_sokoban_push() -> None:
    """Verify compound mental simulation forward-simulates pushing a block into a receptacle."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (1, 1)

    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    engine.learned_cargo_features.add(2)
    engine.learned_goal_features.add(3)

    # 1D corridor: [1 (avatar), 2 (block), 0 (empty), 3 (goal)]
    grid = np.zeros((3, 6), dtype=int)
    grid[1, 1] = 1  # avatar
    grid[1, 2] = 2  # cargo block
    grid[1, 4] = 3  # goal target

    available_actions = [1, 2, 3, 4]

    simulated_plan = engine.simulate_in_mind(grid, available_actions)
    assert simulated_plan is not None
    assert len(simulated_plan) == 2
    # Pushing block twice to the right: (1, 2)->(1, 3), then (1, 3)->(1, 4)
    assert [s.action for s in simulated_plan] == [4, 4]


def test_hierarchical_subgoal_decomposition_multi_stage_switches() -> None:
    """Verify recursive subgoal decomposition navigates multiple switches to unlock successive doors."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (1, 1)

    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel, StateMutationModel

    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)
    engine.learned_goal_features.add(7)

    # Grid: 3 rows, 11 cols
    # Row 1: [0, avatar(1), 0, switch1(5), door1(2), 0, switch2(6), door2(8), 0, goal(7), 0]
    # Walls on row 0 and 2
    grid = np.zeros((3, 11), dtype=int)
    grid[0, :] = 9
    grid[2, :] = 9
    grid[1, 1] = 1  # avatar
    grid[1, 3] = 5  # switch 1
    grid[1, 4] = 2  # door 1 (barrier)
    grid[1, 6] = 6  # switch 2
    grid[1, 7] = 8  # door 2 (barrier)
    grid[1, 9] = 7  # goal

    engine.learned_barrier_features.add(9)
    engine.learned_barrier_features.add(2)
    engine.learned_barrier_features.add(8)

    # Learned mutation rules:
    # Switch 1 (feat 5) removes door 1 (val 2 -> 0)
    engine.state_mutations.append(
        StateMutationModel(
            trigger_type="CONTACT",
            trigger_pos=(1, 3),
            trigger_feature=5,
            prior_value=2,
            posterior_value=0,
            confidence=1.0,
        )
    )
    # Switch 2 (feat 6) removes door 2 (val 8 -> 0)
    engine.state_mutations.append(
        StateMutationModel(
            trigger_type="CONTACT",
            trigger_pos=(1, 6),
            trigger_feature=6,
            prior_value=8,
            posterior_value=0,
            confidence=1.0,
        )
    )

    available_actions = [1, 2, 3, 4]
    plan = engine.simulate_in_mind(grid, available_actions)
    assert plan is not None
    # Path:
    # (1, 1) -> (1, 3): 2 steps RIGHT (action 4, 4)
    # (1, 3) -> (1, 6): 3 steps RIGHT (action 4, 4, 4) through door 1
    # (1, 6) -> (1, 9): 3 steps RIGHT (action 4, 4, 4) through door 2
    # Total = 8 steps RIGHT
    assert len(plan) == 8
    assert all(s.action == 4 for s in plan)


def test_inductive_hcir_agent_disable_archetypes_wiring() -> None:
    """Verify InductiveHCIRAgent with disable_archetypes=True routes via AutonomousEpistemicEngine."""
    from plugins.arc_agi_adapter.inductive_learner import InductiveHCIRAgent

    agent = InductiveHCIRAgent(disable_archetypes=True)
    assert hasattr(agent, "autonomous_engine")
    assert agent.autonomous_engine is not None

    # Test decision routing
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1
    available_actions = [1, 2, 3, 4]

    act, conf = agent.plan_next_action(grid, available_actions)
    assert act in available_actions
    assert conf > 0.9
    assert agent.prev_grid is not None


def test_curiosity_oscillation_loop_breaking() -> None:
    """Verify that curiosity probe target commitment and recency penalties break ping-pong loops."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (2, 2)

    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # Grid with two unknown objects at (2, 2) and (2, 5)
    grid = np.zeros((5, 8), dtype=int)
    grid[2, 2] = 1  # avatar
    grid[2, 5] = 6  # unknown object 1
    grid[4, 2] = 7  # unknown object 2

    available_actions = [1, 2, 3, 4]

    visited_positions: list[tuple[int, int]] = []
    curr_pos = [2, 2]

    # Run 15 exploration steps simulating environment transitions
    for _ in range(15):
        engine.avatar_pos = (curr_pos[0], curr_pos[1])
        act, _ = engine.plan_epistemic_probe(grid, available_actions)
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[act]
        nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
        if 0 <= nr < 5 and 0 <= nc < 8:
            curr_pos[0], curr_pos[1] = nr, nc
        visited_positions.append((curr_pos[0], curr_pos[1]))

    # Verify that the agent did not just pace between 2 cells forever
    unique_positions = set(visited_positions)
    assert len(unique_positions) >= 4, (
        f"Agent was trapped in a narrow loop! Visited: {unique_positions}"
    )
    # Verify both entities were probed and registered
    assert len(engine.probed_entity_ids) >= 1


def test_pushable_cargo_discovery_and_deduplication() -> None:
    """Verify pushable cargo is identified and stationary cargo entities do not generate false movement."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (1, 1)

    grid_prev = np.zeros((6, 6), dtype=int)
    grid_prev[1, 1] = 1  # avatar
    grid_prev[1, 2] = 4  # cargo block A
    grid_prev[3, 3] = 4  # cargo block B (stationary)

    grid_curr = np.zeros((6, 6), dtype=int)
    grid_curr[1, 2] = 1  # avatar stepped into (1, 2)
    grid_curr[1, 3] = 4  # cargo block A pushed to (1, 3)
    grid_curr[3, 3] = 4  # cargo block B remained at (3, 3)

    engine.prev_grid = grid_prev
    engine.last_action = 4
    engine.last_action_data = None
    engine.assimilate_feedback(grid_curr, [1, 2, 3, 4])

    assert 4 in engine.learned_cargo_features
    # Re-observe another push to verify idempotent learning without errors
    grid_next = grid_curr.copy()
    grid_next[1, 2] = 0
    grid_next[1, 3] = 1  # avatar
    grid_next[1, 4] = 4  # cargo pushed again
    engine.prev_grid = grid_curr
    engine.last_action = 4
    engine.last_action_data = None
    engine.assimilate_feedback(grid_next, [1, 2, 3, 4])
    assert 4 in engine.learned_cargo_features


def test_block_stride_avatar_grounding() -> None:
    """Verify AutonomousEpistemicEngine grounds avatar and motor dynamics on multi-cell strides (e.g. wa30)."""
    engine = AutonomousEpistemicEngine()

    # 32x32 grid with a 4x4 avatar at (16, 16)
    grid_0 = np.zeros((32, 32), dtype=int)
    grid_0[16:20, 16:20] = 5

    # Action 1 (UP) translates avatar 4 units upward to (12, 16)
    grid_1 = np.zeros((32, 32), dtype=int)
    grid_1[12:16, 16:20] = 5

    engine.prev_grid = grid_0
    engine.last_action = 1
    engine.last_action_data = None
    engine.assimilate_feedback(grid_1, [1, 2, 3, 4])

    assert engine.avatar_feature == 5
    assert 1 in engine.action_dynamics
    assert engine.action_dynamics[1].get_displacement() == (-4, 0)
    assert engine.action_dynamics[1].confidence >= 0.6

    # Action 1 (UP) again translates avatar to (8, 16), confirming dynamics
    grid_2 = np.zeros((32, 32), dtype=int)
    grid_2[8:12, 16:20] = 5
    engine.prev_grid = grid_1
    engine.last_action = 1
    engine.last_action_data = None
    engine.assimilate_feedback(grid_2, [1, 2, 3, 4])

    assert engine.is_motor_grounded()
    assert engine.action_dynamics[1].confidence >= 0.7


def test_compound_avatar_grounding() -> None:
    """Verify that multi-color compound avatars (e.g. ls20) are grounded into avatar_features without false cargo."""
    engine = AutonomousEpistemicEngine()

    # 32x32 grid with a compound avatar (color 12 on rows 15-16, color 9 on rows 17-19, cols 10-14)
    grid_0 = np.zeros((32, 32), dtype=int)
    grid_0[15:17, 10:15] = 12
    grid_0[17:20, 10:15] = 9

    # Action 1 (UP) translates both parts by (-5, 0)
    grid_1 = np.zeros((32, 32), dtype=int)
    grid_1[10:12, 10:15] = 12
    grid_1[12:15, 10:15] = 9

    engine.prev_grid = grid_0
    engine.last_action = 1
    engine.last_action_data = None
    engine.assimilate_feedback(grid_1, [1, 2, 3, 4])

    assert engine.avatar_features == {9, 12}
    assert engine.avatar_size == 25
    assert engine.learned_cargo_features == set()
    assert engine.action_dynamics[1].get_displacement() == (-5, 0)


def test_hud_status_bar_filtering() -> None:
    """Verify that mutations in status bar / HUD regions are filtered out from environmental toggle models."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (20, 20)

    # 64x64 grid with maze walls (4), solid divider at row 52, and HUD countdown at rows 61-62
    grid_0 = np.full((64, 64), 3, dtype=int)
    grid_0[52, :] = 4
    grid_0[61, 10:20] = 11
    grid_0[20, 20] = 1

    # Step: Avatar moves, and HUD timer changes (11 -> 3)
    grid_1 = grid_0.copy()
    grid_1[20, 20] = 3
    grid_1[19, 20] = 1
    grid_1[61, 10:12] = 3

    engine.prev_grid = grid_0
    engine.last_action = 1
    engine.last_action_data = None
    engine.assimilate_feedback(grid_1, [1, 2, 3, 4])

    assert len(engine.state_mutations) == 0
