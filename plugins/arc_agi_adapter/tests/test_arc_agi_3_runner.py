"""Unit tests for ARC-AGI-3 Interactive Reasoning Benchmark Runner."""

from __future__ import annotations

import numpy as np

from plugins.arc_agi_adapter.arc_agi_3_runner import (
    ActionDynamicsModel,
    ARC3BenchmarkReport,
    ARC3EnvironmentResult,
    ARC3InteractiveAgent,
    ARC3LevelResult,
    CornerDeadlockDetector,
    TopologicalPathPlanner,
)


def test_action_dynamics_model() -> None:
    """Verify ActionDynamicsModel displacement vector formatting."""
    model = ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=0.95)
    assert model.delta_r == -1
    assert model.delta_c == 0
    desc = model.describe()
    assert "Action(1)" in desc
    assert "Δr=-1" in desc


def test_arc3_agent_causal_probing_and_learning() -> None:
    """Verify agent infers avatar color and motor displacement from visual frame diff."""
    agent = ARC3InteractiveAgent()

    # Grid 0: Background 0, obstacle 5, avatar (color 3) at (4, 4)
    g0 = np.zeros((8, 8), dtype=int)
    g0[7, :] = 5  # Floor
    g0[4, 4] = 3  # Avatar dot

    # Grid 1: ACTION1 moves avatar UP to (3, 4)
    g1 = np.zeros((8, 8), dtype=int)
    g1[7, :] = 5
    g1[3, 4] = 3

    agent.update_causal_dynamics(action_id=1, prev_grid=g0, curr_grid=g1)

    assert agent.avatar_color == 3
    assert 1 in agent.action_models
    m1 = agent.action_models[1]
    assert m1.delta_r == -1
    assert m1.delta_c == 0
    assert m1.confidence >= 0.8


def test_arc3_agent_goal_directed_planning() -> None:
    """Verify agent selects the action moving avatar closer to the goal object."""
    agent = ARC3InteractiveAgent()
    agent.avatar_color = 3
    agent.avatar_centroid = (4.0, 4.0)

    # Calibrate action models:
    # Action 1: UP (-1, 0)
    # Action 2: DOWN (+1, 0)
    # Action 3: LEFT (0, -1)
    # Action 4: RIGHT (0, +1)
    agent.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=0.95
    )
    agent.action_models[2] = ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=0.95)
    agent.action_models[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=0.95
    )
    agent.action_models[4] = ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=0.95)

    # Grid with avatar (3) at (4, 4) and goal (color 2) at (1, 4) [ABOVE avatar]
    grid = np.zeros((8, 8), dtype=int)
    grid[4, 4] = 3
    grid[1, 4] = 2  # Goal above

    act, conf = agent.plan_next_action(grid, available_actions=[1, 2, 3, 4])
    # Should select Action 1 (UP)
    assert act == 1
    assert conf >= 0.85


def test_arc3_benchmark_report_markdown() -> None:
    """Verify ARC3BenchmarkReport Markdown rendering and summary statistics."""
    lvl = ARC3LevelResult(
        level_index=0,
        completed=True,
        actions_taken=20,
        baseline_actions=22,
        efficiency_ratio=1.10,
        brier_uncertainty=0.01,
        time_seconds=1.5,
        discovered_dynamics={1: "-1,0", 2: "+1,0"},
    )
    env_res = ARC3EnvironmentResult(
        game_id="ls20",
        total_levels=1,
        levels_completed=1,
        win_rate=1.0,
        total_actions=20,
        total_baseline_actions=22,
        mean_efficiency=1.10,
        mean_brier=0.01,
        duration_seconds=1.5,
        level_results=[lvl],
    )
    report = ARC3BenchmarkReport(
        benchmark_title="ARC-AGI-3 Challenge",
        total_environments=1,
        environments_completed=1,
        total_levels=1,
        levels_completed=1,
        overall_completion_rate=1.0,
        mean_action_efficiency=1.10,
        mean_brier_score=0.01,
        total_actions_taken=20,
        total_time_seconds=1.5,
        environment_results=[env_res],
    )

    md = report.format_markdown()
    assert "# Official ARC-AGI-3 Interactive Reasoning Benchmark Report" in md
    assert "`ls20`" in md
    assert "100.0%" in md
    assert "110.0%" in md


def test_topological_bfs_shortest_path() -> None:
    """Verify BFS navigates around a U-shaped barrier instead of getting stuck."""
    barrier_mask = np.zeros((8, 8), dtype=bool)
    # U-shaped barrier between start (4, 4) and goal (2, 4)
    # Wall on row 3 columns 3, 4, 5 and row 4 columns 3 and 5
    barrier_mask[3, 3:6] = True
    barrier_mask[4, 3] = True
    barrier_mask[4, 5] = True

    path = TopologicalPathPlanner.find_shortest_path(
        start=(4, 4),
        goal=(2, 4),
        grid_shape=(8, 8),
        barrier_mask=barrier_mask,
        step_size=1,
    )
    assert len(path) > 1
    # First step must not walk straight UP into barrier (3, 4)
    assert path[1] != (3, 4)
    # Must exit downward or sideways avoiding barrier
    assert not barrier_mask[path[1][0], path[1][1]]
    assert path[-1] == (2, 4)


def test_sokoban_deadlock_detection() -> None:
    """Verify corner deadlock detector flags unmovable box positions against walls."""
    barrier_mask = np.zeros((8, 8), dtype=bool)
    # Top wall and left wall
    barrier_mask[0, :] = True
    barrier_mask[:, 0] = True

    # Position (1, 1) is adjacent to wall (0, 1) and wall (1, 0) -> corner deadlock
    is_deadlock = CornerDeadlockDetector.is_corner_deadlock(
        box_pos=(1, 1),
        barrier_mask=barrier_mask,
        target_positions=set(),
        grid_shape=(8, 8),
        step_size=1,
    )
    assert is_deadlock is True

    # If (1, 1) is a designated target, it's not a deadlock
    not_deadlock = CornerDeadlockDetector.is_corner_deadlock(
        box_pos=(1, 1),
        barrier_mask=barrier_mask,
        target_positions={(1, 1)},
        grid_shape=(8, 8),
        step_size=1,
    )
    assert not_deadlock is False


def test_state_mutation_induction() -> None:
    """Verify agent induces discrete state mutation when avatar color changes on tile."""
    agent = ARC3InteractiveAgent()
    agent.avatar_color = 3
    agent.avatar_centroid = (4.0, 4.0)

    # Grid 0: Avatar is color 3 at (4, 4)
    g0 = np.zeros((8, 8), dtype=int)
    g0[4, 4] = 3

    # Grid 1: Stepping on transformer tile at (4, 4) mutates avatar color to 7
    g1 = np.zeros((8, 8), dtype=int)
    g1[4, 4] = 7

    agent.update_causal_dynamics(action_id=1, prev_grid=g0, curr_grid=g1)

    assert len(agent.state_mutations) == 1
    mutation = agent.state_mutations[0]
    assert mutation.mutation_type == "COLOR_REMAP"
    assert mutation.prior_value == 3
    assert mutation.posterior_value == 7
    assert agent.avatar_color == 7


def test_arc3_agent_lift_to_hcir() -> None:
    """Verify ARC3InteractiveAgent lifts visual grids into native HCIR graphs."""
    from hbllm.hcir.graph import HCIRNodeType

    agent = ARC3InteractiveAgent()
    agent.avatar_color = 3
    agent.avatar_centroid = (4.0, 4.0)
    agent.pushable_colors.add(8)
    agent.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=0.95
    )

    grid = np.zeros((8, 8), dtype=int)
    grid[4, 4] = 3
    grid[3, 4] = 8  # Pushable box above avatar
    grid[1, 4] = 2  # Goal above

    ws, goal_node, candidate_actions = agent.lift_to_hcir(grid, chosen_goal=(1, 4))

    assert goal_node.id == "goal_arc3"
    assert goal_node.properties["target_position"] == (1, 4)

    entities = ws.graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)
    entity_ids = [e.id for e in entities]
    assert "avatar" in entity_ids
    assert "goal_primary" in entity_ids
    assert "box_3_4" in entity_ids

    assert len(candidate_actions) == 1
    assert candidate_actions[0].properties["action_id"] == 1
    assert candidate_actions[0].properties["delta_r"] == -1


import pytest


@pytest.mark.asyncio
async def test_arc3_agent_counterfactual_planning() -> None:
    """Verify ARC3InteractiveAgent plans actions using core CounterfactualPlanner."""
    agent = ARC3InteractiveAgent()
    agent.avatar_color = 3
    agent.avatar_centroid = (4.0, 4.0)
    agent.goal_centroid = (1.0, 4.0)

    agent.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=0.95
    )
    agent.action_models[2] = ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=0.95)
    agent.action_models[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=0.95
    )
    agent.action_models[4] = ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=0.95)

    grid = np.zeros((8, 8), dtype=int)
    grid[4, 4] = 3
    grid[1, 4] = 2  # Goal above

    best_act, score = await agent.plan_next_action_counterfactual(
        grid, available_actions=[1, 2, 3, 4]
    )
    assert best_act == 1
    assert score > 0.0


def test_arc3_regression_gate_wa30_and_ls20() -> None:
    """Automated Regression Gate: Enforce wa30 == 44 actions and ls20 == 31 actions.

    Any architectural modification or adapter extension that causes action drift
    or failure on these verified Category 1 environments MUST fail the test suite.

    Note on Baseline Updates:
    Hard assertions (== 44, == 31) guard against unintentional code contamination
    (e.g., cross-environment logic leaks). If the underlying upstream arc_agi package
    itself is updated and shifts engine/environment dynamics, this baseline must be
    re-verified and updated deliberately with recorded justification, rather than
    silently masking unintentional regressions.
    """
    from arc_agi import Arcade

    from plugins.arc_agi_adapter.arc_agi_3_runner import ARC3BenchmarkRunner

    arcade_client = Arcade()
    runner = ARC3BenchmarkRunner(max_steps_per_level=100)

    res_wa30 = runner.run_environment(arcade_client, "wa30", max_levels=1)
    assert res_wa30.levels_completed == 1, f"wa30 failed: completed {res_wa30.levels_completed}/1"
    assert res_wa30.level_results[0].actions_taken == 44, (
        f"wa30 action count drifted from 44 to {res_wa30.level_results[0].actions_taken}"
    )

    res_ls20 = runner.run_environment(arcade_client, "ls20", max_levels=1)
    assert res_ls20.levels_completed == 1, f"ls20 failed: completed {res_ls20.levels_completed}/1"
    assert res_ls20.level_results[0].actions_taken == 31, (
        f"ls20 action count drifted from 31 to {res_ls20.level_results[0].actions_taken}"
    )
