"""Unit tests for ARC-AGI-3 Interactive Reasoning Benchmark Runner."""

from __future__ import annotations

import numpy as np

from plugins.developmental_adapter.arc_agi_3_runner import (
    ActionDynamicsModel,
    ARC3BenchmarkReport,
    ARC3EnvironmentResult,
    ARC3InteractiveAgent,
    ARC3LevelResult,
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
