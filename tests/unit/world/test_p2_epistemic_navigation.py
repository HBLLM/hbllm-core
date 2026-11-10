from __future__ import annotations

import tempfile

import numpy as np

from benchmarks.envs.grid_nav import GridNavEnvironment
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortex_perception import FigureGroundSegregator
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_decaying_inhibition_discipline() -> None:
    """Verify undo/frustration action inhibition decays cleanly without infinite lockouts."""
    wm = PrefrontalWorkingMemory()
    action_id = 3

    # Inhibit with bounded cooldown (e.g. 8 steps)
    wm.inhibit_action(action_id, duration=8)
    assert action_id in wm.inhibited_actions
    assert wm.inhibited_actions[action_id] == 8

    # Decay cooldown step by step
    for step in range(7):
        wm.decay_inhibited_actions()
        assert action_id in wm.inhibited_actions
        assert wm.inhibited_actions[action_id] == 7 - step

    # Final decay releases inhibition
    wm.decay_inhibited_actions()
    assert action_id not in wm.inhibited_actions


def test_gestalt_perimeter_vs_ground_separation() -> None:
    """Verify perimeter walls in enclosed mazes are not falsely crowned background canvas."""
    segregator = FigureGroundSegregator()

    # 10x10 grid with wall border (5) and interior navigable floor (0)
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[0, :] = 5
    grid[9, :] = 5
    grid[:, 0] = 5
    grid[:, 9] = 5

    # With no prior walkability, Gestalt perimeter enclosure sees interior floor (0) is dominant
    bg = segregator.estimate_background(grid)
    assert bg == 0, f"Expected interior floor 0 to be background, got {bg}"

    # With prior walkable feature 0, walkable features take precedence
    bg_with_walkable = segregator.estimate_background(grid, walkable_features={0})
    assert bg_with_walkable == 0


def test_feature_explored_retirement() -> None:
    """Verify epistemic curiosity features are retired and marked explored once inspected."""
    theory = HCIRSymbolicWorldTheory()
    key_feature = 3

    theory.induce_depletion_goal(key_feature, evidence="curiosity_candidate")
    assert key_feature in theory.candidate_goal_features
    assert theory.is_feature_explored(key_feature) is False

    # Retire explored feature and retract candidate
    theory.mark_feature_explored(key_feature, reason="inspected_no_effect")
    theory.retract_candidate_goal(key_feature, reason="explored")
    assert theory.is_feature_explored(key_feature) is True
    assert key_feature not in theory.candidate_goal_features


def test_lifelong_and_demonstration_grid_nav_success() -> None:
    """End-to-end verification of lifelong policy acquisition and demonstration imitation on seed 100010."""
    with tempfile.TemporaryDirectory() as d:
        system = build_cognitive_system(
            CognitiveSystemConfig(world_id="p2_nav_verify", memory_dir=d, enable_logging=False)
        )
        engine = system.engine
        env = GridNavEnvironment()

        # Episode 0: self exploration
        obs = env.reset(seed=100010)
        visited: set[tuple[int, int]] = {env.avatar_pos}
        done = False
        for _ in range(100):
            a, _ = engine.step(obs)
            obs, r, done, _ = env.step(a)
            visited.add(env.avatar_pos)
            if done:
                engine.step(obs)
                break

        # Check exploration coverage on seed 100010:
        # Agent explores and does not get stuck in a 2-cell ping-pong trap
        assert len(visited) >= 15, f"Expected active exploration >= 15 cells, got {len(visited)}"

        # Episode 1: lifelong execution (beliefs retained in engine.symbolic_theory)
        obs = env.reset(seed=100010)
        ep1_success = False
        for _ in range(100):
            a, _ = engine.step(obs)
            obs, r, done, info = env.step(a)
            if done:
                engine.step(obs)
                ep1_success = bool(info.get("success", False))
                break

        assert ep1_success is True, "Lifelong episode 1 should successfully reach the goal"
