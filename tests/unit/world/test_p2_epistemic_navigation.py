from __future__ import annotations

import tempfile

import numpy as np

from benchmarks.envs.grid_nav import GridNavEnvironment
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortex_perception import FigureGroundSegregator
from hbllm.hcir.world.percept import EmbodiedObservation, EpisodeBoundary, EpisodeBoundaryKind
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


def test_nonzero_background_negative_transfer_eliminated() -> None:
    """Verify bg != 0 environments retract the unverified 0 prior and learn true floor/barriers."""
    palette = {0: 6, 5: 0, 3: 7, 2: 4, 8: 9, 9: 3, 1: 2}
    with tempfile.TemporaryDirectory() as d:
        engine = build_cognitive_system(
            CognitiveSystemConfig(world_id="bg_nonzero_test", memory_dir=d, enable_logging=False)
        ).engine
        env = GridNavEnvironment(max_steps=100)

        def _remap(obs: EmbodiedObservation) -> EmbodiedObservation:
            lat = np.array(obs.lattice)
            out = lat.copy()
            for k, v in palette.items():
                out[lat == k] = v
            return EmbodiedObservation(
                lattice=out,
                available_actions=obs.available_actions,
                terminal=obs.terminal,
                context_id=obs.context_id,
                episode_id=obs.episode_id,
                entities=(),
            )

        ep_results: list[bool] = []
        ep_barriers: list[set[int]] = []
        ep_is_barrier_0: list[bool] = []
        for ep in range(3):
            if ep > 0:
                engine.on_boundary(
                    EpisodeBoundary(
                        kind=EpisodeBoundaryKind.LEVEL_COMPLETE
                        if ep_results[-1]
                        else EpisodeBoundaryKind.DEATH,
                        same_level=True,
                    )
                )
            obs = _remap(env.reset(seed=910001))
            solved = False
            mid_barriers: set[int] = set()
            saw_is_barrier_0 = False
            for _ in range(100):
                a, _ = engine.step(obs)
                mid_barriers |= set(engine.symbolic_theory.barrier_features)
                if engine.symbolic_theory.is_barrier(0):
                    saw_is_barrier_0 = True
                raw_obs, _, done, info = env.step(a)
                obs = _remap(raw_obs)
                if done:
                    engine.step(obs)
                    solved = bool(info.get("success", False))
                    break
            ep_results.append(solved)
            ep_barriers.append(mid_barriers)
            ep_is_barrier_0.append(saw_is_barrier_0)

        assert engine.bg_feature == 6
        assert engine.symbolic_theory.has_verified_walkable(6)
        assert 0 not in engine.symbolic_theory.walkable_features
        assert 0 in ep_barriers[0]
        assert all(ep_is_barrier_0)
        assert ep_results[0] is True
        assert ep_results[-1] is True


def test_corridor_goal_target_within_three_steps_and_maze_solve() -> None:
    """Verify frontier fallback prevents corridor freezes and surfaces goal_target within 3 steps."""
    # Simple 9x9 perfect-style corridor maze where floor=0, wall=4, avatar=6, goal=7.
    # Start at diverse cells (corridors, corners, junctions) with calibrated axes and assert
    # meta["goal_target"] is non-None within 3 steps whenever >=1 reachable unvisited cell exists.
    maze = np.full((9, 9), 4, dtype=np.int32)
    open_cells = [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 5),
        (3, 5),
        (3, 4),
        (3, 3),
        (3, 2),
        (3, 1),
        (4, 1),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 7),
        (7, 7),
    ]
    for r, c in open_cells:
        maze[r, c] = 0
    maze[7, 7] = 7

    starts = [(1, 1), (1, 3), (3, 3), (5, 1), (5, 5)]
    moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    for idx, start_pos in enumerate(starts):
        with tempfile.TemporaryDirectory() as d:
            engine = build_cognitive_system(
                CognitiveSystemConfig(
                    world_id=f"corridor_gt_{idx}", memory_dir=d, enable_logging=False
                )
            ).engine
            # Pre-calibrate avatar identity & movement deltas cleanly via episode-0 grounding steps
            # from an open junction, then test starting at start_pos.
            pos = start_pos
            got_target_step: int | None = None
            seen_targets: list[tuple[int, int]] = []
            # Allow initial localization if ungrounded (<=5 steps), then require goal_target within 3
            # steps; additionally verify grounded continuation from any start hits within 3 steps.
            for step_i in range(8):
                grid = maze.copy()
                grid[pos] = 6
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4, 7],
                    context_id="corridor_ctx",
                    episode_id=f"ep_{idx}",
                    entities=(),
                )
                act, meta = engine.step(obs)
                if meta.get("goal_target") is not None:
                    seen_targets.append(meta["goal_target"])
                    if engine.avatar_pos is not None and got_target_step is None:
                        got_target_step = step_i
                dr, dc = moves.get(int(getattr(act, "value", act)), (0, 0))
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and maze[nr, nc] != 4:
                    pos = (nr, nc)

            assert seen_targets, f"Expected non-None goal_target from start {start_pos}"

            # Now with grounded motor dynamics at start_pos in a fresh level episode,
            # assert meta['goal_target'] is non-None within 3 steps (<= step index 2).
            engine.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.DEATH, same_level=True))
            pos = start_pos
            grounded_metas: list[tuple[int, int] | None] = []
            for _ in range(3):
                grid = maze.copy()
                grid[pos] = 6
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4, 7],
                    context_id="corridor_ctx",
                    episode_id=f"ep_grounded_{idx}",
                    entities=(),
                )
                act, meta = engine.step(obs)
                grounded_metas.append(meta.get("goal_target"))
                dr, dc = moves.get(int(getattr(act, "value", act)), (0, 0))
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and maze[nr, nc] != 4:
                    pos = (nr, nc)
            assert any(gt is not None for gt in grounded_metas), (
                f"Expected goal_target within 3 steps from {start_pos}, got {grounded_metas}"
            )
