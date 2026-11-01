from __future__ import annotations

from plugins.arc_agi_adapter.local_env import GameState, LocalEnv


def _av(grid):
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == 2:
                return (r, c)
    return None


def test_registry_has_eight_plus_games():
    env = LocalEnv(0)
    assert len(env.game_ids) >= 8
    for gid in env.game_ids:
        g = env.make(gid)
        f = g.reset()
        assert f.state == GameState.NOT_FINISHED
        assert f.win_levels >= 1


def test_nav_basic_movement():
    g = LocalEnv(0).make("nav_basic")
    g.reset()
    f = g.step(2, {})  # down
    assert _av(f.grid) == (2, 1)
    f = g.step(4, {})  # right
    assert _av(f.grid) == (2, 2)


def test_blocked_first_move():
    g = LocalEnv(0).make("blocked_first")
    g.reset()
    f = g.step(1, {})  # up into wall -> zero motion
    assert _av(f.grid) == (2, 2)
    f = g.step(3, {})  # left is also walled -> zero motion
    assert _av(f.grid) == (2, 2)
    f = g.step(4, {})  # right is free
    assert _av(f.grid) == (2, 3)


def test_stride2():
    g = LocalEnv(0).make("stride2")
    g.reset()
    f = g.step(4, {})
    assert _av(f.grid) == (1, 3)


def test_key_door():
    g = LocalEnv(0).make("key_door")
    g.reset()
    assert not g.has_key
    for _ in range(4):
        g.step(4, {})
    assert g.has_key


def test_push_block_and_death_reset():
    g = LocalEnv(0).make("push_block")
    g.reset()
    f = g.step(4, {})  # push block right
    assert _av(f.grid) == (2, 3)
    d = LocalEnv(0).make("death_reset")
    d.reset()
    f = d.step(2, {})
    assert f.state == GameState.GAME_OVER
    f = d.step(0, {})  # RESET
    assert f.state == GameState.NOT_FINISHED


def test_determinism_same_seed():
    a, b = LocalEnv(7).make("hazard"), LocalEnv(7).make("hazard")
    a.reset()
    b.reset()
    for _ in range(10):
        fa, fb = a.step(2, {}), b.step(2, {})
    assert fa.grid == fb.grid


def test_frame_layers_match_arc_shape():
    """ARC fidelity: frame[-1] must be the full 2D lattice the agent extracts."""
    import sys

    sys.path.insert(0, "kaggle_submission")
    import numpy as np
    from my_agent import MyAgent

    env = LocalEnv(0)
    g = env.make("nav_basic")
    f = g.reset()
    grid = MyAgent._extract_grid(None, f) if hasattr(MyAgent._extract_grid, "__get__") else None
    agent = MyAgent(card_id="x", game_id="nav_basic", seed=0)
    grid = agent._extract_grid(f)
    assert grid.shape == (64, 64)
    # rendered layer equals upscaled logical grid
    expected = np.kron(np.asarray(g.grid), np.ones((g.scale, g.scale), dtype=int))
    assert (np.asarray(grid) == expected).all()


def test_extract_grid_rejects_1d():
    import sys

    sys.path.insert(0, "kaggle_submission")
    import pytest
    from my_agent import MyAgent

    agent = MyAgent(card_id="x", game_id="nav_basic", seed=0)
    bad = type("F", (), {"frame": [1, 2, 3]})()  # flat ints -> 1D
    with pytest.raises((ValueError, TypeError)):
        agent._extract_grid(bad)
