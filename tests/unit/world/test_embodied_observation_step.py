from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.percept import EmbodiedObservation, TerminalSignal


def test_engine_step_with_embodied_observation() -> None:
    engine = AutonomousEpistemicEngine(enable_logging=False)
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1] = 1  # avatar
    grid[4, 4] = 2  # goal

    obs = EmbodiedObservation(
        lattice=grid,
        available_actions=[1, 2, 3, 4],
        terminal=None,
        context_id="test_env",
        episode_id="0",
        metadata={"bg": 0},
    )

    action, data = engine.step(obs)
    assert action in [1, 2, 3, 4]


def test_engine_step_with_terminal_signal() -> None:
    engine = AutonomousEpistemicEngine(enable_logging=False)
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1

    obs_win = EmbodiedObservation(
        lattice=grid,
        available_actions=[1, 2],
        terminal=TerminalSignal.SUCCESS,
        context_id="test_env",
        episode_id="1",
    )
    action_win, _ = engine.step(obs_win)
    assert action_win in [1, 2]
