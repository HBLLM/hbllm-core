"""Oracle and protocol validation tests for the R9 widened dev benchmark environments."""

from __future__ import annotations

import pytest

from benchmarks.envs.widened_dev_envs import (
    ContinuousMultiWaypointEnv,
    LanguageGoalSelectEnv,
    LongHorizonQuestEnv,
    MultiAgentCorridorEnv,
    PartialObsMazeEnv,
    StochasticWindNavEnv,
)


@pytest.mark.parametrize(
    "env_cls,seed_base,min_oracle_sr",
    [
        (PartialObsMazeEnv, 1000, 1.0),
        (StochasticWindNavEnv, 1020, 0.8),
        (MultiAgentCorridorEnv, 1040, 0.9),
        (LongHorizonQuestEnv, 1060, 1.0),
        (ContinuousMultiWaypointEnv, 1080, 1.0),
        (LanguageGoalSelectEnv, 1100, 1.0),
    ],
)
def test_widened_dev_oracle_solves_all_regimes(
    env_cls: type, seed_base: int, min_oracle_sr: float
) -> None:
    """Every widened dev environment oracle must reliably solve its regime without cheating observations."""
    solved = 0
    for offset in range(10):
        env = env_cls()
        obs = env.reset(seed=seed_base + offset)
        assert obs.lattice is not None
        assert obs.entities == ()
        done = False
        info: dict = {}
        while not done:
            act = env.get_oracle_action()
            _obs, _rew, done, info = env.step(act)
        if info.get("success"):
            solved += 1
    assert solved / 10.0 >= min_oracle_sr
