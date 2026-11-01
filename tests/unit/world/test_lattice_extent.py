from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine


def test_perceive_uses_real_extent_64x64():
    eng = AutonomousEpistemicEngine(enable_logging=False)
    frame = eng.perceive(np.zeros((64, 64), dtype=int))
    assert frame.shape == (64, 64)


def test_perceive_uses_real_extent_10x7():
    eng = AutonomousEpistemicEngine(enable_logging=False)
    frame = eng.perceive(np.zeros((10, 7), dtype=int))
    assert frame.shape == (10, 7)


def test_perceive_raises_on_unknown_extent(monkeypatch):
    from hbllm.hcir.world.percept import PerceptionRegistry

    eng = AutonomousEpistemicEngine(enable_logging=False)
    monkeypatch.setattr(eng, "perception_adapter", None)
    monkeypatch.setattr(PerceptionRegistry, "get_default", classmethod(lambda cls: None))
    with pytest.raises(ValueError, match="lattice extent unknown"):
        eng.perceive(object())


def test_plan_returns_none_without_lattice():
    from hbllm.hcir.world.object_state_graph import ObjectStateGraphPlanner

    osg = ObjectStateGraphPlanner()
    engine = type("E", (), {"perceive": None, "avatar_pos": None})()
    plan = osg.plan_macro_option(engine=engine, available_actions=[1, 2, 3, 4], curr_grid=None)
    assert plan is None
