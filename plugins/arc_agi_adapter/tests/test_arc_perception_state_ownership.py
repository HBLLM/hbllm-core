"""Regression: extract_arc_objects must not mutate engine-owned state.

Perception lifting runs inside choose_action BEFORE engine.step(obs). A plugin
write to engine.avatar_pos made the assimilator's prev-position tracking see
the post-move position, so every move looked stuck (failed_transitions at
destinations, inhibited actions). Perception must be read-only on the engine.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from plugins.arc_agi_adapter.arc_perception import extract_arc_objects


def _engine_snapshot(engine) -> dict:
    def _set(name):
        return set(getattr(engine, name, set()) or set())

    prev_grid = getattr(engine, "prev_grid", None)
    return {
        "avatar_pos": getattr(engine, "avatar_pos", None),
        "avatar_feature": getattr(engine, "avatar_feature", None),
        "avatar_features": _set("avatar_features"),
        "learned_barriers": _set("learned_barriers"),
        "failed_transitions": _set("failed_transitions"),
        "mobile_threat_features": _set("mobile_threat_features"),
        "prev_grid": None if prev_grid is None else prev_grid.copy(),
    }


def test_extract_arc_objects_does_not_mutate_engine_state() -> None:
    engine = AutonomousEpistemicEngine()
    # Avatar identity calibrated at a DIFFERENT position than the sprite in the
    # grid — this is exactly the branch where the engine used to be overwritten.
    engine.avatar_pos = (20, 20)
    engine.avatar_feature = 5
    engine.avatar_features = {5}

    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4, 4] = 5  # avatar sprite far from engine.avatar_pos
    grid[2, 2] = 3

    before = _engine_snapshot(engine)
    objs = extract_arc_objects(grid, engine=engine, game_obj=None)
    after = _engine_snapshot(engine)

    assert objs is not None
    for attr, prev in before.items():
        if attr == "prev_grid":
            assert (after[attr] is None) == (prev is None) or np.array_equal(after[attr], prev), (
                f"engine.{attr} mutated by extract_arc_objects"
            )
        else:
            assert after[attr] == prev, (
                f"engine.{attr} mutated by extract_arc_objects: {prev!r} -> {after[attr]!r}"
            )
