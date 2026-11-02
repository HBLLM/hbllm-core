"""Regression: synthesize_plan caches _can_navigate_to per destination.

The wall set (learned_barriers ∪ barrier-feature cells) is static for the
whole synthesize_plan call, and GeodesicManifoldRouter.route_geodesic must run
at most once per unique destination. The produced plan must be identical to
the pre-change implementation (loaded from git HEAD).
"""

from __future__ import annotations

import re
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hbllm.hcir.world.engine_state import (
    EngineState,
    FrozenSymbolicTheory,
    FrozenWorkingMemory,
)
from hbllm.hcir.world.macro_actions import (
    MacroActionDriverManager,
    MacroPlanSynthesizer,
    MicroActionSpec,
    ObjectSpec,
)
from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

_REPO = Path(__file__).resolve().parents[3]

_GOALS = [(8, 8), (2, 2), (8, 2)]
_SWITCHES = [(1, 5), (5, 8), (9, 5)]


def _ring(pos: tuple[int, int]) -> set[tuple[int, int]]:
    """All in-bounds 8-neighbours of a cell — fully encloses it for 4-dir moves."""
    r, c = pos
    return {
        (rr, cc)
        for rr in range(r - 1, r + 2)
        for cc in range(c - 1, c + 2)
        if (rr, cc) != pos and 0 <= rr < 10 and 0 <= cc < 10
    }


def _scenario(unenclosed: set[tuple[int, int]] | None = None) -> tuple[Any, np.ndarray, list[int]]:
    """Frozen CognitiveContext: avatar mid-grid, 3 unreachable goals and 3
    switches (optionally some left reachable), plus a barrier-feature cell to
    exercise the vectorized wall build."""
    curr_grid = np.zeros((10, 10), dtype=np.int32)
    curr_grid[5, 5] = 1  # avatar
    curr_grid[0, 0] = 7  # barrier feature cell
    for g in _GOALS:
        curr_grid[g] = 2

    barriers: set[tuple[int, int]] = set()
    for p in _GOALS:
        barriers |= _ring(p)
    for p in _SWITCHES:
        if unenclosed and p in unenclosed:
            continue
        barriers |= _ring(p)

    state = EngineState(
        avatar_pos=(5, 5),
        avatar_feature=1,
        avatar_features={1},
        learned_goal_positions=set(_GOALS),
        learned_goal_features={2},
        learned_barriers=barriers,
        action_dynamics={
            1: MicroActionSpec(action_id=1, delta_r=-1),
            2: MicroActionSpec(action_id=2, delta_r=1),
            3: MicroActionSpec(action_id=3, delta_c=-1),
            4: MicroActionSpec(action_id=4, delta_c=1),
        },
    )
    ctx = state.freeze()
    ctx = replace(
        ctx,
        symbolic_theory=FrozenSymbolicTheory(barrier_features=frozenset({7})),
        # Plain dicts — matches the real engine's mutable working-memory stores
        # (the frozen proxy defaults would raise on trial-count bookkeeping).
        working_memory=FrozenWorkingMemory(
            known_objects={  # type: ignore[arg-type]
                f"sw_{r}_{c}": ObjectSpec(id=f"sw_{r}_{c}", pos=(r, c), feature_id=4, role="switch")
                for r, c in _SWITCHES
            },
            object_trial_counts={},  # type: ignore[arg-type]
        ),
    )
    return ctx, curr_grid, [1, 2, 3, 4, 5]


def _plan_fingerprint(plan: Any) -> list[tuple[str, dict]] | None:
    if plan is None:
        return None

    def _norm(v: Any) -> str:
        # Collapse object reprs (memory address + differing module path for the
        # git-HEAD reference module) down to just the class name.
        return re.sub(r"<[\w.]+\.(\w+) object at 0x[0-9a-fA-F]+>", r"<\1>", repr(v))

    return [(type(a).__name__, {k: _norm(v) for k, v in vars(a).items()}) for a in plan.actions]


def _load_head_macro_actions() -> types.ModuleType | None:
    """Load the pre-change macro_actions module from git HEAD for reference."""
    src = subprocess.run(
        ["git", "show", "HEAD:hbllm/hcir/world/macro_actions.py"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    if src.returncode != 0:
        return None
    mod = types.ModuleType("hbllm_macro_actions_head")
    mod.__dict__["__file__"] = str(_REPO / "hbllm" / "hcir" / "world" / "macro_actions.py")
    sys.modules[mod.__name__] = mod
    exec(
        compile(src.stdout, str(_REPO / "hbllm/hcir/world/macro_actions.py"), "exec"),
        mod.__dict__,
    )
    return mod


@pytest.mark.parametrize(
    "unenclosed,expected_dests",
    [
        # All targets walled off: goals checked twice (second pass hits cache),
        # all switches checked once, plan falls through to exploration.
        (None, set(_GOALS) | set(_SWITCHES)),
        # One open switch first in sort order: next() short-circuits after it.
        ({(5, 8)}, set(_GOALS) | {(5, 8)}),
    ],
)
def test_synthesize_plan_memoizes_router_calls(monkeypatch, unenclosed, expected_dests) -> None:
    old_mod = _load_head_macro_actions()
    if old_mod is None:
        pytest.skip("git show HEAD unavailable for old-code plan reference")

    calls: list[tuple[int, int]] = []
    orig = GeodesicManifoldRouter.route_geodesic

    def spy(*args, **kwargs):
        goals = kwargs.get("goals", args[2] if len(args) > 2 else None)
        if goals:
            calls.append(tuple(goals[0]))
        return orig(*args, **kwargs)

    monkeypatch.setattr(GeodesicManifoldRouter, "route_geodesic", staticmethod(spy))
    # Isolate from any registered domain synthesis hooks (plugin-dependent).
    for manager in (MacroActionDriverManager, old_mod.MacroActionDriverManager):
        monkeypatch.setattr(
            manager,
            "run_synthesis_hooks",
            classmethod(lambda cls, *a, **k: None),
        )

    # Fresh scenario per run: object_trial_counts is mutated inside
    # synthesize_plan — identical inputs require identical fresh state.
    ctx, curr_grid, available_actions = _scenario(unenclosed)
    new_plan = MacroPlanSynthesizer.synthesize_plan(ctx, curr_grid, available_actions)
    new_calls = list(calls)

    assert new_plan is not None and not new_plan.is_empty()
    # At most one router call per unique destination.
    assert len(new_calls) == len(set(new_calls))
    assert set(new_calls) == expected_dests

    calls.clear()
    ctx2, curr_grid2, available_actions2 = _scenario(unenclosed)
    old_plan = old_mod.MacroPlanSynthesizer.synthesize_plan(ctx2, curr_grid2, available_actions2)
    old_calls = list(calls)

    # Same plan, strictly fewer router calls (repeat dests now hit the cache).
    assert _plan_fingerprint(new_plan) == _plan_fingerprint(old_plan)
    assert len(old_calls) > len(new_calls)
