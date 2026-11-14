"""Unit tests for the faculty ablation harness on procedural environments."""

from __future__ import annotations

from benchmarks.ablation.faculty_ablation import (
    CountingProxy,
    NullFaculty,
    _mk_engine,
    _run_env_lifelong,
    faculty_inventory,
)
from benchmarks.envs.continuous_reach import ContinuousReachEnvironment


def test_faculty_inventory_returns_world_faculties():
    engine = _mk_engine(0)
    inv = faculty_inventory(engine)
    assert isinstance(inv, dict)
    assert len(inv) > 50
    # Core owners must not be present
    assert "planner" not in inv
    assert "deliberation_engine" not in inv
    assert "motor" not in inv
    # World faculties must be present
    assert "hazard_tracker" in inv or "habenular_ior" in inv


def test_null_faculty_swallows_calls_and_returns_neutral():
    class DummyFaculty:
        def is_active(self) -> bool:
            return True

        def get_count(self) -> int:
            return 42

        def get_items(self) -> list[str]:
            return ["a", "b"]

    null_fac = NullFaculty(DummyFaculty)
    assert null_fac.is_active() is False
    assert null_fac.get_count() == 0
    assert null_fac.get_items() == []
    assert bool(null_fac) is False
    assert len(null_fac) == 0


def test_counting_proxy_increments_call_counts():
    class DummyFaculty:
        def compute(self, x: int) -> int:
            return x * 2

    counters: dict[str, int] = {}
    proxy = CountingProxy(DummyFaculty(), counters)
    res = proxy.compute(5)
    assert res == 10
    assert counters.get("compute()", 0) == 1
    assert counters.get("__total__", 0) >= 1


def test_run_env_lifelong_returns_metrics():
    engine = _mk_engine(0)
    res = _run_env_lifelong(
        env_cls=ContinuousReachEnvironment,
        seed=42,
        max_budget=20,
        engine=engine,
        episodes=2,
    )
    assert "score" in res
    assert "solve_rate" in res
    assert "mean_steps" in res
    assert len(res["successes"]) == 2
    assert len(res["steps"]) == 2
