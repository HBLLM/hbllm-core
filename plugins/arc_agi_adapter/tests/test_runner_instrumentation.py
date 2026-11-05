"""Instrumentation tests for run_all_25_games telemetry.

Drives the REAL MyAgent + CognitiveAgentAdapter + DriverManager path through
scripted fake envs (small grids, deterministic state transitions) and asserts
the runner records trustworthy telemetry: exact stop reasons, per-game trace
isolation, non-zero latencies, and matching trace-analysis counts.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from plugins.arc_agi_adapter.local_env import GameState  # noqa: E402


# ── Scripted fake ARC envs ────────────────────────────────────────────────────
class _FakeFrame:
    """FrameData-compatible minimal frame (recognized via ``state.name``)."""

    def __init__(
        self,
        grid,
        state,
        levels_completed=0,
        win_levels=1,
        available_actions=None,
    ):
        self.state = state
        self.frame = [grid]
        self.grid = grid
        self.levels_completed = levels_completed
        self.win_levels = win_levels
        self.available_actions = list(available_actions or [1, 2, 3, 4, 5, 6, 7])
        self.step_count = 0


def _grid(fill=0):
    return [[fill] * 8 for _ in range(8)]


class _ScriptedEnv:
    """Minimal ARC-like env: ``on_step(env, action_id, data) -> _FakeFrame``."""

    def __init__(self, on_step, win_levels=1):
        self._on_step = on_step
        self._win_levels = win_levels
        self.calls = 0
        self.grid = _grid()

    def reset(self):
        self.calls = 0
        return _FakeFrame(
            [row[:] for row in self.grid], GameState.NOT_FINISHED, 0, self._win_levels
        )

    def step(self, action_id, data=None):
        return self._on_step(self, action_id, data)


def _win_factory():
    """WIN on the 3rd non-reset step."""

    def on_step(env, aid, data):
        env.calls += 1
        g = _grid()
        g[3][3] = env.calls  # interior cell changes every step
        if env.calls >= 3:
            return _FakeFrame(g, GameState.WIN, levels_completed=1, win_levels=1)
        return _FakeFrame(g, GameState.NOT_FINISHED, 0, 1)

    return _ScriptedEnv(on_step)


def _dead_factory():
    """GAME_OVER on the 2nd step; the reset step (action 0) stays GAME_OVER."""

    def on_step(env, aid, data):
        if aid == 0:
            return _FakeFrame(env.grid, GameState.GAME_OVER, 0, 1)
        env.calls += 1
        if env.calls >= 2:
            return _FakeFrame(env.grid, GameState.GAME_OVER, 0, 1)
        return _FakeFrame(env.grid, GameState.NOT_FINISHED, 0, 1)

    return _ScriptedEnv(on_step)


def _budget_factory():
    """Grid never changes; runs until the step budget is exhausted."""

    def on_step(env, aid, data):
        env.calls += 1
        return _FakeFrame([row[:] for row in env.grid], GameState.NOT_FINISHED, 0, 1)

    return _ScriptedEnv(on_step)


def _boom_factory():
    """env.step always raises."""

    def on_step(env, aid, data):
        raise RuntimeError("boom")

    return _ScriptedEnv(on_step)


def _hud_factory():
    """Only the last (border) row changes each step — a step-counter bar.
    Interior never changes → every step is a no-op under HUD-aware detection."""

    def on_step(env, aid, data):
        env.calls += 1
        g = [row[:] for row in env.grid]
        g[7][env.calls % 8] = 9  # tick the border row like a HUD counter
        return _FakeFrame(g, GameState.NOT_FINISHED, 0, 1)

    return _ScriptedEnv(on_step)


def _interior_factory():
    """An interior cell changes each step — genuinely non-noop."""

    def on_step(env, aid, data):
        env.calls += 1
        g = [row[:] for row in env.grid]
        g[3][3] = env.calls
        return _FakeFrame(g, GameState.NOT_FINISHED, 0, 1)

    return _ScriptedEnv(on_step)


# ── Harness helpers ───────────────────────────────────────────────────────────
def _runner_module(monkeypatch, factories):
    """Import run_all_25_games with a stubbed ``arc_agi`` and fake Arcade."""
    if "arc_agi" not in sys.modules:
        try:
            import arc_agi  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("arc_agi")
            stub.Arcade = object  # type: ignore[attr-defined]
            sys.modules["arc_agi"] = stub
    import plugins.arc_agi_adapter.run_all_25_games as r25

    class _FakeArcade:
        def get_environments(self):
            return [
                SimpleNamespace(game_id=gid, title=gid, baseline_actions=[50], tags=["test"])
                for gid in factories
            ]

        def make(self, gid):
            return factories[gid]()

    monkeypatch.setattr(r25, "Arcade", _FakeArcade)
    return r25


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_runner_records_telemetry_per_game(tmp_path, monkeypatch):
    factories = {
        "boom01": _boom_factory,  # first: run must continue past the exception
        "win01": _win_factory,
        "dead01": _dead_factory,
        "budget01": _budget_factory,
    }
    r25 = _runner_module(monkeypatch, factories)
    scorecard_file = tmp_path / "scorecard.json"
    trace_dir = tmp_path / "traces"

    r25.run_all_25_games(
        scorecard_file=str(scorecard_file),
        trace_dir=str(trace_dir),
        max_steps_per_game=5,
        retries=0,
        resume=False,
        seed=0,
    )

    sc = json.loads(scorecard_file.read_text())
    games = sc["games"]
    assert set(games) == {"boom01", "win01", "dead01", "budget01"}

    # ── exact stop reasons / steps ──────────────────────────────────────
    assert games["boom01"]["stop_reason"] == "runner_exception"
    assert "boom" in (games["boom01"]["stop_detail"] or "")
    assert games["win01"]["stop_reason"] == "win"
    assert games["win01"]["won"] is True
    assert games["win01"]["levels_passed"] == 1
    assert games["win01"]["steps"] == 3
    assert games["dead01"]["stop_reason"] == "game_over"
    assert games["dead01"]["steps"] == 2
    assert games["budget01"]["stop_reason"] == "step_budget"
    assert games["budget01"]["steps"] == 5

    # ── telemetry metrics come from real measurement, not getattr stubs ──
    for gid in ("win01", "dead01", "budget01"):
        assert games[gid]["mean_latency_ms"] > 0, gid
        assert games[gid]["mean_decide_latency_ms"] > 0, gid
    assert games["budget01"]["noop_actions"] == 5
    assert games["budget01"]["max_stagnation_streak"] == 5
    assert games["win01"]["agent_yield_step"] == 3
    assert games["win01"]["agent_yield_reason"] == "win"
    assert "contract_violations" in games["win01"]
    assert "decision_exceptions" in games["win01"]

    # ── run-level latency summary exists ────────────────────────────────
    assert sc["latency_summary"]["mean_cycle_ms"] > 0

    # ── per-game trace isolation ────────────────────────────────────────
    for gid in factories:
        gdir = trace_dir / gid
        agent_trace = gdir / "trace.jsonl"
        runner_trace = gdir / "runner.jsonl"
        assert agent_trace.exists(), f"missing {agent_trace}"
        assert runner_trace.exists(), f"missing {runner_trace}"
        for rec in _read_jsonl(agent_trace):
            assert rec["game"] == gid, f"cross-game record in {agent_trace}: {rec}"
        for rec in _read_jsonl(runner_trace):
            assert rec["game"] == gid, f"cross-game record in {runner_trace}: {rec}"

    # runner.jsonl shape: step records during the loop, one stop per attempt
    win_steps = [
        r for r in _read_jsonl(trace_dir / "win01" / "runner.jsonl") if r["event"] == "step"
    ]
    win_stops = [
        r for r in _read_jsonl(trace_dir / "win01" / "runner.jsonl") if r["event"] == "stop"
    ]
    assert len(win_steps) == 3
    assert len(win_stops) == 1
    assert win_stops[0]["stop_reason"] == "win"
    assert win_stops[0]["stop_step"] == 3
    assert all(isinstance(r["cycle_ms"], float) and r["cycle_ms"] > 0 for r in win_steps)
    assert all(r["decide_ms"] is not None for r in win_steps)

    boom_steps = [
        r for r in _read_jsonl(trace_dir / "boom01" / "runner.jsonl") if r["event"] == "step"
    ]
    boom_stops = [
        r for r in _read_jsonl(trace_dir / "boom01" / "runner.jsonl") if r["event"] == "stop"
    ]
    assert len(boom_steps) == 0  # the step raised; only the stop is recorded
    assert boom_stops[0]["stop_reason"] == "runner_exception"
    assert boom_stops[0]["stop_step"] == 1

    # ── trace_analysis matches the scorecard ────────────────────────────
    from plugins.arc_agi_adapter.trace_analysis import analyze_trace_dir

    report = analyze_trace_dir(trace_dir)
    for gid in factories:
        g = report["games"][gid]
        assert g["steps"] == games[gid]["steps"], gid
        assert g["noop_actions"] == games[gid]["noop_actions"], gid
        assert g["stop_reasons"] == [games[gid]["stop_reason"]], gid
        assert g["contract_violations"] == games[gid]["contract_violations"], gid
        assert g["decision_exceptions"] == games[gid]["decision_exceptions"], gid
        assert g["levels_completed"] == games[gid]["levels_passed"], gid
    assert report["overall"]["steps"] == sum(g["steps"] for g in games.values())


def test_runner_hud_border_noop_detection(tmp_path, monkeypatch):
    """Steps that only tick a border HUD row are no-ops; interior changes are not."""
    factories = {"hud01": _hud_factory, "inter01": _interior_factory}
    r25 = _runner_module(monkeypatch, factories)
    scorecard_file = tmp_path / "scorecard.json"
    trace_dir = tmp_path / "traces"

    r25.run_all_25_games(
        scorecard_file=str(scorecard_file),
        trace_dir=str(trace_dir),
        max_steps_per_game=5,
        retries=0,
        resume=False,
        seed=0,
    )

    sc = json.loads(scorecard_file.read_text())
    games = sc["games"]

    # HUD game: border row ticks every step (raw change) but interior static.
    assert games["hud01"]["noop_actions"] == 5
    assert games["hud01"]["max_stagnation_streak"] == 5
    # Interior game: real change every step → no no-ops.
    assert games["inter01"]["noop_actions"] == 0
    assert games["inter01"]["max_stagnation_streak"] == 0

    hud_steps = [
        r for r in _read_jsonl(trace_dir / "hud01" / "runner.jsonl") if r["event"] == "step"
    ]
    assert len(hud_steps) == 5
    assert all(r["noop"] is True for r in hud_steps)
    assert all(r["effective_changed_cells"] == 0 for r in hud_steps)
    assert all(r["changed_cells"] > 0 for r in hud_steps)  # border ticked raw
    assert all(r["grid_changed"] is True for r in hud_steps)

    inter_steps = [
        r for r in _read_jsonl(trace_dir / "inter01" / "runner.jsonl") if r["event"] == "step"
    ]
    assert all(r["noop"] is False for r in inter_steps)
    assert all(r["effective_changed_cells"] > 0 for r in inter_steps)

    from plugins.arc_agi_adapter.trace_analysis import analyze_trace_dir

    report = analyze_trace_dir(trace_dir)
    assert report["games"]["hud01"]["noop_rate"] == 1.0
    assert report["games"]["hud01"]["raw_change_rate"] == 1.0
    assert report["games"]["hud01"]["max_stagnation_streak"] == 5
    assert report["games"]["inter01"]["noop_rate"] == 0.0
    assert report["games"]["inter01"]["raw_change_rate"] == 1.0


def test_runner_honors_agent_yield(tmp_path, monkeypatch):
    """Official-loop parity: the runner stops when the agent's own guard yields."""
    factories = {"yieldy": _budget_factory}
    r25 = _runner_module(monkeypatch, factories)
    from plugins.arc_agi_adapter.my_agent import MyAgent

    # Tiny action budget → agent yields via "action_budget" at step 2.
    monkeypatch.setattr(MyAgent, "ACTIONS_PER_LEVEL", 2)

    r25.run_all_25_games(
        scorecard_file=str(tmp_path / "sc1.json"),
        trace_dir=str(tmp_path / "t1"),
        max_steps_per_game=5,
        retries=0,
        resume=False,
        seed=0,
    )
    g1 = json.loads((tmp_path / "sc1.json").read_text())["games"]["yieldy"]
    assert g1["stop_reason"] == "agent_yield"
    assert g1["stop_detail"] == "action_budget"
    assert g1["steps"] == 2
    assert g1["agent_yield_step"] == 2
    assert g1["agent_yield_reason"] == "action_budget"

    # Record-only mode: same game runs to the step budget.
    r25.run_all_25_games(
        scorecard_file=str(tmp_path / "sc2.json"),
        trace_dir=str(tmp_path / "t2"),
        max_steps_per_game=5,
        retries=0,
        resume=False,
        seed=0,
        ignore_agent_yield=True,
    )
    g2 = json.loads((tmp_path / "sc2.json").read_text())["games"]["yieldy"]
    assert g2["stop_reason"] == "step_budget"
    assert g2["steps"] == 5
    assert g2["agent_yield_step"] == 2  # still recorded


def test_runner_time_limit_stop_reason(tmp_path, monkeypatch):
    factories = {"slow01": _budget_factory}
    r25 = _runner_module(monkeypatch, factories)
    scorecard_file = tmp_path / "scorecard.json"

    r25.run_all_25_games(
        scorecard_file=str(scorecard_file),
        trace_dir=str(tmp_path / "traces"),
        max_steps_per_game=50,
        retries=0,
        resume=False,
        seed=0,
        per_game_time_limit=0.0,
    )

    sc = json.loads(scorecard_file.read_text())
    assert sc["games"]["slow01"]["stop_reason"] == "time_limit"
    assert sc["games"]["slow01"]["steps"] == 1


def test_trace_recorder_never_overwrites(tmp_path):
    from plugins.arc_agi_adapter.trace import TraceRecorder

    r1 = TraceRecorder(tmp_path)
    assert r1.trace_file.name == "trace.jsonl"
    r1.record_step(
        game="g",
        level=0,
        step=0,
        grid=[[0]],
        available_actions=[1],
        action_id=1,
        action_data=None,
    )
    r1.close()
    assert (tmp_path / "trace.jsonl").exists()

    r2 = TraceRecorder(tmp_path)
    assert r2.trace_file.name == "trace-1.jsonl"
    r2.close()
    r3 = TraceRecorder(tmp_path)
    assert r3.trace_file.name == "trace-2.jsonl"
    r3.close()
    # original trace contents preserved
    recs = _read_jsonl(tmp_path / "trace.jsonl")
    assert len(recs) == 1 and recs[0]["game"] == "g"


def test_trace_from_env_kwarg_beats_env_var(tmp_path, monkeypatch):
    from plugins.arc_agi_adapter.trace import TraceRecorder

    env_dir = tmp_path / "envdir"
    kw_dir = tmp_path / "kwdir"
    monkeypatch.setenv("HBLLM_TRACE_DIR", str(env_dir))

    rec = TraceRecorder.from_env(str(kw_dir))
    assert rec is not None and rec.path == kw_dir
    rec.close()

    rec2 = TraceRecorder.from_env(None)  # env var fallback
    assert rec2 is not None and rec2.path == env_dir
    rec2.close()

    rec3 = TraceRecorder.from_env(None, subdir="gameX")
    assert rec3 is not None and rec3.path == env_dir / "gameX"
    rec3.close()

    monkeypatch.delenv("HBLLM_TRACE_DIR")
    assert TraceRecorder.from_env(None) is None
