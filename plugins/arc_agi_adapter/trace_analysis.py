"""Offline analysis of per-game runner.jsonl traces produced by run_all_25_games.

Reads every ``<trace_dir>/<game>/runner.jsonl`` and reports per-game and overall
telemetry: step counts, cycle/decide latency percentiles, no-op rate, max
stagnation streak, contract violations, decision exceptions, action histogram,
stop reasons, and agent yield points.

Usage::

    python -m plugins.arc_agi_adapter.trace_analysis <trace_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return round(xs[0], 2)
    return round(statistics.quantiles(xs, n=20)[18], 2)


def _percentile_stats(xs: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(statistics.fmean(xs), 2) if xs else None,
        "p50": round(statistics.median(xs), 2) if xs else None,
        "p95": _p95(xs),
        "max": round(max(xs), 2) if xs else None,
    }


def _analyze_records(step_recs: list[dict], stop_recs: list[dict]) -> dict[str, Any]:
    """Summarize one game's runner.jsonl records."""
    cycle = [float(r["cycle_ms"]) for r in step_recs if r.get("cycle_ms") is not None]
    decide = [float(r["decide_ms"]) for r in step_recs if r.get("decide_ms") is not None]

    noop_flags = [
        r.get("grid_changed") is False and r.get("level_before") == r.get("level_after")
        for r in step_recs
    ]
    max_streak = 0
    streak = 0
    for flag in noop_flags:
        streak = streak + 1 if flag else 0
        max_streak = max(max_streak, streak)

    action_hist: Counter[str] = Counter()
    for r in step_recs:
        if r.get("action_id") is not None:
            action_hist[str(r["action_id"])] += 1

    last_stop = stop_recs[-1] if stop_recs else {}
    yield_step = next(
        (r.get("agent_yield_step") for r in stop_recs if r.get("agent_yield_step") is not None),
        None,
    )
    yield_reason = next(
        (r.get("agent_yield_reason") for r in stop_recs if r.get("agent_yield_reason") is not None),
        None,
    )

    return {
        "steps": len(step_recs),
        "attempts": len(stop_recs),
        "levels_completed": last_stop.get("levels_completed"),
        "target_levels": last_stop.get("target_levels"),
        "cycle_ms": _percentile_stats(cycle),
        "decide_ms": {
            "mean": round(statistics.fmean(decide), 2) if decide else None,
            "p95": _p95(decide),
        },
        "noop_rate": round(sum(noop_flags) / len(noop_flags), 4) if noop_flags else None,
        "noop_actions": sum(noop_flags),
        "max_stagnation_streak": max_streak,
        "contract_violations": sum(1 for r in step_recs if r.get("violation")),
        "decision_exceptions": sum(1 for r in step_recs if r.get("exception")),
        "action_histogram": dict(sorted(action_hist.items())),
        "stop_reasons": [r.get("stop_reason") for r in stop_recs],
        "agent_yield_step": yield_step,
        "agent_yield_reason": yield_reason,
    }


def analyze_trace_dir(path: str | Path) -> dict[str, Any]:
    """Analyze all ``<game>/runner.jsonl`` files under ``path``.

    Args:
        path: Root trace directory written by ``run_all_25_games`` (one
            subdirectory per game) or a single game directory containing
            ``runner.jsonl``.

    Returns:
        ``{"games": {game: summary}, "overall": summary}``.
    """
    root = Path(path)
    games: dict[str, dict[str, Any]] = {}
    all_steps: list[dict] = []
    all_stops: list[dict] = []

    candidates = [root] if (root / "runner.jsonl").exists() else []
    candidates += sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []

    for game_dir in candidates:
        rj = game_dir / "runner.jsonl"
        if not rj.exists():
            continue
        step_recs: list[dict] = []
        stop_recs: list[dict] = []
        with open(rj) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("event") == "step":
                    step_recs.append(rec)
                elif rec.get("event") == "stop":
                    stop_recs.append(rec)
        games[game_dir.name] = _analyze_records(step_recs, stop_recs)
        all_steps.extend(step_recs)
        all_stops.extend(stop_recs)

    overall = _analyze_records(all_steps, all_stops) if all_steps or all_stops else {}
    return {"games": games, "overall": overall}


def _fmt(v: Any, suffix: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}{suffix}"
    return f"{v}{suffix}"


def _print_table(report: dict[str, Any]) -> None:
    header = (
        f"{'game':<18} {'steps':>5} {'lvls':>6} {'stop':<18} "
        f"{'cyc_mean':>9} {'cyc_p95':>9} {'dec_mean':>9} {'dec_p95':>9} "
        f"{'noop%':>7} {'streak':>6} {'viol':>4} {'exc':>4} {'yield':>16}"
    )
    print(header)
    print("-" * len(header))

    def _row(name: str, g: dict[str, Any]) -> None:
        lvls = (
            f"{g.get('levels_completed')}/{g.get('target_levels')}"
            if g.get("levels_completed") is not None
            else "-"
        )
        stops = ",".join(str(s) for s in g.get("stop_reasons", [])) or "-"
        y = g.get("agent_yield_step")
        yr = g.get("agent_yield_reason")
        ytxt = f"{yr}@{y}" if y is not None else "-"
        print(
            f"{name:<18} {g.get('steps', 0):>5} {lvls:>6} {stops[:18]:<18} "
            f"{_fmt(g.get('cycle_ms', {}).get('mean')):>9} "
            f"{_fmt(g.get('cycle_ms', {}).get('p95')):>9} "
            f"{_fmt(g.get('decide_ms', {}).get('mean')):>9} "
            f"{_fmt(g.get('decide_ms', {}).get('p95')):>9} "
            f"{_fmt((g.get('noop_rate') or 0) * 100):>7} "
            f"{_fmt(g.get('max_stagnation_streak')):>6} "
            f"{_fmt(g.get('contract_violations')):>4} "
            f"{_fmt(g.get('decision_exceptions')):>4} "
            f"{ytxt[:16]:>16}"
        )

    for name, g in sorted(report.get("games", {}).items()):
        _row(name, g)
    overall = report.get("overall") or {}
    if overall:
        print("-" * len(header))
        _row("OVERALL", overall)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="trace_analysis")
    ap.add_argument("trace_dir", help="Trace dir containing <game>/runner.jsonl files")
    ap.add_argument("--json", dest="json_out", default=None, help="Write full report to JSON")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    report = analyze_trace_dir(args.trace_dir)
    _print_table(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nreport -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
