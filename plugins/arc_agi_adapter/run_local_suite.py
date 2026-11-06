"""Run MyAgent against the deterministic local_env games (WP0.9 harness).

Usage: python kaggle_submission/run_local_suite.py --seed 0 \
         --trace-dir reports/local_suite/traces [--max-steps 200]
Writes the same scorecard shape as all_25_games_scorecard.json to
reports/local_suite/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_core_dir = Path(__file__).resolve().parents[2]
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))


def _policy_action(policy: str, step: int, available: list[int]) -> tuple[int, dict]:
    """Baseline policies for --policy mode (env-only, no engine)."""
    if policy == "round_robin":
        cands = [a for a in available if a != 0] or [7]
        return cands[step % len(cands)], {}
    return 7, {}


def run_game(
    game, game_id: str, agent, max_steps: int, recorder=None, policy: str | None = None
) -> dict:
    frame = game.reset()
    latencies: list[float] = []
    stagnant = 0
    noop = 0
    violations = 0
    prev_hash = None
    steps = 0
    stop_reason = "step_limit"
    for step in range(max_steps):
        steps = step + 1
        t0 = time.perf_counter()
        state = getattr(frame, "state", None)
        sname = getattr(state, "name", str(state))
        if sname in ("GAME_OVER", "WIN", "NOT_PLAYED"):
            action_id, data = 0, {}
            boundary = "death" if sname == "GAME_OVER" else ("win" if sname == "WIN" else "reset")
        elif agent is not None:
            act = agent.choose_action([frame], frame)
            if act is None:
                action_id, data = 7, {}
            else:
                action_id = int(getattr(act, "value", act))
                data = getattr(act, "data", {}) or {}
            boundary = None
        else:
            action_id, data = _policy_action(
                policy or "round_robin", step, list(getattr(frame, "available_actions", []))
            )
            boundary = None
        frame = game.step(action_id, data)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        sname2 = getattr(frame.state, "name", str(frame.state))
        h = hash(tuple(map(tuple, frame.grid)))
        if h == prev_hash and action_id not in (0, 7):
            stagnant += 1
        if action_id in (0, 7):
            noop += 1
        prev_hash = h
        if recorder is not None:
            recorder.record_step(
                game=game_id,
                level=getattr(frame, "levels_completed", 0),
                step=step,
                grid=frame.grid,
                available_actions=getattr(frame, "available_actions", []),
                action_id=action_id,
                action_data=data,
                levels_completed=frame.levels_completed,
                state=sname2,
                latency_ms=lat,
                boundary=boundary,
            )
        if sname2 == "WIN":
            stop_reason = "win"
            break
        if sname2 == "GAME_OVER" and stagnant > 20:
            stop_reason = "game_over_loop"
            break
    return {
        "game_id": game_id,
        "levels_completed": getattr(frame, "levels_completed", 0),
        "win_levels": game.win_levels,
        "state": getattr(frame.state, "name", str(frame.state)),
        "steps": steps,
        "stop_reason": stop_reason,
        "mean_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2)
        if len(latencies) > 1
        else (latencies[0] if latencies else 0.0),
        "stagnant_steps": stagnant,
        "noop_actions": noop,
        "contract_violations": getattr(agent, "_contract_violations", violations),
        "actions_per_level": round(steps / max(1, getattr(frame, "levels_completed", 0) + 1), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--per-game-time-limit", type=int, default=600)
    ap.add_argument("--trace-dir", type=str, default=None)
    ap.add_argument("--env", choices=["local"], default="local")
    ap.add_argument(
        "--policy",
        choices=["random", "round_robin"],
        default=None,
        help="Env-only baseline policy (no MyAgent). Omit to run the engine.",
    )
    args = ap.parse_args()

    from plugins.arc_agi_adapter.local_env import LocalEnv
    from plugins.arc_agi_adapter.trace import TraceRecorder

    env = LocalEnv(seed=args.seed)
    recorder = TraceRecorder(args.trace_dir) if args.trace_dir else None
    results = []
    for gid in env.game_ids:
        agent = None
        if args.policy is None:
            from plugins.arc_agi_adapter.my_agent import MyAgent

            agent = MyAgent(card_id="local_suite", game_id=gid, seed=args.seed)
        g = env.make(gid)
        t0 = time.perf_counter()
        res = run_game(g, gid, agent, args.max_steps, recorder, policy=args.policy)
        if (
            res["stop_reason"] == "step_limit"
            and (time.perf_counter() - t0) > args.per_game_time_limit
        ):
            res["stop_reason"] = "time_limit"
        results.append(res)
        print(
            f"{gid}: {res['state']} {res['levels_completed']}/{res['win_levels']} "
            f"stop={res['stop_reason']} mean={res['mean_latency_ms']}ms p95={res['p95_latency_ms']}ms"
        )
    if recorder:
        recorder.close()
    out = Path("reports/local_suite") / f"scorecard_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    scorecard = {
        "env": "local",
        "seed": args.seed,
        "policy": args.policy or "engine",
        "results": results,
        "levels_passed": sum(r["levels_completed"] for r in results),
        "levels_total": sum(r["win_levels"] for r in results),
    }
    out.write_text(json.dumps(scorecard, indent=2))
    print(f"\nscorecard -> {out}")
    print(f"levels: {scorecard['levels_passed']}/{scorecard['levels_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
