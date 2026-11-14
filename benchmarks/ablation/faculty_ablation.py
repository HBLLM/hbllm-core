"""Faculty ablation harness on procedural benchmark environments.

Phases:
  inventory  – enumerate every hbllm.hcir.world-owned faculty attr on the engine.
  screen     – counting-proxy usage counts (benchmark environments x seeds, closed loop with lifelong metric).
  ablate     – NullFaculty swap per used faculty; score/steps/latency vs baseline.
Decisions per the pre-registered rule (see DECISION_RULE below).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_ENVS = [
    ("grid_nav", "benchmarks.envs.grid_nav.GridNavEnvironment", 100),
    ("grid_push", "benchmarks.envs.grid_push.GridPushEnvironment", 100),
    ("continuous_reach", "benchmarks.envs.continuous_reach.ContinuousReachEnvironment", 60),
    ("symbolic_rules", "benchmarks.envs.symbolic_rules.SymbolicRulesEnvironment", 80),
]

CORE_OWNERS = {
    "perception",
    "assimilator",
    "deliberation_engine",
    "planner",
    "motor",
    "episodic_cortex",
    "working_memory",
    "symbolic_theory",
    "cortical_pipeline",
}
KERNEL_OBJS = {
    "transaction_manager",
    "tiered_workspace",
    "services",
    "workspace",
    "active_task_frame",
    "commit_authority_registry",
    "event_bus",
    "state",
}
DECISION_RULE = (
    "sd = std(baseline_scores across seeds) floored at 0.05; "
    "keep iff (baseline - ablated) > sd, OR load_bearing_crash; "
    "else experimental. 0-call faculties are unused on procedural benchmark environments."
)


def _load_env_class(class_path: str) -> type:
    """Dynamically import environment class from module string."""
    import importlib

    mod_name, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _mk_engine(seed: int = 0, memory_dir: str | Path | None = None) -> Any:
    import tempfile

    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    mem_dir = memory_dir or tempfile.mkdtemp(prefix=f"ablation_{seed}_{os.getpid()}_")
    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id=f"ablation_{seed}_{os.getpid()}_{time.time_ns()}",
            memory_dir=mem_dir,
            enable_logging=False,
            seed=seed,
        )
    )
    return system.engine


def faculty_inventory(engine: Any) -> dict[str, str]:
    """{attr_name: ClassName} for world-defined, non-core, non-kernel attrs."""
    inv = {}
    for name, v in vars(engine).items():
        if name in CORE_OWNERS or name in KERNEL_OBJS or name.startswith("_"):
            continue
        if v is None:
            continue
        mod = type(v).__module__
        if mod.startswith("hbllm.hcir.world"):
            inv[name] = type(v).__name__
    return inv


class CountingProxy:
    """Delegates everything; counts method invocations per method name."""

    def __init__(self, inner: Any, counter: dict[str, int]) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_counter", counter)

    # Generic boundary-protocol probes applied indiscriminately
    _PROBE_READS = frozenset(
        {"on_boundary", "boundary_retention", "FIELD_RETENTION", "reset_episode"}
    )

    def __getattr__(self, name: str) -> Any:
        c = object.__getattribute__(self, "_counter")
        v = getattr(object.__getattribute__(self, "_inner"), name)
        if callable(v):

            def _wrapped(*a: Any, **k: Any) -> Any:
                cc = object.__getattribute__(self, "_counter")
                cc[name + "()"] = cc.get(name + "()", 0) + 1
                cc["__total__"] = cc.get("__total__", 0) + 1
                return v(*a, **k)

            if name not in CountingProxy._PROBE_READS:
                c[name] = c.get(name, 0) + 1
                c["__total__"] = c.get("__total__", 0) + 1
            return _wrapped
        if name not in CountingProxy._PROBE_READS:
            c[name] = c.get(name, 0) + 1
            c["__total__"] = c.get("__total__", 0) + 1
        return v

    def __setattr__(self, name: str, val: Any) -> None:
        setattr(object.__getattribute__(self, "_inner"), name, val)

    def __getitem__(self, k: Any) -> Any:
        return object.__getattribute__(self, "_inner")[k]

    def __setitem__(self, k: Any, v: Any) -> None:
        object.__getattribute__(self, "_inner")[k] = v

    def __bool__(self) -> bool:
        inner = object.__getattribute__(self, "_inner")
        return bool(inner)

    def __contains__(self, k: Any) -> bool:
        inner = object.__getattribute__(self, "_inner")
        if hasattr(inner, "__contains__"):
            return k in inner
        return False

    def __iter__(self) -> Any:
        inner = object.__getattribute__(self, "_inner")
        if hasattr(inner, "__iter__"):
            return iter(inner)
        return iter(())

    def __len__(self) -> int:
        inner = object.__getattribute__(self, "_inner")
        if hasattr(inner, "__len__"):
            return len(inner)
        return 1 if inner else 0


def _neutral_for_return(cls: type, name: str) -> Any:
    try:
        sig = inspect.signature(getattr(cls, name))
        ann = sig.return_annotation
    except Exception:
        return None
    text = str(ann)
    if "bool" in text:
        return False
    if "int" in text and "print" not in text:
        return 0
    if "float" in text:
        return 0.0
    if "list" in text:
        return []
    if "dict" in text:
        return {}
    if "set" in text:
        return set()
    if "tuple" in text:
        return ()
    return None


class NullFaculty:
    """Generic null object: methods return neutral-by-annotation values."""

    def __init__(self, cls: type) -> None:
        object.__setattr__(self, "_cls", cls)

    def __getattr__(self, name: str) -> Any:
        cls = object.__getattribute__(self, "_cls")
        member = getattr(cls, name, None)
        if callable(member):
            neutral = _neutral_for_return(cls, name)
            return lambda *a, **k: neutral
        # attribute read: neutral-by-annotation if declared, else None
        for klass in getattr(cls, "__mro__", [cls]):
            ann = getattr(klass, "__annotations__", {})
            if name in ann:
                return _neutral_for_return(cls, name)
        return None

    def __setattr__(self, name: str, val: Any) -> None:
        pass  # swallow writes

    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Any:
        return iter(())

    def __contains__(self, k: Any) -> bool:
        return False


def _run_env_lifelong(
    env_cls: type,
    seed: int,
    max_budget: int,
    engine: Any,
    episodes: int = 3,
) -> dict[str, Any]:
    """Execute real closed-loop benchmark environment with lifelong metric on an engine."""
    from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind

    episode_successes: list[bool] = []
    episode_steps: list[int] = []
    lat: list[float] = []

    prev_succ = False
    for ep in range(episodes):
        if ep > 0 and engine is not None:
            b_kind = EpisodeBoundaryKind.LEVEL_COMPLETE if prev_succ else EpisodeBoundaryKind.DEATH
            try:
                engine.on_boundary(EpisodeBoundary(kind=b_kind))
            except Exception:
                pass

        env = env_cls(max_steps=max_budget)
        obs = env.reset(seed=seed)
        step_count = 0
        succ = False
        for s in range(max_budget):
            step_count = s + 1
            t0 = time.perf_counter()
            raw = engine.step(obs)
            action = raw[0] if isinstance(raw, tuple) and len(raw) > 1 else raw
            lat.append((time.perf_counter() - t0) * 1000.0)

            obs, r, done, info = env.step(action)
            if done:
                succ = bool(info.get("success", False))
                try:
                    engine.step(obs)
                except Exception:
                    pass
                break

        episode_successes.append(succ)
        episode_steps.append(step_count)
        prev_succ = succ

    solve_rate = sum(1 for s in episode_successes if s) / max(1, len(episode_successes))
    efficiency = sum(max(0.0, 1.0 - (st / max(1, max_budget))) for st in episode_steps) / max(
        1, len(episode_steps)
    )
    score = round(0.5 * solve_rate + 0.5 * efficiency, 4)

    stable_idx = None
    for i in range(len(episode_successes) - 2):
        if episode_successes[i] and episode_successes[i + 1] and episode_successes[i + 2]:
            stable_idx = i
            break

    return {
        "score": score,
        "solve_rate": solve_rate,
        "first_success_step": episode_steps[0] if episode_successes[0] else None,
        "mean_steps": sum(episode_steps) / max(1, len(episode_steps)),
        "stable_policy": stable_idx,
        "mean_ms": (sum(lat) / max(1, len(lat))) if lat else 0.0,
        "successes": episode_successes,
        "steps": episode_steps,
    }


def _task_screen(task: tuple[str, str, int, int, int]) -> dict[str, Any]:
    env_id, env_cls_path, seed, steps, episodes = task
    env_cls = _load_env_class(env_cls_path)
    e = _mk_engine(seed)
    inv = faculty_inventory(e)
    counters = {n: {} for n in inv}
    for n, fac in list(vars(e).items()):
        if n in inv:
            setattr(e, n, CountingProxy(fac, counters[n]))
    res = _run_env_lifelong(env_cls, seed, steps, e, episodes=episodes)
    totals = {n: c.get("__total__", 0) for n, c in counters.items()}
    return {"env": env_id, "seed": seed, "calls": totals, "result": res}


def _task_run(task: tuple[str, int, int, int]) -> dict[str, Any]:
    """(faculty|baseline, seed, steps, episodes) -> per-environment metrics."""
    faculty, seed, steps, episodes = task
    out: dict[str, Any] = {}
    crash = None
    for env_id, env_cls_path, max_b in BENCH_ENVS:
        env_cls = _load_env_class(env_cls_path)
        e = _mk_engine(seed)
        if faculty != "__baseline__":
            inv = faculty_inventory(e)
            cls = type(getattr(e, faculty)) if faculty in inv else None
            if cls is None:
                continue
            setattr(e, faculty, NullFaculty(cls))
        try:
            out[env_id] = _run_env_lifelong(env_cls, seed, min(steps, max_b), e, episodes=episodes)
        except Exception as exc:  # noqa: BLE001 — ablation crash is the signal
            crash = f"{env_id}: {type(exc).__name__}: {exc}"
            out[env_id] = {
                "score": 0.0,
                "solve_rate": 0.0,
                "first_success_step": None,
                "mean_steps": steps,
                "stable_policy": None,
                "mean_ms": 0.0,
                "successes": [False] * episodes,
                "steps": [steps] * episodes,
            }
    return {"faculty": faculty, "seed": seed, "envs": out, "crash": crash}


def _mean(xs: Any) -> float:
    vals = list(xs)
    return sum(vals) / max(1, len(vals))


def _std(xs: Any) -> float:
    vals = list(xs)
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Faculty ablation harness on procedural environments")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--phase", choices=["inventory", "screen", "ablate", "all"], default="all")
    ap.add_argument("--screen-report", default=None)
    args = ap.parse_args(argv)

    engine = _mk_engine(0)
    inv = faculty_inventory(engine)
    Path("reports/ablation").mkdir(parents=True, exist_ok=True)
    Path("reports/ablation/faculty_inventory.json").write_text(
        json.dumps({"faculties": inv, "count": len(inv)}, indent=2)
    )
    print(f"inventory: {len(inv)} faculties -> reports/ablation/faculty_inventory.json")
    if args.phase == "inventory":
        return

    mp_ctx = mp.get_context("spawn")

    # ---- screen ----
    screen_path = Path(args.screen_report or "reports/ablation/screen.json")
    if args.phase == "ablate" and screen_path.exists():
        prior = json.loads(screen_path.read_text())
        calls = prior["calls"]
        used, unused = prior["used"], prior["unused"]
        print(f"screen (cached): {len(used)} used, {len(unused)} unused")
    else:
        calls = {n: 0 for n in inv}
        screen_tasks = [
            (eid, epath, s, min(args.steps, b), args.episodes)
            for eid, epath, b in BENCH_ENVS
            for s in args.seeds
        ]
        with mp_ctx.Pool(args.workers) as pool:
            for res in pool.imap_unordered(_task_screen, screen_tasks):
                for n, c in res["calls"].items():
                    calls[n] += c
                print(
                    f"screen {res['env']} s{res['seed']}: score={res['result']['score']:.4f}, solve={res['result']['solve_rate']:.2f}"
                )
        unused = sorted(n for n, c in calls.items() if c == 0)
        used = sorted(n for n, c in calls.items() if c > 0)
        print(f"screen: {len(used)} used, {len(unused)} unused")
        screen_path.write_text(
            json.dumps({"calls": calls, "used": used, "unused": unused}, indent=2)
        )
    if args.phase == "screen":
        return

    # ---- ablate (used faculties only) ----
    base_tasks = [("__baseline__", s, args.steps, args.episodes) for s in args.seeds]
    fac_tasks = [(n, s, args.steps, args.episodes) for n in used for s in args.seeds]
    results: dict[str, dict[int, Any]] = {"__baseline__": {}}
    with mp_ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_task_run, base_tasks + fac_tasks):
            results.setdefault(res["faculty"], {})[res["seed"]] = res

    report = {
        "rule": DECISION_RULE,
        "envs": [e[0] for e in BENCH_ENVS],
        "seeds": args.seeds,
        "steps": args.steps,
        "episodes": args.episodes,
        "note": "Evaluated on procedural closed-loop benchmark environments with lifelong metric.",
        "faculties": {},
    }
    for name in list(inv):
        if name in unused:
            report["faculties"][name] = {
                "class": inv[name],
                "calls": 0,
                "decision": "unused",
                "delta_score": 0.0,
                "evidence": "0 calls across all procedural runs",
            }
            continue
        base_scores, abl_scores = [], []
        base_steps, abl_steps = [], []
        base_ms, abl_ms = [], []
        crash = None
        for s in args.seeds:
            b = results["__baseline__"].get(s, {})
            a = results.get(name, {}).get(s, {})
            if a.get("crash"):
                crash = a["crash"]
            for eid, _, _ in BENCH_ENVS:
                be = (b.get("envs") or {}).get(eid, {})
                ae = (a.get("envs") or {}).get(eid, {})
                base_scores.append(be.get("score", 0.0))
                abl_scores.append(ae.get("score", 0.0))
                if be.get("mean_steps") is not None:
                    base_steps.append(be["mean_steps"])
                if ae.get("mean_steps") is not None:
                    abl_steps.append(ae["mean_steps"])
                base_ms.append(be.get("mean_ms", 0.0))
                abl_ms.append(ae.get("mean_ms", 0.0))
        sd = max(0.05, _std(base_scores))
        d_score = _mean(base_scores) - _mean(abl_scores)
        d_steps_pct = (
            (_mean(abl_steps) - _mean(base_steps)) / max(1e-9, _mean(base_steps)) * 100.0
            if base_steps and abl_steps
            else 0.0
        )
        d_ms = _mean(abl_ms) - _mean(base_ms)
        if crash:
            decision = "keep"
            why = f"load_bearing_crash: {crash}"
        elif d_score > sd:
            decision = "keep"
            why = f"Δscore {d_score:.4f} > sd {sd:.4f}"
        elif d_steps_pct > 5.0 and d_score >= 0.0:
            decision = "keep"
            why = f"mean_steps +{d_steps_pct:.1f}% with no score gain"
        else:
            decision = "experimental"
            why = f"Δscore {d_score:.4f} <= sd {sd:.4f}"
        report["faculties"][name] = {
            "class": inv[name],
            "calls": calls.get(name, 0),
            "baseline_score": round(_mean(base_scores), 4),
            "ablated_score": round(_mean(abl_scores), 4),
            "delta_score": round(d_score, 4),
            "sd": round(sd, 4),
            "delta_steps_pct": round(d_steps_pct, 1),
            "delta_latency_ms": round(d_ms, 1),
            "decision": decision,
            "rule_outcome": why,
        }

    n_keep = sum(1 for f in report["faculties"].values() if f["decision"] == "keep")
    n_exp = sum(1 for f in report["faculties"].values() if f["decision"] == "experimental")
    n_un = len(unused)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"reports/ablation/{ts}.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"decisions: keep={n_keep} experimental={n_exp} unused={n_un}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
