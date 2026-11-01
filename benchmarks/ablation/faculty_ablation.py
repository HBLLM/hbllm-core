"""WP2 faculty ablation harness.

Phases:
  inventory  – enumerate every hbllm.hcir.world-owned faculty attr on the engine.
  screen     – counting-proxy usage counts (9 games x seeds, closed loop).
  ablate     – NullFaculty swap per used faculty; levels/steps/latency vs baseline.
Decisions per the pre-registered rule (see DECISION_RULE below).
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle_submission"))

GAMES = [
    "nav_basic",
    "blocked_first",
    "stride2",
    "key_door",
    "push_block",
    "click_toggle",
    "recolour_gate",
    "hazard",
    "death_reset",
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
    "sd = std(baseline_levels across seeds) floored at 0.5; "
    "keep iff (baseline - ablated) > sd, OR steps-to-first-level +>5% with no "
    "level gain, OR load_bearing_crash; else experimental. 0-call faculties "
    "are unused. NOTE: 9 small games is limited evidence."
)


def _mk_engine(seed: int = 0):
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

    return AutonomousEpistemicEngine(enable_logging=False, seed=seed)


def faculty_inventory(engine) -> dict[str, str]:
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

    def __init__(self, inner, counter):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_counter", counter)

    # Reads of these names are generic boundary-protocol probes applied to
    # every engine attribute indiscriminately — NOT faculty-specific use.
    # Their actual *invocations* still count (a real on_boundary does work).
    _PROBE_READS = frozenset(
        {"on_boundary", "boundary_retention", "FIELD_RETENTION", "reset_episode"}
    )

    def __getattr__(self, name):
        c = object.__getattribute__(self, "_counter")
        v = getattr(object.__getattribute__(self, "_inner"), name)
        if callable(v):

            def _wrapped(*a, **k):
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

    def __setattr__(self, name, val):
        setattr(object.__getattribute__(self, "_inner"), name, val)

    def __getitem__(self, k):
        return object.__getattribute__(self, "_inner")[k]

    def __setitem__(self, k, v):
        object.__getattribute__(self, "_inner")[k] = v

    def __contains__(self, k):
        return k in object.__getattribute__(self, "_inner")

    def __iter__(self):
        return iter(object.__getattribute__(self, "_inner"))

    def __len__(self):
        return len(object.__getattribute__(self, "_inner"))


def _neutral_for_return(cls, name):
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

    def __init__(self, cls):
        object.__setattr__(self, "_cls", cls)

    def __getattr__(self, name):
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

    def __setattr__(self, name, val):
        pass  # swallow writes

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())

    def __contains__(self, k):
        return False


def _run_game(game_id: str, seed: int, steps: int, engine=None, agent=None):
    """One closed-loop game; returns (levels, steps_to_first_level, mean_ms)."""
    from my_agent import MyAgent

    from plugins.arc_agi_adapter.local_env import LocalEnv

    g = LocalEnv(0).make(game_id)
    if agent is None:
        agent = MyAgent(card_id="abl", game_id=game_id, seed=seed)
        if engine is not None:
            agent.internal_engine = engine
            agent.internal_agent = engine
    f = g.reset()
    first_level_step = None
    lat = []
    for s in range(steps):
        if getattr(f.state, "name", "") != "NOT_FINISHED":
            f = g.step(0, {})
            continue
        t0 = time.perf_counter()
        act = agent.choose_action([f], f)
        lat.append(time.perf_counter() - t0)
        aid = 7 if act is None else int(getattr(act, "value", act))
        f = g.step(aid, getattr(act, "data", {}) or {})
        if first_level_step is None and f.levels_completed > 0:
            first_level_step = s
    return {
        "levels": f.levels_completed,
        "first_level_step": first_level_step,
        "mean_ms": (sum(lat) / max(1, len(lat))) * 1000.0,
    }


def _task_screen(task):
    game_id, seed, steps = task
    e = _mk_engine(seed)
    inv = faculty_inventory(e)
    counters = {n: {} for n in inv}
    for n, fac in list(vars(e).items()):
        if n in inv:
            setattr(e, n, CountingProxy(fac, counters[n]))
    from my_agent import MyAgent

    a = MyAgent(card_id="abl", game_id=game_id, seed=seed)
    a.internal_engine = e
    a.internal_agent = e
    res = _run_game(game_id, seed, steps, agent=a)
    totals = {n: c.get("__total__", 0) for n, c in counters.items()}
    return {"game": game_id, "seed": seed, "calls": totals, "result": res}


def _task_run(task):
    """(faculty|baseline, seed, steps) -> per-game metrics."""
    faculty, seed, steps = task
    out = {}
    crash = None
    for gid in GAMES:
        e = _mk_engine(seed)
        if faculty != "__baseline__":
            inv = faculty_inventory(e)
            cls = type(getattr(e, faculty)) if faculty in inv else None
            if cls is None:
                continue
            setattr(e, faculty, NullFaculty(cls))
        from my_agent import MyAgent

        a = MyAgent(card_id="abl", game_id=gid, seed=seed)
        a.internal_engine = e
        a.internal_agent = e
        try:
            out[gid] = _run_game(gid, seed, steps, agent=a)
        except Exception as exc:  # noqa: BLE001 — ablation crash is the signal
            crash = f"{gid}: {type(exc).__name__}: {exc}"
            out[gid] = {"levels": 0, "first_level_step": None, "mean_ms": 0.0}
        excs = getattr(a, "decision_exceptions", 0)
        if excs:
            crash = crash or f"{gid}: {excs} decision exceptions (swallowed fallbacks)"
            out[gid]["decision_exceptions"] = excs
    return {"faculty": faculty, "seed": seed, "games": out, "crash": crash}


def _mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def _std(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--phase", choices=["inventory", "screen", "ablate", "all"], default="all")
    ap.add_argument("--screen-report", default=None)
    args = ap.parse_args(argv)

    import multiprocessing as mp

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
        screen_tasks = [(g, s, args.steps) for g in GAMES for s in args.seeds]
        with mp_ctx.Pool(args.workers) as pool:
            for res in pool.imap_unordered(_task_screen, screen_tasks):
                for n, c in res["calls"].items():
                    calls[n] += c
                print(f"screen {res['game']} s{res['seed']}: lvl={res['result']['levels']}")
        unused = sorted(n for n, c in calls.items() if c == 0)
        used = sorted(n for n, c in calls.items() if c > 0)
        print(f"screen: {len(used)} used, {len(unused)} unused")
        screen_path.write_text(
            json.dumps({"calls": calls, "used": used, "unused": unused}, indent=2)
        )
    if args.phase == "screen":
        return

    # ---- ablate (used faculties only) ----
    base_tasks = [("__baseline__", s, args.steps) for s in args.seeds]
    fac_tasks = [(n, s, args.steps) for n in used for s in args.seeds]
    results = {"__baseline__": {}}
    with mp_ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_task_run, base_tasks + fac_tasks):
            results.setdefault(res["faculty"], {})[res["seed"]] = res

    report = {
        "rule": DECISION_RULE,
        "games": GAMES,
        "seeds": args.seeds,
        "steps": args.steps,
        "note": "9 small games is limited evidence for retention decisions.",
        "faculties": {},
    }
    for name in list(inv):
        if name in unused:
            report["faculties"][name] = {
                "class": inv[name],
                "calls": 0,
                "decision": "unused",
                "delta_levels": 0.0,
                "evidence": "0 calls across all runs",
            }
            continue
        base_lvls, abl_lvls = [], []
        base_steps, abl_steps = [], []
        base_ms, abl_ms = [], []
        crash = None
        for s in args.seeds:
            b = results["__baseline__"].get(s, {})
            a = results.get(name, {}).get(s, {})
            if a.get("crash"):
                crash = a["crash"]
            for gid in GAMES:
                bg = (b.get("games") or {}).get(gid, {})
                ag = (a.get("games") or {}).get(gid, {})
                base_lvls.append(bg.get("levels", 0))
                abl_lvls.append(ag.get("levels", 0))
                if bg.get("first_level_step") is not None:
                    base_steps.append(bg["first_level_step"])
                if ag.get("first_level_step") is not None:
                    abl_steps.append(ag["first_level_step"])
                base_ms.append(bg.get("mean_ms", 0.0))
                abl_ms.append(ag.get("mean_ms", 0.0))
        sd = max(0.5, _std(base_lvls))
        d_lvl = _mean(base_lvls) - _mean(abl_lvls)
        d_steps_pct = (
            (_mean(abl_steps) - _mean(base_steps)) / max(1e-9, _mean(base_steps)) * 100.0
            if base_steps and abl_steps
            else 0.0
        )
        d_ms = _mean(abl_ms) - _mean(base_ms)
        no_gain = d_lvl <= 0
        if crash:
            decision = "keep"
            why = f"load_bearing_crash: {crash}"
        elif d_lvl > sd:
            decision = "keep"
            why = f"Δlevels {d_lvl:.2f} > sd {sd:.2f}"
        elif d_steps_pct > 5.0 and no_gain:
            decision = "keep"
            why = f"steps-to-first-level +{d_steps_pct:.1f}% with no level gain"
        else:
            decision = "experimental"
            why = f"Δlevels {d_lvl:.2f} <= sd {sd:.2f}"
        report["faculties"][name] = {
            "class": inv[name],
            "calls": calls.get(name, 0),
            "baseline_levels": _mean(base_lvls),
            "ablated_levels": _mean(abl_lvls),
            "delta_levels": round(d_lvl, 3),
            "sd": round(sd, 3),
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
