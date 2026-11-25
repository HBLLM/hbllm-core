"""Faculty ablation harness on procedural benchmark environments.

Phases:
  inventory  – enumerate every hbllm.hcir.world-owned faculty attr on the engine.
  screen     – counting-proxy usage counts (benchmark environments x seeds, closed loop with lifelong metric).
  ablate     – NullFaculty swap per used faculty; score/steps/latency vs baseline.
Decisions per the pre-registered rule (see DECISION_RULE below).
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_ENVS = [
    ("grid_nav", "benchmarks.envs.grid_nav.GridNavEnvironment", 100),
    ("grid_push", "benchmarks.envs.grid_push.GridPushEnvironment", 100),
    ("continuous_reach", "benchmarks.envs.continuous_reach.ContinuousReachEnvironment", 60),
    ("symbolic_rules", "benchmarks.envs.symbolic_rules.SymbolicRulesEnvironment", 80),
    ("visual_transfer", "benchmarks.envs.visual_transfer.VisualMorphologyTransferEnv", 35),
    ("minigrid_doorkey", "benchmarks.envs.external_benchmarks.MiniGridDoorKeyBenchmarkEnv", 80),
]

WIDENED_BENCH_ENVS = [
    ("partial_obs_dev", "benchmarks.envs.widened_dev_envs.PartialObsMazeEnv", 140),
    ("stochastic_wind_dev", "benchmarks.envs.widened_dev_envs.StochasticWindNavEnv", 120),
    ("other_agents_dev", "benchmarks.envs.widened_dev_envs.MultiAgentCorridorEnv", 120),
    ("long_horizon_dev", "benchmarks.envs.widened_dev_envs.LongHorizonQuestEnv", 220),
    (
        "continuous_multi_step_dev",
        "benchmarks.envs.widened_dev_envs.ContinuousMultiWaypointEnv",
        120,
    ),
    ("language_goal_dev", "benchmarks.envs.widened_dev_envs.LanguageGoalSelectEnv", 80),
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


_ALIAS_FACULTY_CANONICAL = {
    "acc_conflict": "acc_conflict_monitor",
}

_FACULTY_ALIAS_TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    "subgoal_stack": (("working_memory", "subgoal_stack"),),
    "body_schema": (("working_memory", "body_schema"),),
    "habenular_ior": (("working_memory", "habenular_ior"),),
    "remote_causal": (("working_memory", "remote_causal"),),
    "acc_conflict_monitor": (("engine", "acc_conflict"), ("working_memory", "acc_conflict")),
}


def _apply_counting_proxy(engine: Any, name: str, counter: dict[str, int]) -> None:
    """Wrap faculty attr (and any working_memory/engine aliases) in CountingProxy."""
    fac = vars(engine).get(name)
    if fac is None:
        return
    proxy = CountingProxy(fac, counter)
    setattr(engine, name, proxy)
    for owner_name, attr_name in _FACULTY_ALIAS_TARGETS.get(name, ()):
        owner = engine if owner_name == "engine" else getattr(engine, owner_name, None)
        if owner is not None and hasattr(owner, attr_name):
            setattr(owner, attr_name, proxy)


EXPERIMENTAL_NONKEEP_CANDIDATES: tuple[str, ...] = (
    "body_schema",
    "hazard_tracker",
    "object_planner",
    "option_discovery",
    "remote_causal",
)

_ABLATION_BUNDLES: dict[str, tuple[str, ...]] = {
    "__all_experimental__": EXPERIMENTAL_NONKEEP_CANDIDATES,
    "__r5_named_demotes__": (
        "surprise_engine",
        "collision_cones",
        "predictive_coding_cortex",
        "snn_predictor",
        "neuromodulatory_system",
        "precision_controller",
        "dorsal_kinetic_stream",
        "deformation_tracker",
        "metacognitive_agency_cortex",
    ),
    "__pipeline_hook_experimental__": (
        "action_discovery",
        "dual_memory",
        "goal_arbiter",
        "goal_induction",
        "mpfc_schema_engine",
        "option_discovery",
        "recursive_value_inducer",
        "room_topology",
        "skill_synthesizer",
        "smith_predictor",
        "social_cognition",
        "temporal_buffer",
        "transition_sequence_model",
    ),
    "__hook_b1__": (
        "action_discovery",
        "goal_arbiter",
        "goal_induction",
        "mpfc_schema_engine",
        "option_discovery",
        "skill_synthesizer",
        "social_cognition",
    ),
    "__hook_b2__": (
        "dual_memory",
        "recursive_value_inducer",
        "room_topology",
        "smith_predictor",
        "temporal_buffer",
        "transition_sequence_model",
    ),
    "__all_zero_r1__": (
        "action_discovery",
        "body_schema",
        "causal_cortex",
        "cerebellar_clock",
        "collision_cones",
        "deformation_tracker",
        "delayed_effect_modeler",
        "dorsal_kinetic_stream",
        "dual_memory",
        "goal_arbiter",
        "goal_induction",
        "library_learner",
        "metacognitive_agency_cortex",
        "mpfc_schema_engine",
        "neuromodulatory_system",
        "physics_engine",
        "precision_controller",
        "predictive_coding_cortex",
        "recursive_value_inducer",
        "room_topology",
        "skill_synthesizer",
        "smith_predictor",
        "snn_predictor",
        "social_cognition",
        "surprise_engine",
        "temporal_buffer",
        "transition_sequence_model",
    ),
    "__all_zero_r1_plus_hazard__": (
        "action_discovery",
        "body_schema",
        "causal_cortex",
        "cerebellar_clock",
        "collision_cones",
        "deformation_tracker",
        "delayed_effect_modeler",
        "dorsal_kinetic_stream",
        "dual_memory",
        "goal_arbiter",
        "goal_induction",
        "hazard_tracker",
        "library_learner",
        "metacognitive_agency_cortex",
        "mpfc_schema_engine",
        "neuromodulatory_system",
        "physics_engine",
        "precision_controller",
        "predictive_coding_cortex",
        "recursive_value_inducer",
        "room_topology",
        "skill_synthesizer",
        "smith_predictor",
        "snn_predictor",
        "social_cognition",
        "surprise_engine",
        "temporal_buffer",
        "transition_sequence_model",
    ),
    "__xs_g1__": (
        "action_discovery",
        "body_schema",
        "causal_cortex",
        "cerebellar_clock",
        "collision_cones",
        "deformation_tracker",
        "delayed_effect_modeler",
    ),
    "__xs_g2__": (
        "dorsal_kinetic_stream",
        "dual_memory",
        "goal_arbiter",
        "goal_induction",
        "library_learner",
        "metacognitive_agency_cortex",
        "mpfc_schema_engine",
    ),
    "__xs_g3__": (
        "neuromodulatory_system",
        "physics_engine",
        "precision_controller",
        "predictive_coding_cortex",
        "recursive_value_inducer",
        "room_topology",
        "skill_synthesizer",
    ),
    "__xs_g4__": (
        "smith_predictor",
        "snn_predictor",
        "social_cognition",
        "surprise_engine",
        "temporal_buffer",
        "transition_sequence_model",
    ),
}


def _apply_null_faculty(engine: Any, name: str, inv: dict[str, str]) -> bool:
    """Replace faculty attr (and any working_memory/engine aliases) with NullFaculty."""
    if name in _ABLATION_BUNDLES:
        ok = False
        for member in _ABLATION_BUNDLES[name]:
            ok = _apply_null_faculty(engine, member, inv) or ok
        return ok
    orig_inst = getattr(engine, name, None) if name in inv else None
    cls = type(orig_inst) if orig_inst is not None else None
    if cls is None:
        return False
    null_fac = NullFaculty(cls, orig_inst)
    setattr(engine, name, null_fac)
    for owner_name, attr_name in _FACULTY_ALIAS_TARGETS.get(name, ()):
        owner = engine if owner_name == "engine" else getattr(engine, owner_name, None)
        if owner is not None and hasattr(owner, attr_name):
            setattr(owner, attr_name, null_fac)
    return True


def faculty_inventory(engine: Any) -> dict[str, str]:
    """{attr_name: ClassName} for world-defined, non-core, non-kernel attrs."""
    inv = {}
    attrs = vars(engine)
    for name, v in attrs.items():
        if name in CORE_OWNERS or name in KERNEL_OBJS or name.startswith("_"):
            continue
        if (
            name in _ALIAS_FACULTY_CANONICAL
            and attrs.get(_ALIAS_FACULTY_CANONICAL[name]) is not None
        ):
            continue
        if v is None:
            continue
        mod = type(v).__module__
        if mod == "hbllm.hcir.world" or mod.startswith("hbllm.hcir.world."):
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


class NullResult:
    """Permissive null return object for custom dataclass/result types in faculty ablation."""

    def __getattr__(self, name: str) -> Any:
        if name in ("self_avatar", "winner", "detected_macro_period", "active_goal"):
            return None
        if name == "shape":
            return (0, 0)
        if name.startswith(("is_", "has_", "should_", "can_", "enable_")):
            return False
        if name.endswith(("_features", "_positions", "_cells", "_transitions", "_ids", "_set")):
            return set()
        if name.endswith(
            ("_list", "_history", "_agents", "_objects", "_steps", "_subgoals", "_trajectories")
        ):
            return []
        if name.endswith(("_map", "_dict", "_scores", "_values", "_graph")) or name in (
            "variables",
            "relations",
            "metadata",
            "scores",
            "weights",
            "params",
        ):
            return {}
        if name in ("max_depth", "depth", "count", "steps", "recommended_simulation_depth"):
            return 0
        if name.startswith(("total_", "num_", "n_")) and not name.endswith(
            ("_energy", "_precision", "_weight", "_score")
        ):
            return 0
        return NullResult()

    def __setattr__(self, name: str, val: Any) -> None:
        pass

    def __getitem__(self, k: Any) -> Any:
        return 1.0

    def __iter__(self) -> Any:
        return iter(())

    def keys(self) -> Any:
        return ()

    def items(self) -> Any:
        return ()

    def values(self) -> Any:
        return ()

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False

    def __float__(self) -> float:
        return 0.0

    def __int__(self) -> int:
        return 0

    def __lt__(self, other: Any) -> bool:
        try:
            return 0.0 < float(other)
        except (TypeError, ValueError):
            return False

    def __le__(self, other: Any) -> bool:
        try:
            return 0.0 <= float(other)
        except (TypeError, ValueError):
            return False

    def __gt__(self, other: Any) -> bool:
        try:
            return 0.0 > float(other)
        except (TypeError, ValueError):
            return False

    def __ge__(self, other: Any) -> bool:
        try:
            return 0.0 >= float(other)
        except (TypeError, ValueError):
            return False

    def __add__(self, other: Any) -> float:
        try:
            return 0.0 + float(other)
        except (TypeError, ValueError):
            return 0.0

    def __radd__(self, other: Any) -> float:
        try:
            return float(other) + 0.0
        except (TypeError, ValueError):
            return 0.0

    def __sub__(self, other: Any) -> float:
        try:
            return 0.0 - float(other)
        except (TypeError, ValueError):
            return 0.0

    def __rsub__(self, other: Any) -> float:
        try:
            return float(other) - 0.0
        except (TypeError, ValueError):
            return 0.0

    def __mul__(self, other: Any) -> float:
        try:
            return 0.0 * float(other)
        except (TypeError, ValueError):
            return 0.0

    def __rmul__(self, other: Any) -> float:
        try:
            return float(other) * 0.0
        except (TypeError, ValueError):
            return 0.0

    def __truediv__(self, other: Any) -> float:
        try:
            denom = float(other)
            return 0.0 / denom if denom != 0.0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, *a: Any, **k: Any) -> NullResult:
        return self


def _split_top_level_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _neutral_for_type_text(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("typing."):
        text = text[len("typing.") :]
    # Unwrap Optional[...] or X | None
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1].strip()
    if " | None" in text:
        text = text.replace(" | None", "").strip()
    elif "None | " in text:
        text = text.replace("None | ", "").strip()
    if text.startswith("Union[") and text.endswith("]"):
        args = [a for a in _split_top_level_args(text[6:-1]) if a != "None"]
        if args:
            text = args[0]

    low = text.lower()
    if text in ("None", "NoneType", "<class 'NoneType'>", "<class 'inspect._empty'>"):
        return None
    if low.startswith(("tuple[", "typing.tuple[")):
        inner = text[text.find("[") + 1 : -1]
        args = _split_top_level_args(inner)
        if not args or args == ["()"] or (len(args) == 2 and args[1] == "..."):
            return ()
        return tuple(_neutral_for_type_text(a) for a in args)
    if low in ("tuple", "<class 'tuple'>"):
        return ()
    if low.startswith(("list[", "sequence[", "mutable", "iterable[")) or low in (
        "list",
        "<class 'list'>",
    ):
        return []
    if low.startswith("frozenset") or low in ("frozenset", "<class 'frozenset'>"):
        return frozenset()
    if low.startswith(("set[", "abstractset[", "mutableset[")) or low in ("set", "<class 'set'>"):
        return set()
    if low.startswith(("dict[", "mapping[", "mutablemapping[")) or low in (
        "dict",
        "<class 'dict'>",
    ):
        return defaultdict(lambda: 1.0)
    if "ndarray" in low:
        return np.zeros((0, 0), dtype=float)
    if low in ("bool", "<class 'bool'>"):
        return False
    if low in ("int", "<class 'int'>"):
        return 0
    if low in ("float", "<class 'float'>"):
        return 0.0
    if low in ("str", "<class 'str'>"):
        return ""
    if "callable" in low:
        return lambda *a, **k: None
    return NullResult()


def _neutral_for_return(cls: type, name: str, instance: Any = None) -> Any:
    if instance is not None:
        bound = getattr(instance, name, None)
        try:
            sig = inspect.signature(bound if callable(bound) else getattr(cls, name))
            ann = sig.return_annotation
        except Exception:
            return None
    else:
        try:
            sig = inspect.signature(getattr(cls, name))
            ann = sig.return_annotation
        except Exception:
            return None
    if ann is inspect.Signature.empty:
        return None
    if isinstance(ann, str):
        ann_str = ann
    elif isinstance(ann, type):
        ann_str = ann.__name__
    else:
        ann_str = str(ann)
    val = _neutral_for_type_text(ann_str)
    if name == "get_modulation_params" and isinstance(val, dict):
        return defaultdict(lambda: 1.0)
    return val


class NullFaculty:
    """Generic null object: methods return neutral-by-annotation values."""

    def __init__(self, cls: type, instance: Any = None) -> None:
        object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "_instance", instance)

    def __getattr__(self, name: str) -> Any:
        cls = object.__getattribute__(self, "_cls")
        instance = object.__getattribute__(self, "_instance")
        member = getattr(cls, name, None)
        if callable(member) and not isinstance(member, property):
            if name == "compete_and_ignite":
                return lambda candidates=(), *a, **k: candidates[0] if candidates else None
            neutral = _neutral_for_return(cls, name, instance)
            return lambda *a, _ret=neutral, **k: copy.copy(_ret)
        if instance is not None:
            has_attr = True
            try:
                val = getattr(instance, name)
            except Exception:
                has_attr = False
                val = None
            if has_attr and val is None:
                return None
            if callable(val):
                neutral = _neutral_for_return(cls, name, instance)
                return lambda *a, _ret=neutral, **k: copy.copy(_ret)
            if isinstance(val, bool):
                return False
            if isinstance(val, int):
                return 0
            if isinstance(val, float):
                return 0.0
            if isinstance(val, str):
                return ""
            if isinstance(val, set):
                return set()
            if isinstance(val, frozenset):
                return frozenset()
            if isinstance(val, list):
                return []
            if isinstance(val, tuple):
                return ()
            if isinstance(val, dict):
                return defaultdict(lambda: 1.0)
            if val is not None:
                return NullFaculty(type(val), val)
        # Fallback by attribute annotation or name convention
        for klass in getattr(cls, "__mro__", [cls]):
            ann = getattr(klass, "__annotations__", {})
            if name in ann:
                ann_val = ann[name]
                ann_str = ann_val.__name__ if isinstance(ann_val, type) else str(ann_val)
                return _neutral_for_type_text(ann_str)
        if name.endswith(("_features", "_positions", "_transitions", "_cells", "_ids")):
            return set()
        if name.endswith(("_history", "_trajectories", "_agents", "_objects")):
            return []
        if name.endswith(("_values", "_scores", "_map")):
            return {}
        if name.startswith(("total_",)) or name.endswith(("_energy", "_precision", "_weight")):
            return 0.0
        if name.startswith(("max_",)) or name.endswith(("_count", "_depth")):
            return 0
        if name.startswith(("is_", "has_")):
            return False
        return NullResult()

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return NullResult()


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
            orig_inst = getattr(e, faculty, None) if faculty in inv else None
            cls = type(orig_inst) if orig_inst is not None else None
            if cls is None:
                continue
            setattr(e, faculty, NullFaculty(cls, orig_inst))
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


def _task_dev_cross_seed(args: tuple[str, str, str, str, list[int], int]) -> dict[str, Any]:
    """Run one R1 cross-seed transfer chain across all seeds of a task."""
    import tempfile

    from benchmarks.suites.runner import _run_loop_with_info
    from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    faculty, task_id, family, env_cls_path, seeds, max_budget = args
    env_cls = _load_env_class(env_cls_path)
    transfer_succ = 0
    transfer_total = 0
    transfer_deaths = 0
    crash: str | None = None
    with tempfile.TemporaryDirectory(prefix=f"abl_xs_{task_id}_{os.getpid()}_") as mem_dir:
        system = build_cognitive_system(
            CognitiveSystemConfig(
                world_id=f"abl_xs_{task_id}_{os.getpid()}_{time.time_ns()}",
                memory_dir=mem_dir,
                enable_logging=False,
                seed=seeds[0],
            )
        )
        engine = system.engine
        inv = faculty_inventory(engine)
        if faculty != "__baseline__":
            _apply_null_faculty(engine, faculty, inv)

        prev_succ = False
        for idx, s in enumerate(seeds):
            if idx > 0:
                b_kind = (
                    EpisodeBoundaryKind.LEVEL_COMPLETE if prev_succ else EpisodeBoundaryKind.DEATH
                )
                try:
                    engine.on_boundary(EpisodeBoundary(kind=b_kind, same_level=False))
                except Exception as exc:  # noqa: BLE001
                    crash = f"{task_id}@{s} boundary: {type(exc).__name__}: {exc}"
                    break
            env = env_cls(max_steps=max_budget)
            obs = env.reset(seed=s)
            lat: list[float] = []
            errs: list[str] = []
            try:
                succ, _, lat, errs, died = _run_loop_with_info(
                    env, obs, engine, "system", max_budget, lat, errs, seed=s
                )
            except Exception as exc:  # noqa: BLE001
                crash = f"{task_id}@{s}: {type(exc).__name__}: {exc}"
                succ, died = False, True
            if errs and crash is None:
                crash = f"{task_id}@{s}: {errs[0]}"
                succ = False
            if idx > 0:
                transfer_total += 1
                transfer_succ += int(bool(succ))
                transfer_deaths += int(bool(died))
            prev_succ = bool(succ)

    return {
        "faculty": faculty,
        "task_id": task_id,
        "family": family,
        "transfer_succ": transfer_succ,
        "transfer_total": transfer_total,
        "transfer_deaths": transfer_deaths,
        "crash": crash,
    }


def _task_dev_episode(args: tuple[str, str, str, str, int, int, bool]) -> dict[str, Any]:
    """Run one R1 zero-shot dev episode for (__baseline__ | faculty)."""
    import tempfile

    import numpy as np

    from benchmarks.suites.runner import _run_loop_with_info
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    faculty, task_id, family, env_cls_path, seed, max_budget, count_calls = args
    env_cls = _load_env_class(env_cls_path)
    counters: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix=f"abl_dev_{seed}_{os.getpid()}_") as mem_dir:
        system = build_cognitive_system(
            CognitiveSystemConfig(
                world_id=f"abl_dev_{task_id}_{seed}_{os.getpid()}_{time.time_ns()}",
                memory_dir=mem_dir,
                enable_logging=False,
                seed=seed,
            )
        )
        engine = system.engine
        inv = faculty_inventory(engine)
        if count_calls:
            counters = {n: {} for n in inv}
            for n in inv:
                _apply_counting_proxy(engine, n, counters[n])
        elif faculty != "__baseline__":
            _apply_null_faculty(engine, faculty, inv)

        env = env_cls(max_steps=max_budget)
        obs = env.reset(seed=seed)
        lat: list[float] = []
        errs: list[str] = []
        crash: str | None = None
        try:
            succ, steps, lat, errs, died = _run_loop_with_info(
                env, obs, engine, "system", max_budget, lat, errs, seed=seed
            )
        except Exception as exc:  # noqa: BLE001 — ablation crash is the signal
            crash = f"{task_id}@{seed}: {type(exc).__name__}: {exc}"
            succ, steps, died = False, max_budget, True
        if errs and crash is None:
            crash = f"{task_id}@{seed}: {errs[0]}"
            succ = False
        p50 = float(np.percentile(lat, 50)) if lat else 0.0
        p95 = float(np.percentile(lat, 95)) if lat else 0.0
        return {
            "faculty": faculty,
            "task_id": task_id,
            "family": family,
            "seed": seed,
            "max_budget": max_budget,
            "success": bool(succ),
            "steps": int(steps),
            "died": bool(died),
            "p50_ms": p50,
            "p95_ms": p95,
            "crash": crash,
            "calls": {n: c.get("__total__", 0) for n, c in counters.items()},
        }


def _aggregate_dev_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate 80 R1 dev zero-shot rows using exact EvaluationProtocol scoring."""
    from hbllm.benchmarks.protocol import wilson_ci

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        families[r["family"]].append(r)

    fam_scores: dict[str, float] = {}
    fam_solve: dict[str, float] = {}
    for fam, items in families.items():
        n_fam = len(items)
        solved = [it for it in items if it["success"]]
        k_fam = len(solved)
        fam_solve[fam] = round(k_fam / max(1, n_fam), 4)
        if not solved:
            fam_scores[fam] = 0.0
        else:
            sr = k_fam / n_fam
            eff = (
                sum(max(0.0, 1.0 - (it["steps"] / max(1, it["max_budget"]))) for it in solved)
                / k_fam
            )
            fam_scores[fam] = round(0.5 * sr + 0.5 * eff, 4)

    zs_gen = round(sum(fam_scores.values()) / max(1, len(fam_scores)), 4) if fam_scores else 0.0
    n_tot = len(rows)
    k_tot = sum(1 for r in rows if r["success"])
    deaths = sum(1 for r in rows if r["died"])
    mean_steps = _mean([r["steps"] for r in rows])
    p50_ms = _mean([r["p50_ms"] for r in rows])
    p95_ms = _mean([r["p95_ms"] for r in rows])
    crashes = [r["crash"] for r in rows if r.get("crash")]
    return {
        "generality_score": zs_gen,
        "solve_rate": round(k_tot / max(1, n_tot), 4),
        "n_solved": k_tot,
        "n_trials": n_tot,
        "wilson_95_ci": list(wilson_ci(k_tot, n_tot)),
        "deaths": deaths,
        "mean_steps": round(mean_steps, 2),
        "mean_p50_ms": round(p50_ms, 2),
        "mean_p95_ms": round(p95_ms, 2),
        "family_scores": fam_scores,
        "family_solve_rates": fam_solve,
        "crash": crashes[0] if crashes else None,
    }


def _run_dev_ablation(args: argparse.Namespace, inv: dict[str, str], mp_ctx: Any) -> None:
    """Run screening and per-faculty leave-one-out ablation on R1 dev/widened suites."""
    import yaml

    suite_name = str(getattr(args, "suite", "dev") or "dev")
    manifest_path = Path(f"benchmarks/suites/{suite_name}.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    tasks_def = manifest["tasks"]
    dev_slots = [
        (t["id"], t["family"], t["env_class"], int(s), int(t.get("max_budget", 100)))
        for t in tasks_def
        for s in t["seeds"]
    ]
    n_slots = len(dev_slots)

    screen_path = Path(args.screen_report or f"reports/ablation/screen_{suite_name}.json")
    if args.phase == "ablate" and screen_path.exists():
        prior = json.loads(screen_path.read_text(encoding="utf-8"))
        calls = prior["calls"]
        used, unused = prior["used"], prior["unused"]
        print(f"dev screen (cached): {len(used)} used, {len(unused)} unused across n={n_slots}")
    else:
        t0 = time.perf_counter()
        screen_jobs = [
            ("__baseline__", tid, fam, epath, s, mb, True) for tid, fam, epath, s, mb in dev_slots
        ]
        calls = {n: 0 for n in inv}
        base_rows: list[dict[str, Any]] = []
        with mp_ctx.Pool(args.workers) as pool:
            for row in pool.imap_unordered(_task_dev_episode, screen_jobs):
                base_rows.append(row)
                for n, c in row["calls"].items():
                    calls[n] = calls.get(n, 0) + c
        unused = sorted(n for n, c in calls.items() if c == 0)
        used = sorted(n for n, c in calls.items() if c > 0)
        base_agg = _aggregate_dev_results(base_rows)
        dt = time.perf_counter() - t0
        print(
            f"dev screen ({dt:.1f}s): baseline gen={base_agg['generality_score']:.4f} "
            f"sr={base_agg['solve_rate']:.4f} deaths={base_agg['deaths']} | "
            f"{len(used)} used, {len(unused)} unused"
        )
        screen_path.write_text(
            json.dumps(
                {
                    "n_trials": n_slots,
                    "baseline": base_agg,
                    "calls": calls,
                    "used": used,
                    "unused": unused,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.phase == "screen":
        return

    targets = args.faculties if args.faculties else used
    targets = [t for t in targets if t in inv or t in _ABLATION_BUNDLES]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t_abl = time.perf_counter()
    if not getattr(args, "cross_seed_only", False):
        base_jobs = [
            ("__baseline__", tid, fam, epath, s, mb, False) for tid, fam, epath, s, mb in dev_slots
        ]
        fac_jobs = [
            (fac, tid, fam, epath, s, mb, False)
            for fac in targets
            for tid, fam, epath, s, mb in dev_slots
        ]
        with mp_ctx.Pool(args.workers) as pool:
            for row in pool.imap_unordered(_task_dev_episode, base_jobs + fac_jobs):
                grouped[row["faculty"]].append(row)

    xs_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if getattr(args, "with_cross_seed", False) or getattr(args, "cross_seed_only", False):
        xs_jobs = [
            (
                fac,
                t["id"],
                t["family"],
                t["env_class"],
                [int(s) for s in t["seeds"]],
                int(t.get("max_budget", 100)),
            )
            for fac in ["__baseline__", *targets]
            for t in tasks_def
        ]
        with mp_ctx.Pool(args.workers) as pool:
            for row in pool.imap_unordered(_task_dev_cross_seed, xs_jobs):
                xs_grouped[row["faculty"]].append(row)
    dt_abl = time.perf_counter() - t_abl

    def _agg_xs(rows: list[dict[str, Any]]) -> dict[str, Any]:
        tot = sum(r["transfer_total"] for r in rows)
        suc = sum(r["transfer_succ"] for r in rows)
        dth = sum(r["transfer_deaths"] for r in rows)
        crashes = [r["crash"] for r in rows if r.get("crash")]
        return {
            "transfer_solve_rate": round(suc / max(1, tot), 4),
            "transfer_deaths": dth,
            "n_transfer": tot,
            "crash": crashes[0] if crashes else None,
        }

    base_agg = (
        _aggregate_dev_results(grouped["__baseline__"])
        if grouped.get("__baseline__")
        else {
            "generality_score": 0.0,
            "solve_rate": 0.0,
            "deaths": 0,
            "mean_steps": 1.0,
            "mean_p50_ms": 0.0,
            "family_solve_rates": {},
            "crash": None,
        }
    )
    base_xs = _agg_xs(xs_grouped["__baseline__"]) if xs_grouped.get("__baseline__") else None
    report: dict[str, Any] = {
        "suite": suite_name,
        "n_trials": n_slots,
        "wall_seconds": round(dt_abl, 1),
        "baseline": base_agg,
        "baseline_cross_seed": base_xs,
        "faculties": {},
    }
    eval_names = list(inv) + [t for t in targets if t not in inv]
    for name in eval_names:
        if name in unused and name not in targets:
            report["faculties"][name] = {
                "class": inv[name],
                "calls": 0,
                "decision": "unused",
                "delta_generality": 0.0,
                "delta_solve_rate": 0.0,
                "delta_deaths": 0,
                "evidence": f"0 calls across n={n_slots} R1 dev trials",
            }
            continue
        if name not in grouped and name not in xs_grouped:
            continue
        agg = (
            _aggregate_dev_results(grouped[name])
            if name in grouped
            else {
                "generality_score": 0.0,
                "solve_rate": 0.0,
                "deaths": 0,
                "mean_steps": 1.0,
                "mean_p50_ms": 0.0,
                "family_solve_rates": {},
                "crash": None,
            }
        )
        xs = _agg_xs(xs_grouped[name]) if name in xs_grouped else None
        d_gen = round(base_agg["generality_score"] - agg["generality_score"], 4)
        d_sr = round(base_agg["solve_rate"] - agg["solve_rate"], 4)
        d_deaths = agg["deaths"] - base_agg["deaths"]  # positive => ablation caused more deaths
        d_p50 = round(agg["mean_p50_ms"] - base_agg["mean_p50_ms"], 2)
        d_steps_pct = round(
            (agg["mean_steps"] - base_agg["mean_steps"])
            / max(1e-9, base_agg["mean_steps"])
            * 100.0,
            1,
        )
        d_xs = (
            round(base_xs["transfer_solve_rate"] - xs["transfer_solve_rate"], 4)
            if base_xs and xs
            else 0.0
        )
        xs_crash = xs.get("crash") if xs else None
        if agg["crash"] or xs_crash:
            decision = "keep"
            why = f"load_bearing_crash: {agg['crash'] or xs_crash}"
        elif d_gen > 0.005 or d_sr > 0.0 or d_deaths > 0 or d_xs > 0.0:
            decision = "keep"
            why = f"Δgen={d_gen:+.4f}, Δsr={d_sr:+.4f}, Δxs={d_xs:+.4f}, Δdeaths={d_deaths:+d}"
        else:
            decision = "experimental"
            why = f"no measurable positive R1 effect (Δgen={d_gen:+.4f}, Δsr={d_sr:+.4f}, Δxs={d_xs:+.4f}, Δdeaths={d_deaths:+d})"
        cls_label = inv.get(name, f"Bundle({len(_ABLATION_BUNDLES.get(name, ()))})")
        report["faculties"][name] = {
            "class": cls_label,
            "calls": calls.get(name, 0),
            "baseline_generality": base_agg["generality_score"],
            "ablated_generality": agg["generality_score"],
            "delta_generality": d_gen,
            "baseline_solve_rate": base_agg["solve_rate"],
            "ablated_solve_rate": agg["solve_rate"],
            "delta_solve_rate": d_sr,
            "baseline_transfer_solve_rate": base_xs["transfer_solve_rate"] if base_xs else None,
            "ablated_transfer_solve_rate": xs["transfer_solve_rate"] if xs else None,
            "delta_transfer_solve_rate": d_xs if xs else None,
            "baseline_deaths": base_agg["deaths"],
            "ablated_deaths": agg["deaths"],
            "delta_deaths": d_deaths,
            "delta_steps_pct": d_steps_pct,
            "delta_p50_ms": d_p50,
            "ablated_family_solve_rates": agg["family_solve_rates"],
            "decision": decision,
            "rule_outcome": why,
        }
        print(f"  [{decision:12s}] {name:28s} {why} (Δp50={d_p50:+.1f}ms)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"reports/ablation/{ts}_r1_{suite_name}.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"R1 {suite_name} ablation report: {out_path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Faculty ablation harness on procedural environments")
    ap.add_argument("--suite", choices=["legacy", "dev", "widened_dev"], default="legacy")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--phase", choices=["inventory", "screen", "ablate", "all"], default="all")
    ap.add_argument("--screen-report", default=None)
    ap.add_argument("--faculties", nargs="*", default=None)
    ap.add_argument("--with-cross-seed", action="store_true")
    ap.add_argument("--cross-seed-only", action="store_true")
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
    if args.suite in ("dev", "widened_dev"):
        _run_dev_ablation(args, inv, mp_ctx)
        return

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
