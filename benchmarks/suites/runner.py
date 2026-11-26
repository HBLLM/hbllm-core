"""Test suite runner for dev and held-out benchmark splits.

Executes true closed-loop evaluation across ALL seeds in each task manifest:
1. Primary headline: Zero-shot across all manifest seeds (fresh CognitiveSystem per seed)
   with exact Wilson 95% confidence intervals and exploration-death counting.
2. Secondary headline: Cross-seed procedural transfer across all manifest seeds with one
   shared CognitiveSystem (`same_level=False` boundaries) and Wilson 95% intervals.
3. Auxiliary metrics: Same-layout replay (`same_layout_replay`, legacy `lifelong`) and
   one-shot imitation (`demonstration`).
4. Zero swallowed exceptions: any engine/boundary/terminal exception is recorded in `errors`
   and marks the episode as failed.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure core root is on sys.path
_CORE_ROOT = Path(__file__).resolve().parents[2]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

import yaml

from hbllm.benchmarks.protocol import EvaluationProtocol, TaskResult, wilson_ci
from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind, TerminalSignal
from hbllm.runtime.bootstrap import (
    CognitiveSystemConfig,
    build_cognitive_system,
    count_consolidated_skills,
)

_DEATH_REASONS = frozenset({"hazard_collision", "died", "lava", "trap", "failure", "fatal"})


def _load_env_class(class_path: str) -> type:
    """Dynamically import environment class from module string."""
    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _is_death_outcome(success: bool, terminal_obs: Any, info: dict[str, Any]) -> bool:
    if success:
        return False
    reason = str(info.get("reason", "")).lower()
    if reason in _DEATH_REASONS:
        return True
    return getattr(terminal_obs, "terminal", None) == TerminalSignal.FAILURE


_EXPLORATORY_DEATH_BRANCHES = frozenset(
    {"epistemic_probe", "affordance_choice", "explore", "exploration"}
)


def _is_exploration_death(
    engine: Any | None,
    pre_obs: Any,
    action: Any,
    meta: dict[str, Any] | None,
    died: bool,
    baseline: str,
    *,
    pre_uncalibrated: bool = False,
) -> bool:
    """True when a terminal death occurred during exploration rather than policy exploitation."""
    if not died:
        return False
    if baseline in (
        "fatal_memory_bfs",
        "dyna_q",
        "random",
        "rules_known_planner",
        "object_centric_hypothesis",
    ):
        return True
    if engine is None:
        return baseline != "oracle"
    branch = str((meta or {}).get("branch", "")).lower()
    if branch in _EXPLORATORY_DEATH_BRANCHES or pre_uncalibrated:
        return True
    grid = getattr(pre_obs, "visual_lattice", None)
    if grid is None:
        grid = getattr(pre_obs, "grid", None)
    action_dynamics = getattr(engine, "action_dynamics", None)
    avatar_pos = getattr(engine, "avatar_pos", None)
    hazard_tracker = getattr(engine, "hazard_tracker", None)
    if (
        grid is not None
        and action_dynamics is not None
        and avatar_pos is not None
        and hazard_tracker is not None
    ):
        dyn = action_dynamics.get(action)
        if dyn is not None:
            nr = int(avatar_pos[0]) + int(dyn.delta_r)
            nc = int(avatar_pos[1]) + int(dyn.delta_c)
            H, W = len(grid), len(grid[0]) if len(grid) > 0 else 0
            if 0 <= nr < H and 0 <= nc < W:
                cell_feat = int(grid[nr][nc])
                known_lethal = getattr(hazard_tracker, "known_lethal_features", set())
                if cell_feat in known_lethal:
                    return False
                is_unverified = getattr(engine, "is_feature_unverified", None)
                bg_feature = getattr(engine, "bg_feature", 0)
                if callable(is_unverified):
                    return bool(is_unverified(cell_feat, bg_feature))
    return branch in {"macro_plan", "mental_plan", "mental_simulation"}


def run_closed_loop_task(
    env_cls: type,
    task_def: dict[str, Any],
    max_budget: int,
    baseline: str = "system",
    perception_mode: str = "handcrafted",
    episodes: int = 5,
    shared_memory: bool = True,
    cross_seed: bool = True,
) -> tuple[bool, int, list[float], list[str], dict[str, Any]]:
    """Execute a task across ALL manifest seeds for zero-shot and cross-seed transfer,
    plus N-episode same-layout replay on the primary seed.

    Returns:
        (overall_success, steps, latencies_ms, errors, extra)
    """
    seeds = [int(s) for s in task_def.get("seeds", [0])] or [0]
    primary_seed = seeds[0]
    task_id = task_def.get("id", f"task_{primary_seed}")
    episodes = max(1, int(episodes))

    episode_successes: list[bool] = []
    episode_steps: list[int] = []
    all_latencies: list[float] = []
    all_errors: list[str] = []

    zs_seeds: list[int] = []
    zs_successes: list[bool] = []
    zs_steps: list[int] = []
    zs_deaths = 0
    zs_exploration_deaths = 0
    replay_deaths = 0
    replay_exploration_deaths = 0

    def _make_system(tmp_dir: str, tag: str, grounding: bool):
        config = CognitiveSystemConfig(
            world_id=f"bench_{task_id}_{tag}",
            memory_dir=tmp_dir,
            enable_grounding=grounding,
            perception_mode=perception_mode,
        )
        return build_cognitive_system(config)

    # 1. Same-layout replay over `episodes` on `seeds[0]` (ep 0 doubles as seed 0's zero-shot)
    def _run_primary_replay(use_engine: bool) -> None:
        nonlocal zs_deaths, zs_exploration_deaths, replay_deaths, replay_exploration_deaths
        with tempfile.TemporaryDirectory() as tmp_dir:
            system = None
            engine = None
            if baseline == "fatal_memory_bfs":
                from benchmarks.baselines import FatalMemoryBFSAgent

                engine = FatalMemoryBFSAgent(retain_fatal_memory=shared_memory)
            elif baseline == "dyna_q":
                from benchmarks.baselines import TabularDynaQAgent

                engine = TabularDynaQAgent()
            elif baseline == "rules_known_planner":
                from benchmarks.baselines import RulesKnownPlannerAgent

                engine = RulesKnownPlannerAgent(retain_fatal_memory=shared_memory)
            elif baseline == "object_centric_hypothesis":
                from benchmarks.baselines import ObjectCentricHypothesisAgent

                engine = ObjectCentricHypothesisAgent()
            elif use_engine and shared_memory:
                system = _make_system(
                    tmp_dir, f"{baseline}_s{primary_seed}", baseline in ("system", "demonstration")
                )
                engine = system.engine

            prev_succ = False
            for ep_idx in range(episodes):
                ep_errors: list[str] = []
                ep_death_info: dict[str, bool] = {}
                if baseline in (
                    "fatal_memory_bfs",
                    "dyna_q",
                    "rules_known_planner",
                    "object_centric_hypothesis",
                ):
                    if ep_idx > 0 and hasattr(engine, "reset_episode"):
                        engine.reset_episode(reset_memory=not shared_memory)
                    env_instance = env_cls(max_steps=max_budget)
                    obs = env_instance.reset(seed=primary_seed)
                    succ, steps, lats, errs, died = _run_loop_with_info(
                        env_instance,
                        obs,
                        engine,
                        baseline,
                        max_budget,
                        [],
                        ep_errors,
                        seed=primary_seed,
                        death_info=ep_death_info,
                    )
                    exp_died = bool(ep_death_info.get("exploration_death", died))
                elif use_engine and not shared_memory:
                    with tempfile.TemporaryDirectory() as ep_dir:
                        system = _make_system(
                            ep_dir,
                            f"{baseline}_s{primary_seed}_ep{ep_idx}",
                            baseline in ("system", "demonstration"),
                        )
                        engine = system.engine
                        env_instance = env_cls(max_steps=max_budget)
                        obs = env_instance.reset(seed=primary_seed)
                        ep_base = (
                            ("demonstration_oracle" if ep_idx == 0 else "system")
                            if baseline == "demonstration"
                            else baseline
                        )
                        succ, steps, lats, errs, died = _run_loop_with_info(
                            env_instance,
                            obs,
                            engine,
                            ep_base,
                            max_budget,
                            [],
                            ep_errors,
                            seed=primary_seed,
                            death_info=ep_death_info,
                        )
                        exp_died = bool(ep_death_info.get("exploration_death", died))
                else:
                    if engine is not None and ep_idx > 0:
                        b_kind = (
                            EpisodeBoundaryKind.LEVEL_COMPLETE
                            if prev_succ
                            else EpisodeBoundaryKind.DEATH
                        )
                        try:
                            engine.on_boundary(
                                EpisodeBoundary(kind=b_kind, same_level=shared_memory)
                            )
                        except Exception as e:
                            ep_errors.append(
                                f"Boundary error at seed {primary_seed} episode {ep_idx}: {e}"
                            )
                    env_instance = (
                        env_cls(max_budget=max_budget) if False else env_cls(max_steps=max_budget)
                    )
                    obs = env_instance.reset(seed=primary_seed)
                    ep_base = (
                        ("demonstration_oracle" if ep_idx == 0 else "system")
                        if baseline == "demonstration"
                        else baseline
                    )
                    if ep_errors:
                        succ, steps, lats, errs, died = False, 0, [], ep_errors, False
                        exp_died = False
                    else:
                        ep_death_info = {}
                        succ, steps, lats, errs, died = _run_loop_with_info(
                            env_instance,
                            obs,
                            engine,
                            ep_base,
                            max_budget,
                            [],
                            ep_errors,
                            seed=primary_seed,
                            death_info=ep_death_info,
                        )
                        exp_died = bool(ep_death_info.get("exploration_death", died))

                if errs:
                    succ = False
                episode_successes.append(succ)
                episode_steps.append(steps)
                all_latencies.extend(lats)
                all_errors.extend(errs)
                prev_succ = succ
                if died:
                    replay_deaths += 1
                    if exp_died:
                        replay_exploration_deaths += 1

                if ep_idx == 0 and baseline != "demonstration":
                    zs_seeds.append(primary_seed)
                    zs_successes.append(succ)
                    zs_steps.append(steps)
                    if died:
                        zs_deaths += 1
                        if exp_died:
                            zs_exploration_deaths += 1

    _run_primary_replay(
        use_engine=(
            baseline
            in (
                "system",
                "no_grounding",
                "demonstration",
                "fatal_memory_bfs",
                "dyna_q",
                "rules_known_planner",
                "object_centric_hypothesis",
            )
        )
    )

    # 2. Independent zero-shot evaluation across ALL remaining seeds (seeds[1:])
    if baseline != "demonstration" and len(seeds) > 1:
        for s in seeds[1:]:
            seed_errs: list[str] = []
            seed_death_info: dict[str, bool] = {}
            if baseline in ("system", "no_grounding"):
                with tempfile.TemporaryDirectory() as s_dir:
                    s_sys = _make_system(s_dir, f"{baseline}_zs_s{s}", baseline == "system")
                    s_env = env_cls(max_steps=max_budget)
                    s_obs = s_env.reset(seed=s)
                    succ, steps, lats, errs, died = _run_loop_with_info(
                        s_env,
                        s_obs,
                        s_sys.engine,
                        baseline,
                        max_budget,
                        [],
                        seed_errs,
                        seed=s,
                        death_info=seed_death_info,
                    )
            elif baseline == "fatal_memory_bfs":
                from benchmarks.baselines import FatalMemoryBFSAgent

                s_agent = FatalMemoryBFSAgent(retain_fatal_memory=False)
                s_env = env_cls(max_steps=max_budget)
                s_obs = s_env.reset(seed=s)
                succ, steps, lats, errs, died = _run_loop_with_info(
                    s_env,
                    s_obs,
                    s_agent,
                    baseline,
                    max_budget,
                    [],
                    seed_errs,
                    seed=s,
                    death_info=seed_death_info,
                )
            elif baseline == "dyna_q":
                from benchmarks.baselines import TabularDynaQAgent

                s_agent = TabularDynaQAgent()
                s_env = env_cls(max_steps=max_budget)
                s_obs = s_env.reset(seed=s)
                succ, steps, lats, errs, died = _run_loop_with_info(
                    s_env,
                    s_obs,
                    s_agent,
                    baseline,
                    max_budget,
                    [],
                    seed_errs,
                    seed=s,
                    death_info=seed_death_info,
                )
            elif baseline == "rules_known_planner":
                from benchmarks.baselines import RulesKnownPlannerAgent

                s_agent = RulesKnownPlannerAgent(retain_fatal_memory=False)
                s_env = env_cls(max_steps=max_budget)
                s_obs = s_env.reset(seed=s)
                succ, steps, lats, errs, died = _run_loop_with_info(
                    s_env,
                    s_obs,
                    s_agent,
                    baseline,
                    max_budget,
                    [],
                    seed_errs,
                    seed=s,
                    death_info=seed_death_info,
                )
            elif baseline == "object_centric_hypothesis":
                from benchmarks.baselines import ObjectCentricHypothesisAgent

                s_agent = ObjectCentricHypothesisAgent()
                s_env = env_cls(max_steps=max_budget)
                s_obs = s_env.reset(seed=s)
                succ, steps, lats, errs, died = _run_loop_with_info(
                    s_env,
                    s_obs,
                    s_agent,
                    baseline,
                    max_budget,
                    [],
                    seed_errs,
                    seed=s,
                    death_info=seed_death_info,
                )
            else:
                s_env = env_cls(max_steps=max_budget)
                s_obs = s_env.reset(seed=s)
                succ, steps, lats, errs, died = _run_loop_with_info(
                    s_env,
                    s_obs,
                    None,
                    baseline,
                    max_budget,
                    [],
                    seed_errs,
                    seed=s,
                    death_info=seed_death_info,
                )
            if errs:
                succ = False
            zs_seeds.append(s)
            zs_successes.append(succ)
            zs_steps.append(steps)
            if died:
                zs_deaths += 1
                if seed_death_info.get("exploration_death", died):
                    zs_exploration_deaths += 1
            all_latencies.extend(lats)
            all_errors.extend(errs)
    elif baseline == "demonstration":
        # Demonstration mode reports episode 1 (post-oracle imitation)
        im_ok = episode_successes[1] if len(episode_successes) > 1 else False
        im_st = episode_steps[1] if len(episode_steps) > 1 else max_budget
        zs_seeds.append(primary_seed)
        zs_successes.append(im_ok)
        zs_steps.append(im_st)

    # Compute stable policy metric on same-layout replay
    stable_idx = None
    for i in range(len(episode_successes) - 2):
        if episode_successes[i] and episode_successes[i + 1] and episode_successes[i + 2]:
            stable_idx = i
            break

    first_succ_steps = None
    for i, s_ok in enumerate(episode_successes):
        if s_ok:
            first_succ_steps = episode_steps[i]
            break

    replay_final_success = bool(episode_successes[-1]) if episode_successes else False
    replay_final_steps = (
        episode_steps[-1]
        if (episode_steps and replay_final_success)
        else int(sum(episode_steps) / max(1, len(episode_steps)))
    )

    # 3. One-shot imitation on primary seed
    one_shot_imitation: dict[str, Any] = {}
    if baseline == "demonstration":
        one_shot_imitation = {
            "success": episode_successes[1] if len(episode_successes) > 1 else False,
            "steps": episode_steps[1] if len(episode_steps) > 1 else max_budget,
            "oracle_success": episode_successes[0] if episode_successes else False,
            "oracle_steps": episode_steps[0] if episode_steps else max_budget,
        }
    elif baseline == "system" and callable(getattr(env_cls, "get_oracle_action", None)):
        with tempfile.TemporaryDirectory() as tmp_dir:
            demo_sys = _make_system(tmp_dir, "demo", True)
            demo_engine = demo_sys.engine
            demo_env = env_cls(max_steps=max_budget)
            d_obs = demo_env.reset(seed=primary_seed)
            d_errs: list[str] = []
            d_succ, d_steps, _, _, _ = _run_loop_with_info(
                demo_env,
                d_obs,
                demo_engine,
                "demonstration_oracle",
                max_budget,
                [],
                d_errs,
                seed=primary_seed,
            )
            try:
                demo_engine.on_boundary(
                    EpisodeBoundary(
                        kind=(
                            EpisodeBoundaryKind.LEVEL_COMPLETE
                            if d_succ
                            else EpisodeBoundaryKind.DEATH
                        )
                    )
                )
            except Exception as e:
                d_errs.append(f"Demo boundary error: {e}")
            d_obs = demo_env.reset(seed=primary_seed)
            if d_errs:
                i_succ, i_steps = False, max_budget
            else:
                i_succ, i_steps, _, _, _ = _run_loop_with_info(
                    demo_env,
                    d_obs,
                    demo_engine,
                    "system",
                    max_budget,
                    [],
                    d_errs,
                    seed=primary_seed,
                )
                if d_errs:
                    i_succ = False
            all_errors.extend(d_errs)
            one_shot_imitation = {
                "success": i_succ,
                "steps": i_steps,
                "oracle_success": d_succ,
                "oracle_steps": d_steps,
            }

    # 4. Cross-seed procedural transfer across ALL distinct seeds in the manifest
    cross_seed_transfer: dict[str, Any] = {}
    if (
        cross_seed
        and baseline
        in (
            "system",
            "no_grounding",
            "fatal_memory_bfs",
            "dyna_q",
            "rules_known_planner",
            "object_centric_hypothesis",
        )
        and len(seeds) > 1
    ):
        xs_seeds = list(seeds)
        xs_succ: list[bool] = []
        xs_steps: list[int] = []
        xs_deaths = 0
        xs_exploration_deaths = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            xs_sys = None
            if baseline in ("system", "no_grounding"):
                xs_sys = _make_system(tmp_dir, f"xseed_{baseline}", baseline == "system")
                xs_engine = xs_sys.engine
            elif baseline == "fatal_memory_bfs":
                from benchmarks.baselines import FatalMemoryBFSAgent

                xs_engine = FatalMemoryBFSAgent(retain_fatal_memory=True)
            elif baseline == "dyna_q":
                from benchmarks.baselines import TabularDynaQAgent

                xs_engine = TabularDynaQAgent()
            elif baseline == "rules_known_planner":
                from benchmarks.baselines import RulesKnownPlannerAgent

                xs_engine = RulesKnownPlannerAgent(retain_fatal_memory=True)
            elif baseline == "object_centric_hypothesis":
                from benchmarks.baselines import ObjectCentricHypothesisAgent

                xs_engine = ObjectCentricHypothesisAgent()
            prev_ok = False
            for ep_idx, ep_seed in enumerate(xs_seeds):
                xs_errs: list[str] = []
                if ep_idx > 0:
                    if hasattr(xs_engine, "on_boundary"):
                        try:
                            xs_engine.on_boundary(
                                EpisodeBoundary(
                                    kind=(
                                        EpisodeBoundaryKind.LEVEL_COMPLETE
                                        if prev_ok
                                        else EpisodeBoundaryKind.DEATH
                                    ),
                                    same_level=False,
                                )
                            )
                        except Exception as e:
                            xs_errs.append(f"Cross-seed boundary error at seed {ep_seed}: {e}")
                    elif hasattr(xs_engine, "reset_episode"):
                        xs_engine.reset_episode(reset_memory=False)
                xs_env = env_cls(max_steps=max_budget)
                xs_obs = xs_env.reset(seed=ep_seed)
                xs_death_info: dict[str, bool] = {}
                if xs_errs:
                    ok, st, died = False, max_budget, False
                else:
                    ok, st, _, xs_errs, died = _run_loop_with_info(
                        xs_env,
                        xs_obs,
                        xs_engine,
                        baseline,
                        max_budget,
                        [],
                        xs_errs,
                        seed=ep_seed,
                        death_info=xs_death_info,
                    )
                    if xs_errs:
                        ok = False
                if died:
                    xs_deaths += 1
                    if xs_death_info.get("exploration_death", died):
                        xs_exploration_deaths += 1
                all_errors.extend(xs_errs)
                xs_succ.append(ok)
                xs_steps.append(st)
                prev_ok = ok
            if hasattr(xs_engine, "on_boundary"):
                try:
                    xs_engine.on_boundary(
                        EpisodeBoundary(
                            kind=(
                                EpisodeBoundaryKind.LEVEL_COMPLETE
                                if prev_ok
                                else EpisodeBoundaryKind.DEATH
                            ),
                            same_level=False,
                        )
                    )
                except Exception as e:
                    all_errors.append(f"Cross-seed final boundary error: {e}")
            skills = count_consolidated_skills(xs_sys if xs_sys is not None else xs_engine)
        n_xs = len(xs_succ)
        k_xs = sum(xs_succ)
        k_tr = sum(xs_succ[1:])
        n_tr = max(0, n_xs - 1)
        cross_seed_transfer = {
            "seeds": xs_seeds,
            "successes": xs_succ,
            "steps": xs_steps,
            "solve_rate": round(k_xs / max(1, n_xs), 4),
            "solve_rate_wilson_95_ci": list(wilson_ci(k_xs, n_xs)),
            "transfer_solve_rate": round(k_tr / max(1, n_tr), 4) if n_tr > 0 else 0.0,
            "transfer_wilson_95_ci": list(wilson_ci(k_tr, n_tr)) if n_tr > 0 else [0.0, 0.0],
            "total_deaths": xs_deaths,
            "exploration_deaths": xs_exploration_deaths,
            "skills_consolidated": skills,
        }

    n_zs = len(zs_successes)
    k_zs = sum(zs_successes)
    zs_solve_rate = round(k_zs / max(1, n_zs), 4)
    zs_ci = list(wilson_ci(k_zs, n_zs))
    zs_mean_steps = int(round(sum(zs_steps) / max(1, n_zs)))

    replay_payload = {
        "seed": primary_seed,
        "success": replay_final_success,
        "final_steps": replay_final_steps,
        "episodes": episodes,
        "successes": episode_successes,
        "steps": episode_steps,
        "steps_to_stable_policy": stable_idx,
        "total_deaths": replay_deaths,
        "exploration_deaths": replay_exploration_deaths,
    }

    extra = {
        "seed_count": len(seeds),
        "total_deaths": zs_deaths,
        "exploration_deaths": zs_exploration_deaths,
        "episode_successes": episode_successes,
        "episode_steps": episode_steps,
        "steps_to_stable_policy": stable_idx,
        "steps_to_first_success": first_succ_steps,
        "zero_shot": {
            "n": n_zs,
            "seeds": zs_seeds,
            "successes": zs_successes,
            "steps_per_seed": zs_steps,
            "solve_rate": zs_solve_rate,
            "wilson_95_ci": zs_ci,
            "total_deaths": zs_deaths,
            "death_rate": round(zs_deaths / max(1, n_zs), 4),
            "exploration_deaths": zs_exploration_deaths,
            "exploration_death_rate": round(zs_exploration_deaths / max(1, n_zs), 4),
            "success": zs_successes[0] if n_zs == 1 else (k_zs * 2 >= n_zs and k_zs > 0),
            "steps": zs_steps[0] if n_zs == 1 else zs_mean_steps,
        },
        "same_layout_replay": replay_payload,
        "lifelong": replay_payload,
        "one_shot_imitation": one_shot_imitation,
        "cross_seed_transfer": cross_seed_transfer,
    }

    overall_success = replay_final_success and not all_errors
    return overall_success, replay_final_steps, all_latencies, all_errors, extra


def run_closed_loop_episode(
    env: Any,
    seed: int,
    max_budget: int,
    baseline: str = "system",
    perception_mode: str = "handcrafted",
) -> tuple[bool, int, list[float], list[str]]:
    """Execute a single episode in closed loop.

    Returns:
        (success, steps, latencies_ms, errors)
    """
    latencies: list[float] = []
    errors: list[str] = []
    obs = env.reset(seed=seed)

    if baseline in ("system", "no_grounding"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = CognitiveSystemConfig(
                world_id=f"bench_{seed}",
                memory_dir=tmp_dir,
                enable_grounding=(baseline == "system"),
                perception_mode=perception_mode,
            )
            system = build_cognitive_system(config)
            return _run_loop(
                env, obs, system.engine, baseline, max_budget, latencies, errors, seed=seed
            )
    elif baseline == "fatal_memory_bfs":
        from benchmarks.baselines import FatalMemoryBFSAgent

        agent = FatalMemoryBFSAgent(retain_fatal_memory=False)
        return _run_loop(env, obs, agent, baseline, max_budget, latencies, errors, seed=seed)
    elif baseline == "dyna_q":
        from benchmarks.baselines import TabularDynaQAgent

        agent = TabularDynaQAgent()
        return _run_loop(env, obs, agent, baseline, max_budget, latencies, errors, seed=seed)
    elif baseline == "rules_known_planner":
        from benchmarks.baselines import RulesKnownPlannerAgent

        agent = RulesKnownPlannerAgent(retain_fatal_memory=False)
        return _run_loop(env, obs, agent, baseline, max_budget, latencies, errors, seed=seed)
    elif baseline == "object_centric_hypothesis":
        from benchmarks.baselines import ObjectCentricHypothesisAgent

        agent = ObjectCentricHypothesisAgent()
        return _run_loop(env, obs, agent, baseline, max_budget, latencies, errors, seed=seed)

    return _run_loop(env, obs, None, baseline, max_budget, latencies, errors, seed=seed)


def _run_loop(
    env: Any,
    obs: Any,
    engine: Any | None,
    baseline: str,
    max_budget: int,
    latencies: list[float],
    errors: list[str],
    seed: int = 0,
) -> tuple[bool, int, list[float], list[str]]:
    succ, step_count, lats, errs, _died = _run_loop_with_info(
        env, obs, engine, baseline, max_budget, latencies, errors, seed=seed
    )
    return succ, step_count, lats, errs


def _run_loop_with_info(
    env: Any,
    obs: Any,
    engine: Any | None,
    baseline: str,
    max_budget: int,
    latencies: list[float],
    errors: list[str],
    seed: int = 0,
    death_info: dict[str, bool] | None = None,
) -> tuple[bool, int, list[float], list[str], bool]:
    success = False
    died = False
    step_count = 0
    rng = random.Random(seed)
    step_journal: list[dict[str, Any]] = []
    last_oracle_action: Any | None = None

    for step in range(max_budget):
        step_count = step + 1
        t0 = time.perf_counter()

        action = None
        meta = None
        try:
            if baseline == "stub_agent":
                action = 1
            elif baseline == "random":
                avail = getattr(obs, "available_actions", [1, 2, 3, 4])
                action = rng.choice(list(avail))
            elif baseline == "oracle":
                action = env.get_oracle_action()
            elif baseline == "fatal_memory_bfs":
                if engine is None or not hasattr(engine, "step"):
                    from benchmarks.baselines import FatalMemoryBFSAgent

                    engine = FatalMemoryBFSAgent(retain_fatal_memory=True)
                action = engine.step(obs)
            elif baseline == "dyna_q":
                if engine is None or not hasattr(engine, "step"):
                    from benchmarks.baselines import TabularDynaQAgent

                    engine = TabularDynaQAgent()
                action = engine.step(obs)
            elif baseline == "rules_known_planner":
                if engine is None or not hasattr(engine, "step"):
                    from benchmarks.baselines import RulesKnownPlannerAgent

                    engine = RulesKnownPlannerAgent(retain_fatal_memory=True)
                action = engine.step(obs)
            elif baseline == "object_centric_hypothesis":
                if engine is None or not hasattr(engine, "step"):
                    from benchmarks.baselines import ObjectCentricHypothesisAgent

                    engine = ObjectCentricHypothesisAgent()
                action = engine.step(obs)
            elif baseline == "demonstration_oracle":
                if engine is not None:
                    raw_action = engine.step(obs, last_action=last_oracle_action)
                    if isinstance(raw_action, tuple) and len(raw_action) > 1:
                        meta = raw_action[1]
                action = env.get_oracle_action()
                last_oracle_action = action
            else:
                assert engine is not None
                raw_action = engine.step(obs)
                if isinstance(raw_action, tuple) and len(raw_action) > 1:
                    action, meta = raw_action[0], raw_action[1]
                else:
                    action = raw_action
        except Exception as e:
            errors.append(f"Action selection error at step {step_count}: {e}")
            success = False
            break

        step_journal.append({"step": step_count, "action": action, "meta": meta})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        pre_obs = obs
        calib_fn = getattr(engine, "is_action_calibrated", None)
        pre_uncalibrated = (
            (not calib_fn(action)) if (calib_fn is not None and callable(calib_fn)) else False
        )

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            errors.append(f"Environment step error at step {step_count}: {e}")
            success = False
            break

        if done:
            success = bool(info.get("success", False))
            died = _is_death_outcome(success, obs, info)
            if death_info is not None:
                death_info["died"] = died
                death_info["exploration_death"] = _is_exploration_death(
                    engine,
                    pre_obs,
                    action,
                    meta,
                    died,
                    baseline,
                    pre_uncalibrated=pre_uncalibrated,
                )
            if hasattr(engine, "record_transition") and baseline in (
                "fatal_memory_bfs",
                "rules_known_planner",
                "object_centric_hypothesis",
            ):
                engine.record_transition(pre_obs, action, obs, died, info)
            elif baseline == "dyna_q" and hasattr(engine, "record_transition"):
                engine.record_transition(pre_obs, action, obs, reward, done)
            if engine is not None and baseline in (
                "system",
                "no_grounding",
                "demonstration_oracle",
            ):
                try:
                    engine.step(
                        obs,
                        last_action=(
                            last_oracle_action if baseline == "demonstration_oracle" else None
                        ),
                    )
                except Exception as e:
                    errors.append(f"Terminal feedback error at step {step_count}: {e}")
                    success = False
            break
        elif hasattr(engine, "record_transition") and baseline in (
            "fatal_memory_bfs",
            "rules_known_planner",
            "object_centric_hypothesis",
        ):
            engine.record_transition(pre_obs, action, obs, False, info)
        elif baseline == "dyna_q" and hasattr(engine, "record_transition"):
            engine.record_transition(pre_obs, action, obs, reward, False)

    return success, step_count, latencies, errors, died


def run_suite(
    split: str = "dev",
    reports_dir: str = "reports",
    perception_mode: str = "handcrafted",
    baseline: str = "system",
    manifest_path: str | None = None,
    episodes_per_task: int = 5,
    shared_memory: bool = True,
    cross_seed: bool = True,
) -> int:
    """Executes the closed-loop benchmark evaluation for the given split."""
    manifest_file = (
        Path(manifest_path) if manifest_path else Path(__file__).parent / f"{split}.yaml"
    )

    if not manifest_file.exists():
        print(f"Error: Manifest file {manifest_file} not found.")
        return 1

    with open(manifest_file, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    protocol = EvaluationProtocol(
        reports_dir=reports_dir,
        perception_mode=perception_mode,
        config={
            "model": "deterministic_hcir",
            "version": "2.0.0",
            "perception_mode": perception_mode,
            "baseline": baseline,
            "split": split,
            "episodes_per_task": episodes_per_task,
            "shared_memory": shared_memory,
        },
    )

    if split == "heldout" and baseline == "system":
        try:
            protocol.verify_heldout_gate()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2

    print(
        f"=== Running HBLLM Generality Suite [{split.upper()}] (baseline={baseline}, perception={perception_mode}, episodes={episodes_per_task}, shared_memory={shared_memory}) ==="
    )
    print(f"Git SHA: {protocol.git_sha} | Config Hash: {protocol.config_hash}")

    tasks: list[TaskResult] = []

    for task_def in manifest.get("tasks", []):
        task_id = task_def["id"]
        family = task_def["family"]
        env_cls = _load_env_class(task_def["env_class"])
        max_budget = task_def.get("max_budget", 100)
        n_seeds = len(task_def.get("seeds", [0]))

        print(
            f"-> Evaluating task [{task_id}] ({family}, seeds={n_seeds}, replay_eps={episodes_per_task}, budget={max_budget})..."
        )

        def make_task_runner(e_cls: type, t_def: dict[str, Any], t_budget: int):
            def run_task():
                return run_closed_loop_task(
                    env_cls=e_cls,
                    task_def=t_def,
                    max_budget=t_budget,
                    baseline=baseline,
                    perception_mode=perception_mode,
                    episodes=episodes_per_task,
                    shared_memory=shared_memory,
                    cross_seed=cross_seed,
                )

            return run_task

        t_res = protocol.evaluate_task(
            task_id=f"{task_id}_{baseline}_{perception_mode}",
            family=family,
            split=split,
            run_fn=make_task_runner(env_cls, task_def, max_budget),
            max_budget=max_budget,
        )
        tasks.append(t_res)

    baselines_dict: dict[str, Any] = {}
    if baseline == "system":
        print(
            "\n--- Evaluating Reference Baselines (random, fatal_memory_bfs, dyna_q, rules_known_planner, object_centric_hypothesis, oracle, no_grounding, demonstration) ---"
        )
        for b_name in (
            "random",
            "fatal_memory_bfs",
            "dyna_q",
            "rules_known_planner",
            "object_centric_hypothesis",
            "oracle",
            "no_grounding",
            "demonstration",
        ):
            b_tasks: list[TaskResult] = []
            for task_def in manifest.get("tasks", []):
                t_id = task_def["id"]
                t_fam = task_def["family"]
                t_cls = _load_env_class(task_def["env_class"])
                t_budget = task_def.get("max_budget", 100)

                b_ok, b_st, b_lt, b_er, b_extra = run_closed_loop_task(
                    env_cls=t_cls,
                    task_def=task_def,
                    max_budget=t_budget,
                    baseline=b_name,
                    perception_mode=perception_mode,
                    episodes=episodes_per_task,
                    shared_memory=shared_memory,
                    cross_seed=(
                        b_name
                        in (
                            "no_grounding",
                            "fatal_memory_bfs",
                            "dyna_q",
                            "rules_known_planner",
                            "object_centric_hypothesis",
                        )
                        and cross_seed
                    ),
                )
                b_task_res = TaskResult(
                    task_id=f"{t_id}_{b_name}",
                    family=t_fam,
                    split=split,
                    steps=b_st,
                    max_budget=t_budget,
                    seed_count=b_extra.get("seed_count", 1),
                    total_deaths=b_extra.get("total_deaths", b_extra.get("exploration_deaths", 0)),
                    exploration_deaths=b_extra.get("exploration_deaths", 0),
                    success_at_budget=b_ok,
                    steps_to_first_success=b_extra.get("steps_to_first_success"),
                    steps_to_stable_policy=b_extra.get("steps_to_stable_policy"),
                    errors=b_er,
                    p50_latency_ms=round(sorted(b_lt)[len(b_lt) // 2], 2) if b_lt else 0.0,
                    p95_latency_ms=(
                        round(sorted(b_lt)[min(len(b_lt) - 1, int(len(b_lt) * 0.95))], 2)
                        if b_lt
                        else 0.0
                    ),
                    episode_successes=b_extra.get("episode_successes", []),
                    episode_steps=b_extra.get("episode_steps", []),
                    zero_shot=b_extra.get("zero_shot", {}),
                    same_layout_replay=b_extra.get("same_layout_replay", {}),
                    lifelong=b_extra.get("lifelong", {}),
                    cross_seed_transfer=b_extra.get("cross_seed_transfer", {}),
                )
                b_tasks.append(b_task_res)

            b_fams: dict[str, list[TaskResult]] = {}
            for bt in b_tasks:
                b_fams.setdefault(bt.family, []).append(bt)

            zs_b_fam_scores: dict[str, float] = {}
            b_fam_scores: dict[str, float] = {}
            zs_tot_trials = 0
            zs_tot_solved = 0

            for f_name, f_list in b_fams.items():
                trials: list[tuple[bool, int, int]] = []
                for x in f_list:
                    trials.extend(EvaluationProtocol.extract_zero_shot_trials(x))
                n_tr = len(trials)
                ok_tr = [tr for tr in trials if tr[0]]
                zs_tot_trials += n_tr
                zs_tot_solved += len(ok_tr)
                if not ok_tr:
                    zs_b_fam_scores[f_name] = 0.0
                else:
                    sr = len(ok_tr) / n_tr
                    eff = sum(max(0.0, 1.0 - (st / max(1, bud))) for _, st, bud in ok_tr) / len(
                        ok_tr
                    )
                    zs_b_fam_scores[f_name] = round(0.5 * sr + 0.5 * eff, 4)

                solved_lf = [x for x in f_list if x.success_at_budget]
                if not solved_lf:
                    b_fam_scores[f_name] = 0.0
                else:
                    sr_lf = len(solved_lf) / len(f_list)
                    eff_lf = sum(
                        max(0.0, 1.0 - (x.steps / max(1, x.max_budget))) for x in solved_lf
                    ) / len(solved_lf)
                    b_fam_scores[f_name] = round(0.5 * sr_lf + 0.5 * eff_lf, 4)

            zs_b_gen = round(sum(zs_b_fam_scores.values()) / max(1, len(zs_b_fam_scores)), 4)
            b_gen = round(sum(b_fam_scores.values()) / max(1, len(b_fam_scores)), 4)
            tot_solved_lf = sum(1 for bt in b_tasks if bt.success_at_budget)
            zs_ci = list(wilson_ci(zs_tot_solved, zs_tot_trials))
            b_deaths = sum(bt.exploration_deaths for bt in b_tasks)
            b_tot_deaths = sum(max(bt.total_deaths, bt.exploration_deaths) for bt in b_tasks)

            b_curves = [bt.episode_successes for bt in b_tasks if bt.episode_successes]
            avg_curve = (
                [all(c[ep] for c in b_curves) for ep in range(episodes_per_task)]
                if b_curves
                else []
            )
            stable_list = [
                bt.steps_to_stable_policy for bt in b_tasks if bt.steps_to_stable_policy is not None
            ]
            b_stable = round(sum(stable_list) / len(stable_list), 2) if stable_list else None

            xs_b: dict[str, Any] = {}
            xs_bt = [bt for bt in b_tasks if bt.cross_seed_transfer.get("successes")]
            if xs_bt:
                tr_eps = sum(max(0, len(bt.cross_seed_transfer["successes"]) - 1) for bt in xs_bt)
                tr_ok = sum(sum(bt.cross_seed_transfer["successes"][1:]) for bt in xs_bt)
                xs_b = {
                    "transfer_solve_rate": round(tr_ok / max(1, tr_eps), 4),
                    "transfer_wilson_95_ci": list(wilson_ci(tr_ok, tr_eps)),
                }

            baselines_dict[b_name] = {
                "generality_score": zs_b_gen,
                "zero_shot_generality_score": zs_b_gen,
                "zero_shot_family_scores": zs_b_fam_scores,
                "zero_shot_n_trials": zs_tot_trials,
                "zero_shot_solve_rate": round(zs_tot_solved / max(1, zs_tot_trials), 4),
                "zero_shot_wilson_95_ci": zs_ci,
                "total_deaths": b_tot_deaths,
                "exploration_deaths": b_deaths,
                "same_layout_replay_generality_score": b_gen,
                "lifelong_generality_score": b_gen,
                "lifelong_family_scores": b_fam_scores,
                "family_scores": zs_b_fam_scores,
                "solve_rate": round(tot_solved_lf / max(1, len(b_tasks)), 4),
                "lifelong_curve": avg_curve,
                "mean_steps_to_stable_policy": b_stable,
                "cross_seed_transfer": xs_b,
            }
            print(
                f"  Baseline [{b_name}]: zero_shot={zs_b_gen:.4f} ({zs_tot_solved}/{zs_tot_trials}, CI={zs_ci}, deaths={b_deaths}), replay={b_gen:.4f}"
            )

    suite_name = Path(manifest_path).stem if manifest_path else split
    report = protocol.compute_report(
        split=split, tasks=tasks, baselines=baselines_dict, suite_name=suite_name
    )

    print("\n=== Benchmark Results Summary ===")
    print(f"Split: {report.split.upper()} | Baseline: {baseline}")
    print(
        f"Headline 1 — Zero-shot (n={report.zero_shot.get('n_trials', 0)}): "
        f"generality={report.generality_score:.4f}, solve_rate={report.zero_shot.get('solve_rate', 0.0):.4f}, "
        f"95%_CI={report.zero_shot.get('wilson_95_ci')}, deaths={report.total_deaths} (exploration={report.exploration_deaths})"
    )
    if report.cross_seed_transfer:
        print(
            f"Headline 2 — Cross-seed transfer (n={report.cross_seed_transfer.get('n_transfer_episodes', 0)}): "
            f"transfer_solve_rate={report.cross_seed_transfer.get('transfer_solve_rate', 0.0):.4f}, "
            f"95%_CI={report.cross_seed_transfer.get('transfer_wilson_95_ci')}, "
            f"skills={report.cross_seed_transfer.get('skills_consolidated', 0)}"
        )
    print(
        f"Auxiliary  — Same-layout replay ({episodes_per_task} eps): "
        f"generality={report.same_layout_replay.get('generality_score', 0.0):.4f}, "
        f"stable_policy={report.same_layout_replay.get('mean_steps_to_stable_policy')}"
    )
    for fam, score in report.family_scores.items():
        fam_ci = report.zero_shot.get("family_wilson_95_ci", {}).get(fam)
        fam_sr = report.zero_shot.get("family_solve_rates", {}).get(fam, 0.0)
        print(f"  Family [{fam}]: zero_shot={score:.4f} (sr={fam_sr:.4f}, CI={fam_ci})")
    print(
        f"Mean Latency: p50={report.mean_p50_latency_ms:.2f}ms | p95={report.mean_p95_latency_ms:.2f}ms"
    )
    for t in report.tasks:
        print(
            f"  Task [{t.task_id}]: zs_sr={t.zero_shot.get('solve_rate')} CI={t.zero_shot.get('wilson_95_ci')} "
            f"deaths={t.exploration_deaths} replay={t.episode_successes}"
        )
    print("Report written successfully.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HBLLM Closed-Loop Benchmark Suite Runner")
    parser.add_argument("split", choices=["dev", "heldout"], help="Split to execute")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for reports")
    parser.add_argument(
        "--perception-mode",
        default="handcrafted",
        choices=["handcrafted", "snn", "learned"],
        help="Perception lifting mode",
    )
    parser.add_argument(
        "--baseline",
        default="system",
        choices=[
            "system",
            "stub_agent",
            "random",
            "fatal_memory_bfs",
            "dyna_q",
            "rules_known_planner",
            "object_centric_hypothesis",
            "oracle",
            "no_grounding",
            "demonstration",
        ],
        help="Evaluation policy baseline",
    )
    parser.add_argument("--manifest", default=None, help="Custom manifest YAML path")
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=5,
        help="Number of episodes per task for same-layout replay evaluation",
    )
    parser.add_argument(
        "--shared-memory",
        dest="shared_memory",
        action="store_true",
        default=True,
        help="Retain memory across episodes in replay evaluation",
    )
    parser.add_argument(
        "--no-shared-memory",
        dest="shared_memory",
        action="store_false",
        help="Do not retain memory across episodes",
    )
    parser.add_argument(
        "--no-cross-seed",
        dest="cross_seed",
        action="store_false",
        default=True,
        help="Skip cross-seed procedural transfer evaluation",
    )
    args = parser.parse_args()

    return run_suite(
        split=args.split,
        reports_dir=args.reports_dir,
        perception_mode=args.perception_mode,
        baseline=args.baseline,
        manifest_path=args.manifest,
        episodes_per_task=args.episodes_per_task,
        shared_memory=args.shared_memory,
        cross_seed=args.cross_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
