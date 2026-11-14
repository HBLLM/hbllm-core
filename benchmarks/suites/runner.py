"""Test suite runner for dev and held-out benchmark splits.

Executes true closed-loop evaluation where agent actions are fed back to environments
and success is determined strictly by the environment simulation.
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

from hbllm.benchmarks.protocol import EvaluationProtocol, TaskResult
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def _load_env_class(class_path: str) -> type:
    """Dynamically import environment class from module string."""
    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


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
    """Execute a task over N episodes in closed loop with optional shared memory.

    Returns:
        (overall_success, steps, latencies_ms, errors, extra)
    """
    seeds = task_def.get("seeds", [0])
    task_seed = seeds[0]
    task_id = task_def.get("id", f"task_{task_seed}")

    episode_successes: list[bool] = []
    episode_steps: list[int] = []
    all_latencies: list[float] = []
    all_errors: list[str] = []

    def _execute(engine_factory=None):
        system = None
        engine = None
        if engine_factory is not None and shared_memory:
            system = engine_factory()
            engine = system.engine

        prev_succ = False
        for ep_idx in range(episodes):
            if engine_factory is not None and not shared_memory:
                system = engine_factory()
                engine = system.engine
            elif engine is not None and ep_idx > 0:
                from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind

                b_kind = (
                    EpisodeBoundaryKind.LEVEL_COMPLETE if prev_succ else EpisodeBoundaryKind.DEATH
                )
                try:
                    engine.on_boundary(EpisodeBoundary(kind=b_kind, same_level=shared_memory))
                except Exception:
                    pass

            env_instance = env_cls(max_steps=max_budget)
            obs = env_instance.reset(seed=task_seed)
            ep_baseline = baseline
            if baseline == "demonstration":
                ep_baseline = "demonstration_oracle" if ep_idx == 0 else "system"
            succ, steps, lats, errs = _run_loop(
                env_instance,
                obs,
                engine,
                ep_baseline,
                max_budget,
                [],
                [],
                seed=task_seed,
            )
            episode_successes.append(succ)
            episode_steps.append(steps)
            all_latencies.extend(lats)
            all_errors.extend(errs)
            prev_succ = succ

    if baseline in ("system", "no_grounding", "demonstration"):
        with tempfile.TemporaryDirectory() as tmp_dir:

            def make_system():
                config = CognitiveSystemConfig(
                    world_id=f"bench_{task_id}_{baseline}",
                    memory_dir=tmp_dir,
                    enable_grounding=(baseline in ("system", "demonstration")),
                    perception_mode=perception_mode,
                )
                return build_cognitive_system(config)

            _execute(make_system)
    else:
        _execute(None)

    # Compute stable policy metric (first i after which 3 consecutive successes occur)
    stable_idx = None
    for i in range(len(episode_successes) - 2):
        if episode_successes[i] and episode_successes[i + 1] and episode_successes[i + 2]:
            stable_idx = i
            break

    # First success
    first_succ_steps = None
    for i, s in enumerate(episode_successes):
        if s:
            first_succ_steps = episode_steps[i]
            break

    overall_success = bool(episode_successes[-1]) if episode_successes else False
    mean_steps = (
        episode_steps[-1]
        if (episode_steps and overall_success)
        else (int(sum(episode_steps) / max(1, len(episode_steps))))
    )

    one_shot_imitation: dict[str, Any] = {}
    if baseline == "demonstration":
        one_shot_imitation = {
            "success": episode_successes[1] if len(episode_successes) > 1 else False,
            "steps": episode_steps[1] if len(episode_steps) > 1 else max_budget,
            "oracle_success": episode_successes[0] if episode_successes else False,
            "oracle_steps": episode_steps[0] if episode_steps else max_budget,
        }
    elif baseline == "system" and hasattr(env_cls, "get_oracle_action"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = CognitiveSystemConfig(
                world_id=f"bench_{task_id}_demo",
                memory_dir=tmp_dir,
                enable_grounding=True,
                perception_mode=perception_mode,
            )
            demo_sys = build_cognitive_system(config)
            demo_engine = demo_sys.engine
            demo_env = env_cls(max_steps=max_budget)
            d_obs = demo_env.reset(seed=task_seed)
            d_succ, d_steps, _, _ = _run_loop(
                demo_env,
                d_obs,
                demo_engine,
                "demonstration_oracle",
                max_budget,
                [],
                [],
                seed=task_seed,
            )
            from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind

            demo_engine.on_boundary(
                EpisodeBoundary(
                    kind=(
                        EpisodeBoundaryKind.LEVEL_COMPLETE if d_succ else EpisodeBoundaryKind.DEATH
                    )
                )
            )
            d_obs = demo_env.reset(seed=task_seed)
            i_succ, i_steps, _, _ = _run_loop(
                demo_env, d_obs, demo_engine, "system", max_budget, [], [], seed=task_seed
            )
            one_shot_imitation = {
                "success": i_succ,
                "steps": i_steps,
                "oracle_success": d_succ,
                "oracle_steps": d_steps,
            }

    # Cross-seed procedural transfer: one shared engine across DISTINCT procedural
    # layouts (seeds[0..k]) with same_level=False boundaries, measuring whether
    # feature-level world theory and consolidated skills transfer between layouts.
    cross_seed_transfer: dict[str, Any] = {}
    if cross_seed and baseline in ("system", "no_grounding") and len(seeds) > 1:
        xs_seeds = [seeds[i % len(seeds)] for i in range(episodes)]
        xs_succ: list[bool] = []
        xs_steps: list[int] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = CognitiveSystemConfig(
                world_id=f"bench_{task_id}_xseed_{baseline}",
                memory_dir=tmp_dir,
                enable_grounding=(baseline == "system"),
                perception_mode=perception_mode,
            )
            xs_engine = build_cognitive_system(config).engine
            from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind

            prev_ok = False
            for ep_idx, ep_seed in enumerate(xs_seeds):
                if ep_idx > 0:
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
                    except Exception:
                        pass
                xs_env = env_cls(max_steps=max_budget)
                xs_obs = xs_env.reset(seed=ep_seed)
                ok, st, _, _ = _run_loop(
                    xs_env, xs_obs, xs_engine, "system", max_budget, [], [], seed=ep_seed
                )
                xs_succ.append(ok)
                xs_steps.append(st)
                prev_ok = ok
            skills = len(getattr(getattr(xs_engine, "library_learner", None), "macro_skills", {}))
        cross_seed_transfer = {
            "seeds": xs_seeds,
            "successes": xs_succ,
            "steps": xs_steps,
            "solve_rate": round(sum(xs_succ) / max(1, len(xs_succ)), 4),
            "transfer_solve_rate": (
                round(sum(xs_succ[1:]) / max(1, len(xs_succ) - 1), 4) if len(xs_succ) > 1 else 0.0
            ),
            "skills_consolidated": skills,
        }

    extra = {
        "episode_successes": episode_successes,
        "episode_steps": episode_steps,
        "steps_to_stable_policy": stable_idx,
        "steps_to_first_success": first_succ_steps,
        "zero_shot": {
            "success": episode_successes[0] if episode_successes else False,
            "steps": episode_steps[0] if episode_steps else max_budget,
        },
        "lifelong": {
            "success": overall_success,
            "episodes": episodes,
            "successes": episode_successes,
            "steps": episode_steps,
            "steps_to_stable_policy": stable_idx,
        },
        "one_shot_imitation": one_shot_imitation,
        "cross_seed_transfer": cross_seed_transfer,
    }

    return overall_success, mean_steps, all_latencies, all_errors, extra


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

    system = None
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
    success = False
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
                # Always emits constant action (e.g. ACTION_UP = 1)
                action = 1
            elif baseline == "random":
                # Select random action from available actions (seeded for determinism)
                avail = getattr(obs, "available_actions", [1, 2, 3, 4])
                action = rng.choice(list(avail))
            elif baseline == "oracle":
                # Compute optimal action from ground-truth state
                action = env.get_oracle_action()
            elif baseline == "demonstration_oracle":
                if engine is not None:
                    try:
                        raw_action = engine.step(obs, last_action=last_oracle_action)
                        if isinstance(raw_action, tuple) and len(raw_action) > 1:
                            meta = raw_action[1]
                    except Exception:
                        pass
                action = env.get_oracle_action()
                last_oracle_action = action
            else:
                # AutonomousEpistemicEngine step: unpack (action, metadata)
                assert engine is not None
                raw_action = engine.step(obs)
                if isinstance(raw_action, tuple) and len(raw_action) > 1:
                    action, meta = raw_action[0], raw_action[1]
                else:
                    action = raw_action
        except Exception as e:
            errors.append(f"Action selection error at step {step_count}: {e}")
            break

        step_journal.append({"step": step_count, "action": action, "meta": meta})
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Environment closed-loop transition
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            errors.append(f"Environment step error at step {step_count}: {e}")
            break

        if done:
            success = bool(info.get("success", False))
            # Feed terminal observation back to engine to trigger reward/goal learning
            if engine is not None:
                try:
                    engine.step(
                        obs,
                        last_action=(
                            last_oracle_action if baseline == "demonstration_oracle" else None
                        ),
                    )
                except Exception:
                    pass
            break

    return success, step_count, latencies, errors


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
            "version": "1.0.0",
            "perception_mode": perception_mode,
            "baseline": baseline,
            "split": split,
            "episodes_per_task": episodes_per_task,
            "shared_memory": shared_memory,
        },
    )

    if split == "heldout" and baseline == "system":
        protocol.verify_heldout_gate()

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

        print(
            f"-> Evaluating task [{task_id}] ({family}, episodes={episodes_per_task}, shared_memory={shared_memory}, budget={max_budget})..."
        )

        def make_task_runner(t_def: dict[str, Any], t_budget: int):
            def run_task():
                return run_closed_loop_task(
                    env_cls=env_cls,
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
            run_fn=make_task_runner(task_def, max_budget),
            max_budget=max_budget,
        )
        tasks.append(t_res)

    baselines_dict: dict[str, Any] = {}
    if baseline == "system":
        print(
            "\n--- Evaluating Reference Baselines (random, oracle, no_grounding, demonstration) ---"
        )
        for b_name in ("random", "oracle", "no_grounding", "demonstration"):
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
                    cross_seed=False,
                )
                b_task_res = TaskResult(
                    task_id=f"{t_id}_{b_name}",
                    family=t_fam,
                    split=split,
                    steps=b_st,
                    max_budget=t_budget,
                    success_at_budget=b_ok,
                    steps_to_first_success=b_extra.get("steps_to_first_success"),
                    steps_to_stable_policy=b_extra.get("steps_to_stable_policy"),
                    p50_latency_ms=round(sorted(b_lt)[len(b_lt) // 2], 2) if b_lt else 0.0,
                    p95_latency_ms=(
                        round(sorted(b_lt)[min(len(b_lt) - 1, int(len(b_lt) * 0.95))], 2)
                        if b_lt
                        else 0.0
                    ),
                    episode_successes=b_extra.get("episode_successes", []),
                    episode_steps=b_extra.get("episode_steps", []),
                    zero_shot=b_extra.get("zero_shot", {}),
                    lifelong=b_extra.get("lifelong", {}),
                )
                b_tasks.append(b_task_res)

            b_fams: dict[str, list[TaskResult]] = {}
            for bt in b_tasks:
                b_fams.setdefault(bt.family, []).append(bt)
            b_fam_scores: dict[str, float] = {}
            for f_name, f_list in b_fams.items():
                solved = [x for x in f_list if x.success_at_budget]
                if not solved:
                    b_fam_scores[f_name] = 0.0
                else:
                    sr = len(solved) / len(f_list)
                    eff = sum(
                        max(0.0, 1.0 - (x.steps / max(1, x.max_budget))) for x in solved
                    ) / len(solved)
                    b_fam_scores[f_name] = round(0.5 * sr + 0.5 * eff, 4)
            b_gen = round(sum(b_fam_scores.values()) / max(1, len(b_fam_scores)), 4)
            tot_solved = sum(1 for bt in b_tasks if bt.success_at_budget)

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

            baselines_dict[b_name] = {
                "generality_score": b_gen,
                "family_scores": b_fam_scores,
                "solve_rate": round(tot_solved / max(1, len(b_tasks)), 4),
                "lifelong_curve": avg_curve,
                "mean_steps_to_stable_policy": b_stable,
            }
            print(
                f"  Baseline [{b_name}]: generality={b_gen:.4f}, solve_rate={tot_solved}/{len(b_tasks)}, curve={avg_curve}, stable={b_stable}"
            )

    report = protocol.compute_report(split=split, tasks=tasks, baselines=baselines_dict)

    print("\n=== Benchmark Results Summary ===")
    print(f"Split: {report.split.upper()} | Baseline: {baseline}")
    print(
        f"Zero-shot: generality={report.zero_shot.get('generality_score', 0.0):.4f}, solve_rate={report.zero_shot.get('solve_rate', 0.0):.4f}"
    )
    print(
        f"Lifelong:  generality={report.lifelong.get('generality_score', 0.0):.4f}, solve_rate={report.lifelong.get('solve_rate', 0.0):.4f}, stable_policy={report.lifelong.get('mean_steps_to_stable_policy')}"
    )
    if report.cross_seed_transfer:
        print(
            f"Cross-seed: solve_rate={report.cross_seed_transfer.get('solve_rate', 0.0):.4f}, transfer_solve_rate={report.cross_seed_transfer.get('transfer_solve_rate', 0.0):.4f}, skills={report.cross_seed_transfer.get('skills_consolidated', 0)}"
        )
    for fam, score in report.family_scores.items():
        print(f"  Family [{fam}]: {score:.4f}")
    print(f"Generality Score: {report.generality_score:.4f}")
    print(
        f"Mean Latency: p50={report.mean_p50_latency_ms:.2f}ms | p95={report.mean_p95_latency_ms:.2f}ms"
    )
    for t in report.tasks:
        print(
            f"  Task [{t.task_id}]: curve={t.episode_successes}, steps={t.episode_steps}, stable={t.steps_to_stable_policy}"
        )
        if t.cross_seed_transfer:
            print(
                f"    cross-seed seeds={t.cross_seed_transfer.get('seeds')} curve={t.cross_seed_transfer.get('successes')} steps={t.cross_seed_transfer.get('steps')}"
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
        choices=["system", "stub_agent", "random", "oracle", "no_grounding"],
        help="Evaluation policy baseline",
    )
    parser.add_argument("--manifest", default=None, help="Custom manifest YAML path")
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=5,
        help="Number of episodes per task for lifelong evaluation",
    )
    parser.add_argument(
        "--shared-memory",
        dest="shared_memory",
        action="store_true",
        default=True,
        help="Retain memory across episodes in lifelong evaluation",
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
