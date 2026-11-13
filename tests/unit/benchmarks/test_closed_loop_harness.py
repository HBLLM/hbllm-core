"""Tests for the closed-loop evaluation harness and procedural environments."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.envs.continuous_reach import ContinuousReachEnvironment
from benchmarks.envs.grid_nav import GridNavEnvironment
from benchmarks.envs.grid_push import GridPushEnvironment
from benchmarks.envs.symbolic_rules import SymbolicRulesEnvironment
from benchmarks.suites.runner import run_closed_loop_episode, run_closed_loop_task, run_suite
from hbllm.benchmarks.protocol import EvaluationProtocol


def test_closed_loop_stub_agent_fails_all_tasks():
    """Stub agent emitting constant action must fail 100% of tasks (0.0 success rate)."""
    env = GridNavEnvironment(size=10, max_steps=60)
    successes: list[bool] = []

    for seed in range(10):
        succ, _, _, _ = run_closed_loop_episode(
            env=env,
            seed=seed,
            max_budget=60,
            baseline="stub_agent",
        )
        successes.append(succ)

    solve_rate = sum(1 for s in successes if s) / len(successes)
    assert solve_rate == 0.0, f"Stub agent unexpectedly solved {solve_rate * 100}% of tasks"


def test_closed_loop_random_baseline_bounded():
    """Random agent must have low success rate (< 20%) on maze navigation with key-door lock."""
    env = GridNavEnvironment(size=10, max_steps=80)
    successes: list[bool] = []

    for seed in range(20):
        succ, _, _, _ = run_closed_loop_episode(
            env=env,
            seed=seed,
            max_budget=80,
            baseline="random",
        )
        successes.append(succ)

    solve_rate = sum(1 for s in successes if s) / len(successes)
    assert solve_rate < 0.20, f"Random baseline exceeded 20% success bound: {solve_rate:.2f}"


def test_closed_loop_oracle_baseline_solves_tasks():
    """Oracle agent with ground-truth search must achieve > 90% success across environments."""
    # 1. Grid navigation
    nav_env = GridNavEnvironment(size=10, max_steps=100)
    nav_successes = [
        run_closed_loop_episode(nav_env, s, 100, baseline="oracle")[0] for s in range(10)
    ]
    nav_rate = sum(1 for s in nav_successes if s) / len(nav_successes)
    assert nav_rate >= 0.90, f"GridNav oracle solve rate too low: {nav_rate:.2f}"

    # 2. Grid push
    push_env = GridPushEnvironment(size=8, max_steps=100)
    push_successes = [
        run_closed_loop_episode(push_env, s, 100, baseline="oracle")[0] for s in range(10)
    ]
    push_rate = sum(1 for s in push_successes if s) / len(push_successes)
    assert push_rate >= 0.90, f"GridPush oracle solve rate too low: {push_rate:.2f}"

    # 3. Continuous reach
    cont_env = ContinuousReachEnvironment(max_steps=60)
    cont_successes = [
        run_closed_loop_episode(cont_env, s, 60, baseline="oracle")[0] for s in range(10)
    ]
    cont_rate = sum(1 for s in cont_successes if s) / len(cont_successes)
    assert cont_rate >= 0.90, f"Continuous reach oracle solve rate too low: {cont_rate:.2f}"

    # 4. Symbolic rules
    sym_env = SymbolicRulesEnvironment(size=9, max_steps=80)
    sym_successes = [
        run_closed_loop_episode(sym_env, s, 80, baseline="oracle")[0] for s in range(10)
    ]
    sym_rate = sum(1 for s in sym_successes if s) / len(sym_successes)
    assert sym_rate >= 0.90, f"Symbolic rules oracle solve rate too low: {sym_rate:.2f}"


def test_observation_protocol_zero_entity_role_leakage():
    """Observations must supply raw lattices without oracle entity role tags."""
    envs = [
        GridNavEnvironment(size=10),
        GridPushEnvironment(size=8),
        ContinuousReachEnvironment(),
        SymbolicRulesEnvironment(size=9),
    ]

    for env in envs:
        obs = env.reset(seed=42)
        assert obs.lattice is not None
        assert len(obs.entities) == 0, f"Oracle entities leaked in {env.__class__.__name__}"
        assert len(obs.available_actions) > 0


def test_suite_manifests_disjoint_seeds_and_heldout_family():
    """Verify suite manifests have strictly disjoint seeds and novel held-out families."""
    dev_path = Path("benchmarks/suites/dev.yaml")
    heldout_path = Path("benchmarks/suites/heldout.yaml")
    assert dev_path.exists()
    assert heldout_path.exists()

    with open(dev_path, encoding="utf-8") as f:
        dev_manifest = yaml.safe_load(f)
    with open(heldout_path, encoding="utf-8") as f:
        heldout_manifest = yaml.safe_load(f)

    dev_seeds: set[int] = set()
    for task in dev_manifest["tasks"]:
        dev_seeds.update(task["seeds"])

    heldout_seeds: set[int] = set()
    for task in heldout_manifest["tasks"]:
        heldout_seeds.update(task["seeds"])

    # Disjoint seeds
    assert len(dev_seeds & heldout_seeds) == 0, "Seeds overlap between dev and heldout!"

    # Disjoint seed ranges
    assert max(dev_seeds) < min(heldout_seeds), "Dev and heldout seed ranges are not separated"

    # Disjoint task IDs & held-out coverage across all families
    dev_ids = {task["id"] for task in dev_manifest["tasks"]}
    heldout_ids = {task["id"] for task in heldout_manifest["tasks"]}
    assert len(dev_ids & heldout_ids) == 0, "Task IDs overlap between dev and heldout!"
    dev_families = {task["family"] for task in dev_manifest["tasks"]}
    heldout_families = {task["family"] for task in heldout_manifest["tasks"]}
    assert "symbolic" in heldout_families
    assert heldout_families >= dev_families


def test_config_hash_covers_envs_and_heldout_manifest():
    """Config hash must reflect changes in benchmark envs and heldout manifest."""
    protocol = EvaluationProtocol()
    h = protocol.config_hash
    assert isinstance(h, str)
    assert len(h) == 16


def test_successful_oracle_episode_populates_goal_features():
    """Verify that a successful episode with terminal signal populates symbolic_theory.goal_features."""
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    env = GridNavEnvironment(size=10, max_steps=100)
    obs = env.reset(seed=0)
    system = build_cognitive_system(
        CognitiveSystemConfig(world_id="test_oracle_goal", enable_logging=False)
    )

    done = False
    while not done:
        system.engine.step(obs)
        action = env.get_oracle_action()
        obs, reward, done, info = env.step(action)
    if info.get("success"):
        system.engine.step(obs)

    assert info.get("success") is True
    assert 8 in system.engine.symbolic_theory.goal_features


def test_closed_loop_lifelong_oracle_flat_and_stub_zero():
    """Lifelong evaluation: oracle shows flat True curve, stub shows flat False curve."""
    task_def = {
        "id": "test_grid_nav",
        "family": "grid",
        "env_class": "benchmarks.envs.grid_nav.GridNavEnvironment",
        "seeds": [10],
    }

    # 1. Oracle baseline (N=5 episodes)
    ok_o, steps_o, _, _, extra_o = run_closed_loop_task(
        env_cls=GridNavEnvironment,
        task_def=task_def,
        max_budget=100,
        baseline="oracle",
        episodes=5,
        shared_memory=True,
    )
    assert ok_o is True
    assert extra_o["episode_successes"] == [True, True, True, True, True]
    assert len(extra_o["episode_steps"]) == 5
    assert extra_o["steps_to_stable_policy"] == 0
    assert extra_o["zero_shot"]["success"] is True
    assert extra_o["lifelong"]["success"] is True

    # 2. Stub agent baseline (N=5 episodes)
    ok_s, steps_s, _, _, extra_s = run_closed_loop_task(
        env_cls=GridNavEnvironment,
        task_def=task_def,
        max_budget=60,
        baseline="stub_agent",
        episodes=5,
        shared_memory=True,
    )
    assert ok_s is False
    assert extra_s["episode_successes"] == [False, False, False, False, False]
    assert extra_s["steps_to_stable_policy"] is None
    assert extra_s["zero_shot"]["success"] is False
    assert extra_s["lifelong"]["success"] is False


def test_closed_loop_lifelong_json_schema(tmp_path: Path):
    """Evaluation report JSON must contain both zero_shot and lifelong schemas."""
    import json

    dev_manifest = tmp_path / "mini_dev.yaml"
    dev_manifest.write_text(
        yaml.dump(
            {
                "split": "dev",
                "tasks": [
                    {
                        "id": "mini_reach",
                        "family": "continuous",
                        "env_class": "benchmarks.envs.continuous_reach.ContinuousReachEnvironment",
                        "seeds": [50],
                        "max_budget": 30,
                    }
                ],
            }
        )
    )

    code = run_suite(
        split="dev",
        reports_dir=str(tmp_path / "reports"),
        perception_mode="handcrafted",
        baseline="oracle",
        manifest_path=str(dev_manifest),
        episodes_per_task=4,
        shared_memory=True,
    )
    assert code == 0

    report_files = list((tmp_path / "reports" / "dev").glob("dev_*.json"))
    assert len(report_files) == 1

    report_data = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert "zero_shot" in report_data
    assert "lifelong" in report_data
    assert "generality_score" in report_data["zero_shot"]
    assert "generality_score" in report_data["lifelong"]

    task = report_data["tasks"][0]
    assert "episode_successes" in task
    assert "episode_steps" in task
    assert "steps_to_stable_policy" in task
    assert len(task["episode_successes"]) == 4
