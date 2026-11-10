from __future__ import annotations

import tempfile

from benchmarks.envs.grid_nav import GridNavEnvironment
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_step_returns_provenance_meta() -> None:
    """P2-1: Verify that every step() returns a non-None meta dict with all provenance keys."""
    with tempfile.TemporaryDirectory() as d:
        system = build_cognitive_system(
            CognitiveSystemConfig(world_id="test_prov", memory_dir=d, enable_logging=False)
        )
        engine = system.engine
        env = GridNavEnvironment(size=10, max_steps=10)
        obs = env.reset(seed=100010)

        action, meta = engine.step(obs)
        assert action is not None
        assert meta is not None and isinstance(meta, dict)

        required_keys = {
            "goal_target",
            "goal_source",
            "subgoal_stack",
            "plan",
            "plan_source",
            "candidates",
            "chosen_action",
            "veto_reasons",
            "phase",
        }
        missing = required_keys - set(meta.keys())
        assert not missing, f"Missing required provenance keys in meta: {missing}"

        assert meta["chosen_action"] == action
        assert meta["goal_source"] in ("reward", "learned_position", "epistemic", "none")
        assert meta["plan_source"] in ("astar", "macro", "escape", "arbitration")
        assert isinstance(meta["subgoal_stack"], list)
        assert isinstance(meta["veto_reasons"], list)
        assert isinstance(meta["candidates"], dict)
        assert isinstance(meta["phase"], str)


def test_scenario_d_planning_provenance() -> None:
    """P2-1 acceptance test: Scenario D planning behavioral test.

    Runs episodes A, B, and C (oracle observation), then verifies that in episode D
    the engine formulates an active plan toward its known goal targets.
    Expected to fail until P2-3 fixes the goal->plan->motor deliberation chain.
    """
    with tempfile.TemporaryDirectory() as d:
        system = build_cognitive_system(
            CognitiveSystemConfig(world_id="diag_prov", memory_dir=d, enable_logging=False)
        )
        engine = system.engine
        env = GridNavEnvironment()

        def _run_ep(driver: str) -> None:
            obs = env.reset(seed=100010)
            for _ in range(100):
                if driver == "oracle":
                    a = env.get_oracle_action()
                    engine.step(obs)
                else:
                    a, _ = engine.step(obs)
                obs, r, done, info = env.step(a)
                if done:
                    engine.step(obs)
                    break

        # Episode A: self fresh
        _run_ep("self")
        # Episode B: self same layout
        _run_ep("self")
        # Episode C: oracle demonstrates solution (key -> door -> goal)
        _run_ep("oracle")

        # Episode D: self on identical layout — must explain decisions and exploit
        obs_d = env.reset(seed=100010)
        for step_idx in range(5):
            action, meta = engine.step(obs_d)
            assert meta is not None, f"Step {step_idx}: meta is None"
            assert meta.get("goal_target") is not None, (
                f"Step {step_idx}: goal_target is None (meta={meta})"
            )
            plan = meta.get("plan")
            assert plan is not None and len(plan) > 0, (
                f"Step {step_idx}: plan is empty or None (meta={meta})"
            )
            obs_d, _, done, _ = env.step(action)
            if done:
                break
