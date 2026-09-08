"""
Unit and Integration Tests for Stochastic Robustness Adapter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_test_dir = Path(__file__).resolve().parent
_plugin_dir = _test_dir.parent
_plugins_root = _plugin_dir.parent
_core_root = _plugins_root.parent

for p in [str(_core_root), str(_plugins_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stochastic_adapter import (
    PLUGIN_NAME,
    PLUGIN_VERSION,
    StochasticAction,
    StochasticPerceptionAdapter,
    StochasticTier,
    make_stochastic_env,
    run_stochastic_benchmark,
    run_stochastic_tier,
)


def test_plugin_manifest() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["name"] == PLUGIN_NAME
    assert data["version"] == PLUGIN_VERSION
    assert "stochastic_filter_belief" in data["capabilities"]


def test_stochastic_environment_deterministic() -> None:
    env = make_stochastic_env(tier=StochasticTier.TIER_1_DETERMINISTIC_BASELINE, seed=42)
    obs = env.reset()
    assert obs.target_pos is not None
    assert not obs.was_slipped
    assert not obs.was_occluded


def test_stochastic_environment_noise() -> None:
    # Tier 2: actuator slip
    env_slip = make_stochastic_env(tier=StochasticTier.TIER_2_ACTUATOR_SLIP, seed=42)
    env_slip.reset()
    slips = 0
    for _ in range(50):
        obs, _r, done, info = env_slip.step(StochasticAction.RIGHT)
        if info["was_slipped"]:
            slips += 1
        if done:
            break
    assert slips > 0, "Tier 2 should induce actuator slips"

    # Tier 3: sensory dropout
    env_drop = make_stochastic_env(tier=StochasticTier.TIER_3_SENSORY_DROPOUT, seed=42)
    env_drop.reset()
    drops = 0
    for _ in range(50):
        obs, _r, done, info = env_drop.step(StochasticAction.RIGHT)
        if info["was_occluded"]:
            drops += 1
        if done:
            break
    assert drops > 0, "Tier 3 should induce sensory occlusions"


def test_epistemic_object_permanence() -> None:
    perception = StochasticPerceptionAdapter()
    env = make_stochastic_env(tier=StochasticTier.TIER_1_DETERMINISTIC_BASELINE, seed=42)
    obs = env.reset()
    percept1 = perception.process_observation(obs)
    assert percept1["target_pos"] == (7, 7)
    assert percept1["target_belief"].confidence == 1.0

    # Simulate sudden sensory dropout
    obs.target_pos = None
    percept2 = perception.process_observation(obs)
    # Object permanence must maintain belief at (7, 7)
    assert percept2["target_pos"] == (7, 7)
    assert percept2["target_belief"].occluded is True


def test_surprise_detection() -> None:
    perception = StochasticPerceptionAdapter()
    env = make_stochastic_env(tier=StochasticTier.TIER_1_DETERMINISTIC_BASELINE, seed=42)
    obs = env.reset()
    perception.process_observation(obs)

    # Register expectation that agent will move to (1, 2)
    perception.register_expected_transition((1, 2))

    # Observed actual pos is (1, 1) -> surprise!
    percept = perception.process_observation(obs)
    assert percept["surprise_detected"] is True


def test_stochastic_all_tiers_completion() -> None:
    for tier in StochasticTier:
        res = run_stochastic_tier(tier, episodes=3, seed=100)
        assert res["success_rate"] == 1.0, (
            f"Tier {tier.value} failed to achieve 100% success rate: {res}"
        )


def test_stochastic_full_benchmark_smoke() -> None:
    bench = run_stochastic_benchmark(episodes_per_tier=2, seed=42)
    assert bench["overall_success_rate"] == 1.0
    assert len(bench["tiers"]) == 5
    for tier_name, tier_res in bench["tiers"].items():
        assert tier_res["success_rate"] == 1.0
