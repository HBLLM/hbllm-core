"""
Unit and Integration Tests for Semantic Ambiguity Adapter Plugin.
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

from semantic_adapter import (
    PLUGIN_NAME,
    PLUGIN_VERSION,
    SemanticObservation,
    SemanticPerceptionAdapter,
    SemanticTier,
    make_semantic_env,
    run_semantic_benchmark,
    run_semantic_tier,
)


def test_plugin_manifest() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "plugin.json"
    assert manifest_path.exists(), "plugin.json must exist"
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["name"] == PLUGIN_NAME
    assert data["version"] == PLUGIN_VERSION
    assert "semantic_detect_ambiguity" in data["capabilities"]


def test_semantic_environment_mechanics() -> None:
    env = make_semantic_env(tier=SemanticTier.TIER_1_CANONICAL_EXPLICIT, seed=42)
    obs = env.reset()
    assert len(obs.pending_subgoals) == 2
    assert not obs.done

    obs, r, done, info = env.step("pick:red_key")
    assert len(obs.completed_subgoals) == 1
    assert not done

    obs, r, done, info = env.step("unlock:blue_door")
    assert len(obs.completed_subgoals) == 2
    assert obs.won
    assert done


def test_ambiguity_detection() -> None:
    perception = SemanticPerceptionAdapter()

    # Canonical explicit
    obs1 = SemanticObservation(
        instruction="Pick up the red key and unlock the blue door.",
        scene_objects={},
    )
    p1 = perception.process_observation(obs1)
    assert p1["is_canonical"] is True
    assert p1["ambiguity_detected"] is False

    # Elliptical / underspecified
    obs2 = SemanticObservation(
        instruction="Please tidy up the workbench.",
        scene_objects={},
    )
    p2 = perception.process_observation(obs2)
    assert p2["ambiguity_detected"] is True
    assert p2["grounding_confidence"] < 0.5


def test_guided_hcir_all_tiers_completion() -> None:
    for tier in SemanticTier:
        res = run_semantic_tier(tier, cohort="guided_hcir", episodes=2, seed=42)
        assert res["success_rate"] == 1.0, f"Guided HCIR failed on {tier.value}: {res}"


def test_pure_vs_guided_benchmark_comparison() -> None:
    guided_bench = run_semantic_benchmark(cohort="guided_hcir", episodes_per_tier=2, seed=42)
    assert guided_bench["overall_success_rate"] == 1.0

    pure_bench = run_semantic_benchmark(cohort="pure_hcir", episodes_per_tier=2, seed=42)
    # Pure HCIR succeeds on explicit (Tier 1) and synonym (Tier 2), but flags ambiguity on 3-5
    assert pure_bench["tiers"][SemanticTier.TIER_1_CANONICAL_EXPLICIT.value]["success_rate"] == 1.0
    assert pure_bench["tiers"][SemanticTier.TIER_2_SYNONYM_PARAPHRASE.value]["success_rate"] == 1.0
    assert (
        pure_bench["tiers"][SemanticTier.TIER_3_UNDERSPECIFIED_ELLIPTICAL.value]["success_rate"]
        == 0.0
    )
