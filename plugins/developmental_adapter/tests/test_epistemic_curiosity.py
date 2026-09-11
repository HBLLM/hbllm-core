"""Tests for Autonomous Epistemic Curiosity Engine (Stage D8)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.curiosity import EpistemicCuriosityEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment


def test_epistemic_curiosity_cycle_reduces_entropy():
    """Verify that intrinsic curiosity reduces epistemic entropy without external reward."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="affordance_discovery_world")
    engine = EpistemicCuriosityEngine(substrate, env)

    report = engine.run_curiosity_cycle(max_steps=8)

    assert report.initial_entropy > 0.0
    assert report.final_entropy < report.initial_entropy
    assert report.entropy_reduction > 0.0
    assert report.interventions_executed > 0
    assert len(report.discovered_rules) >= 1
    assert len(substrate.causal_rules) >= 1
