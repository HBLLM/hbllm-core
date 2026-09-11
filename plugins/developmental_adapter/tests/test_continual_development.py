"""Tests for Continual Development & Memory Consolidation Engine (Stage D12)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.continual_development import ContinualDevelopmentEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment


def test_dual_store_memory_consolidation_and_zero_forgetting():
    """Verify offline consolidation sleep cycle and Backward Transfer (BWT >= 0)."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="affordance_discovery_world")
    engine = ContinualDevelopmentEngine(substrate, env)

    # Record baseline performances on initial stages
    engine.record_stage_baseline("D3_CausalDiscovery", 1.0)
    engine.record_stage_baseline("D4_Affordances", 1.0)
    engine.record_stage_baseline("D5_ToolUse", 1.0)

    # Populate some rules and affordances
    substrate.causal_rules.append({"action": "PUSH", "consequence": "MOVES", "confidence": 0.95})
    substrate.causal_rules.append(
        {"action": "TOUCH", "consequence": "NONE", "confidence": 0.2}  # Should be pruned
    )
    substrate.affordances["ball"] = ["ROLL", "PUSH"]

    # Perform sleep consolidation
    report = engine.consolidate_memory_sleep_cycle()
    assert report["retained_rules"] == 1
    assert "ball" in substrate.affordances

    # Test evaluations after subsequent learning
    current_evals = {
        "D3_CausalDiscovery": 1.0,
        "D4_Affordances": 1.0,
        "D5_ToolUse": 1.0,
    }

    mean_bwt, no_forgetting = engine.evaluate_backward_transfer(current_evals)
    assert mean_bwt >= 0.0
    assert no_forgetting is True
