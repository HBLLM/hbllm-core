"""Tests for Cross-World and Cross-Domain Transfer (Stages D14 & D15)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.cross_transfer import CrossTransferEngine


def test_cross_world_zero_shot_transfer():
    """Verify zero-shot schema transfer to visually and spatially altered World B."""
    substrate = create_blank_brain_substrate()
    # Learned knowledge from World A
    substrate.affordances["ball"] = ["ROLL", "PUSH"]
    substrate.affordances["block"] = ["PUSH"]

    engine = CrossTransferEngine(substrate)
    eval_result = engine.evaluate_cross_world_transfer()

    assert eval_result.zero_shot_transfer_accuracy >= 0.8
    assert len(eval_result.reused_schemas) >= 1


def test_cross_domain_schema_projection():
    """Verify projection of abstract relational schemas to discrete domain (Sokoban)."""
    substrate = create_blank_brain_substrate()
    substrate.causal_rules.append({"action": "PUSH", "consequence": "MOVES", "confidence": 0.95})
    substrate.spatial_schemas.append({"type": "CONTAINMENT", "relation": "INSIDE"})

    engine = CrossTransferEngine(substrate)
    eval_result = engine.evaluate_cross_domain_transfer(target_domain="Sokoban")

    assert eval_result.zero_shot_transfer_accuracy == 1.0
    assert eval_result.sample_efficiency_ratio > 1.0
    assert "CAUSAL_PUSH_DYNAMICS" in eval_result.reused_schemas
