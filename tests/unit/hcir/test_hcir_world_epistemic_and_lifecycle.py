"""
Unit tests for Commit 2 of Phase 11 — Predictive World Kernel Integration.
Tests EpistemicState, WorldCausalGraph, WorldKernelTransaction, PredictionLifecycle, VerificationGate, PredictionPromotion, and WorldModelRegistry.
"""

from __future__ import annotations

from hbllm.hcir.world.epistemic_state import CertaintyLevel, EpistemicState, EvidenceSource
from hbllm.hcir.world.prediction_lifecycle import PredictionLifecycle, PredictionState
from hbllm.hcir.world.prediction_promotion import PredictionPromotion
from hbllm.hcir.world.verification_gate import VerificationGate
from hbllm.hcir.world.world_causal import CausalEdgeType, WorldCausalGraph
from hbllm.hcir.world.world_model_registry import (
    ModelLifecycleState,
    WorldModelDescriptor,
    WorldModelRegistry,
)
from hbllm.hcir.world.world_transaction import WorldKernelTransaction


def test_epistemic_state_and_causal_graph():
    ep = EpistemicState(
        belief_id="b1",
        confidence=0.92,
        evidence_sources=[EvidenceSource.SENSOR, EvidenceSource.MODEL],
        certainty=CertaintyLevel.OBSERVATION,
    )
    assert ep.is_empirically_grounded()

    causal = WorldCausalGraph(world_id="factory_1")
    causal.add_causal_relation("cooling_failure", "temp_rise", relationship=CausalEdgeType.CAUSES)
    causes = causal.get_causes_for("temp_rise")
    assert len(causes) == 1
    assert causes[0].source_id == "cooling_failure"


def test_transaction_lifecycle_and_promotion():
    wtx = WorldKernelTransaction(snapshot_hash="hash123")
    assert wtx.snapshot_hash == "hash123"

    lifecycle = PredictionLifecycle(prediction_id="pred_1")
    assert lifecycle.state == PredictionState.CREATED
    lifecycle.transition_to(PredictionState.VERIFIED)
    assert lifecycle.state == PredictionState.VERIFIED

    gate = VerificationGate(min_confirmations=2, min_confidence=0.80)
    assert not gate.evaluate_eligibility(confirmations=1, confidence=0.85)
    assert gate.evaluate_eligibility(confirmations=2, confidence=0.85)

    promotion = PredictionPromotion(verification_gate=gate)
    receipt = promotion.promote_prediction(
        prediction_id="pred_1",
        action_intent="reduce_speed",
        confirmed_outcome="temp_decreased",
        confirmations=2,
        confidence=0.88,
    )
    assert receipt is not None
    assert receipt.promotion_id == "promo_pred_1"


def test_world_model_registry():
    registry = WorldModelRegistry()
    desc = WorldModelDescriptor(
        model_id="physics_v1",
        model_version="1.0.0",
        domain="robotics",
        supported_horizons_ms=[1000, 60000],
        status=ModelLifecycleState.ACTIVE,
    )
    registry.register_model(desc)
    active = registry.list_active_models_for_domain("robotics")
    assert len(active) == 1
    assert active[0].model_id == "physics_v1"
