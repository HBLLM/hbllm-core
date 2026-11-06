"""
Unit tests for Commit 4 of Phase 11 — Predictive World Kernel Integration.
Tests PredictorWeightManager, WeightUpdatePolicy, ConfidenceCalibrator, and PredictionBudget.
"""

from __future__ import annotations

import pytest

from hbllm.hcir.world.confidence_calibrator import ConfidenceCalibrator
from hbllm.hcir.world.prediction_budget import PredictionBudget
from hbllm.hcir.world.predictor_weights import PredictorWeightManager
from hbllm.hcir.world.weight_policy import WeightUpdatePolicy
from hbllm.hcir.world.world_context import WorldModelScope


def test_predictor_weight_manager_and_policy():
    mgr = PredictorWeightManager()
    scope = WorldModelScope(tenant_id="t1", world_id="w1", domain="robotics")
    state = mgr.get_predictor_weight(scope, "physics")
    assert state.accuracy_weight == 0.50

    policy = WeightUpdatePolicy(learning_rate=0.05)
    delta = policy.calculate_delta(expected_value=80.0, actual_value=80.02)
    assert delta == 1.0

    updated = mgr.update_weight(scope, "physics", delta, lr=0.05)
    assert updated.accuracy_weight > 0.50

    # Forward architecture exports for weights and policy
    pred_st = updated.to_causal_predicate()
    assert pred_st.variable == "weight:physics"
    assert pred_st.value == updated.accuracy_weight

    tx_st = updated.to_transaction_operation("t1_w1_robotics")
    assert tx_st.node_id == "weight_t1_w1_robotics_physics"

    preds_mgr = mgr.export_causal_predicates(scope)
    assert len(preds_mgr) == 1
    txs_mgr = mgr.to_transaction_operations(scope)
    assert len(txs_mgr) == 1

    pred_pol = policy.to_causal_predicate()
    assert pred_pol.variable == "weight_update_policy"
    assert pred_pol.value == 0.05


def test_confidence_calibrator_decay():
    calibrator = ConfidenceCalibrator()
    vec_fresh = calibrator.calibrate(raw_confidence=0.95, historical_accuracy=0.90, age_seconds=0.0)
    assert vec_fresh.calibrated_confidence == pytest.approx(0.855, rel=1e-2)

    vec_decayed = calibrator.calibrate(
        raw_confidence=0.95, historical_accuracy=0.90, age_seconds=3600.0, half_life_seconds=3600.0
    )
    assert vec_decayed.calibrated_confidence < vec_fresh.calibrated_confidence

    pred_vec = vec_fresh.to_causal_predicate()
    assert pred_vec.variable == "confidence"
    assert pred_vec.value == vec_fresh.calibrated_confidence


def test_prediction_budget():
    budget = PredictionBudget()
    fast_predictors = budget.select_predictor_tier(horizon_ms=60000, available_cognitive_budget=0.2)
    assert fast_predictors == ["physics", "statistical"]

    full_predictors = budget.select_predictor_tier(horizon_ms=60000, available_cognitive_budget=0.9)
    assert len(full_predictors) == 5

    pred_bgt = budget.to_causal_predicate()
    assert pred_bgt.variable == "prediction_budget_latency"
    assert pred_bgt.value == 500

    tx_bgt = budget.to_transaction_operations()
    assert len(tx_bgt) == 1
    assert tx_bgt[0].node_id == "prediction_budget_governor"
