"""
Unit tests for Commit 5 of Phase 11 — Predictive World Kernel Integration.
Tests PredictionErrorNode, PredictionErrorTypology, and SurpriseEngine.
"""

from __future__ import annotations

from hbllm.hcir.world.prediction_error import PredictionErrorTypology, WorldPredictionError
from hbllm.hcir.world.surprise_engine import SurpriseEngine


def test_world_prediction_error():
    err = WorldPredictionError(
        prediction_id="p1",
        typology=PredictionErrorTypology.MODEL_ERROR,
        expected_state={"temp": 80.0},
        actual_state={"temp": 95.0},
        error_variance=15.0,
    )
    assert err.prediction_id == "p1"
    assert err.typology == PredictionErrorTypology.MODEL_ERROR

    # Forward architecture exports
    pred_err = err.to_causal_predicate()
    assert pred_err.variable == f"prediction_error:{err.error_id}"
    assert pred_err.value == 15.0

    tx_err = err.to_transaction_operations()
    assert len(tx_err) == 1
    assert tx_err[0].node_id == f"error_{err.error_id}"


def test_surprise_engine_evaluation():
    engine = SurpriseEngine(surprise_threshold=0.15)

    # Minor expected variance -> low surprise
    low_eval = engine.evaluate_surprise(
        prediction_id="p1",
        expected_state={"temp": 80.0},
        actual_state={"temp": 80.5},
        confidence=0.90,
    )
    assert not low_eval.is_surprising
    assert low_eval.prediction_error_node is None

    # High unexpected variance -> high surprise
    high_eval = engine.evaluate_surprise(
        prediction_id="p2",
        expected_state={"temp": 80.0},
        actual_state={"temp": 120.0},
        confidence=0.95,
    )
    assert high_eval.is_surprising
    assert high_eval.salience_boost == 0.35
    assert high_eval.prediction_error_node is not None
    assert high_eval.prediction_error_node.typology == PredictionErrorTypology.MODEL_ERROR
