"""
Unit tests for Commit 3 of Phase 11 — Predictive World Kernel Integration.
Tests ensemble predictors, PredictorDisagreementAnalyzer, and PredictiveRealityModel.
"""

from __future__ import annotations

from hbllm.hcir.world.disagreement_analyzer import PredictorDisagreementAnalyzer
from hbllm.hcir.world.predictive_reality import PredictiveRealityModel
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot


def test_individual_predictors():
    snapshot = WorldStateSnapshot(
        world_id="w1",
        variables={"temp_celsius": 80.0, "vibration_g": 0.06},
    )

    phys = PhysicsPredictor()
    p_state, p_conf = phys.predict_state(snapshot, "reduce_speed", horizon_ms=10000)
    assert p_state["temp_celsius"] < 80.0
    assert p_conf == 0.92

    snn = SNNTemporalPredictor()
    s_state, s_conf = snn.predict_state(snapshot, "reduce_speed", horizon_ms=10000)
    assert s_state["vibration_g"] < 0.06
    assert s_conf == 0.88


def test_disagreement_analyzer_and_ensemble_model():
    analyzer = PredictorDisagreementAnalyzer()
    comp_preds = {
        "p1": ({"temp": 80}, 0.90),
        "p2": ({"temp": 80}, 0.50),
    }
    report = analyzer.analyze_disagreement(comp_preds)
    assert report.high_disagreement

    snapshot = WorldStateSnapshot(world_id="w1", variables={"temp_celsius": 80.0})
    reality_model = PredictiveRealityModel()
    ensemble = reality_model.predict(snapshot, "reduce_speed", horizon_ms=10000)

    assert ensemble.prediction_id.startswith("pred_")
    assert ensemble.calibrated_confidence > 0.0
    assert "physics" in ensemble.component_predictions
    assert "snn" in ensemble.component_predictions

    # Forward architecture neuro-symbolic and transaction exports
    preds = ensemble.to_causal_predicates()
    assert len(preds) > 0
    assert any(p.relation == "reduce_speed" for p in preds)

    txs = ensemble.to_transaction_operations()
    assert len(txs) == 1
    assert txs[0].node_id == f"pred_{ensemble.prediction_id}"

    affordance = ensemble.to_affordance()
    assert affordance.action_id == "reduce_speed"
    assert affordance.confidence == ensemble.calibrated_confidence

    pred_res, tx_res = reality_model.predict_and_transact(
        snapshot, "reduce_speed", horizon_ms=10000
    )
    assert pred_res.prediction_id.startswith("pred_")
    assert len(tx_res) == 1

    all_preds = reality_model.export_causal_predicates(ensemble)
    assert len(all_preds) >= 2
    assert any(p.variable == "prediction_provenance" for p in all_preds)
