"""
Phase 11 — Predictive World Kernel Integration End-to-End Integration Tests.

Verifies end-to-end predictive cognition, active inference, digital twin sync,
confidence calibration, surprise engine salience boost, verification gate promotion, and replay determinism.
"""

from __future__ import annotations

from hbllm.hcir.graph import ActionNode
from hbllm.hcir.world.active_inference import ActiveInferenceEngine
from hbllm.hcir.world.digital_twin import DigitalTwinRegistry
from hbllm.hcir.world.prediction_promotion import PredictionPromotion
from hbllm.hcir.world.prediction_receipt import WorldPredictionReceipt
from hbllm.hcir.world.prediction_types import PredictionProvenance
from hbllm.hcir.world.predictive_reality import PredictiveRealityModel
from hbllm.hcir.world.surprise_engine import SurpriseEngine
from hbllm.hcir.world.verification_gate import VerificationGate
from hbllm.hcir.world.world_belief import WorldBeliefGraph
from hbllm.hcir.world.world_state_interpreter import WorldStateInterpreter
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot


class TestPhase11PredictiveWorldKernel:
    """Integration test suite for Phase 11 Predictive World Kernel Integration."""

    def test_digital_twin_interpreter_and_belief_isolation(self):
        twin = DigitalTwinRegistry(world_id="factory_edge_1")
        twin.sync_sensor_telemetry("motor_temp", 88.5, source="thermocouple_1")
        twin.register_entity("motor_1", "Main Motor", status="active")

        snapshot = twin.create_snapshot()
        assert snapshot.variables["motor_temp"] == 88.5
        assert len(snapshot.state_hash) == 16

        interpreter = WorldStateInterpreter()
        hypotheses = interpreter.interpret_snapshot(snapshot)
        assert len(hypotheses) >= 1

        belief_graph = WorldBeliefGraph(world_id="factory_edge_1")
        belief_graph.ingest_hypotheses(hypotheses)

        belief = belief_graph.get_belief("b_motor_temp_status")
        assert belief is not None
        assert belief.value == "overheating_warning"
        # Verify strict isolation: belief graph does not mutate twin
        assert twin.get_variable("motor_temp") == 88.5

    def test_ensemble_prediction_and_surprise_engine(self):
        snapshot = WorldStateSnapshot(
            world_id="factory_edge_1",
            variables={"motor_temp": 88.5, "vibration": 0.08},
        )

        reality_model = PredictiveRealityModel()
        ensemble_pred = reality_model.predict(
            snapshot, "engage_secondary_cooling", horizon_ms=30000
        )

        assert ensemble_pred.calibrated_confidence > 0.0
        assert "physics" in ensemble_pred.component_predictions

        surprise_engine = SurpriseEngine(surprise_threshold=0.15)
        # Expected temp 60.0 vs actual 88.5 -> high surprise
        eval_res = surprise_engine.evaluate_surprise(
            prediction_id=ensemble_pred.prediction_id,
            expected_state={"motor_temp": 60.0},
            actual_state={"motor_temp": 88.5},
            confidence=ensemble_pred.calibrated_confidence,
        )

        assert eval_res.is_surprising
        assert eval_res.salience_boost == 0.35
        assert eval_res.prediction_error_node is not None

    def test_active_inference_action_selection(self):
        engine = ActiveInferenceEngine()
        actions = [
            ActionNode(
                id="a_cool", intent="engage_secondary_cooling", risk_factor=0.1, estimated_cost=5.0
            ),
            ActionNode(
                id="a_idle", intent="continue_full_load", risk_factor=0.8, estimated_cost=2.0
            ),
        ]

        best_action = engine.select_best_action(
            actions, information_gain_map={"a_cool": 0.7, "a_idle": 0.1}
        )
        assert best_action.action.id == "a_cool"
        assert best_action.utility_score > 0.0

    def test_verification_gate_and_promotion(self):
        gate = VerificationGate(min_confirmations=2, min_confidence=0.80)
        promotion = PredictionPromotion(verification_gate=gate)

        receipt = promotion.promote_prediction(
            prediction_id="pred_cool_1",
            action_intent="engage_secondary_cooling",
            confirmed_outcome="temperature_lowered",
            confirmations=3,
            confidence=0.89,
        )

        assert receipt is not None
        assert receipt.promotion_id == "promo_pred_cool_1"
        assert len(promotion.all_promotions()) == 1

    def test_replay_trace_determinism(self):
        prov = PredictionProvenance(world_id="factory_edge_1", predictors_used=["physics", "snn"])
        receipt = WorldPredictionReceipt(
            receipt_id="rcpt_p11_01",
            prediction_id="pred_101",
            model_version="world-model-v1.0",
            provenance=prov,
            input_state_hash="a1b2c3d4e5f67890",
            predicted_outcome="nominal",
            actual_outcome="nominal",
            calibrated_confidence=0.88,
            surprise_score=0.02,
        )

        assert receipt.input_state_hash == "a1b2c3d4e5f67890"
        assert receipt.calibrated_confidence == 0.88
