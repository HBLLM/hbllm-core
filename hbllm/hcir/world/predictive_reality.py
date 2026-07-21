"""
Predictive Reality Model — Ensemble Reality State Transition Predictor.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from hbllm.hcir.world.disagreement_analyzer import PredictorDisagreementAnalyzer
from hbllm.hcir.world.prediction_types import EnsemblePrediction, PredictionProvenance
from hbllm.hcir.world.predictors.llm import LLMReasoningPredictor
from hbllm.hcir.world.predictors.neural import NeuralWorldModel
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor
from hbllm.hcir.world.predictors.statistical import StatisticalPredictor
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class PredictiveRealityModel:
    """Multi-horizon ensemble predictive reality model."""

    def __init__(self) -> None:
        self.physics = PhysicsPredictor()
        self.statistical = StatisticalPredictor()
        self.snn = SNNTemporalPredictor()
        self.neural = NeuralWorldModel()
        self.llm = LLMReasoningPredictor()
        self.disagreement_analyzer = PredictorDisagreementAnalyzer()

    def predict(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
        provenance: PredictionProvenance | None = None,
    ) -> EnsemblePrediction:
        """Evaluate ensemble predictors and return unified EnsemblePrediction."""
        comp_results: dict[str, tuple[dict[str, Any], float]] = {}

        p_state, p_conf = self.physics.predict_state(snapshot, action_intent, horizon_ms)
        comp_results["physics"] = (p_state, p_conf)

        st_state, st_conf = self.statistical.predict_state(snapshot, action_intent, horizon_ms)
        comp_results["statistical"] = (st_state, st_conf)

        snn_state, snn_conf = self.snn.predict_state(snapshot, action_intent, horizon_ms)
        comp_results["snn"] = (snn_state, snn_conf)

        neu_state, neu_conf = self.neural.predict_state(snapshot, action_intent, horizon_ms)
        comp_results["neural"] = (neu_state, neu_conf)

        llm_state, llm_conf = self.llm.predict_state(snapshot, action_intent, horizon_ms)
        comp_results["llm"] = (llm_state, llm_conf)

        disagreement = self.disagreement_analyzer.analyze_disagreement(comp_results)

        # Unified state prediction uses physics + snn weighted average
        unified_state = dict(p_state)
        avg_confidence = (
            p_conf * 0.4 + snn_conf * 0.3 + st_conf * 0.15 + neu_conf * 0.1 + llm_conf * 0.05
        )
        if disagreement.high_disagreement:
            avg_confidence *= 0.85  # Confidence penalty on high disagreement

        prov = provenance or PredictionProvenance(
            world_id=snapshot.world_id,
            predictors_used=["physics", "statistical", "snn", "neural", "llm"],
        )

        pred_id = f"pred_{uuid.uuid4().hex[:8]}"
        logger.info(
            "PredictiveRealityModel generated prediction '%s' for action '%s' confidence=%.2f",
            pred_id,
            action_intent,
            avg_confidence,
        )

        return EnsemblePrediction(
            prediction_id=pred_id,
            action_intent=action_intent,
            predicted_state=unified_state,
            calibrated_confidence=avg_confidence,
            component_predictions=comp_results,
            provenance=prov,
        )
