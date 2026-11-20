"""Predictive Reality Model — Multi-Horizon Ensemble Reality State Transition Predictor."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from hbllm.hcir.transactions import TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.disagreement_analyzer import PredictorDisagreementAnalyzer
from hbllm.hcir.world.prediction.models import EnsemblePrediction, PredictionProvenance
from hbllm.hcir.world.prediction.weights import PredictorWeightManager
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor
from hbllm.hcir.world.predictors.whole_grid import WholeGridPredictor
from hbllm.hcir.world.world_context import WorldModelScope
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class PredictiveRealityModel:
    """Multi-horizon ensemble predictive reality model."""

    def __init__(
        self,
        enable_whole_grid: bool = False,
        weight_manager: PredictorWeightManager | None = None,
    ) -> None:
        self.physics = PhysicsPredictor()
        self.snn = SNNTemporalPredictor()
        self.whole_grid = WholeGridPredictor()
        self.enable_whole_grid = enable_whole_grid
        self.disagreement_analyzer = PredictorDisagreementAnalyzer()
        self.weight_manager = weight_manager or PredictorWeightManager()

    def predict(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
        provenance: PredictionProvenance | None = None,
        scope: WorldModelScope | None = None,
    ) -> EnsemblePrediction:
        """Evaluate ensemble predictors and return unified EnsemblePrediction."""
        comp_results: dict[str, tuple[dict[str, Any], float]] = {}

        p_state, p_conf = self.physics.predict_state(snapshot, action_intent, horizon_ms)
        comp_results["physics"] = (p_state, p_conf)

        snn_state, snn_conf = self.snn.predict_state(snapshot, action_intent, horizon_ms)
        comp_results["snn"] = (snn_state, snn_conf)

        has_grid = (
            self.enable_whole_grid
            or "grid" in snapshot.variables
            or "grid_2d" in snapshot.variables
        )
        if has_grid:
            wg_state, wg_conf = self.whole_grid.predict_state(snapshot, action_intent, horizon_ms)
            comp_results["whole_grid"] = (wg_state, wg_conf)

        predictors_used = ["physics", "snn"]
        if has_grid:
            predictors_used.append("whole_grid")

        model_scope = scope or WorldModelScope(world_id=snapshot.world_id)
        raw_scores = {
            name: max(
                0.001,
                self.weight_manager.get_predictor_weight(model_scope, name).compute_score(),
            )
            for name in predictors_used
        }
        total_score = sum(raw_scores.values()) or 1.0
        norm_weights = {name: score / total_score for name, score in raw_scores.items()}
        avg_confidence = sum(
            comp_results[name][1] * norm_weights[name]
            for name in predictors_used
            if name in comp_results
        )

        disagreement = self.disagreement_analyzer.analyze_disagreement(comp_results)

        unified_state = dict(p_state)
        if "whole_grid" in comp_results and "grid" in comp_results["whole_grid"][0]:
            unified_state["grid"] = comp_results["whole_grid"][0]["grid"]
        if disagreement.high_disagreement:
            avg_confidence *= 0.85

        prov = provenance or PredictionProvenance(
            world_id=snapshot.world_id,
            predictors_used=predictors_used,
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

    def predict_and_transact(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
        provenance: PredictionProvenance | None = None,
        scope: WorldModelScope | None = None,
    ) -> tuple[EnsemblePrediction, list[TransactionOperation]]:
        """Compute ensemble prediction and generate corresponding HCIR transaction proposals."""
        pred = self.predict(
            snapshot=snapshot,
            action_intent=action_intent,
            horizon_ms=horizon_ms,
            provenance=provenance,
            scope=scope,
        )
        return pred, pred.to_transaction_operations()

    def export_causal_predicates(self, ensemble: EnsemblePrediction) -> list[CausalPredicate]:
        """Export prediction state variables and provenance as declarative causal predicates."""
        preds = ensemble.to_causal_predicates()
        preds.append(ensemble.provenance.to_causal_predicate())
        return preds


__all__ = [
    "PredictiveRealityModel",
]
