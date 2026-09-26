"""
Surprise Engine — Computes Confidence-Scaled Surprise & Attention Salience Boosting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.hcir.world.prediction_error import PredictionErrorTypology, WorldPredictionError

logger = logging.getLogger(__name__)


@dataclass
class SurpriseEvaluation:
    """Evaluation result of surprise computation."""

    surprise_score: float
    is_surprising: bool
    prediction_error_node: WorldPredictionError | None = None
    salience_boost: float = 0.0
    is_persistent: bool = False


class SurpriseEngine:
    """Surprise engine computing error variance scaled by confidence, salience, and persistence."""

    def __init__(
        self,
        surprise_threshold: float = 0.15,
        persistent_surprise_threshold: int = 2,
    ) -> None:
        self.surprise_threshold = surprise_threshold
        self.persistent_surprise_threshold = persistent_surprise_threshold
        self._surprise_ledger: dict[str, list[float]] = {}

    def reset_ledger(self, context_key: str | None = None) -> None:
        """Clear surprise history for a context key or entire ledger."""
        if context_key is not None:
            self._surprise_ledger.pop(context_key, None)
        else:
            self._surprise_ledger.clear()

    def evaluate_surprise(
        self,
        prediction_id: str,
        expected_state: dict,
        actual_state: dict,
        confidence: float = 0.90,
        attention_salience: float = 1.0,
        prediction_source: str = "physics",
        context_signature: str = "",
    ) -> SurpriseEvaluation:
        """Compute surprise score: Error * (0.5 + Confidence) * AttentionSalience."""
        # Calculate raw state variance
        keys = set(expected_state.keys()).union(actual_state.keys())
        total_diff = 0.0
        count = 0

        for k in keys:
            v_exp = expected_state.get(k)
            v_act = actual_state.get(k)
            if isinstance(v_exp, (int, float)) and isinstance(v_act, (int, float)):
                diff = abs(v_exp - v_act)
                norm = max(1.0, abs(v_exp))
                total_diff += diff / norm
                count += 1
            elif v_exp != v_act:
                # Qualitative state mismatch
                total_diff += 1.0
                count += 1

        raw_error = (total_diff / count) if count > 0 else 0.0

        # Surprise formula: Error * (0.5 + Confidence) * Salience
        surprise_score = raw_error * (0.5 + confidence) * attention_salience
        is_surprising = surprise_score >= self.surprise_threshold

        err_node = None
        salience_boost = 0.0
        is_persistent = False

        if is_surprising:
            salience_boost = 0.35
            context_key = context_signature or prediction_source
            history = self._surprise_ledger.setdefault(context_key, [])
            history.append(surprise_score)

            if len(history) >= self.persistent_surprise_threshold:
                is_persistent = True
                typology = PredictionErrorTypology.LATENT_CONFOUNDER
                logger.info(
                    "SurpriseEngine PERSISTENT SURPRISE DETECTED (count=%d, key='%s') -> LATENT_CONFOUNDER",
                    len(history),
                    context_key,
                )
            else:
                typology = (
                    PredictionErrorTypology.MODEL_ERROR
                    if confidence > 0.70
                    else PredictionErrorTypology.ENVIRONMENT_CHANGE
                )

            err_node = WorldPredictionError(
                prediction_id=prediction_id,
                prediction_source=prediction_source,
                typology=typology,
                expected_state=expected_state,
                actual_state=actual_state,
                error_variance=raw_error,
                confidence_before=confidence,
                confidence_after=max(0.1, confidence - 0.20),
                weight_delta=-0.15 if typology == PredictionErrorTypology.MODEL_ERROR else 0.0,
            )
            logger.info(
                "SurpriseEngine SURPRISE DETECTED: score=%.4f (boost=+%.2f, typology=%s)",
                surprise_score,
                salience_boost,
                typology.value,
            )

        return SurpriseEvaluation(
            surprise_score=surprise_score,
            is_surprising=is_surprising,
            prediction_error_node=err_node,
            salience_boost=salience_boost,
            is_persistent=is_persistent,
        )
