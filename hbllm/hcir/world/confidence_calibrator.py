"""
Confidence Calibrator — Prediction Confidence Vector Scaling & Temporal Decay Calculator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionConfidenceVector:
    """Multi-dimensional confidence vector capturing raw, calibrated, decay, and historical metrics."""

    raw_confidence: float
    calibrated_confidence: float
    temporal_decay: float
    historical_accuracy: float
    uncertainty: float


class ConfidenceCalibrator:
    """Calibrates raw predictor confidence and computes temporal exponential decay."""

    def calibrate(
        self,
        raw_confidence: float,
        historical_accuracy: float = 0.90,
        age_seconds: float = 0.0,
        half_life_seconds: float = 3600.0,
    ) -> PredictionConfidenceVector:
        """Compute calibrated confidence vector and exponential temporal decay."""
        # Exponential temporal decay: exp(-t / half_life)
        temporal_decay = math.exp(-max(0.0, age_seconds) / max(1.0, half_life_seconds))
        calibrated = raw_confidence * historical_accuracy * temporal_decay
        uncertainty = max(0.0, 1.0 - calibrated)

        return PredictionConfidenceVector(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            temporal_decay=temporal_decay,
            historical_accuracy=historical_accuracy,
            uncertainty=uncertainty,
        )
