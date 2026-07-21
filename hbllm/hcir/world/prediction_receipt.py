"""
World Prediction Receipt — Structured Replay & Trace Receipt for Predictions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from hbllm.hcir.world.prediction_types import PredictionProvenance


@dataclass
class WorldPredictionReceipt:
    """Deterministic trace receipt of a prediction execution."""

    receipt_id: str
    prediction_id: str
    model_version: str
    provenance: PredictionProvenance
    input_state_hash: str
    predicted_outcome: str
    actual_outcome: str | None = None
    calibrated_confidence: float = 0.90
    surprise_score: float = 0.0
    created_at: float = field(default_factory=time.time)
