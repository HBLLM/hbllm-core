"""
Prediction Types & Provenance — Structured Containers for Ensemble Predictions & Provenance.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionProvenance:
    """Provenance chain tracking requestor, tenant, world, and predictor models used."""

    requestor: str = "executive_controller"
    tenant_id: str = "default_tenant"
    world_id: str = "default_world"
    entity_ids: Sequence[str] = field(default_factory=tuple)
    predictors_used: Sequence[str] = field(default_factory=tuple)
    model_version: str = "world-model-v1.0"
    timestamp: float = field(default_factory=time.time)


@dataclass
class EnsemblePrediction:
    """Aggregated prediction result from ensemble reality predictors."""

    prediction_id: str
    action_intent: str
    predicted_state: dict[str, Any]
    calibrated_confidence: float
    component_predictions: dict[str, Any] = field(default_factory=dict)
    provenance: PredictionProvenance = field(default_factory=PredictionProvenance)
