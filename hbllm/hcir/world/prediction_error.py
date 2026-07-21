"""
Prediction Error Node & Typology — Distinguishes Model Error from Environment Change.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PredictionErrorTypology(str, Enum):
    """Typology classifying prediction error root cause."""

    MODEL_ERROR = "model_error"
    ENVIRONMENT_CHANGE = "environment_change"


@dataclass
class PredictionErrorNode:
    """Graph node capturing expectation vs reality variance and error typology."""

    error_id: str = field(default_factory=lambda: f"perr_{uuid.uuid4().hex[:8]}")
    prediction_id: str = ""
    prediction_source: str = "physics"
    typology: PredictionErrorTypology = PredictionErrorTypology.MODEL_ERROR
    expected_state: dict[str, Any] = field(default_factory=dict)
    actual_state: dict[str, Any] = field(default_factory=dict)
    error_variance: float = 0.0
    cause_hypothesis: str = "unknown"
    confidence_before: float = 0.90
    confidence_after: float = 0.70
    weight_delta: float = -0.10
    created_at: float = field(default_factory=time.time)
