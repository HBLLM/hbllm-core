"""
Prediction Lifecycle State Machine — Tracks Prediction Evolution & Real-world Verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PredictionState(str, Enum):
    """Lifecycle states of a prediction."""

    CREATED = "created"
    EXECUTED = "executed"
    OBSERVED = "observed"
    VERIFIED = "verified"
    ERROR_DETECTED = "error_detected"
    LEARNED = "learned"
    ARCHIVED = "archived"


@dataclass
class PredictionLifecycle:
    """State machine tracking the verification lifecycle of a prediction."""

    prediction_id: str
    state: PredictionState = PredictionState.CREATED

    def transition_to(self, new_state: PredictionState) -> None:
        """Advance prediction lifecycle to new state."""
        logger.debug(
            "PredictionLifecycle [%s]: %s -> %s",
            self.prediction_id,
            self.state.value,
            new_state.value,
        )
        self.state = new_state
