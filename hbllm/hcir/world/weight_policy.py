"""
Weight Update Policy — Mathematical Delta Learning Rule for Predictor Weights.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WeightUpdatePolicy:
    """Mathematical delta learning policy updating predictor weights."""

    def __init__(self, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate

    def calculate_delta(self, expected_value: float, actual_value: float) -> float:
        """Calculate accuracy delta from prediction error variance."""
        error = abs(expected_value - actual_value)
        # Low error yields positive accuracy delta; high error yields negative delta
        if error < 0.05:
            return 1.0
        elif error < 0.20:
            return 0.5
        else:
            return -1.0
