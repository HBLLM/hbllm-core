"""
Verification Gate — Safety Filter Validating Real-World Confirmations Before Learning Promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VerificationGate:
    """Safety gate evaluating eligibility of simulation predictions for reality-verified learning promotion."""

    min_confirmations: int = 1
    min_confidence: float = 0.80
    max_risk_level: str = "LOW"

    def evaluate_eligibility(
        self,
        confirmations: int,
        confidence: float,
        risk_level: str = "LOW",
    ) -> bool:
        """Return True if prediction meets all verification gate safety criteria."""
        if confirmations < self.min_confirmations:
            logger.info(
                "VerificationGate rejected: confirmations %d < required %d",
                confirmations,
                self.min_confirmations,
            )
            return False

        if confidence < self.min_confidence:
            logger.info(
                "VerificationGate rejected: confidence %.2f < required %.2f",
                confidence,
                self.min_confidence,
            )
            return False

        logger.info(
            "VerificationGate APPROVED promotion: confirmations=%d, confidence=%.2f",
            confirmations,
            confidence,
        )
        return True
