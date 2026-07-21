"""
Prediction Promotion — Promotes Verified Simulation Predictions to Reality Learning Updates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from hbllm.hcir.world.verification_gate import VerificationGate

logger = logging.getLogger(__name__)


@dataclass
class PromotedSkillReceipt:
    """Receipt of a promoted simulation prediction converted into procedural skill update."""

    promotion_id: str
    prediction_id: str
    action_intent: str
    confirmed_outcome: str
    confidence: float
    promoted_at: float = field(default_factory=time.time)


class PredictionPromotion:
    """Manages promotion of verified simulation predictions to reality weight updates and procedural memory."""

    def __init__(self, verification_gate: VerificationGate | None = None) -> None:
        self.gate = verification_gate or VerificationGate()
        self._promotions: list[PromotedSkillReceipt] = []

    def promote_prediction(
        self,
        prediction_id: str,
        action_intent: str,
        confirmed_outcome: str,
        confirmations: int,
        confidence: float,
        risk_level: str = "LOW",
    ) -> PromotedSkillReceipt | None:
        """Evaluate prediction against VerificationGate and emit PromotedSkillReceipt if approved."""
        if not self.gate.evaluate_eligibility(confirmations, confidence, risk_level):
            return None

        receipt = PromotedSkillReceipt(
            promotion_id=f"promo_{prediction_id}",
            prediction_id=prediction_id,
            action_intent=action_intent,
            confirmed_outcome=confirmed_outcome,
            confidence=confidence,
        )
        self._promotions.append(receipt)
        logger.info(
            "PredictionPromotion SUCCESS: Promoted prediction '%s' for action '%s'",
            prediction_id,
            action_intent,
        )
        return receipt

    def all_promotions(self) -> list[PromotedSkillReceipt]:
        """Retrieve all promoted receipts."""
        return list(self._promotions)
