"""Prediction Promotion — Promotes Verified Simulation Predictions to Reality Learning Updates.

Manages promotion of verified simulation predictions to reality weight updates,
procedural memory skills, declarative CausalPredicates, and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.verification_gate import RiskLevel, VerificationGate

if TYPE_CHECKING:
    from hbllm.hcir.world.cortex_causal import ActionAffordance

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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export promoted skill receipt as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"promoted_skill:{self.promotion_id}",
            operator="==",
            value=self.confirmed_outcome,
            relation=self.action_intent,
            target_entity=self.prediction_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction committing promoted skill to procedural memory."""
        return [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=f"promoted_skill_{self.promotion_id}",
                node_data={
                    "promotion_id": self.promotion_id,
                    "prediction_id": self.prediction_id,
                    "action_intent": self.action_intent,
                    "confirmed_outcome": self.confirmed_outcome,
                    "confidence": self.confidence,
                    "promoted_at": self.promoted_at,
                },
            )
        ]

    def to_affordance(self) -> ActionAffordance:
        """Synthesize verified procedural action affordance."""
        from hbllm.hcir.world.cortex_causal import ActionAffordance

        return ActionAffordance(
            action_id=self.action_intent,
            name=f"promoted_{self.action_intent}",
            confidence=self.confidence,
            metadata={
                "promotion_id": self.promotion_id,
                "confirmed_outcome": self.confirmed_outcome,
            },
        )


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
        risk_level: RiskLevel | str = RiskLevel.LOW,
    ) -> PromotedSkillReceipt | None:
        """Evaluate prediction against VerificationGate and emit PromotedSkillReceipt if approved."""
        if not self.gate.evaluate_eligibility(confirmations, confidence, str(risk_level)):
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

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all promoted skills as declarative causal predicates."""
        return [p.to_causal_predicate() for p in self._promotions]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions for all promoted skill receipts."""
        ops: list[TransactionOperation] = []
        for p in self._promotions:
            ops.extend(p.to_transaction_operations())
        return ops
