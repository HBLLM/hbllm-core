"""Prediction Lifecycle State Machine — Tracks Prediction Evolution & Real-world Verification.

State machine tracking prediction verification and learning lifecycles, lifted
to declarative CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class PredictionState(StrEnum):
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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export lifecycle state as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"lifecycle:{self.prediction_id}",
            operator="==",
            value=self.state.value,
            target_entity=self.prediction_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction updating prediction lifecycle state."""
        return [
            TransactionOperation(
                op=TransactionOp.MODIFY_NODE,
                node_id=f"pred_{self.prediction_id}",
                changes={"lifecycle_state": self.state.value},
            )
        ]
