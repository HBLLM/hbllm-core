"""
LLM Reasoning Predictor — High-level Qualitative Reasoner.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class LLMReasoningPredictor:
    """High-level qualitative LLM reasoning predictor."""

    name: str = "llm"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export qualitative predictor status as a declarative causal predicate."""
        return CausalPredicate(
            variable="predictor:llm",
            operator="==",
            value="active",
            relation="qualitative_reasoner",
            target_entity="llm",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering qualitative predictor."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="predictor_llm",
                node_data={
                    "name": self.name,
                    "type": "qualitative_reasoner",
                    "active": True,
                },
            )
        ]

    def predict_state(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
    ) -> tuple[dict[str, Any], float]:
        """Compute high-level qualitative prediction."""
        predicted = dict(snapshot.variables)
        predicted["qualitative_summary"] = (
            f"Action '{action_intent}' will likely maintain operational stability."
        )
        logger.debug(
            "LLMReasoningPredictor calculated state for action '%s' confidence=0.78", action_intent
        )
        return predicted, 0.78
