"""Prediction & Counterfactual Budget — Compute & Latency Resource Governor for Predictions.

Resource governor constraining simulation depth and runtime latency, lifted to
declarative CausalPredicates and atomic TransactionOperations for cognitive governance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualBudget:
    """Governor constraints on counterfactual simulation branch depth and compute."""

    max_depth: int = 5
    max_nodes: int = 100
    max_compute_pct: float = 0.30

    def to_causal_predicate(self) -> CausalPredicate:
        """Export counterfactual budget as a declarative causal predicate."""
        return CausalPredicate(
            variable="counterfactual_budget_depth",
            operator="<=",
            value=self.max_depth,
            relation="max_depth",
        )


@dataclass
class PredictionBudget:
    """Resource governor selecting predictor execution tiers based on cognitive budget."""

    max_latency_ms: int = 500
    max_cost_units: float = 10.0
    counterfactual_budget: CounterfactualBudget = field(default_factory=CounterfactualBudget)

    def select_predictor_tier(
        self, horizon_ms: int, available_cognitive_budget: float = 1.0
    ) -> list[str]:
        """Dynamically select predictor list based on horizon and available cognitive budget."""
        if available_cognitive_budget < 0.3:
            logger.info(
                "PredictionBudget LOW cognitive budget (%.2f): using fast predictors",
                available_cognitive_budget,
            )
            return ["physics", "statistical"]
        elif available_cognitive_budget < 0.7:
            return ["physics", "statistical", "snn"]
        else:
            return ["physics", "statistical", "snn", "neural", "llm"]

    def compute_search_budget(
        self,
        has_temporal: bool = False,
        has_threats: bool = False,
        consecutive_failures: int = 0,
        is_under_strict_budget: bool = False,
        remaining_steps: int | None = None,
    ) -> int:
        """Compute anytime A* / MCTS state expansion budget based on temporal hazard complexity
        and acute cognitive step limits.
        """
        # Under acute step urgency (<= 25 steps remaining), cap search expansions to 600
        if is_under_strict_budget or (remaining_steps is not None and remaining_steps <= 25):
            return 600

        base = 1500
        if consecutive_failures > 0 and not has_temporal:
            return int(base * 0.8)
        if has_temporal:
            return int(base * 3.33)
        return base

    def to_causal_predicate(self) -> CausalPredicate:
        """Export prediction budget constraint as a declarative causal predicate."""
        return CausalPredicate(
            variable="prediction_budget_latency",
            operator="<=",
            value=self.max_latency_ms,
            relation="max_latency_ms",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction configuring cognitive prediction budget."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="prediction_budget_governor",
                node_data={
                    "max_latency_ms": self.max_latency_ms,
                    "max_cost_units": self.max_cost_units,
                    "max_depth": self.counterfactual_budget.max_depth,
                    "max_nodes": self.counterfactual_budget.max_nodes,
                },
            )
        ]
