"""
Prediction & Counterfactual Budget — Compute & Latency Resource Governor for Predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualBudget:
    """Governor constraints on counterfactual simulation branch depth and compute."""

    max_depth: int = 5
    max_nodes: int = 100
    max_compute_pct: float = 0.30


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
