"""Risk Engine for Utility Scoring and Scenario Arbitration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from hbllm.brain.self_state import SelfStateEngine
from hbllm.brain.simulation.models import CounterfactualScenario

logger = logging.getLogger(__name__)


class RiskCategory(StrEnum):
    """Explicit categories for modeling simulation risk."""

    RESOURCE = "resource"  # e.g., battery drain, high token usage
    COGNITIVE = "cognitive"  # e.g., recursion storms, context window pressure
    SOCIAL = "social"  # e.g., interrupting the user during focus time
    RELIABILITY = "reliability"  # e.g., low confidence or highly speculative origin
    CASCADING = "cascading"  # e.g., destabilizing downstream dependencies


@dataclass
class RiskProfile:
    """Risk constraints for the autonomy engine."""

    max_speculative_actions_per_hour: int = 5
    max_counterfactual_branches: int = 10
    max_simulation_depth: int = 5
    risk_thresholds: dict[RiskCategory, float] | None = None

    def __post_init__(self) -> None:
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                RiskCategory.RESOURCE: 0.8,
                RiskCategory.COGNITIVE: 0.7,
                RiskCategory.SOCIAL: 0.5,
                RiskCategory.RELIABILITY: 0.9,
                RiskCategory.CASCADING: 0.6,
            }


class RiskEngine:
    """Evaluates utility and explicitly models risk for proposed scenarios."""

    def __init__(
        self, profile: RiskProfile | None = None, self_state: SelfStateEngine | None = None
    ) -> None:
        self.profile = profile or RiskProfile()
        self.self_state = self_state

    def evaluate_scenario(self, scenario: CounterfactualScenario) -> float:
        """Score a scenario. Higher score means better utility/lower risk."""
        if not scenario.predicted_state:
            return 0.0

        utility = scenario.utility_score

        # 1. Epistemic Calibration & Tool Reliability injection
        if self.self_state:
            # Check reliability of proposed tools
            lowest_reliability = 1.0
            for task in scenario.proposed_tasks:
                rel = self.self_state.tools.get_reliability(task.action_topic)
                lowest_reliability = min(lowest_reliability, rel)

            # Inverse of reliability contributes to RELIABILITY risk
            current_rel_risk = scenario.risk_categories.get(RiskCategory.RELIABILITY.value, 0.0)
            scenario.risk_categories[RiskCategory.RELIABILITY.value] = max(
                current_rel_risk, 1.0 - lowest_reliability
            )

            # Add cognitive stress risk
            stress = self.self_state.get_cognitive_pressure()
            if stress > 0.6:
                scenario.risk_categories[RiskCategory.COGNITIVE.value] = stress

        composite_risk = 0.0
        categories_count = 0

        for category_name, risk_val in scenario.risk_categories.items():
            try:
                cat_enum = RiskCategory(category_name)
            except ValueError:
                logger.warning("Unknown risk category: %s", category_name)
                continue

            threshold = self.profile.risk_thresholds.get(cat_enum, 1.0)  # type: ignore

            # If any risk exceeds its threshold, penalize heavily
            if risk_val >= threshold:
                utility *= 0.1  # Severe penalty
                logger.debug(
                    "Scenario %s penalized for exceeding risk threshold in %s",
                    scenario.scenario_id,
                    category_name,
                )

            composite_risk += risk_val
            categories_count += 1

        if categories_count > 0:
            composite_risk /= categories_count

        scenario.predicted_state.risk_score = composite_risk

        # Final score balances baseline utility against composite risk
        final_score = max(0.0, utility * (1.0 - composite_risk))
        return final_score
