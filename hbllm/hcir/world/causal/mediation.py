"""Pearl Non-Parametric Causal Mediation Data Classes (W079)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MediationIdentifiabilityAssumptions:
    """Represents Pearl's 4 sequential conditional independence identifiability conditions (W079).

    1. Y(x, m) ⟂ X | W (no unmeasured treatment-outcome confounding)
    2. Y(x, m) ⟂ M | (X, W) (no unmeasured mediator-outcome confounding)
    3. M(x) ⟂ X | W (no unmeasured treatment-mediator confounding)
    4. No mediator-outcome confounder is affected by treatment X.
    """

    no_treatment_outcome_confounding: bool = True
    no_mediator_outcome_confounding: bool = True
    no_treatment_mediator_confounding: bool = True
    no_cross_world_counterfactual_confounding: bool = True
    covariates_controlled: list[str] = field(default_factory=list)

    def is_fully_identified(self) -> bool:
        """True iff all 4 Pearl/Imai identifiability conditions hold."""
        return (
            self.no_treatment_outcome_confounding
            and self.no_mediator_outcome_confounding
            and self.no_treatment_mediator_confounding
            and self.no_cross_world_counterfactual_confounding
        )


@dataclass
class CausalMediationResult:
    """Non-parametric causal mediation decomposition results (W079)."""

    treatment: str
    mediator: str
    outcome: str
    total_effect: float  # TE = E[Y(x1) - Y(x0)]
    natural_direct_effect: float  # NDE = E[Y(x1, M(x0)) - Y(x0, M(x0))]
    natural_indirect_effect: float  # NIE = E[Y(x1, M(x1)) - Y(x1, M(x0))]
    controlled_direct_effect: dict[Any, float]  # CDE(m) = E[Y(x1, m) - Y(x0, m)]
    proportion_mediated: float  # NIE / TE
    identifiability: MediationIdentifiabilityAssumptions
    is_causally_identified: bool
    unmeasured_confounding_warnings: list[str] = field(default_factory=list)
    estimation_regime: str = "pearl_counterfactual_stratification"
