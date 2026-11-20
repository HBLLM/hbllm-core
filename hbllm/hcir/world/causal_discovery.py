"""Active Interventional Causal Discovery Engine for HCIR World Kernel (Facade).

Implements domain-agnostic active hypothesis generation, interventional probing,
strict falsification, Bayesian posterior updating, composite multi-object predicate discovery,
and causal rule induction under confounding.
This facade re-exports all symbols from `hbllm.hcir.world.causal` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.causal import (
    BaseCausalDiscoveryEngine,
    BeliefTransitionEvent,
    BeliefTransitionType,
    CausalHypothesis,
    CausalMediationResult,
    CausalPredicate,
    LogicOperator,
    MediationIdentifiabilityAssumptions,
)

__all__ = [
    "BaseCausalDiscoveryEngine",
    "BeliefTransitionEvent",
    "BeliefTransitionType",
    "CausalHypothesis",
    "CausalMediationResult",
    "CausalPredicate",
    "LogicOperator",
    "MediationIdentifiabilityAssumptions",
]
