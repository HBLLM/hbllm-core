"""Prefrontal Cortex Inductive Logic & Relational Hypothesis Testing Faculty (Facade).

Biologically modeled on mammalian prefrontal cortex inductive hypothesis generation,
Popperian counterexample refutation, and causal rule abstraction:
1. Data Models & Observation Context (models.py)
2. Specialized Hypothesis Observers (observers.py)
3. PCFG Prior, Compositional Programs & Popperian Refutation (pcfg.py)
4. Inductive Hypothesis Engine (engine.py)
"""

from __future__ import annotations

from hbllm.hcir.world.hypothesis import (
    ActionMutationObserver,
    CompositionalProgramObserver,
    EntityStateDelta,
    HazardContactObserver,
    HypothesisObserver,
    HypothesisStatus,
    HypothesisTransitionEvent,
    HypothesisType,
    InductiveHypothesisEngine,
    PopperianFalsifier,
    RelationalCueObserver,
    RelationalRuleHypothesis,
    RemoteMechanismObserver,
    RulePCFGPrior,
    SynthesizedRuleAST,
    TransitionObservationContext,
    _shift_numeric_locus,
)

__all__ = [
    "ActionMutationObserver",
    "CompositionalProgramObserver",
    "EntityStateDelta",
    "HazardContactObserver",
    "HypothesisObserver",
    "HypothesisStatus",
    "HypothesisTransitionEvent",
    "HypothesisType",
    "InductiveHypothesisEngine",
    "PopperianFalsifier",
    "RelationalCueObserver",
    "RelationalRuleHypothesis",
    "RemoteMechanismObserver",
    "RulePCFGPrior",
    "SynthesizedRuleAST",
    "TransitionObservationContext",
    "_shift_numeric_locus",
]
