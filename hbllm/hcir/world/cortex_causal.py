"""Ventromedial Prefrontal Cortex (vmPFC) & Inferior Parietal Causal Induction Faculty.

Biologically modeled on mammalian vmPFC distal causal credit assignment (Pearl's do(a) intervention)
and parietal tool affordance resonance (Iriki-Maravita body schema plasticity):
1. Interventional Action-Effect Binding:
   - Evaluates do(a) interventions: correlates intentional actions at origin/target locations
     with local and distal state mutations (diffs).
   - Generates and verifies causal hypotheses for switches, levers, plates, and keys.
2. Neuro-Symbolic World Theory:
   - Grounds first-order causal predicates for goals, barriers, walkable areas, and pushable cargo.
   - Enforces the HCIR Attractor Invariant across diverse environments without heuristics.
3. Parietal Tool-Barrier Resonance:
   - Tracks held objects in the extended body schema and evaluates barrier permeability.
4. Remote Mechanism Reverse Affordance Lookup:
   - When an objective is blocked by a barrier, queries vmPFC for governing distal triggers.

This module is a lightweight backward-compatible facade delegating to .
"""

from __future__ import annotations

from hbllm.hcir.world.causal import (
    ActionAffordance,
    ConditionalAffordance,
    GovernedSetView,
    HCIRSymbolicWorldTheory,
    OntologyEvidenceRank,
    SpelkePhysicalAxiom,
    invalidate_stale_context_ontology,
    parse_evidence,
)
from hbllm.hcir.world.causal_discovery import CausalPredicate

__all__ = [
    "ActionAffordance",
    "CausalPredicate",
    "ConditionalAffordance",
    "GovernedSetView",
    "HCIRSymbolicWorldTheory",
    "OntologyEvidenceRank",
    "SpelkePhysicalAxiom",
    "invalidate_stale_context_ontology",
    "parse_evidence",
]
