"""Object-Centric Macro-Action State Graph Planner (Options Framework) (Facade).

Decouples high-level discrete choice across K salient objects from low-level physical distance.
This facade re-exports all symbols from `hbllm.hcir.world.objects` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.objects import (
    ExecutiveDirective,
    MacroOption,
    ObjectCategory,
    ObjectStateGraphPlanner,
    ObjectStateLedger,
    RelationalEdge,
    RelationalGraphMatcher,
    RelationalNode,
    SalientObject,
    _analogous_concept_hazard_risk,
    _extract_structural_signature,
    _scipy_label,
    compute_stance_priority,
    estimate_entity_risk,
)

__all__ = [
    "ExecutiveDirective",
    "MacroOption",
    "ObjectCategory",
    "ObjectStateGraphPlanner",
    "ObjectStateLedger",
    "RelationalEdge",
    "RelationalGraphMatcher",
    "RelationalNode",
    "SalientObject",
    "_analogous_concept_hazard_risk",
    "_extract_structural_signature",
    "_scipy_label",
    "compute_stance_priority",
    "estimate_entity_risk",
]
