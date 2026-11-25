"""Autonomous Learning Integration — Phases B, C, D (Facade).

Connects the existing HCIR cognitive faculties into a closed self-improving loop.
This facade re-exports all symbols from `hbllm.hcir.world.learning` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.learning import (
    CausalPlanningContext,
    CausalTheoryPlanner,
    CrossContextKnowledgeTransfer,
    GameFeatureSignature,
    MetaLearningGovernor,
    RelationalSubgoalMacro,
    RelationalWaypoint,
    StrategyOutcome,
    TransferableInvariant,
)

__all__ = [
    "CausalPlanningContext",
    "CausalTheoryPlanner",
    "CrossContextKnowledgeTransfer",
    "GameFeatureSignature",
    "MetaLearningGovernor",
    "RelationalSubgoalMacro",
    "RelationalWaypoint",
    "StrategyOutcome",
    "TransferableInvariant",
]
