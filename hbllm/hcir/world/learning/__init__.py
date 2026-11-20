"""Autonomous Learning Subsystem — Causal Planning, Transfer, and Meta-Learning."""

from __future__ import annotations

from hbllm.hcir.world.learning.causal_planner import (
    CausalPlanningContext,
    CausalTheoryPlanner,
    RelationalSubgoalMacro,
    RelationalWaypoint,
    get_sub_attr,
)
from hbllm.hcir.world.learning.meta import (
    GameFeatureSignature,
    MetaLearningGovernor,
    StrategyOutcome,
)
from hbllm.hcir.world.learning.transfer import (
    CrossContextKnowledgeTransfer,
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
    "get_sub_attr",
]
