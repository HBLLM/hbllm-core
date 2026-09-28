"""HCIR Skill Acquisition System — Domain-Agnostic Base.

Core provides only the skill protocol contract and reusable subskills.
All domain-specific skills (ARC-AGI) live in their plugin package:
    plugins/arc_agi_adapter/arc_skills/

Plugins register their skills with CognitiveBlackbox.register_skill()
at load time for generic dispatch via the BaseHierarchicalSkill protocol.
"""

from __future__ import annotations

from hbllm.hcir.skills.base import (
    BaseHierarchicalSkill,
)
from hbllm.hcir.skills.common_subskills import (
    BlockPushPlanner,
    ColorCluster,
    DiscreteVectorTranslator,
    GF2LinearSolver,
    LatticeNavigator,
    LatticeQuantizer,
    PerceptualClusterDetector,
    Raycaster2D,
    RemoteActuator,
    TemporalSyncPlanner,
)
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    AnyOf,
    ColorDistributionPredicate,
    CutSetPredicate,
    DeclarativeNeuroSymbolicSkill,
    EntityBoundingBoxPredicate,
    EntityCountPredicate,
    GridDimensionPredicate,
    Not,
    PanelConstraint,
    SkillEvaluationContext,
    SubgoalSequence,
    SymbolicPredicate,
    SymbolicSubgoal,
)

__all__ = [
    "ActionAffordancePredicate",
    "AllOf",
    "AnyOf",
    "BaseHierarchicalSkill",
    "BlockPushPlanner",
    "ColorCluster",
    "ColorDistributionPredicate",
    "CutSetPredicate",
    "DeclarativeNeuroSymbolicSkill",
    "DiscreteVectorTranslator",
    "EntityBoundingBoxPredicate",
    "EntityCountPredicate",
    "GF2LinearSolver",
    "GridDimensionPredicate",
    "LatticeNavigator",
    "LatticeQuantizer",
    "Not",
    "PanelConstraint",
    "PerceptualClusterDetector",
    "Raycaster2D",
    "RemoteActuator",
    "SkillEvaluationContext",
    "SubgoalSequence",
    "SymbolicPredicate",
    "SymbolicSubgoal",
    "TemporalSyncPlanner",
]
