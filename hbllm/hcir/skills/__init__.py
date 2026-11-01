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
    ColorCluster,
    DiscreteVectorTranslator,
    LatticeNavigator,
    LatticeQuantizer,
    PerceptualClusterDetector,
    Raycaster2D,
    RemoteActuator,
    TemporalSyncPlanner,
)
from hbllm.hcir.skills.curriculum import (
    CurricularSkillGraph,
    CurricularSkillNode,
    SkillTier,
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
    HomogeneousRegionPredicate,
    MetadataPredicate,
    Not,
    PearlCausalDoPredicate,
    PixelDensityPredicate,
    PredictiveConfidencePredicate,
    SkillEvaluationContext,
    SubgoalSequence,
    SymbolicPredicate,
    SymbolicSubgoal,
    SymmetryPredicate,
)
from hbllm.hcir.skills.induction import (
    SkillInductionEngine,
)

__all__ = [
    "ActionAffordancePredicate",
    "AllOf",
    "AnyOf",
    "BaseHierarchicalSkill",
    "ColorCluster",
    "ColorDistributionPredicate",
    "CurricularSkillGraph",
    "CurricularSkillNode",
    "CutSetPredicate",
    "DeclarativeNeuroSymbolicSkill",
    "DiscreteVectorTranslator",
    "EntityBoundingBoxPredicate",
    "EntityCountPredicate",
    "GridDimensionPredicate",
    "HomogeneousRegionPredicate",
    "LatticeNavigator",
    "LatticeQuantizer",
    "MetadataPredicate",
    "Not",
    "PearlCausalDoPredicate",
    "PerceptualClusterDetector",
    "PixelDensityPredicate",
    "PredictiveConfidencePredicate",
    "Raycaster2D",
    "RemoteActuator",
    "SkillEvaluationContext",
    "SkillInductionEngine",
    "SkillTier",
    "SubgoalSequence",
    "SymbolicPredicate",
    "SymbolicSubgoal",
    "SymmetryPredicate",
    "TemporalSyncPlanner",
]
