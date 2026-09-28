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

__all__ = [
    "BaseHierarchicalSkill",
    "BlockPushPlanner",
    "ColorCluster",
    "DiscreteVectorTranslator",
    "GF2LinearSolver",
    "LatticeNavigator",
    "LatticeQuantizer",
    "PerceptualClusterDetector",
    "Raycaster2D",
    "RemoteActuator",
    "TemporalSyncPlanner",
]
