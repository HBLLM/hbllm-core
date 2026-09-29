"""ARC-AGI Domain Skills — Hierarchical Cognitive Skill Library.

All ARC-AGI specific pattern recognition and plan synthesis skills,
migrated from core/hbllm/hcir/skills/ to the plugin layer.

These skills implement the BaseHierarchicalSkill protocol and are
registered with the CognitiveBlackbox's SkillRegistry at plugin load time.

The core retains only domain-agnostic infrastructure:
- BaseHierarchicalSkill (protocol contract)
- common_subskills (reusable grid primitives)
- primitives.py (topology, segmentation)
"""

from __future__ import annotations

# Re-export base and common subskills (these also live in core)
from hbllm.hcir.skills.base import BaseHierarchicalSkill
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

# Domain-specific ARC skills (canonical location: this package)
from plugins.arc_agi_adapter.arc_skills.automaton_synthesis import (
    AutomatonProgramSynthesisSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.coupled_controllables import (
    CoupledControllableModel,
    CoupledControllableSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.grammar_translation import (
    GrammarTranslationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.hierarchical_pattern_grammar import (
    HierarchicalPatternGrammarSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.inductive_skill_factory import (
    InductiveSkillFactory,
)
from plugins.arc_agi_adapter.arc_skills.inverted_buoyancy import (
    InvertedBuoyancySkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.kinematic_arm_linkage import (
    KinematicLinkageSolver,
)
from plugins.arc_agi_adapter.arc_skills.kinematics import (
    KinematicModel,
    KinematicMomentumSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.kinetic_coupling import (
    KineticCouplingSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.laser_routing import (
    LaserRoutingSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.modal_incantation import (
    IncantationGlyph,
    IncantationType,
    ModalIncantationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.morphological_mutation import (
    MorphologicalStateMutationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.morphology_synthesis import (
    MorphologicalProgramSynthesis,
    MorphologicalTransformation,
)
from plugins.arc_agi_adapter.arc_skills.optical_mirror_reflection import (
    OpticalMirrorReflectionSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.perceptual_context import (
    PerceptualSkillContext,
)
from plugins.arc_agi_adapter.arc_skills.permutation_algebra import (
    PermutationAlgebraSkillAcquisition,
    ToggleIncidenceModel,
)
from plugins.arc_agi_adapter.arc_skills.relational_affordance import (
    RelationalAffordanceRule,
    RelationalAffordanceSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.reticle_superposition import (
    ReticleSuperpositionSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.rigid_assembly import (
    RigidAssemblySkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.spatial_resource_navigation import (
    SpatialResourceNavigationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.spatiotemporal import (
    AutonomousTrajectoryModel,
    EntityTrajectory,
    SpatiotemporalSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.structural_fingerprint import (
    StructuralFingerprint,
    StructuralTransferRegistry,
    get_global_transfer_registry,
)
from plugins.arc_agi_adapter.arc_skills.temporal_echo import (
    TemporalEchoSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.topology_transformation import (
    TopologyTransformationSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.visual_canvas import (
    VisualCanvasSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.visual_program import (
    VisualProgramSynthesisSkillAcquisition,
)
from plugins.arc_agi_adapter.arc_skills.vortex_attractor import (
    VortexAttractorSkillAcquisition,
)

__all__ = [
    # Base protocol
    "BaseHierarchicalSkill",
    # Perceptual and transfer layers
    "PerceptualSkillContext",
    "InductiveSkillFactory",
    "StructuralFingerprint",
    "StructuralTransferRegistry",
    "get_global_transfer_registry",
    # Common subskills
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
    # Domain skills
    "AutomatonProgramSynthesisSkillAcquisition",
    "AutonomousTrajectoryModel",
    "CoupledControllableModel",
    "CoupledControllableSkillAcquisition",
    "EntityTrajectory",
    "GrammarTranslationSkillAcquisition",
    "HierarchicalPatternGrammarSkillAcquisition",
    "IncantationGlyph",
    "IncantationType",
    "InvertedBuoyancySkillAcquisition",
    "KinematicLinkageSolver",
    "KinematicModel",
    "KinematicMomentumSkillAcquisition",
    "KineticCouplingSkillAcquisition",
    "LaserRoutingSkillAcquisition",
    "ModalIncantationSkillAcquisition",
    "MorphologicalProgramSynthesis",
    "MorphologicalStateMutationSkillAcquisition",
    "MorphologicalTransformation",
    "OpticalMirrorReflectionSkillAcquisition",
    "PermutationAlgebraSkillAcquisition",
    "RelationalAffordanceRule",
    "RelationalAffordanceSkillAcquisition",
    "ReticleSuperpositionSkillAcquisition",
    "RigidAssemblySkillAcquisition",
    "SpatialResourceNavigationSkillAcquisition",
    "SpatiotemporalSkillAcquisition",
    "TemporalEchoSkillAcquisition",
    "ToggleIncidenceModel",
    "TopologyTransformationSkillAcquisition",
    "VisualCanvasSkillAcquisition",
    "VisualProgramSynthesisSkillAcquisition",
    "VortexAttractorSkillAcquisition",
]
