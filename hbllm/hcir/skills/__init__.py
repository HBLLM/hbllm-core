"""HCIR Skill Acquisition System.

Provides domain-agnostic cognitive skill induction modules across game archetypes:
1. Spatiotemporal & Periodic Phase Acquisition (moving hazards, velocity vectors, space-time A*)
2. Algebraic & Permutation Operator Inversion (toggle incidence, GF(2) linear solver)
3. Kinematics, Inertia & Momentum Models (sliding ice, raycasting, gravity)
4. Relational Object-to-Object Affordances (tool use, Sokoban pushing, bridging)
5. Morphological Program Synthesis (visual analogies, symmetry, rotation, canvas stamping)
6. Coupled Controllable Coordination (mirrored agents, joint configuration search)
"""

from __future__ import annotations

from hbllm.hcir.skills.automaton_synthesis import (
    AutomatonProgramSynthesisSkillAcquisition,
)
from hbllm.hcir.skills.coupled_controllables import (
    CoupledControllableModel,
    CoupledControllableSkillAcquisition,
)
from hbllm.hcir.skills.grammar_translation import (
    GrammarTranslationSkillAcquisition,
)
from hbllm.hcir.skills.hierarchical_pattern_grammar import (
    HierarchicalPatternGrammarSkillAcquisition,
)
from hbllm.hcir.skills.inverted_buoyancy import (
    InvertedBuoyancySkillAcquisition,
)
from hbllm.hcir.skills.kinematics import (
    KinematicModel,
    KinematicMomentumSkillAcquisition,
)
from hbllm.hcir.skills.kinetic_coupling import (
    KineticCouplingSkillAcquisition,
)
from hbllm.hcir.skills.laser_routing import (
    LaserRoutingSkillAcquisition,
)
from hbllm.hcir.skills.modal_incantation import (
    IncantationGlyph,
    IncantationType,
    ModalIncantationSkillAcquisition,
)
from hbllm.hcir.skills.morphological_mutation import (
    MorphologicalStateMutationSkillAcquisition,
)
from hbllm.hcir.skills.morphology_synthesis import (
    MorphologicalProgramSynthesis,
    MorphologicalTransformation,
)
from hbllm.hcir.skills.permutation_algebra import (
    GF2LinearSolver,
    PermutationAlgebraSkillAcquisition,
    ToggleIncidenceModel,
)
from hbllm.hcir.skills.relational_affordance import (
    RelationalAffordanceRule,
    RelationalAffordanceSkillAcquisition,
)
from hbllm.hcir.skills.reticle_superposition import (
    ReticleSuperpositionSkillAcquisition,
)
from hbllm.hcir.skills.rigid_assembly import (
    RigidAssemblySkillAcquisition,
)
from hbllm.hcir.skills.spatiotemporal import (
    AutonomousTrajectoryModel,
    EntityTrajectory,
    SpatiotemporalSkillAcquisition,
)
from hbllm.hcir.skills.temporal_echo import (
    TemporalEchoSkillAcquisition,
)
from hbllm.hcir.skills.topology_transformation import (
    TopologyTransformationSkillAcquisition,
)
from hbllm.hcir.skills.visual_canvas import (
    VisualCanvasSkillAcquisition,
)
from hbllm.hcir.skills.visual_program import (
    VisualProgramSynthesisSkillAcquisition,
)
from hbllm.hcir.skills.vortex_attractor import (
    VortexAttractorSkillAcquisition,
)

__all__ = [
    "AutomatonProgramSynthesisSkillAcquisition",
    "AutonomousTrajectoryModel",
    "CoupledControllableModel",
    "CoupledControllableSkillAcquisition",
    "EntityTrajectory",
    "GF2LinearSolver",
    "GrammarTranslationSkillAcquisition",
    "HierarchicalPatternGrammarSkillAcquisition",
    "IncantationGlyph",
    "IncantationType",
    "InvertedBuoyancySkillAcquisition",
    "KinematicModel",
    "KinematicMomentumSkillAcquisition",
    "KineticCouplingSkillAcquisition",
    "LaserRoutingSkillAcquisition",
    "ModalIncantationSkillAcquisition",
    "MorphologicalProgramSynthesis",
    "MorphologicalStateMutationSkillAcquisition",
    "MorphologicalTransformation",
    "PermutationAlgebraSkillAcquisition",
    "RelationalAffordanceRule",
    "RelationalAffordanceSkillAcquisition",
    "ReticleSuperpositionSkillAcquisition",
    "RigidAssemblySkillAcquisition",
    "SpatiotemporalSkillAcquisition",
    "TemporalEchoSkillAcquisition",
    "ToggleIncidenceModel",
    "TopologyTransformationSkillAcquisition",
    "VisualCanvasSkillAcquisition",
    "VisualProgramSynthesisSkillAcquisition",
    "VortexAttractorSkillAcquisition",
]
