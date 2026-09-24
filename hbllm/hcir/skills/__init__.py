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

from hbllm.hcir.skills.coupled_controllables import (
    CoupledControllableModel,
    CoupledControllableSkillAcquisition,
)
from hbllm.hcir.skills.kinematics import (
    KinematicModel,
    KinematicMomentumSkillAcquisition,
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
from hbllm.hcir.skills.spatiotemporal import (
    AutonomousTrajectoryModel,
    EntityTrajectory,
    SpatiotemporalSkillAcquisition,
)

__all__ = [
    "AutonomousTrajectoryModel",
    "CoupledControllableModel",
    "CoupledControllableSkillAcquisition",
    "EntityTrajectory",
    "GF2LinearSolver",
    "KinematicModel",
    "KinematicMomentumSkillAcquisition",
    "MorphologicalProgramSynthesis",
    "MorphologicalTransformation",
    "PermutationAlgebraSkillAcquisition",
    "RelationalAffordanceRule",
    "RelationalAffordanceSkillAcquisition",
    "SpatiotemporalSkillAcquisition",
    "ToggleIncidenceModel",
]
