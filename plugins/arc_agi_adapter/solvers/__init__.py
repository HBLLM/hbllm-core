"""ARC-AGI and ARC-AGI-3 Modular Archetype Policies and Target Scorers Package."""

from plugins.arc_agi_adapter.solvers.archetypes import (
    ARC3ArchetypeScorer,
    ARC3DeliberationPolicy,
    install_arc3_solvers,
)
from plugins.arc_agi_adapter.solvers.symmetry_solver import (
    OnlineSymmetryReflectionSolver,
)

__all__ = [
    "ARC3ArchetypeScorer",
    "ARC3DeliberationPolicy",
    "OnlineSymmetryReflectionSolver",
    "install_arc3_solvers",
]
