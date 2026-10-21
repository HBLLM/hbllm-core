"""ARC-AGI and ARC-AGI-3 Modular Solvers and Archetype Policies Package."""

from plugins.arc_agi_adapter.solvers.archetypes import (
    ARC3ArchetypeScorer,
    ARC3DeliberationPolicy,
    install_arc3_solvers,
)
from plugins.arc_agi_adapter.solvers.game_solvers import (
    GravitySpillingPlatformSolver,
    LightsOutSolver,
    MemoryCardMatchingSolver,
    MinesweeperGameSolver,
    MirroredConvergenceSolver,
    PegSolitaireSolver,
    SentryCombatSolver,
    TrackMazeSolver,
    VortexAttractorSolver,
)

__all__ = [
    "ARC3ArchetypeScorer",
    "ARC3DeliberationPolicy",
    "install_arc3_solvers",
    "SentryCombatSolver",
    "MinesweeperGameSolver",
    "MemoryCardMatchingSolver",
    "VortexAttractorSolver",
    "PegSolitaireSolver",
    "LightsOutSolver",
    "TrackMazeSolver",
    "MirroredConvergenceSolver",
    "GravitySpillingPlatformSolver",
]
