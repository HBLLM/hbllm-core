"""ARC-AGI and ARC-AGI-3 Cognitive USB Peripheral Driver Adapter."""

from .arc_driver import ARC3Driver, ArcadeDriver
from .arc_evaluator import ARCBenchmarkRunner, ARCTaskAdapter
from .solvers import (
    ARC3ArchetypeScorer,
    ARC3DeliberationPolicy,
    GravitySpillingPlatformSolver,
    LightsOutSolver,
    MemoryCardMatchingSolver,
    MinesweeperGameSolver,
    MirroredConvergenceSolver,
    PegSolitaireSolver,
    SentryCombatSolver,
    TrackMazeSolver,
    VortexAttractorSolver,
    install_arc3_solvers,
)

__all__ = [
    "ARC3Driver",
    "ArcadeDriver",
    "ARCTaskAdapter",
    "ARCBenchmarkRunner",
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
