"""ARC-AGI and ARC-AGI-3 Cognitive USB Peripheral Driver Adapter."""

from hbllm.hcir.world.percept import PerceptionRegistry

from .arc_driver import ARC3Driver, ArcadeDriver
from .arc_evaluator import ARCBenchmarkRunner, ARCTaskAdapter
from .arc_macro_actions import (
    ArcClickCell,
    ToggleSwitch,
    arc_macro_plan_synthesis_hook,
    register_arc_macro_actions,
)
from .arc_perception import ARCPerceptionAdapter, extract_arc_objects
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

# Register plugin adapter into core registry
PerceptionRegistry.register("arc_agi", ARCPerceptionAdapter(), set_default=True)

# Register ARC domain macro actions and driver hooks into Core MacroActionDriverManager
register_arc_macro_actions()

__all__ = [
    "ARC3Driver",
    "ArcadeDriver",
    "ARCPerceptionAdapter",
    "extract_arc_objects",
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
    "ToggleSwitch",
    "ArcClickCell",
    "arc_macro_plan_synthesis_hook",
    "register_arc_macro_actions",
]
