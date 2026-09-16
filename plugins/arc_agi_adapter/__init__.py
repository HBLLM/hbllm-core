"""ARC-AGI and ARC-AGI-3 Interactive Reasoning Benchmark Adapter.

Provides native HCIR cognitive evaluation harnesses for:
1. Chollet ARC-AGI-1 / 2 static grid puzzles (relational transformation, topological parsing).
2. Official ARC-AGI-3 interactive games (active motor calibration, physical affordance induction,
   hierarchical subgoal trees, and multi-step mental rollouts).
"""

from .arc_agi_3_runner import (
    ActionDynamicsModel,
    ARC3BenchmarkReport,
    ARC3BenchmarkRunner,
    ARC3EnvironmentResult,
    ARC3InteractiveAgent,
    ARC3LevelResult,
    ARCGameState,
    StateMutationModel,
)
from .arc_agi_runner import (
    ARCBenchmarkReport,
    ARCGrid,
    ARCRelationalSolver,
    ARCTaskResult,
    GridObject,
    GridTopologyExtractor,
)
from .config import ARC3BenchmarkConfig

__all__ = [
    # ARC-1 / 2 Static Grids
    "ARCGrid",
    "GridObject",
    "GridTopologyExtractor",
    "ARCRelationalSolver",
    "ARCTaskResult",
    "ARCBenchmarkReport",
    # ARC-3 Interactive Games
    "ARCGameState",
    "ActionDynamicsModel",
    "StateMutationModel",
    "ARC3LevelResult",
    "ARC3EnvironmentResult",
    "ARC3BenchmarkReport",
    "ARC3InteractiveAgent",
    "ARC3BenchmarkRunner",
    "ARC3BenchmarkConfig",
]
