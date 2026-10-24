"""ARC-AGI and ARC-AGI-3 Modular Archetype Policies and Target Scorers Package."""

from plugins.arc_agi_adapter.solvers.archetypes import (
    ARC3ArchetypeScorer,
    ARC3DeliberationPolicy,
    install_arc3_solvers,
)
from plugins.arc_agi_adapter.solvers.forward_search_solver import (
    OnlineForwardSearchSolver,
)
from plugins.arc_agi_adapter.solvers.geometric_stamping_solver import (
    OnlineGeometricStampingSolver,
)
from plugins.arc_agi_adapter.solvers.grammar_translation_solver import (
    OnlineGrammarTranslationSolver,
)
from plugins.arc_agi_adapter.solvers.kinematic_pull_solver import (
    OnlineKinematicPullSolver,
)
from plugins.arc_agi_adapter.solvers.permutation_cycle_solver import (
    OnlinePermutationCycleSolver,
)
from plugins.arc_agi_adapter.solvers.program_synthesis_solver import (
    OnlineProgramSynthesisSolver,
)
from plugins.arc_agi_adapter.solvers.symmetry_solver import (
    OnlineSymmetryReflectionSolver,
)

__all__ = [
    "ARC3ArchetypeScorer",
    "ARC3DeliberationPolicy",
    "OnlineForwardSearchSolver",
    "OnlineGeometricStampingSolver",
    "OnlineGrammarTranslationSolver",
    "OnlineKinematicPullSolver",
    "OnlinePermutationCycleSolver",
    "OnlineProgramSynthesisSolver",
    "OnlineSymmetryReflectionSolver",
    "install_arc3_solvers",
]
