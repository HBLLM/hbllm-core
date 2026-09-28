"""ARC-AGI Solver & Analysis Modules.

Submodules:
    game_solvers     — Game-specific algorithmic solvers
    visual_analysis  — Visual perception & topology analysis
    spatial_navigation — Spatial navigation helpers
    knowledge_base   — Cross-level knowledge & causal inference

Import directly from submodules to avoid circular import chains:
    from plugins.arc_agi_adapter.arc_solvers.game_solvers import LightsOutSolver
"""

# Lazy re-exports via __all__ — no eager imports to prevent circular chains.
# Consumers should import from specific submodules directly.

__all__ = [
    # game_solvers
    "VortexAttractorSolver",
    "PegSolitaireSolver",
    "TrackMazeSolver",
    "LightsOutSolver",
    "MirroredConvergenceSolver",
    "GravitySpillingPlatformSolver",
    # visual_analysis
    "DiffType",
    "PuzzleTypology",
    "FrameDiff",
    "FrameDiffAnalyzer",
    "VisualEntity",
    "VisualTopologyExtractor",
    "CanvasRegion",
    "DynamicCanvasMatcher",
    "VisualSymmetryAnalyzer",
    "VisualCanvasMatcher",
    "CoupledMIMOIdentifier",
    # spatial_navigation
    "RoomDoor",
    "RoomTopologyExtractor",
    "DynamicSpatialNavigator",
    "SpatiotemporalNavigator",
    "SpatialResourceNavigator",
    # knowledge_base
    "ActionAffordance",
    "ControllableSignature",
    "ObjectInteractionRecipe",
    "CrossLevelKnowledgeBase",
    "TrialOutcome",
    "TrialFeedbackMemory",
    "GoalHypothesis",
    "GoalStateInductor",
    "CausalHypothesis",
    "CausalAffordanceEngine",
    "TemporalHazardTracker",
    "DynamicPermutationSolver",
]


def __getattr__(name: str):
    """Lazy import to avoid circular import chains."""
    if name in __all__[:6]:
        from plugins.arc_agi_adapter.arc_solvers import game_solvers

        return getattr(game_solvers, name)
    elif name in __all__[6:17]:
        from plugins.arc_agi_adapter.arc_solvers import visual_analysis

        return getattr(visual_analysis, name)
    elif name in __all__[17:22]:
        from plugins.arc_agi_adapter.arc_solvers import spatial_navigation

        return getattr(spatial_navigation, name)
    elif name in __all__[22:]:
        from plugins.arc_agi_adapter.arc_solvers import knowledge_base

        return getattr(knowledge_base, name)
    raise AttributeError(f"module 'arc_solvers' has no attribute {name!r}")
