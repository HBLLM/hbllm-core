"""Experimental Cortical Faculties (WP2).

Faculties that are not part of the active canonical 5-stage inference pipeline
reside here and can be lazily loaded or ablated without polluting production exports.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import for experimental faculties."""
    import hbllm.hcir.world as world

    if hasattr(world, name):
        return getattr(world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AllostaticRegime",
    "AmygdalarThreatConditioningEngine",
    "AnteriorCingulateEffortAllocator",
    "AnteriorInsularTimingEngine",
    "BoundaryGridPhaseAligner",
    "CerebellarMicrozoneAdaptor",
    "ClaustrumInterhemisphericIntegrator",
    "ClaustrumSalienceConductor",
    "DorsolateralRuleAbstractionEngine",
    "DualValueOpponencyCortex",
    "EntorhinalBorderCellEngine",
    "FrontopolarExplorationHorizonEngine",
    "GlobalWorkspaceIgnitionEngine",
    "HippocampalPhasePrecessionEngine",
    "HippocampalPrioritizedReplayBuffer",
    "HippocampalTimeCellEngine",
    "InferotemporalShapeHasher",
    "InteroceptiveAllostasisCortex",
    "LateralPrefrontalTaskSwitcher",
    "LocusCoeruleusNoradrenalineEngine",
    "MedialPrefrontalSchemaEngine",
    "ParietalFoveatedAttentionEngine",
    "ParietalPotentialFieldNavigator",
    "PedunculopontineLocomotorEngine",
    "PrefrontalBlameAttributor",
    "PulvinarAttentionRouter",
    "RetrosplenialHeadingEngine",
    "RetrosplenialLandmarkAnchorEngine",
    "SomatosensoryHapticEngine",
    "SpatialDistanceTransformEngine",
    "SubthalamicDriftDiffusionEngine",
    "SubthalamicThresholdEscalator",
    "SuperiorColliculusTargetArbiter",
    "ThalamicReticularGatingEngine",
    "VentralPremotorAffordanceResonator",
    "VentralTegmentalCuriosityEngine",
    "VentrolateralMotorInhibitionEngine",
    "VentromedialSomaticMarkerEngine",
]
