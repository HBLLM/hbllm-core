"""Perception Cortex Package."""

from hbllm.hcir.world.perception.diff import (
    EpistemicObservationDiff,
    FrameDiffType,
    OrientedThreat,
)
from hbllm.hcir.world.perception.engine import PerceptionEngine
from hbllm.hcir.world.perception.gestalt import (
    GestaltAffordanceDetector,
    SpatialRoleClassifier,
)
from hbllm.hcir.world.perception.thalamus import (
    FigureGroundSegregator,
    SensoryThalamus,
    _fast_unique_counts,
)

__all__ = [
    "EpistemicObservationDiff",
    "FigureGroundSegregator",
    "FrameDiffType",
    "GestaltAffordanceDetector",
    "OrientedThreat",
    "PerceptionEngine",
    "SensoryThalamus",
    "SpatialRoleClassifier",
    "_fast_unique_counts",
]
