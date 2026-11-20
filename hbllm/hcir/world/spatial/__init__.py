"""Spatial Coordinate Transformation Subsystem — Dual Reference Frames & PPC/RSC."""

from __future__ import annotations

from hbllm.hcir.world.spatial.models import (
    DihedralTransform,
    EgocentricCoordinate,
    EgocentricEntityRepresentation,
    EgocentricPolarCoord,
    GeometricMappingResult,
    GeometricModelType,
    HeadingOrientation,
    SpatialBearing,
    SpatialEgoVector,
)
from hbllm.hcir.world.spatial.transformers import (
    ParietalCoordinateTransformer,
    RetrosplenialCoordinateTransformer,
)
from hbllm.hcir.world.spatial.transforms import (
    AffineTransform2D,
    AxonometricIsometricTransform,
    GeometricModelSelector,
    ProjectiveHomography2D,
)

__all__ = [
    "AffineTransform2D",
    "AxonometricIsometricTransform",
    "DihedralTransform",
    "EgocentricCoordinate",
    "EgocentricEntityRepresentation",
    "EgocentricPolarCoord",
    "GeometricMappingResult",
    "GeometricModelSelector",
    "GeometricModelType",
    "HeadingOrientation",
    "ParietalCoordinateTransformer",
    "ProjectiveHomography2D",
    "RetrosplenialCoordinateTransformer",
    "SpatialBearing",
    "SpatialEgoVector",
]
