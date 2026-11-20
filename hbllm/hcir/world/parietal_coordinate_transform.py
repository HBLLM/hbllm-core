"""Posterior Parietal Cortex (PPC) Egocentric-to-Allocentric Spatial Coordinate Transformer (Facade).

Biologically modeled on mammalian posterior parietal cortex and hippocampal place/grid networks.
This facade re-exports all symbols from `hbllm.hcir.world.spatial` for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.world.spatial import (
    AffineTransform2D,
    AxonometricIsometricTransform,
    DihedralTransform,
    EgocentricCoordinate,
    EgocentricEntityRepresentation,
    EgocentricPolarCoord,
    GeometricMappingResult,
    GeometricModelSelector,
    GeometricModelType,
    HeadingOrientation,
    ParietalCoordinateTransformer,
    ProjectiveHomography2D,
    RetrosplenialCoordinateTransformer,
    SpatialBearing,
    SpatialEgoVector,
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
