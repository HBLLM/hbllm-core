"""Spatial Coordinates, Reference Frames, and Geometric Mapping Models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hbllm.hcir.world.spatial.transforms import (
        AffineTransform2D,
        ProjectiveHomography2D,
    )

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class SpatialBearing(StrEnum):
    """Cardinal, intercardinal, and self bearings in egocentric space."""

    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"
    NORTH_EAST = "NE"
    NORTH_WEST = "NW"
    SOUTH_EAST = "SE"
    SOUTH_WEST = "SW"
    SELF = "SELF"


class HeadingOrientation(StrEnum):
    """Gaze and heading orientations in 2D coordinate space."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    @property
    def unit_vector(self) -> tuple[int, int]:
        """Directional unit vector in (delta_r, delta_c)."""
        if self == HeadingOrientation.NORTH:
            return (-1, 0)
        elif self == HeadingOrientation.SOUTH:
            return (1, 0)
        elif self == HeadingOrientation.EAST:
            return (0, 1)
        elif self == HeadingOrientation.WEST:
            return (0, -1)
        return (-1, 0)

    @property
    def right_vector(self) -> tuple[int, int]:
        """Right-hand orthogonal lateral unit vector in (delta_r, delta_c)."""
        hr, hc = self.unit_vector
        return (hc, -hr)


class DihedralTransform(StrEnum):
    """Canonical D4 dihedral group transformations."""

    REFLECT_HORIZONTAL = "reflect_horizontal"
    REFLECT_VERTICAL = "reflect_vertical"
    REFLECT_MAIN_DIAG = "reflect_main_diag"
    ROTATE_180 = "rotate_180"
    ROTATE_90_CW = "rotate_90_cw"
    ROTATE_90_CCW = "rotate_90_ccw"
    IDENTITY = "identity"


class GeometricModelType(StrEnum):
    """Geometric transformation family classifications (W040)."""

    ISOMETRIC = "isometric"
    AFFINE = "affine"
    PROJECTIVE = "projective"
    DEGENERATE = "degenerate"


@dataclass(frozen=True)
class EgocentricPolarCoord:
    """Viewpoint-invariant polar coordinate and forward/lateral projections."""

    radius: float
    azimuth_rad: float
    relative_bearing: str
    forward_depth: float
    lateral_offset: float

    def to_causal_predicate(self, entity_id: str = "entity") -> CausalPredicate:
        """Export egocentric polar coordinates as a formal causal predicate."""
        return CausalPredicate(
            variable="egocentric_polar",
            operator="==",
            value=f"r:{self.radius:.2f}:az:{self.azimuth_rad:.2f}:{self.relative_bearing}",
            relation=f"fwd:{self.forward_depth:.2f}:lat:{self.lateral_offset:.2f}",
            target_entity=entity_id,
        )


@dataclass(frozen=True)
class EgocentricEntityRepresentation:
    """High-level lifted spatial entity in avatar-centric egocentric frame."""

    entity_id: str
    feature_id: int
    allocentric_pos: tuple[int, int]
    polar_coord: EgocentricPolarCoord
    egocentric_displacement: tuple[int, int]

    def to_causal_predicate(self) -> CausalPredicate:
        """Export entity egocentric representation as a causal predicate."""
        return CausalPredicate(
            variable=f"ego_entity:{self.entity_id}",
            operator="==",
            value=f"feat:{self.feature_id}:bearing:{self.polar_coord.relative_bearing}",
            relation=f"dist:{self.polar_coord.radius:.2f}",
            target_entity=self.entity_id,
        )


@dataclass(frozen=True)
class SpatialEgoVector:
    """Egocentric displacement vector relative to avatar position."""

    delta_r: int
    delta_c: int
    manhattan_dist: int
    bearing: SpatialBearing | str

    def is_adjacent(self, chebyshev: bool = False) -> bool:
        """True if the target position is within immediate 1-step reach."""
        if chebyshev:
            return max(abs(self.delta_r), abs(self.delta_c)) <= 1
        return self.manhattan_dist <= 1

    def unit_step(self) -> tuple[int, int]:
        """Normalize vector to single cardinal or diagonal unit step."""
        step_r = 0 if self.delta_r == 0 else (1 if self.delta_r > 0 else -1)
        step_c = 0 if self.delta_c == 0 else (1 if self.delta_c > 0 else -1)
        return step_r, step_c

    def to_rotated_ego(
        self,
        heading: HeadingOrientation | str = HeadingOrientation.NORTH,
    ) -> tuple[int, int]:
        """Rotate displacement vector to align with avatar heading (forward_depth, lateral_offset)."""
        if isinstance(heading, str):
            heading = HeadingOrientation(heading.upper())
        hr, hc = heading.unit_vector
        rr, rc = heading.right_vector
        fwd = self.delta_r * hr + self.delta_c * hc
        lat = self.delta_r * rr + self.delta_c * rc
        return int(round(fwd)), int(round(lat))

    def to_polar(
        self,
        heading: HeadingOrientation | str = HeadingOrientation.NORTH,
    ) -> EgocentricPolarCoord:
        """Transform displacement into viewpoint-invariant polar coordinate relative to heading."""
        if isinstance(heading, str):
            heading = HeadingOrientation(heading.upper())
        hr, hc = heading.unit_vector
        rr, rc = heading.right_vector

        fwd = float(self.delta_r * hr + self.delta_c * hc)
        lat = float(self.delta_r * rr + self.delta_c * rc)
        radius = float(np.hypot(self.delta_r, self.delta_c))

        if radius < 1e-6:
            return EgocentricPolarCoord(
                radius=0.0,
                azimuth_rad=0.0,
                relative_bearing="SELF",
                forward_depth=0.0,
                lateral_offset=0.0,
            )

        azimuth = float(np.arctan2(lat, fwd))
        pi = float(np.pi)

        # Discretize relative bearing into 8 octants
        if abs(azimuth) < pi / 8.0:
            rel_bearing = "FORWARD"
        elif pi / 8.0 <= azimuth < 3.0 * pi / 8.0:
            rel_bearing = "FORWARD_RIGHT"
        elif 3.0 * pi / 8.0 <= azimuth < 5.0 * pi / 8.0:
            rel_bearing = "RIGHT"
        elif 5.0 * pi / 8.0 <= azimuth < 7.0 * pi / 8.0:
            rel_bearing = "BACKWARD_RIGHT"
        elif abs(azimuth) >= 7.0 * pi / 8.0:
            rel_bearing = "BACKWARD"
        elif -7.0 * pi / 8.0 <= azimuth < -5.0 * pi / 8.0:
            rel_bearing = "BACKWARD_LEFT"
        elif -5.0 * pi / 8.0 <= azimuth < -3.0 * pi / 8.0:
            rel_bearing = "LEFT"
        else:
            rel_bearing = "FORWARD_LEFT"

        return EgocentricPolarCoord(
            radius=radius,
            azimuth_rad=azimuth,
            relative_bearing=rel_bearing,
            forward_depth=fwd,
            lateral_offset=lat,
        )

    def to_causal_predicate(
        self, agent_id: str = "avatar", target_id: str = "target"
    ) -> CausalPredicate:
        """Export egocentric relational bearing into a formal declarative causal predicate."""
        return CausalPredicate(
            variable="spatial_bearing",
            operator="==",
            value=str(self.bearing),
            relation=f"dist_{self.manhattan_dist}",
            target_entity=target_id,
        )


@dataclass
class GeometricMappingResult:
    """Epistemic result of geometric mapping selection between coordinate frames (W040)."""

    best_model_type: GeometricModelType | str
    best_model: AffineTransform2D | ProjectiveHomography2D | None
    residual: float
    residuals: dict[str, float]
    ambiguity_score: float
    parallelism_deviation: float
    is_isometric: bool
    perspective_distortion: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard dictionary representation."""
        return {
            "best_model_type": str(self.best_model_type),
            "best_model": self.best_model,
            "residual": self.residual,
            "residuals": dict(self.residuals),
            "ambiguity_score": self.ambiguity_score,
            "parallelism_deviation": self.parallelism_deviation,
            "is_isometric": self.is_isometric,
            "perspective_distortion": self.perspective_distortion,
        }

    def __getitem__(self, key: str) -> Any:
        """Dictionary access emulation for backward compatibility."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Safe dictionary getter emulation."""
        return self.to_dict().get(key, default)

    def to_causal_predicate(
        self, source_frame: str = "source", target_frame: str = "target"
    ) -> CausalPredicate:
        """Export geometric mapping invariance as a formal causal predicate."""
        return CausalPredicate(
            variable="geometric_mapping",
            operator="==",
            value=str(self.best_model_type),
            relation=str(round(self.residual, 4)),
            target_entity=target_frame,
        )


@dataclass
class EgocentricCoordinate:
    """An agent-centered coordinate with forward and lateral components."""

    forward: int  # Positive = ahead of agent, negative = behind
    lateral: int  # Positive = right of agent, negative = left of agent

    def to_tuple(self) -> tuple[int, int]:
        return (self.forward, self.lateral)
