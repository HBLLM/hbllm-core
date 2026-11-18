from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.grounding.contracts.enums import ModalitySource, TrafficClass


class CoordinateType(StrEnum):
    """Classification of coordinate representations."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class ReferenceFrame:
    """Mathematical reference frame defining the coordinate system for spatial observations.

    Supports arbitrary dimensions (discrete integer lattices or continuous metric spaces),
    canonical axis naming, physical units, scaling factors, bounding extents, and origin definitions.
    """

    frame_id: str
    dimensions: int
    coordinate_type: CoordinateType
    axes: tuple[str, ...]
    scale: tuple[float, ...] = field(default_factory=tuple)
    bounds: tuple[tuple[float, float], ...] | None = None
    unit: str = ""
    origin_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("ReferenceFrame frame_id must be non-empty")
        if self.dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {self.dimensions}")
        if len(self.axes) != self.dimensions:
            raise ValueError(
                f"axes length ({len(self.axes)}) must match dimensions ({self.dimensions})"
            )

        # CoordinateType validation / conversion
        if isinstance(self.coordinate_type, str) and not isinstance(
            self.coordinate_type, CoordinateType
        ):
            object.__setattr__(self, "coordinate_type", CoordinateType(self.coordinate_type))

        # Scale initialization and validation
        if not self.scale:
            object.__setattr__(self, "scale", (1.0,) * self.dimensions)
        else:
            if len(self.scale) != self.dimensions:
                raise ValueError(
                    f"scale length ({len(self.scale)}) must match dimensions ({self.dimensions})"
                )
            if any(s <= 0 for s in self.scale):
                raise ValueError(f"All scale components must be positive, got {self.scale}")

        # Bounds validation
        if self.bounds is not None:
            if len(self.bounds) != self.dimensions:
                raise ValueError(
                    f"bounds length ({len(self.bounds)}) must match dimensions ({self.dimensions})"
                )
            for i, (low, high) in enumerate(self.bounds):
                if low > high:
                    raise ValueError(f"Invalid bound at dim {i}: low ({low}) > high ({high})")

    @classmethod
    def grid_2d(
        cls,
        frame_id: str = "grid_2d",
        axes: tuple[str, ...] = ("row", "col"),
        shape: tuple[int, int] | None = None,
        scale: tuple[float, float] = (1.0, 1.0),
        unit: str = "cell",
        origin_description: str = "top_left",
        metadata: dict[str, Any] | None = None,
    ) -> ReferenceFrame:
        """Create a discrete 2D grid reference frame."""
        bounds = (
            ((0.0, float(shape[0] - 1)), (0.0, float(shape[1] - 1)))
            if shape is not None and len(shape) == 2
            else None
        )
        return cls(
            frame_id=frame_id,
            dimensions=2,
            coordinate_type=CoordinateType.DISCRETE,
            axes=axes,
            scale=scale,
            bounds=bounds,
            unit=unit,
            origin_description=origin_description,
            metadata=metadata or {},
        )

    @classmethod
    def cartesian_3d(
        cls,
        frame_id: str = "cartesian_3d",
        axes: tuple[str, ...] = ("x", "y", "z"),
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        unit: str = "meter",
        origin_description: str = "world_origin",
        metadata: dict[str, Any] | None = None,
    ) -> ReferenceFrame:
        """Create a continuous 3D Cartesian reference frame."""
        return cls(
            frame_id=frame_id,
            dimensions=3,
            coordinate_type=CoordinateType.CONTINUOUS,
            axes=axes,
            scale=scale,
            bounds=bounds,
            unit=unit,
            origin_description=origin_description,
            metadata=metadata or {},
        )

    @classmethod
    def joint_space(
        cls,
        dof: int = 7,
        frame_id: str = "joint_space",
        joint_names: tuple[str, ...] | None = None,
        bounds: tuple[tuple[float, float], ...] | None = None,
        unit: str = "radian",
        origin_description: str = "zero_configuration",
        metadata: dict[str, Any] | None = None,
    ) -> ReferenceFrame:
        """Create a continuous N-DOF robot joint space reference frame."""
        names = joint_names if joint_names is not None else tuple(f"joint_{i}" for i in range(dof))
        return cls(
            frame_id=frame_id,
            dimensions=dof,
            coordinate_type=CoordinateType.CONTINUOUS,
            axes=names,
            scale=(1.0,) * dof,
            bounds=bounds,
            unit=unit,
            origin_description=origin_description,
            metadata=metadata or {},
        )

    @classmethod
    def generic_nd(
        cls,
        dimensions: int,
        coordinate_type: CoordinateType | str = CoordinateType.CONTINUOUS,
        frame_id: str = "",
        axes: tuple[str, ...] | None = None,
        scale: tuple[float, ...] | None = None,
        bounds: tuple[tuple[float, float], ...] | None = None,
        unit: str = "",
        origin_description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ReferenceFrame:
        """Create a generic N-dimensional reference frame."""
        coord_type = (
            coordinate_type
            if isinstance(coordinate_type, CoordinateType)
            else CoordinateType(coordinate_type)
        )
        fid = frame_id or f"frame_{dimensions}d"
        axis_names = axes if axes is not None else tuple(f"dim_{i}" for i in range(dimensions))
        scale_tup = scale if scale is not None else (1.0,) * dimensions
        return cls(
            frame_id=fid,
            dimensions=dimensions,
            coordinate_type=coord_type,
            axes=axis_names,
            scale=scale_tup,
            bounds=bounds,
            unit=unit,
            origin_description=origin_description,
            metadata=metadata or {},
        )

    def transform_point(
        self,
        point: tuple[float, ...] | Sequence[float],
        target_frame: ReferenceFrame,
    ) -> tuple[float, ...]:
        """Transform a coordinate point from this reference frame to target_frame.

        Normalizes coordinates by source scale and rescales by target scale.
        """
        if len(point) != self.dimensions:
            raise ValueError(
                f"Source point dimension ({len(point)}) does not match frame dimensions ({self.dimensions})"
            )
        if target_frame.dimensions != self.dimensions:
            raise ValueError(
                f"Cannot transform point between mismatched dimensions: {self.dimensions} vs {target_frame.dimensions}"
            )

        # Scale-space projection: point -> normalized -> target
        transformed = tuple(
            float(p) / float(s_src) * float(s_tgt)
            for p, s_src, s_tgt in zip(point, self.scale, target_frame.scale)
        )
        return transformed

    def is_within_bounds(self, point: tuple[float, ...] | Sequence[float]) -> bool:
        """Check whether a given point lies within the frame's bounds."""
        if len(point) != self.dimensions:
            return False
        if self.bounds is None:
            return True
        for p, (low, high) in zip(point, self.bounds):
            if p < low or p > high:
                return False
        return True


@dataclass(frozen=True)
class ObservationEnvelope:
    """Immutable packet representing an original sensory capture and its provenance.

    Carries explicit sequence numbering, wall-clock and monotonic timestamps for
    clock-drift-free alignment across asynchronous multimodal sensors, along with
    its formal spatial ReferenceFrame defining coordinate semantics.
    """

    observation_id: str
    source_modality: ModalitySource
    source_channel_id: str
    traffic_class: TrafficClass
    sequence_number: int
    wall_timestamp: float
    monotonic_timestamp: float
    raw_payload_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    clock_domain: str = "host_monotonic"
    payload_ref: str = ""
    payload_hash: str = ""
    reference_frame: ReferenceFrame | None = None

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("ObservationEnvelope observation_id must be non-empty")
        if self.sequence_number < 0:
            raise ValueError(f"sequence_number must be non-negative, got {self.sequence_number}")
        if self.wall_timestamp <= 0:
            raise ValueError(f"wall_timestamp must be positive, got {self.wall_timestamp}")
        if self.monotonic_timestamp < 0:
            raise ValueError(
                f"monotonic_timestamp must be non-negative, got {self.monotonic_timestamp}"
            )

        # Synchronize payload_ref and raw_payload_ref
        effective_payload_ref = self.payload_ref or self.raw_payload_ref
        if not effective_payload_ref:
            raise ValueError("ObservationEnvelope payload_ref / raw_payload_ref must be non-empty")
        object.__setattr__(self, "payload_ref", effective_payload_ref)
        object.__setattr__(self, "raw_payload_ref", effective_payload_ref)

        # Synchronize payload_hash and provenance_hash
        effective_hash = self.payload_hash or self.provenance_hash
        if not effective_hash:
            seed = (
                f"{self.observation_id}:{self.source_channel_id}:"
                f"{self.sequence_number}:{self.wall_timestamp}:{effective_payload_ref}"
            )
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            effective_hash = digest

        object.__setattr__(self, "payload_hash", effective_hash)
        object.__setattr__(self, "provenance_hash", effective_hash)

        # Reference frame auto-inference if omitted
        if self.reference_frame is None:
            inferred_frame: ReferenceFrame | None = None
            if self.source_modality == ModalitySource.VISION:
                shape = self.metadata.get("grid_shape")
                if shape and len(shape) == 2:
                    inferred_frame = ReferenceFrame.grid_2d(shape=(int(shape[0]), int(shape[1])))
                else:
                    inferred_frame = ReferenceFrame.grid_2d()
            elif self.source_modality in (
                ModalitySource.PROPRIOCEPTIVE,
                ModalitySource.KINESTHETIC,
            ):
                pos = self.metadata.get("position")
                if pos and len(pos) == 3:
                    inferred_frame = ReferenceFrame.cartesian_3d()
                elif pos and len(pos) == 2:
                    inferred_frame = ReferenceFrame.generic_nd(
                        2, coordinate_type=CoordinateType.CONTINUOUS, axes=("x", "y"), unit="meter"
                    )
                elif pos:
                    inferred_frame = ReferenceFrame.generic_nd(
                        len(pos), coordinate_type=CoordinateType.CONTINUOUS, unit="meter"
                    )
                else:
                    inferred_frame = ReferenceFrame.cartesian_3d()
            object.__setattr__(self, "reference_frame", inferred_frame)

    @classmethod
    def create(
        cls,
        observation_id: str,
        source_modality: ModalitySource,
        source_channel_id: str,
        traffic_class: TrafficClass,
        sequence_number: int,
        raw_payload_ref: str = "",
        payload_ref: str = "",
        metadata: dict[str, Any] | None = None,
        wall_timestamp: float | None = None,
        monotonic_timestamp: float | None = None,
        provenance_hash: str = "",
        payload_hash: str = "",
        clock_domain: str = "host_monotonic",
        reference_frame: ReferenceFrame | None = None,
    ) -> ObservationEnvelope:
        ref = payload_ref or raw_payload_ref
        hash_val = payload_hash or provenance_hash
        return cls(
            observation_id=observation_id,
            source_modality=source_modality,
            source_channel_id=source_channel_id,
            traffic_class=traffic_class,
            sequence_number=sequence_number,
            wall_timestamp=wall_timestamp if wall_timestamp is not None else time.time(),
            monotonic_timestamp=(
                monotonic_timestamp if monotonic_timestamp is not None else time.monotonic()
            ),
            raw_payload_ref=ref,
            payload_ref=ref,
            metadata=metadata or {},
            provenance_hash=hash_val,
            payload_hash=hash_val,
            clock_domain=clock_domain,
            reference_frame=reference_frame,
        )
