"""Proposition & Spatial/Temporal Semantics — HCIR Cognitive Runtime §1.

Modality-neutral foundational types for the universal EvidenceNode contract.
These types make Whisper, YOLO, an IMU, and a LiDAR indistinguishable
from HCIR's perspective: all produce Propositions about the world.

Types:
    Proposition       — Modality-neutral semantic triple (subject/predicate/object)
    BoundingBox       — Normalized image-space bounding box
    SpatialContext    — Spatial reference frame for grounded observations
    TemporalValidity  — Observation, ingestion, and validity timestamps

Design invariants:
    - ``object_value`` is ``Any``, not ``str``.  Structured data
      (vectors, dicts, numbers) must never be serialized into strings.
    - Spatial context carries structured bounding boxes, polygons,
      and depth — never string-encoded geometry.
    - Temporal validity distinguishes observation time from ingestion
      time, critical for streaming perception and deduplication.

Examples::

    # Vision (YOLO)
    Proposition(subject="person_17", predicate="located_at", object_value="kitchen")

    # Audio (Whisper)
    Proposition(subject="utterance_42", predicate="transcribed_as",
                object_value="turn on the lights")

    # IMU
    Proposition(subject="robot", predicate="acceleration",
                object_value=[0.1, -0.3, 9.8], object_type="vector3")

    # Temperature sensor
    Proposition(subject="sensor_4", predicate="temperature",
                object_value=22.5, object_type="celsius")

    # Door state
    Proposition(subject="door_7", predicate="state",
                object_value="OPEN", object_type="enum")
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Proposition — modality-neutral semantic triple
# ═══════════════════════════════════════════════════════════════════════════


class Proposition(BaseModel):
    """Modality-neutral semantic triple: subject/predicate/object.

    This is the universal representation for what was observed.
    Every perception provider — audio, visual, sensor, robot joint —
    ultimately produces propositions about the world.

    The ``object_value`` field is ``Any`` to avoid serializing structured
    data into strings.  Pydantic handles serialization cleanly.

    Attributes:
        subject: The entity being described (``"person_17"``,
            ``"utterance_42"``, ``"robot"``).
        predicate: The relationship or property (``"located_at"``,
            ``"transcribed_as"``, ``"acceleration"``).
        object_value: The value — can be string, number, list, dict,
            or any JSON-serializable structure.
        object_type: Optional type hint for the object value
            (``"location"``, ``"vector3"``, ``"transcript"``,
            ``"event_class"``, ``"celsius"``, ``"enum"``).
    """

    subject: str = ""
    predicate: str = ""
    object_value: Any = None
    object_type: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Context — structured geometry, never strings
# ═══════════════════════════════════════════════════════════════════════════


class BoundingBox(BaseModel):
    """Normalized bounding box in image space.

    Coordinates are normalized to ``[0, 1]`` so they are
    resolution-independent.

    Attributes:
        x1: Left edge (normalized).
        y1: Top edge (normalized).
        x2: Right edge (normalized).
        y2: Bottom edge (normalized).
    """

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @property
    def area(self) -> float:
        """Normalized area of the bounding box."""
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def center(self) -> tuple[float, float]:
        """Center point (cx, cy) of the bounding box."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class SpatialContext(BaseModel):
    """Spatial reference frame for grounded observations.

    Supports multiple spatial representations that different sensors
    and providers can contribute to:

    - ``bounding_box``: Image-space detection (YOLO, OCR)
    - ``position``: 3D world coordinates (LiDAR, robot localization)
    - ``depth_meters``: Depth estimate (depth camera, stereo)
    - ``polygon``: Arbitrary region vertices (segmentation)
    - ``orientation``: Quaternion orientation (IMU, robot joint)

    Attributes:
        frame_id: Reference frame identifier (``"camera_0"``,
            ``"world"``, ``"robot_base"``).
        position: 3D position ``[x, y, z]`` in the reference frame.
        orientation: Quaternion ``[qw, qx, qy, qz]`` orientation.
        bounding_box: Normalized image-space bounding box.
        polygon: Arbitrary region as list of ``[x, y]`` vertices.
        depth_meters: Depth estimate in meters.
        confidence: Spatial measurement confidence ``[0.0, 1.0]``.
    """

    frame_id: str = ""
    position: list[float] | None = None
    orientation: list[float] | None = None
    bounding_box: BoundingBox | None = None
    polygon: list[list[float]] | None = None
    depth_meters: float | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Validity — observation ≠ ingestion ≠ validity
# ═══════════════════════════════════════════════════════════════════════════


class TemporalValidity(BaseModel):
    """When this evidence was observed, received, and how long it's valid.

    Distinguishes three critical timestamps that streaming perception
    and deduplication depend on:

    - ``observed_at``: When the sensor captured the signal.
    - ``received_at``: When HCIR received the evidence.
    - ``valid_from`` / ``valid_until``: Temporal validity window.

    Example timeline::

        Camera frame captured:           12:00:00.100  → observed_at
        Model finishes inference:         12:00:00.180
        HCIR receives evidence:           12:00:00.185  → received_at
        Evidence validity expires:        12:00:05.100  → valid_until

    Attributes:
        observed_at: Sensor/observation timestamp (epoch seconds).
        received_at: HCIR ingestion timestamp (epoch seconds).
        valid_from: Start of validity window (defaults to observed_at).
        valid_until: End of validity window (``None`` = until contradicted).
    """

    observed_at: float = Field(default_factory=time.time)
    received_at: float = Field(default_factory=time.time)
    valid_from: float | None = None
    valid_until: float | None = None

    @property
    def latency_ms(self) -> float:
        """Sensor-to-HCIR latency in milliseconds."""
        return (self.received_at - self.observed_at) * 1000.0

    @property
    def is_expired(self) -> bool:
        """Whether this evidence has exceeded its validity window."""
        if self.valid_until is None:
            return False
        return time.time() > self.valid_until
