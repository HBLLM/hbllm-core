from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.contracts.enums import ModalitySource, TrafficClass


@dataclass(frozen=True)
class ObservationEnvelope:
    """Immutable packet representing an original sensory capture and its provenance.

    Carries explicit sequence numbering, wall-clock and monotonic timestamps for
    clock-drift-free alignment across asynchronous multimodal sensors.
    """

    observation_id: str
    source_modality: ModalitySource
    source_channel_id: str
    traffic_class: TrafficClass
    sequence_number: int
    wall_timestamp: float
    monotonic_timestamp: float
    raw_payload_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

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

        # Compute provenance hash if not provided
        if not self.provenance_hash:
            seed = (
                f"{self.observation_id}:{self.source_channel_id}:"
                f"{self.sequence_number}:{self.wall_timestamp}:{self.raw_payload_ref}"
            )
            # Use BLAKE2s / SHA-256 for deterministic provenance
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            object.__setattr__(self, "provenance_hash", digest)

    @classmethod
    def create(
        cls,
        observation_id: str,
        source_modality: ModalitySource,
        source_channel_id: str,
        traffic_class: TrafficClass,
        sequence_number: int,
        raw_payload_ref: str,
        metadata: dict[str, Any] | None = None,
        wall_timestamp: float | None = None,
        monotonic_timestamp: float | None = None,
        provenance_hash: str = "",
    ) -> ObservationEnvelope:
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
            raw_payload_ref=raw_payload_ref,
            metadata=metadata or {},
            provenance_hash=provenance_hash,
        )
