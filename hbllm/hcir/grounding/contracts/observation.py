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
    raw_payload_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    clock_domain: str = "host_monotonic"
    payload_ref: str = ""
    payload_hash: str = ""

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
        )
