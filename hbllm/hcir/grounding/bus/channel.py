from __future__ import annotations

import threading
from dataclasses import dataclass

from hbllm.hcir.grounding.bus.policies import ChannelPolicy
from hbllm.hcir.grounding.bus.ring_buffer import BoundedRingBuffer
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope


@dataclass
class ChannelMetrics:
    channel_id: str
    total_ingested: int = 0
    total_delivered: int = 0
    dropped_count: int = 0
    gap_count: int = 0
    last_sequence_number: int = -1
    last_wall_timestamp: float = 0.0
    last_monotonic_timestamp: float = 0.0


class SequenceViolationError(ValueError):
    """Raised when sequence numbering violates channel consistency rules."""


class ObservationChannel:
    """A managed sensory input channel enforcing traffic classes and sequence guarantees."""

    def __init__(self, channel_id: str, policy: ChannelPolicy) -> None:
        if not channel_id:
            raise ValueError("channel_id must be non-empty")
        self._channel_id = channel_id
        self._policy = policy
        self._buffer: BoundedRingBuffer[ObservationEnvelope] = BoundedRingBuffer(
            capacity=policy.buffer_capacity,
            overflow_policy=policy.overflow_policy,
        )
        self._lock = threading.RLock()
        self._metrics = ChannelMetrics(channel_id=channel_id)
        self._seen_observation_ids: set[str] = set()

    @property
    def channel_id(self) -> str:
        return self._channel_id

    @property
    def policy(self) -> ChannelPolicy:
        return self._policy

    @property
    def metrics(self) -> ChannelMetrics:
        with self._lock:
            # Sync dropped count from ring buffer
            self._metrics.dropped_count = self._buffer.dropped_count
            return ChannelMetrics(**self._metrics.__dict__)

    def ingest(self, envelope: ObservationEnvelope, timeout: float | None = None) -> bool:
        """Ingest an observation envelope into the channel.

        Validates channel binding, detects duplicate observation IDs, checks
        monotonic timestamp consistency, and detects sequence number gaps.
        """
        if envelope.source_channel_id != self._channel_id:
            raise ValueError(
                f"Envelope source_channel_id ({envelope.source_channel_id}) "
                f"does not match channel ({self._channel_id})"
            )

        with self._lock:
            # Check duplicate ID
            if envelope.observation_id in self._seen_observation_ids:
                raise ValueError(
                    f"Duplicate observation_id {envelope.observation_id} detected on channel {self._channel_id}"
                )

            # Check sequence ordering & gap detection
            last_seq = self._metrics.last_sequence_number
            if last_seq >= 0:
                if envelope.sequence_number <= last_seq:
                    raise SequenceViolationError(
                        f"Non-increasing sequence number on {self._channel_id}: "
                        f"current={envelope.sequence_number}, last={last_seq}"
                    )
                if self._policy.detect_sequence_gaps and envelope.sequence_number > last_seq + 1:
                    gap = envelope.sequence_number - (last_seq + 1)
                    self._metrics.gap_count += gap

            # Check monotonic timestamp ordering
            if self._policy.enforce_monotonic_time and self._metrics.last_monotonic_timestamp > 0.0:
                if envelope.monotonic_timestamp < self._metrics.last_monotonic_timestamp:
                    raise ValueError(
                        f"Monotonic clock inversion on {self._channel_id}: "
                        f"{envelope.monotonic_timestamp} < {self._metrics.last_monotonic_timestamp}"
                    )

            # Push to buffer
            pushed = self._buffer.push(envelope, timeout=timeout)
            if pushed:
                self._metrics.total_ingested += 1
                self._metrics.last_sequence_number = envelope.sequence_number
                self._metrics.last_wall_timestamp = envelope.wall_timestamp
                self._metrics.last_monotonic_timestamp = envelope.monotonic_timestamp
                self._seen_observation_ids.add(envelope.observation_id)
                # Keep seen_ids bounded to prevent unbounded memory growth
                if len(self._seen_observation_ids) > 10000:
                    # Prune older half (simple strategy)
                    self._seen_observation_ids = set(list(self._seen_observation_ids)[5000:])
                return True
            else:
                return False

    def pull(self, timeout: float | None = None) -> ObservationEnvelope | None:
        """Pull the next available observation envelope."""
        item = self._buffer.pop(timeout=timeout)
        if item is not None:
            with self._lock:
                self._metrics.total_delivered += 1
        return item

    def pull_all(self) -> list[ObservationEnvelope]:
        """Pull all available envelopes without blocking."""
        items = self._buffer.pop_all()
        if items:
            with self._lock:
                self._metrics.total_delivered += len(items)
        return items

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._seen_observation_ids.clear()
