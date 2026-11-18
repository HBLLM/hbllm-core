from __future__ import annotations

from dataclasses import dataclass

from hbllm.hcir.grounding.contracts.enums import OverflowPolicy, TrafficClass


@dataclass(frozen=True)
class ChannelPolicy:
    """Quality of Service and buffering policy for an observation channel."""

    traffic_class: TrafficClass
    buffer_capacity: int
    overflow_policy: OverflowPolicy
    max_latency_seconds: float = 5.0
    detect_sequence_gaps: bool = True
    enforce_monotonic_time: bool = True

    @classmethod
    def default_for(cls, traffic_class: TrafficClass) -> ChannelPolicy:
        """Returns the canonical recommended policy for a traffic class."""
        if traffic_class == TrafficClass.CLASS_HIGH_FREQ:
            return cls(
                traffic_class=traffic_class,
                buffer_capacity=128,
                overflow_policy=OverflowPolicy.DROP_OLDEST,
                max_latency_seconds=1.0,
                detect_sequence_gaps=True,
            )
        elif traffic_class == TrafficClass.CLASS_STREAMING:
            return cls(
                traffic_class=traffic_class,
                buffer_capacity=256,
                overflow_policy=OverflowPolicy.DROP_OLDEST,
                max_latency_seconds=2.0,
                detect_sequence_gaps=True,
            )
        elif traffic_class == TrafficClass.CLASS_TELEMETRY:
            return cls(
                traffic_class=traffic_class,
                buffer_capacity=512,
                overflow_policy=OverflowPolicy.DROP_OLDEST,
                max_latency_seconds=5.0,
                detect_sequence_gaps=True,
            )
        elif traffic_class == TrafficClass.CLASS_DISCRETE:
            return cls(
                traffic_class=traffic_class,
                buffer_capacity=1024,
                overflow_policy=OverflowPolicy.BLOCK_PRODUCER,
                max_latency_seconds=30.0,
                detect_sequence_gaps=True,
            )
        elif traffic_class == TrafficClass.CLASS_TRANSACTION:
            return cls(
                traffic_class=traffic_class,
                buffer_capacity=2048,
                overflow_policy=OverflowPolicy.BLOCK_PRODUCER,
                max_latency_seconds=60.0,
                detect_sequence_gaps=True,
            )
        return cls(
            traffic_class=traffic_class,
            buffer_capacity=128,
            overflow_policy=OverflowPolicy.DROP_OLDEST,
        )
