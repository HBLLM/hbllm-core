from __future__ import annotations

import pytest

from hbllm.hcir.grounding.bus import (
    ChannelPolicy,
    ObservationBus,
    ObservationChannel,
    SequenceViolationError,
)
from hbllm.hcir.grounding.contracts import (
    ModalitySource,
    ObservationEnvelope,
    TrafficClass,
)


def test_monotonic_sequence_enforcement() -> None:
    """Verify that non-increasing sequence numbers raise SequenceViolationError."""
    policy = ChannelPolicy.default_for(TrafficClass.CLASS_TELEMETRY)
    channel = ObservationChannel(channel_id="imu_channel", policy=policy)

    # Ingest sequence 10
    env1 = ObservationEnvelope.create(
        observation_id="obs_imu_10",
        source_modality=ModalitySource.PROPRIOCEPTIVE,
        source_channel_id="imu_channel",
        traffic_class=TrafficClass.CLASS_TELEMETRY,
        sequence_number=10,
        raw_payload_ref="blob://imu/10",
        monotonic_timestamp=10.0,
    )
    assert channel.ingest(env1) is True

    # Ingest sequence 9 (backward sequence inversion)
    env_backward = ObservationEnvelope.create(
        observation_id="obs_imu_9",
        source_modality=ModalitySource.PROPRIOCEPTIVE,
        source_channel_id="imu_channel",
        traffic_class=TrafficClass.CLASS_TELEMETRY,
        sequence_number=9,
        raw_payload_ref="blob://imu/9",
        monotonic_timestamp=11.0,
    )
    with pytest.raises(SequenceViolationError, match="Non-increasing sequence number"):
        channel.ingest(env_backward)

    # Ingest sequence 10 again (duplicate sequence number)
    env_duplicate = ObservationEnvelope.create(
        observation_id="obs_imu_10_dup",
        source_modality=ModalitySource.PROPRIOCEPTIVE,
        source_channel_id="imu_channel",
        traffic_class=TrafficClass.CLASS_TELEMETRY,
        sequence_number=10,
        raw_payload_ref="blob://imu/10_dup",
        monotonic_timestamp=11.0,
    )
    with pytest.raises(SequenceViolationError, match="Non-increasing sequence number"):
        channel.ingest(env_duplicate)


def test_sequence_gap_detection() -> None:
    """Verify that gaps in sequence numbers increment gap_count."""
    policy = ChannelPolicy.default_for(TrafficClass.CLASS_TELEMETRY)
    channel = ObservationChannel(channel_id="imu_gap", policy=policy)

    # Sequence 1
    env1 = ObservationEnvelope.create(
        observation_id="obs_1",
        source_modality=ModalitySource.PROPRIOCEPTIVE,
        source_channel_id="imu_gap",
        traffic_class=TrafficClass.CLASS_TELEMETRY,
        sequence_number=1,
        raw_payload_ref="blob://1",
        monotonic_timestamp=1.0,
    )
    channel.ingest(env1)
    assert channel.metrics.gap_count == 0

    # Sequence 5 (missing 2, 3, 4 -> gap of 3)
    env5 = ObservationEnvelope.create(
        observation_id="obs_5",
        source_modality=ModalitySource.PROPRIOCEPTIVE,
        source_channel_id="imu_gap",
        traffic_class=TrafficClass.CLASS_TELEMETRY,
        sequence_number=5,
        raw_payload_ref="blob://5",
        monotonic_timestamp=2.0,
    )
    channel.ingest(env5)
    assert channel.metrics.gap_count == 3


def test_monotonic_clock_inversion_detection() -> None:
    """Verify that inverted monotonic timestamps are rejected."""
    policy = ChannelPolicy.default_for(TrafficClass.CLASS_TELEMETRY)
    channel = ObservationChannel(channel_id="imu_clock", policy=policy)

    env1 = ObservationEnvelope.create(
        observation_id="obs_c1",
        source_modality=ModalitySource.PROPRIOCEPTIVE,
        source_channel_id="imu_clock",
        traffic_class=TrafficClass.CLASS_TELEMETRY,
        sequence_number=1,
        raw_payload_ref="blob://1",
        monotonic_timestamp=100.5,
    )
    channel.ingest(env1)

    # Timestamp rolls backwards
    env_inversion = ObservationEnvelope.create(
        observation_id="obs_c2",
        source_modality=ModalitySource.PROPRIOCEPTIVE,
        source_channel_id="imu_clock",
        traffic_class=TrafficClass.CLASS_TELEMETRY,
        sequence_number=2,
        raw_payload_ref="blob://2",
        monotonic_timestamp=100.2,  # earlier than 100.5
    )
    with pytest.raises(ValueError, match="Monotonic clock inversion"):
        channel.ingest(env_inversion)


def test_duplicate_observation_id_rejection() -> None:
    """Verify that duplicate observation_ids on the same channel raise ValueError."""
    policy = ChannelPolicy.default_for(TrafficClass.CLASS_HIGH_FREQ)
    channel = ObservationChannel(channel_id="cam_dup", policy=policy)

    env1 = ObservationEnvelope.create(
        observation_id="obs_unique_1",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_dup",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=1,
        raw_payload_ref="blob://1",
    )
    assert channel.ingest(env1) is True

    # Same observation_id
    env_dup = ObservationEnvelope.create(
        observation_id="obs_unique_1",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_dup",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=2,
        raw_payload_ref="blob://2",
    )
    with pytest.raises(ValueError, match="Duplicate observation_id"):
        channel.ingest(env_dup)


def test_bus_replay_filtering() -> None:
    """Test historical replay ordered retrieval and filtering by wall timestamp and modality."""
    bus = ObservationBus(replay_capacity=100)

    t0 = 1700000000.0
    for i in range(1, 6):
        modality = ModalitySource.VISION if i % 2 != 0 else ModalitySource.AUDIO
        channel_id = "cam_front" if modality == ModalitySource.VISION else "mic_main"
        env = ObservationEnvelope.create(
            observation_id=f"obs_replay_{i}",
            source_modality=modality,
            source_channel_id=channel_id,
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=i,
            raw_payload_ref=f"blob://{i}",
            wall_timestamp=t0 + float(i * 10),
            monotonic_timestamp=float(i),
        )
        bus.publish(env)

    # Replay all
    all_replayed = bus.replay(since_wall_time=t0)
    assert len(all_replayed) == 5
    assert [e.sequence_number for e in all_replayed] == [1, 2, 3, 4, 5]

    # Replay since t0 + 25 (items 3, 4, 5)
    recent = bus.replay(since_wall_time=t0 + 25.0)
    assert len(recent) == 3
    assert [e.sequence_number for e in recent] == [3, 4, 5]

    # Replay filtered by modality
    vision_only = bus.replay(since_wall_time=t0, modality=ModalitySource.VISION)
    assert len(vision_only) == 3
    assert [e.sequence_number for e in vision_only] == [1, 3, 5]
