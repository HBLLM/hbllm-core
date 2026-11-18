from __future__ import annotations

import concurrent.futures
import time

import pytest

from hbllm.hcir.grounding.bus import (
    BoundedRingBuffer,
    BufferOverflowError,
    ChannelPolicy,
    ObservationBus,
    ObservationChannel,
)
from hbllm.hcir.grounding.contracts import (
    ModalitySource,
    ObservationEnvelope,
    OverflowPolicy,
    TrafficClass,
)


def test_bus_channel_registration_and_publishing() -> None:
    """Test channel registration, default policies, and publishing."""
    bus = ObservationBus()
    env = ObservationEnvelope.create(
        observation_id="obs_001",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_front",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=1,
        raw_payload_ref="blob://frames/001.bin",
    )

    published = bus.publish(env)
    assert published is True

    channel = bus.get_channel("cam_front")
    assert channel is not None
    assert channel.metrics.total_ingested == 1
    assert channel.metrics.last_sequence_number == 1

    # Pull envelope from channel
    pulled = channel.pull()
    assert pulled is not None
    assert pulled.observation_id == "obs_001"
    assert channel.metrics.total_delivered == 1


def test_bus_subscriber_filtering() -> None:
    """Verify that subscribers receive only matching observations."""
    bus = ObservationBus()
    received_vision: list[ObservationEnvelope] = []
    received_audio: list[ObservationEnvelope] = []

    bus.subscribe(
        callback=lambda env: received_vision.append(env),
        modality=ModalitySource.VISION,
    )
    bus.subscribe(
        callback=lambda env: received_audio.append(env),
        modality=ModalitySource.AUDIO,
    )

    # Publish vision envelope
    env_vis = ObservationEnvelope.create(
        observation_id="obs_v1",
        source_modality=ModalitySource.VISION,
        source_channel_id="cam_front",
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=1,
        raw_payload_ref="blob://vis.bin",
    )
    bus.publish(env_vis)

    # Publish audio envelope
    env_aud = ObservationEnvelope.create(
        observation_id="obs_a1",
        source_modality=ModalitySource.AUDIO,
        source_channel_id="mic_array",
        traffic_class=TrafficClass.CLASS_STREAMING,
        sequence_number=1,
        raw_payload_ref="blob://aud.bin",
    )
    bus.publish(env_aud)

    assert len(received_vision) == 1
    assert received_vision[0].observation_id == "obs_v1"

    assert len(received_audio) == 1
    assert received_audio[0].observation_id == "obs_a1"


def test_concurrent_producers_thread_safety() -> None:
    """Test thread safety with multiple concurrent sensor channels publishing simultaneously."""
    bus = ObservationBus(replay_capacity=2000)
    num_producers = 5
    messages_per_producer = 100

    def produce(channel_idx: int) -> None:
        channel_id = f"sensor_{channel_idx}"
        for seq in range(1, messages_per_producer + 1):
            env = ObservationEnvelope.create(
                observation_id=f"obs_c{channel_idx}_{seq}",
                source_modality=ModalitySource.PROPRIOCEPTIVE,
                source_channel_id=channel_id,
                traffic_class=TrafficClass.CLASS_TELEMETRY,
                sequence_number=seq,
                raw_payload_ref=f"blob://proprio/{channel_idx}/{seq}",
                wall_timestamp=time.time(),
                monotonic_timestamp=time.monotonic(),
            )
            bus.publish(env)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_producers) as executor:
        futures = [executor.submit(produce, i) for i in range(num_producers)]
        for f in futures:
            f.result()

    metrics = bus.get_all_metrics()
    assert len(metrics) == num_producers
    for cid, m in metrics.items():
        assert m.total_ingested == messages_per_producer
        assert m.last_sequence_number == messages_per_producer
        assert m.gap_count == 0


def test_buffer_overflow_drop_oldest_policy() -> None:
    """Test DROP_OLDEST policy under buffer capacity limit."""
    policy = ChannelPolicy(
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        buffer_capacity=3,
        overflow_policy=OverflowPolicy.DROP_OLDEST,
    )
    channel = ObservationChannel(channel_id="cam_fast", policy=policy)

    # Push 5 items into capacity-3 buffer
    for i in range(1, 6):
        env = ObservationEnvelope.create(
            observation_id=f"obs_{i}",
            source_modality=ModalitySource.VISION,
            source_channel_id="cam_fast",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=i,
            raw_payload_ref=f"blob://{i}",
            monotonic_timestamp=float(i),
        )
        channel.ingest(env)

    # Oldest 2 items should have been dropped
    assert channel.metrics.dropped_count == 2
    assert channel.metrics.total_ingested == 5

    # Should only read items 3, 4, 5
    items = channel.pull_all()
    assert len(items) == 3
    assert [item.sequence_number for item in items] == [3, 4, 5]


def test_buffer_overflow_reject_with_error() -> None:
    """Test REJECT_WITH_ERROR policy when capacity is reached."""
    ring: BoundedRingBuffer[str] = BoundedRingBuffer(
        capacity=2,
        overflow_policy=OverflowPolicy.REJECT_WITH_ERROR,
    )
    assert ring.push("item1") is True
    assert ring.push("item2") is True

    with pytest.raises(BufferOverflowError):
        ring.push("item3")

    assert ring.dropped_count == 1
