"""Tests for the Environmental Simulation and Perception Hooks."""

from __future__ import annotations

import asyncio

import pytest

from hbllm.brain.simulation.environment_sim import EnvironmentSimulator
from hbllm.perception.reality_bus import (
    PerceptionModality,
    RealityEventBus,
)


@pytest.fixture
def reality_bus():
    return RealityEventBus()


@pytest.fixture
def env_simulator(reality_bus):
    return EnvironmentSimulator(bus=reality_bus, tick_rate_seconds=0.1)


@pytest.mark.asyncio
async def test_simulate_vision_event(reality_bus, env_simulator):
    received_events = []

    reality_bus.subscribe(received_events.append)
    await env_simulator.simulate_vision_event(
        detected_objects=["test_object_1", "test_object_2"], confidence=0.98
    )

    await asyncio.sleep(0.01)

    assert len(received_events) == 1
    assert received_events[0].event_type == "vision_frame"
    assert received_events[0].modality == PerceptionModality.SENSOR
    assert received_events[0].confidence == 0.98
    assert "test_object_1" in received_events[0].payload["detected_objects"]


@pytest.mark.asyncio
async def test_simulate_audio_event(reality_bus, env_simulator):
    received_events = []

    reality_bus.subscribe(received_events.append)
    await env_simulator.simulate_audio_event(transcription="mock voice instruction", duration=1.5)

    await asyncio.sleep(0.01)

    assert len(received_events) == 1
    assert received_events[0].event_type == "audio_waveform"
    assert received_events[0].payload["transcription"] == "mock voice instruction"
    assert received_events[0].payload["duration_seconds"] == 1.5


@pytest.mark.asyncio
async def test_simulate_sensor_event(reality_bus, env_simulator):
    received_events = []

    reality_bus.subscribe(received_events.append)
    await env_simulator.simulate_sensor_event()

    await asyncio.sleep(0.01)

    assert len(received_events) == 1
    assert received_events[0].event_type == "sensor_telemetry"
    assert "cpu_usage_percent" in received_events[0].payload


@pytest.mark.asyncio
async def test_loop_starts_and_stops(reality_bus, env_simulator):
    received_events = []
    reality_bus.subscribe(received_events.append)

    await env_simulator.start()
    assert env_simulator.is_running is True

    # Allow loop to execute a couple of ticks
    await asyncio.sleep(0.25)

    await env_simulator.stop()
    assert env_simulator.is_running is False

    assert len(received_events) >= 1
