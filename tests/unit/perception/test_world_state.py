"""Tests for WorldStateEngine — unified environmental awareness."""

import time

import pytest_asyncio

from hbllm.perception.world_state import WorldStateEngine


@pytest_asyncio.fixture
async def engine():
    """WorldStateEngine without bus (direct state manipulation)."""
    e = WorldStateEngine(state_ttl_s=300.0)
    return e


@pytest_asyncio.fixture
async def engine_with_bus(bus):
    """WorldStateEngine connected to a bus."""
    e = WorldStateEngine(bus=bus, state_ttl_s=300.0)
    await e.start()
    return e


def test_empty_summary(engine):
    """Empty engine returns a valid summary."""
    summary = engine.get_summary()
    assert "⏰" in summary  # Always has time context


def test_empty_state(engine):
    """Empty engine returns structured state."""
    state = engine.get_state()
    assert "timestamp" in state
    assert "temporal" in state
    assert "hardware" in state
    assert "iot_devices" in state


def test_hardware_state_update(engine):
    """Hardware state is tracked."""
    engine._hardware = {"battery_level": 0.85, "cpu_load": 0.45}
    engine._timestamps["hardware"] = time.time()
    summary = engine.get_summary()
    assert "Battery" in summary or "85" in summary


def test_iot_device_tracking(engine):
    """IoT devices are included in state."""
    engine._iot_devices["kitchen_light"] = {
        "name": "Kitchen Light",
        "state": {"on": True},
        "_updated_at": time.time(),
    }
    engine._timestamps["iot"] = time.time()
    state = engine.get_state()
    assert "kitchen_light" in state["iot_devices"]


def test_stale_data_excluded(engine):
    """Data older than TTL is excluded from state."""
    engine._hardware = {"battery_level": 0.5}
    engine._timestamps["hardware"] = time.time() - 600  # 10 min old, TTL is 5 min
    state = engine.get_state()
    assert state["hardware"] == {}


def test_summary_includes_audio(engine):
    """Audio environment is included in summary."""
    engine._audio_env = {
        "sound_class": "speech",
        "confidence": 0.85,
        "_updated_at": time.time(),
    }
    engine._timestamps["audio"] = time.time()
    summary = engine.get_summary()
    assert "speech" in summary


def test_summary_suppresses_silence(engine):
    """Silence is not reported in summary."""
    engine._audio_env = {
        "sound_class": "silence",
        "confidence": 0.9,
        "_updated_at": time.time(),
    }
    engine._timestamps["audio"] = time.time()
    summary = engine.get_summary()
    assert "silence" not in summary


def test_fused_events_tracked(engine):
    """Fused events are stored and shown in state."""
    engine._fused_events.append(
        {
            "narrative": "Someone entered the front door",
            "timestamp": time.time(),
        }
    )
    engine._timestamps["fused"] = time.time()
    state = engine.get_state()
    assert len(state["recent_events"]) == 1


def test_time_period_mapping(engine):
    """Time period helper returns reasonable strings."""
    assert engine._get_time_period(6) == "early morning"
    assert engine._get_time_period(10) == "morning"
    assert engine._get_time_period(15) == "afternoon"
    assert engine._get_time_period(19) == "evening"
    assert engine._get_time_period(22) == "night"
    assert engine._get_time_period(2) == "late night"


def test_stats(engine):
    """Stats reports correct telemetry."""
    s = engine.stats()
    assert s["updates_received"] == 0
    assert s["summaries_generated"] == 0
    engine.get_summary()
    assert engine.stats()["summaries_generated"] == 1
