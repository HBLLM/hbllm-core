"""Tests for the WorldStateEngine (Phase 3.2)."""

from __future__ import annotations

import time

import pytest

from hbllm.brain.world_state import WorldStateEngine
from hbllm.perception.event_log import EventLog
from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
)


@pytest.fixture
def temp_event_log(tmp_path):
    """Provides a temporary SQLite EventLog."""
    return EventLog(data_dir=tmp_path)


@pytest.mark.asyncio
class TestWorldStateEngine:
    async def test_entity_state_update(self):
        """Test that handling an event updates the live graph."""
        engine = WorldStateEngine()

        event = PerceptionEvent(
            entity_id="device_1",
            event_type="os",
            sub_type="app_switch",
            modality=PerceptionModality.SYSTEM,
            origin=EventOrigin.SYSTEM,
            confidence=0.9,
            source_trust=1.0,
            payload={"app": "terminal"},
        )

        await engine.handle_normalized_event(event)

        state = engine.get_entity_state("device_1")
        assert state is not None
        assert state.properties["app"] == "terminal"
        assert state.confidence == 0.9
        assert EventOrigin.SYSTEM.value in state.source_set

    async def test_confidence_decay(self):
        """Test that older events lose confidence over time."""
        engine = WorldStateEngine()

        event = PerceptionEvent(
            entity_id="user_1",
            event_type="presence",
            confidence=1.0,
            source_trust=1.0,
            event_timestamp=time.time() - 3600.0,  # 1 hour ago
        )
        await engine.handle_normalized_event(event)

        state = engine.get_entity_state("user_1")
        assert state is not None

        # Manually trigger decay based on current time.
        # With half_life = 3600s, confidence should halve.
        state.decay_confidence(current_time=time.time(), half_life_s=3600.0)
        assert 0.45 < state.confidence < 0.55

    async def test_boot_recovery_from_log(self, temp_event_log):
        """Test that the engine rebuilds state from the EventLog."""
        engine = WorldStateEngine(event_log=temp_event_log)

        # Pre-seed the log
        e1 = PerceptionEvent(
            entity_id="node_a",
            event_type="status",
            confidence=0.8,
            payload={"status": "online"},
            event_timestamp=time.time(),
        )
        e2 = PerceptionEvent(
            entity_id="node_b",
            event_type="status",
            confidence=1.0,
            payload={"status": "offline"},
            event_timestamp=time.time(),
        )
        temp_event_log.append(e1)
        temp_event_log.append(e2)

        # Trigger boot recovery
        await engine.boot_recovery()

        state_a = engine.get_entity_state("node_a")
        state_b = engine.get_entity_state("node_b")

        assert state_a is not None
        assert state_a.properties["status"] == "online"

        assert state_b is not None
        assert state_b.properties["status"] == "offline"

    async def test_simulation_interface_retained(self):
        """Ensure the legacy simulator is still available via WorldStateEngine
        module exports (SimulationInterface)."""
        from hbllm.brain.world_state import SimulationInterface

        sim = SimulationInterface()
        result = await sim.simulate("Test goal", [{"name": "strategy_1", "steps": ["step1"]}])

        assert result.best_scenario is not None
        assert result.best_scenario.strategy == "strategy_1"
