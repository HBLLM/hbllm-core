"""
Unit tests for the Spiking Neural Network (SNN) primitives, spiking human attention gating,
and spiking reflex rules in HBLLM Core.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from hbllm.brain.control.attention import HumanAttentionModel, HumanAttentionState
from hbllm.brain.snn import LIFConfig, LIFNeuron, SpikeEvent, SpikingAccumulator
from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)
from hbllm.perception.reflex_arc import ReflexArc, ReflexRule, SpikingReflexRule


class TestLIFNeuron:
    """Tests for the Leaky Integrate-and-Fire (LIF) neuron model."""

    def test_basic_stimulation_and_spike(self):
        config = LIFConfig(threshold=1.0, decay_half_life=10.0, reset_potential=0.0, refractory_period=0.0)
        neuron = LIFNeuron(config)

        # Initial step
        event = neuron.step(0.5, timestamp=100.0)
        assert not event.fired
        assert event.strength == 0.0
        assert neuron.v == 0.5

        # Second step brings potential to 1.1 (above threshold 1.0)
        event2 = neuron.step(0.6, timestamp=100.0)
        assert event2.fired
        assert event2.strength == 1.1  # 1.1 / 1.0
        assert neuron.v == 0.0  # Reset potential

    def test_time_based_decay(self):
        # With half-life of 2.0s, potential decays by 50% in 2.0 seconds
        config = LIFConfig(threshold=2.0, decay_half_life=2.0, reset_potential=0.0, refractory_period=0.0)
        neuron = LIFNeuron(config)

        neuron.step(1.0, timestamp=100.0)
        assert neuron.v == 1.0

        # Move forward 2.0s, potential should decay from 1.0 to 0.5
        neuron.step(0.0, timestamp=102.0)
        assert pytest.approx(neuron.v) == 0.5

        # Move forward 2.0s more, should decay from 0.5 to 0.25
        neuron.step(0.0, timestamp=104.0)
        assert pytest.approx(neuron.v) == 0.25

    def test_instant_decay(self):
        config = LIFConfig(threshold=1.0, decay_half_life=0.0, reset_potential=0.0, refractory_period=0.0)
        neuron = LIFNeuron(config)

        neuron.step(0.5, timestamp=100.0)
        # 1 second later with instant decay (half-life = 0) should decay completely to 0.0
        neuron.step(0.0, timestamp=101.0)
        assert neuron.v == 0.0

    def test_refractory_period(self):
        config = LIFConfig(threshold=1.0, decay_half_life=100.0, reset_potential=0.0, refractory_period=2.0)
        neuron = LIFNeuron(config)

        # Fire spike at t=100
        event = neuron.step(1.2, timestamp=100.0)
        assert event.fired
        assert neuron.refractory_time_remaining == 2.0
        assert neuron.v == 0.0

        # Stimulate at t=101 (within refractory period). Input should be ignored.
        event2 = neuron.step(0.5, timestamp=101.0)
        assert not event2.fired
        assert neuron.v == 0.0
        assert neuron.refractory_time_remaining == 1.0

        # Stimulate at t=103 (past refractory period). Input should be accumulated.
        event3 = neuron.step(0.5, timestamp=103.0)
        assert not event3.fired
        assert neuron.v == 0.5
        assert neuron.refractory_time_remaining == 0.0


class TestSpikingAccumulator:
    """Tests for the generic SpikingAccumulator wrapper."""

    def test_accumulator_operations(self):
        config = LIFConfig(threshold=1.0, decay_half_life=5.0)
        accumulator = SpikingAccumulator(config)

        # Stimulate
        ev = accumulator.stimulate(0.4, timestamp=100.0)
        assert not ev.fired

        # get_potential fetches decayed potential
        pot = accumulator.get_potential(timestamp=105.0)  # Exactly 1 half-life later
        assert pytest.approx(pot) == 0.2

        # Check reset
        accumulator.reset()
        assert accumulator.get_potential(timestamp=105.0) == 0.0


class TestHumanAttentionModelSpiking:
    """Tests verifying the refactored HumanAttentionModel using spiking fatigue accumulation."""

    def test_consecutive_interruptions_spike(self):
        model = HumanAttentionModel()

        # Single mild interruption
        model.record_interruption(severity=0.1) # stimulus = 0.15 + 0.1 = 0.25
        assert model.state.approval_fatigue == 0.25
        assert not model.state.focus_mode_active

        # Multiple interruptions in quick succession (within 10 seconds)
        # Total charge = 0.25 + 0.25 + 0.25 + 0.25 = 1.00 (which exceeds threshold 0.8)
        model.record_interruption(severity=0.1)
        model.record_interruption(severity=0.1)

        # Third quick interruption fires a spike and triggers focus protection
        model.record_interruption(severity=0.1)
        assert model.state.focus_mode_active
        assert model.state.approval_fatigue == 0.0  # Reset potential is 0.0
        assert model.state.attention_budget == 100.0

    def test_sparse_interruptions_dont_spike(self):
        model = HumanAttentionModel()
        now = 1000.0

        # Record interruption at t=1000
        # Charge = 0.25
        model.fatigue_accumulator.stimulate(0.25, timestamp=now)

        # Move forward 5 minutes (300s, exactly 1 half-life). Fatigue decays to 0.125
        # Charge = 0.125 + 0.25 = 0.375
        model.fatigue_accumulator.stimulate(0.25, timestamp=now + 300.0)

        # Move forward another 5 minutes. Fatigue decays to 0.1875
        # Charge = 0.1875 + 0.25 = 0.4375 (never crosses threshold 0.8)
        model.fatigue_accumulator.stimulate(0.25, timestamp=now + 600.0)

        pot = model.fatigue_accumulator.get_potential(timestamp=now + 600.0)
        assert pot < 0.8
        assert not model.fatigue_accumulator.neuron.refractory_time_remaining > 0.0

    def test_natural_recovery_clears_focus_mode(self):
        # Use refractory of 10s for fast test
        config = LIFConfig(threshold=0.8, decay_half_life=10.0, refractory_period=10.0)
        model = HumanAttentionModel(config=config)

        # Fire a spike
        model.record_interruption(severity=0.7)  # charge = 0.85 > 0.8 threshold
        assert model.state.focus_mode_active

        # Natural recovery 5 seconds later: still in refractory period
        model.natural_recovery()  # internal time-decay step
        # Since we mock/use time.time() under the hood when no timestamp is passed,
        # we can step the internal accumulator directly using timestamps.
        model.fatigue_accumulator.neuron.step(0.0, timestamp=time.time() + 5.0)
        model.natural_recovery()
        assert model.state.focus_mode_active

        # Natural recovery 15 seconds later: refractory period is over
        model.fatigue_accumulator.neuron.step(0.0, timestamp=time.time() + 15.0)
        model.natural_recovery()
        assert not model.state.focus_mode_active

    def test_can_interrupt_gating(self):
        config = LIFConfig(threshold=0.8, decay_half_life=300.0, refractory_period=600.0)
        model = HumanAttentionModel(config=config)

        # Initial: can interrupt
        assert model.can_interrupt(action_criticality=0.5)

        # Heavy fatigue spike
        model.record_interruption(severity=0.8) # triggers spike, focus mode activated

        # Critical action (>= 0.9) bypasses focus protection
        assert model.can_interrupt(action_criticality=0.95)
        # Normal action is blocked
        assert not model.can_interrupt(action_criticality=0.5)


class TestSpikingReflexRule:
    """Tests for the SpikingReflexRule and its integration with ReflexArc."""

    @pytest.mark.asyncio
    async def test_spiking_reflex_accumulation_and_firing(self):
        bus = RealityEventBus()

        # Setup spiking rule: threshold = 100.0, decay half-life = 2.0s, refractory = 0.0
        rule = SpikingReflexRule(
            trigger={"event_type": "resource_monitor", "cpu_usage": 1.0},
            action_topic="system.action.throttle",
            action_payload={"reason": "CPU Overload"},
            config=LIFConfig(threshold=100.0, decay_half_life=2.0, reset_potential=0.0, refractory_period=0.0),
            current_multiplier=50.0  # Each cpu_usage input will be scaled by 50
        )

        reflex_arc = ReflexArc(bus, [rule])

        # Ingest one event: cpu_usage = 1.0 -> stimulus = 50.0. Under threshold (100.0). No reflex fired.
        ev1 = PerceptionEvent(event_type="resource_monitor", payload={"cpu_usage": 1.0})
        await bus.ingest(ev1)
        assert len(reflex_arc.fired_actions) == 0

        # Ingest second event: cpu_usage = 1.2 -> stimulus = 60.0. Total = 110.0. Fires spike!
        ev2 = PerceptionEvent(event_type="resource_monitor", payload={"cpu_usage": 1.2})
        await bus.ingest(ev2)
        assert len(reflex_arc.fired_actions) == 1

        topic, payload = reflex_arc.fired_actions[0]
        assert topic == "system.action.throttle"
        # Verify spike strength was added to payload (using approx to allow for slight time decay)
        assert "spike_strength" in payload
        assert pytest.approx(payload["spike_strength"], abs=1e-2) == 1.10

    @pytest.mark.asyncio
    async def test_transient_spike_decays_without_firing(self):
        bus = RealityEventBus()

        # Setup rule: threshold = 10.0, decay half-life = 0.5 seconds
        rule = SpikingReflexRule(
            trigger={"event_type": "error_burst", "error_count": 1.0},
            action_topic="system.action.alert",
            action_payload={},
            config=LIFConfig(threshold=10.0, decay_half_life=0.5),
            current_multiplier=8.0
        )
        reflex_arc = ReflexArc(bus, [rule])

        # Ingest alert at t=now. Stimulus = 8.0. No firing.
        ev1 = PerceptionEvent(event_type="error_burst", payload={"error_count": 1.0})
        await bus.ingest(ev1)
        assert len(reflex_arc.fired_actions) == 0

        # Wait 2.0 seconds (4 half-lives). Potential decays from 8.0 down to 0.5.
        await asyncio.sleep(2.0)

        # Ingest second alert: stimulus = 8.0. Total = 8.5 < 10.0. Still no firing.
        ev2 = PerceptionEvent(event_type="error_burst", payload={"error_count": 1.0})
        await bus.ingest(ev2)
        assert len(reflex_arc.fired_actions) == 0
