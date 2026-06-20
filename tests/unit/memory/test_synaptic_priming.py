"""
Unit and integration tests for SNN observability (Metrics + Telemetry) and the Synaptic Priming retrieval layer.
"""

from __future__ import annotations

import asyncio

import pytest

from hbllm.brain.snn import LIFConfig
from hbllm.memory.priming import WorkingMemoryPrimer
from hbllm.memory.semantic import SemanticMemory
from hbllm.network.metrics import MetricsCollector
from hbllm.perception.reality_bus import PerceptionEvent, RealityEventBus
from hbllm.perception.reflex_arc import ReflexArc, SpikingReflexRule


class TestSNNMetricsAndTelemetry:
    """Verifies Prometheus metrics and RealityEventBus SnTelemetry."""

    def test_metrics_collection(self):
        # Reset MetricsCollector singleton to ensure a fresh test
        MetricsCollector.reset()
        collector = MetricsCollector.get_instance()

        # Stimulate a dummy LIF neuron through standard metrics recording
        collector.record_snn_potential("test_neuron_A", 0.75)
        collector.record_snn_spike("test_neuron_A", 1.25)

        # Check that metrics were correctly recorded in the in-memory fallback
        snapshot = collector.snapshot()
        assert snapshot["backend"] == "inmemory"
        assert snapshot["gauges"]["snn_potential:test_neuron_A"] == 0.75
        assert snapshot["counters"]["snn_spikes:test_neuron_A"] == 1.25

    @pytest.mark.asyncio
    async def test_reality_bus_snn_telemetry(self):
        bus = RealityEventBus()

        # Setup reflex rule to stimulate and publish SnTelemetry
        rule = SpikingReflexRule(
            trigger={"event_type": "resource_monitor", "cpu_usage": 1.0},
            action_topic="system.action.throttle",
            action_payload={},
            config=LIFConfig(threshold=1.0, decay_half_life=10.0),
            current_multiplier=0.6,  # 0.6 stimulus (over delta threshold 0.05)
        )
        ReflexArc(bus, [rule])

        # Track telemetry events published on the bus
        telemetry_events = []

        def on_telemetry(event: PerceptionEvent):
            if event.event_type == "snn_telemetry":
                telemetry_events.append(event)

        bus.subscribe(on_telemetry)

        # Ingest one event
        ev = PerceptionEvent(event_type="resource_monitor", payload={"cpu_usage": 1.0})
        await bus.ingest(ev)

        # Give asyncio tasks a moment to run out-of-band SnTelemetry ingestion
        await asyncio.sleep(0.1)

        assert len(telemetry_events) == 1
        tel_event = telemetry_events[0]
        assert tel_event.sub_type == "reflex_potential"
        assert tel_event.payload["neuron_id"] == "reflex_system_action_throttle"
        assert pytest.approx(tel_event.payload["potential"]) == 0.6
        assert tel_event.payload["refractory_time_remaining"] == 0.0
        assert tel_event.payload["fired"] is False


class TestWorkingMemoryPrimer:
    """Verifies the WorkingMemoryPrimer keyword matching and direct stimulation."""

    def test_keyword_stimulation(self):
        primer = WorkingMemoryPrimer()

        # Stimulate by text containing physics keywords
        primer.stimulate_by_text("Evaluating quantum gravity equations on the cluster.")
        boosts = primer.get_boosts()

        assert boosts["physics"] > 0.0
        assert boosts["math"] > 0.0
        assert boosts["coding"] == 0.0

    def test_direct_stimulation(self):
        primer = WorkingMemoryPrimer()

        # Directly stimulate category
        primer.stimulate_category("finance", charge=0.8)
        boosts = primer.get_boosts()

        assert boosts["finance"] == pytest.approx(0.8, abs=1e-3)
        assert boosts["general"] == 0.0

    def test_decay_over_time(self):
        # Setup config with short half-life of 2.0s
        config = LIFConfig(threshold=1.0, decay_half_life=2.0)
        primer = WorkingMemoryPrimer(config=config)

        primer.stimulate_category("coding", charge=0.8)
        assert primer.get_boosts()["coding"] == pytest.approx(0.8, abs=1e-3)

        # Decay by simulating step with time delta (step is applied in get_potential internally via step(0, now))
        # Modify internal category neuron state to simulate decay over 2 seconds
        # (Since time.time() is used, we'll manually step using a test epoch timestamp)
        neuron = primer.categories["coding"].neuron
        neuron.last_update_time = 100.0
        neuron.v = 0.8

        neuron.step(0.0, timestamp=102.0)  # 2.0s later
        assert pytest.approx(neuron.v) == 0.4


class TestSemanticMemoryPrimingBoosts:
    """Verifies that WorkingMemoryPrimer boosts matching category documents in search."""

    def test_priming_boosts_search_scores(self):
        # Start fresh in-memory SemanticMemory
        mem = SemanticMemory()

        # Store documents in different domains
        doc_phys = mem.store(
            "Introduction to string theory and quantum loop gravity.",
            metadata={"domain": "physics"},
        )
        doc_math = mem.store(
            "Solving differential equations using calculus principles.", metadata={"domain": "math"}
        )
        mem.store(
            "Writing clean asynchronous Python code and modules.", metadata={"domain": "coding"}
        )

        # Query: "theory"
        # Without priming
        results_unprimed = mem.search("theory", top_k=3)
        # Physics document should rank high because it mentions "theory"
        assert results_unprimed[0]["id"] == doc_phys

        # Query: "calculus"
        # Without priming
        res_math_unprimed = mem.search("calculus", top_k=3)
        assert res_math_unprimed[0]["id"] == doc_math

        # Query: "theory" with math category highly primed (potential = 1.0)
        priming_boosts = {"math": 1.0, "physics": 0.0, "coding": 0.0}
        results_primed = mem.search(
            "theory", top_k=3, priming_boosts=priming_boosts, priming_boost_weight=0.5
        )

        # Due to math boost of 0.5, doc_math (score ~ similarity + 0.5) should outrank doc_phys (similarity ~ 0.3)
        assert results_primed[0]["id"] == doc_math
