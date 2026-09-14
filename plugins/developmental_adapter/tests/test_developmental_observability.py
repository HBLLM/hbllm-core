"""Unit tests for Developmental Learning Observability, Metrics & Distributed Tracing."""

from __future__ import annotations

import time

from hbllm.network.metrics import MetricsCollector
from hbllm.observability import trace_span
from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.causal_discovery import InterventionalCausalDiscoveryEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.metrics import (
    DevelopmentalTelemetryEmitter,
)
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
from plugins.developmental_adapter.trainer import CognitiveSchoolTrainer, SchoolTrainingConfig
from plugins.developmental_adapter.types import BabyActionType


def test_telemetry_emitter_standalone() -> None:
    """Telemetry emitter must accurately record events, gauges, and latencies in memory."""
    DevelopmentalTelemetryEmitter.reset()
    emitter = DevelopmentalTelemetryEmitter.get_instance()

    # 1. Record events
    emitter.record_intervention("push", "moved")
    emitter.record_intervention("push", "static")
    emitter.record_hypothesis_event("generated")
    emitter.record_hypothesis_event("falsified")
    emitter.record_hypothesis_event("confirmed")
    emitter.record_concept_acquired("spatial")
    emitter.record_concept_acquired("tool")
    emitter.record_entropy("causal_beliefs", 1.585)

    with emitter.measure_latency("test_stage"):
        time.sleep(0.005)

    snapshot = emitter.get_telemetry_snapshot()

    # Verify counters
    assert snapshot["counters"]["intervention:push:moved"] == 1
    assert snapshot["counters"]["intervention:push:static"] == 1
    assert snapshot["counters"]["hypothesis:generated"] == 1
    assert snapshot["counters"]["hypothesis:falsified"] == 1
    assert snapshot["counters"]["hypothesis:confirmed"] == 1
    assert snapshot["counters"]["concept:spatial"] == 1
    assert snapshot["counters"]["concept:tool"] == 1

    # Verify gauges & latencies
    assert snapshot["gauges"]["entropy:causal_beliefs"] == 1.585
    assert "test_stage" in snapshot["latencies"]
    assert snapshot["latencies"]["test_stage"]["count"] == 1
    assert snapshot["latencies"]["test_stage"]["avg"] > 0.0
    assert snapshot["recent_events_count"] == 7


def test_metrics_collector_developmental_integration() -> None:
    """Core MetricsCollector must correctly record developmental events across backends."""
    collector = MetricsCollector.get_instance()

    collector.record_developmental_intervention(action_type="lift", result="moved")
    collector.record_developmental_hypothesis(outcome="generated")
    collector.record_developmental_hypothesis(outcome="falsified")
    collector.record_developmental_concept(domain="relational")
    collector.set_developmental_entropy(system="affordances", entropy_val=0.75)

    with collector.measure_developmental_latency("test_dev_stage"):
        time.sleep(0.002)

    text = collector.get_metrics_text()
    assert text is not None

    snap = collector.snapshot()
    assert snap is not None


def test_trace_span_graceful_degradation() -> None:
    """Trace spans must function cleanly as context managers with attributes."""
    with trace_span("developmental.test_span", attributes={"iteration": 1, "test": "true"}) as span:
        assert span is not None
        time.sleep(0.001)


def test_causal_discovery_emits_telemetry_end_to_end() -> None:
    """InterventionalCausalDiscoveryEngine must emit real telemetry during its lifecycle."""
    DevelopmentalTelemetryEmitter.reset()
    emitter = DevelopmentalTelemetryEmitter.get_instance()

    env = BabyWorldEnvironment(seed=42)
    obs = env.reset("confounded_train_world")
    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(
        substrate=substrate, perception=perception, environment=env
    )

    demos = env.generate_observational_demonstrations()
    hyps = engine.observe_and_generate_hypotheses(obs, episodes_data=demos)
    assert len(hyps) > 0

    snapshot_after_gen = emitter.get_telemetry_snapshot()
    assert snapshot_after_gen["counters"]["hypothesis:generated"] == len(hyps)
    assert "entropy:causal_beliefs" in snapshot_after_gen["gauges"]

    # Execute probe
    target_id = list(env.objects.keys())[0]
    did_move, _ = engine.execute_interventional_probe(
        target_id=target_id, action=BabyActionType.PUSH
    )

    snapshot_after_probe = emitter.get_telemetry_snapshot()
    res_tag = "moved" if did_move else "static"
    assert snapshot_after_probe["counters"][f"intervention:push:{res_tag}"] >= 1
    assert "intervention" in snapshot_after_probe["latencies"]
    assert snapshot_after_probe["latencies"]["intervention"]["count"] >= 1


def test_school_trainer_emits_telemetry() -> None:
    """CognitiveSchoolTrainer must emit telemetry and trace spans across semester training."""
    DevelopmentalTelemetryEmitter.reset()
    emitter = DevelopmentalTelemetryEmitter.get_instance()

    config = SchoolTrainingConfig(
        student_name="Telemetry Student",
        teacher_name="Dr. Maria Vygotsky",
        num_semesters=1,
        episodes_per_semester=1,
        seed=100,
    )
    trainer = CognitiveSchoolTrainer(config=config)

    summary = trainer.train()
    assert summary.total_semesters == 1

    snapshot = emitter.get_telemetry_snapshot()
    assert snapshot["counters"]["concept:perceptual_lexicon"] == 1
    assert "semester_1" in snapshot["latencies"]
    assert snapshot["latencies"]["semester_1"]["count"] == 1
