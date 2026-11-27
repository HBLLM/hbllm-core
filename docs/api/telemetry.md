---
title: "API Reference — Observability & Telemetry"
description: "API reference for CognitiveTimeline, DecisionReplayEngine, ReplayDebugger, TracingMiddleware, and OpenTelemetry tracing helpers."
---

# Observability & Telemetry API

The **Observability Subsystem** provides microsecond latency tracking, causal event replay, distributed tracing, and Prometheus metrics export.

**Packages:** `hbllm.telemetry`, `hbllm.observability`, `hbllm.hcir.replay_debugger`

---

## Subsystem Index

| Class / Function | Module | Role |
|---|---|---|
| `CognitiveTimeline` / `TimelineEntry` | `hbllm.telemetry.timeline` | Ring buffer recording chronological cognitive events & provenance |
| `DecisionReplayEngine` / `DecisionRecord` | `hbllm.telemetry.replay` | Records and queries full decision context windows |
| `ReplayDebugger` | `hbllm.hcir.replay_debugger` | Step-by-step deterministic replay over HCIR `InstructionStream`s with breakpoints |
| `init_tracing` / `get_tracer` / `trace_span` | `hbllm.observability.tracing` | OpenTelemetry-compatible distributed trace span helpers |
| `TracingMiddleware` | `hbllm.observability.middleware` | HTTP request telemetry and trace instrumentation |

---

## `CognitiveTimeline`

**Module:** `hbllm.telemetry.timeline.CognitiveTimeline`

```python
from hbllm.telemetry.timeline import CognitiveTimeline, TimelineEntry

timeline = CognitiveTimeline(max_entries=10_000)

# Record an event with causal provenance
entry: TimelineEntry = timeline.record(
    subsystem="epistemic_revision",
    event_name="belief_updated",
    data={"belief_id": "b-12"},
)

# Query recent entries
events = timeline.query(last_n=50)
print(f"Recorded events: {len(events)}")
```

---

## `DecisionReplayEngine` & `ReplayDebugger`

**Modules:** `hbllm.telemetry.replay.DecisionReplayEngine`, `hbllm.hcir.replay_debugger.ReplayDebugger`

```python
from hbllm.telemetry.replay import DecisionRecord, DecisionReplayEngine

engine = DecisionReplayEngine()
engine.record(
    DecisionRecord(
        decision_type="plan_selection",
        selected_capabilities=["motor.navigate"],
        execution_result={"success": True},
    )
)
records = engine.get_replay_window(start_time=0.0, end_time=1e12)
```

---

## Distributed Tracing (`hbllm.observability.tracing`)

**Module:** `hbllm.observability.tracing`

```python
from hbllm.observability.tracing import get_tracer, init_tracing, trace_span

init_tracing(service_name="hbllm-core")
tracer = get_tracer("hbllm-core")

with trace_span("cognitive_cycle", attributes={"tenant.id": "acme-corp"}) as span:
    span.set_attribute("snn.spikes", 142)
```
