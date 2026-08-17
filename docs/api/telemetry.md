---
title: "API Reference — Observability & Telemetry"
description: "API reference for TimelineRecorder, ReplayDebugger, TraceCollector, and Prometheus observability middleware."
---

# Observability & Telemetry API

The **Observability Subsystem** provides microsecond latency tracking, causal event replay, distributed tracing, and Prometheus metrics export.

**Packages:** `hbllm.telemetry`, `hbllm.observability`

---

## Subsystem Index

| Class | Module | Role |
|---|---|---|
| `TimelineRecorder` | `hbllm.telemetry.timeline` | Ring buffer recording sliding window cognitive events & stage latencies |
| `ReplayDebugger` | `hbllm.telemetry.replay` | Replays historical cognitive sessions step-by-step for root-cause analysis |
| `TraceCollector` | `hbllm.observability.tracing` | OpenTelemetry-compatible distributed trace span collector |
| `ObservabilityMiddleware` | `hbllm.observability.middleware` | HTTP request telemetry and latency histogram instrumentation |

---

## `TimelineRecorder`

**Module:** `hbllm.telemetry.timeline.TimelineRecorder`

```python
from hbllm.telemetry.timeline import TimelineEvent, TimelineRecorder

recorder = TimelineRecorder(max_events=10_000)

# Record event with latency
recorder.record(
    event=TimelineEvent(
        trace_id="tr-101",
        stage="epistemic_revision",
        node_id="belief_manager",
        duration_us=420,
        metadata={"belief_id": "b-12"},
    )
)

# Export recent timeline
events = recorder.get_recent(limit=50)
print(f"Recorded events: {len(events)}")
```

---

## `ReplayDebugger`

**Module:** `hbllm.telemetry.replay.ReplayDebugger`

```python
from hbllm.telemetry.replay import ReplayDebugger

debugger = ReplayDebugger(db_path="./data/cognitive_events.db")

# Load session for deterministic replay
session = await debugger.load_session(session_id="sess-prod-99")

async for step in session.step_forward():
    print(f"Replaying Node: {step.node_id}, State Delta: {step.mutation}")
```

---

## `TraceCollector`

**Module:** `hbllm.observability.tracing.TraceCollector`

```python
from hbllm.observability.tracing import TraceCollector

collector = TraceCollector(service_name="hbllm-core")

with collector.start_span("cognitive_cycle") as span:
    span.set_attribute("tenant.id", "acme-corp")
    span.set_attribute("snn.spikes", 142)
    # Execution occurs here
```
