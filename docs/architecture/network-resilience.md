---
title: "Network Resilience — Durable Bus, Load Balancing & Fault Tolerance"
description: "Architecture deep-dive into HBLLM's production-grade network resilience layer — durable message persistence, load balancing strategies, circuit breakers, graceful degradation, and distributed tracing."
---

# Network Resilience

Beyond the core `InProcessBus` and `RedisBus`, HBLLM provides a full resilience layer for production deployments. These modules ensure the cognitive brain remains operational even when individual nodes fail, networks partition, or load spikes occur.

---

## Module Index

| File | Class | Purpose |
|---|---|---|
| `durable_bus.py` | `DurableBus` | SQLite-backed at-least-once delivery with dead letter queue |
| `load_balancer.py` | `LoadBalancer` | Distribute requests across healthy nodes (round-robin, least-loaded, capability-match) |
| `fallback.py` | Fallback chain | Graceful provider failover when primary LLM is unavailable |
| `degraded.py` | Degraded mode handler | Reduced-capability operation during partial outages |
| `health.py` | Health monitoring | Node health checks and status aggregation |
| `circuit_breaker.py` | `CircuitBreakerRegistry` | Per-node circuit breakers with half-open recovery |
| `cognition_router.py` | `CognitionRouter` | Intelligent routing based on query complexity and node availability |
| `cluster_config.py` | `ClusterConfig` | Multi-node cluster topology configuration |
| `metrics.py` | Network metrics | Bus throughput, latency histograms, node activation counters |
| `tracing.py` | Distributed tracing | OpenTelemetry spans for cross-node message flows |
| `serialization.py` | Message serialization | JSON + msgpack codecs for bus messages |
| `plugin_manager.py` | Plugin system | Dynamic node loading and lifecycle management |

---

## Durable Bus

**Module:** `hbllm.network.durable_bus.DurableBus`

Wraps any `MessageBus` (InProcess or Redis) with SQLite-backed persistence for at-least-once delivery guarantees.

### Features

- **SQLite WAL journal** — Messages persisted before publish
- **Exponential backoff retry** — Configurable base delay, max delay, max retries
- **Dead letter queue** — Failed messages after max retries are preserved for inspection
- **Message deduplication** — Bounded set prevents duplicate processing
- **Transparent wrapping** — Subscribe/unsubscribe/request pass through to inner bus

### Usage

```python
from hbllm.network.bus import InProcessBus
from hbllm.network.durable_bus import DurableBus

inner = InProcessBus()
bus = DurableBus(inner, db_path="messages.db", max_retries=3)
await bus.start()

# Use exactly like any MessageBus
await bus.publish("topic", message)
await bus.subscribe("topic", handler)

# Inspect failures
dead_letters = bus.get_dead_letters(limit=10)
print(f"Dead: {bus.dead_letter_count()}, Pending: {bus.pending_count()}")
print(bus.stats())
```

### Message Lifecycle

```
publish() → journal(SQLite) → inner.publish()
                                  ↓
                           handler succeeds → mark_delivered
                           handler fails    → mark_failed → schedule retry
                                                              ↓
                                                     retry succeeds → delivered
                                                     retry fails    → retry again
                                                     max retries    → dead letter queue
```

---

## Load Balancer

**Module:** `hbllm.network.load_balancer.LoadBalancer`

Distributes requests across multiple nodes providing the same capability. Integrates with `ServiceRegistry` for discovery and `CircuitBreakerRegistry` to skip unhealthy nodes.

### Strategies

| Strategy | Algorithm | Best For |
|---|---|---|
| `round_robin` | Cycle through healthy nodes | Even distribution, stateless |
| `least_loaded` | Weighted score: latency × 0.3 + health × 0.7 | Heterogeneous hardware |
| `capability_match` | Prefer exact capability match, fallback to round-robin | Specialized nodes |

### Usage

```python
from hbllm.network.load_balancer import LoadBalancer

lb = LoadBalancer(
    registry=registry,
    circuit_breakers=breaker_registry,
    strategy="least_loaded",
)

# Select best node for a capability
node = await lb.select("reasoning")
if node:
    await bus.publish(f"node.{node.node_id}", message)
```

---

## Cluster Configuration

**Module:** `hbllm.network.cluster_config.ClusterConfig`

Pydantic model for multi-node cluster topology, configurable via `hbllm.yaml`:

```yaml
cluster:
  node_id: "node-1"
  role: "primary"           # primary | worker | observer
  bus_type: "redis"         # inprocess | redis | durable
  redis_url: "redis://localhost:6379"
  redis_hmac_key: "secret"
  heartbeat_interval_s: 10
  discovery_method: "static"  # static | dns | consul
```

---

## Additional Modules

### Cognition Router

Intelligently routes queries based on complexity analysis, available node health, and historical performance data.

### Distributed Tracing

OpenTelemetry instrumentation for cross-node message flows:

```python
from hbllm.network.tracing import TracingMiddleware

# Automatically adds spans for publish/subscribe/request
bus.add_interceptor(TracingMiddleware())
```

### Plugin Manager

Dynamic node loading and lifecycle management for extending the brain at runtime without restarting.
