---
title: "API Reference — Network Layer"
description: "API documentation for HBLLM's message bus, messages, and circuit breaker."
---

# Network API

The network layer provides asynchronous Pub/Sub communication between all cognitive nodes.

## MessageBus

### InProcessBus

```python
from hbllm.network.bus import InProcessBus

bus = InProcessBus()
await bus.start()

# Subscribe to topics (supports wildcards)
sub = await bus.subscribe("perception.*", handler)

# Publish messages
await bus.publish("perception.temperature", message)

# Request-response pattern with timeout
response = await bus.request("query.route", message, timeout=5.0)

# Cleanup
await bus.unsubscribe(sub)
await bus.stop()
```

### RedisBus

For distributed multi-server deployments with HMAC authentication:

```python
from hbllm.network.redis_bus import RedisBus

bus = RedisBus(
    redis_url="redis://localhost:6379",
    hmac_key="your-secret-key",
)
await bus.start()
```

## Message Model

All inter-node communication uses the `Message` Pydantic model:

```python
from hbllm.network.messages import Message, MessageType, Priority

msg = Message(
    type=MessageType.QUERY,
    source_node_id="router-01",
    topic="query.route",
    payload={"text": "Hello, world!"},
    priority=Priority.HIGH,
    tenant_id="tenant-001",
    ttl_seconds=30.0,
)

# Create correlated responses
response = msg.create_response(payload={"answer": "Hi!"})
error = msg.create_error(error="Something went wrong", code="ERR_01")
```

### Message Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | Auto UUID | Unique message identifier |
| `type` | `MessageType` | *(required)* | Message category |
| `source_node_id` | `str` | *(required)* | Sending node ID |
| `target_node_id` | `str \| None` | `None` | Target node (None = broadcast) |
| `tenant_id` | `str` | `"default"` | Multi-tenant isolation key |
| `session_id` | `str` | `"default"` | Session correlation |
| `topic` | `str` | *(required)* | Message routing topic |
| `payload` | `dict[str, Any]` | `{}` | Message data |
| `priority` | `Priority` | `NORMAL` | Priority level |
| `timestamp` | `datetime` | Auto UTC now | Creation timestamp |
| `correlation_id` | `str \| None` | `None` | Links requests to responses |
| `ttl_seconds` | `float \| None` | `None` | Time-to-live in seconds |

### MessageType

| Type | Usage |
|---|---|
| `QUERY` | User query needing response |
| `RESPONSE` | Reply to a query |
| `ERROR` | Error report |
| `EVENT` | Fire-and-forget notification |
| `ROUTE_REQUEST` | Intent routing request |
| `ROUTE_DECISION` | Routing decision result |
| `TASK_DECOMPOSE` | GoT task breakdown |
| `TASK_RESULT` | Task execution result |
| `MEMORY_STORE` / `MEMORY_SEARCH` / `MEMORY_RESULT` | Memory operations |
| `HEARTBEAT` / `HEARTBEAT_ACK` | Health monitoring |
| `FEEDBACK` / `LEARNING_UPDATE` | Learning pipeline |
| `SPAWN_REQUEST` / `SPAWN_COMPLETE` | Neurogenesis |
| `SYSTEM_IMPROVE` | Self-improvement trigger |
| `SALIENCE_SCORE` | Experience salience |

### Priority (int Enum)

Higher values are dispatched first:

| Constant | Value | Use Case |
|---|---|---|
| `Priority.LOW` | `0` | Background tasks, learning |
| `Priority.NORMAL` | `1` | Standard processing |
| `Priority.HIGH` | `2` | User-facing queries |
| `Priority.CRITICAL` | `3` | Safety, shutdown commands |

## CircuitBreaker

Protects against cascading failures with automatic recovery:

```python
from hbllm.network.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
)
```

| State | Behavior |
|---|---|
| `CLOSED` | Normal operation, tracking failures |
| `OPEN` | All calls rejected, waiting for recovery |
| `HALF_OPEN` | One test call allowed to check recovery |

A `CircuitBreakerRegistry` manages multiple breakers across nodes.
