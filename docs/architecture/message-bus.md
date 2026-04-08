---
title: "Message Bus Architecture — Async Pub/Sub Communication"
description: "Architecture deep-dive into HBLLM's async message bus — topic-based routing, distributed deployment with RedisBus, circuit breakers, and message correlation patterns."
---

# Network API

The network layer provides asynchronous Pub/Sub communication between all cognitive nodes.

## MessageBus

### InProcessBus

```python
from hbllm.network.bus import InProcessBus

bus = InProcessBus()
await bus.start()

# Subscribe to topics
sub = await bus.subscribe("perception.*", handler)

# Publish messages
await bus.publish("perception.temperature", message)

# Request-response pattern
response = await bus.request("query.route", message, timeout=5.0)

# Cleanup
await bus.unsubscribe(sub)
await bus.stop()
```

### RedisBus

```python
from hbllm.network.redis_bus import RedisBus

bus = RedisBus(
    redis_url="redis://localhost:6379",
    hmac_key="your-secret-key",
)
await bus.start()
```

## Message

```python
from hbllm.network.messages import Message, MessageType, Priority

msg = Message(
    type=MessageType.QUERY,
    source_node_id="router-01",
    topic="query.route",
    payload={"text": "Hello"},
    priority=Priority.HIGH,
    ttl_seconds=30.0,
    correlation_id="req-001",
    tenant_id="tenant-001",
    session_id="session-abc",
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | Auto UUID | Unique message identifier |
| `type` | `MessageType` | — | Message category (required) |
| `source_node_id` | `str` | — | ID of the sending node (required) |
| `target_node_id` | `str \| None` | `None` | Target node (None = broadcast) |
| `tenant_id` | `str` | `"default"` | Multi-tenant isolation key |
| `session_id` | `str` | `"default"` | Session correlation key |
| `topic` | `str` | — | Message topic (required) |
| `payload` | `dict` | `{}` | Message data |
| `priority` | `Priority` | `NORMAL` | Message priority level |
| `timestamp` | `datetime` | Auto UTC now | Creation timestamp |
| `correlation_id` | `str \| None` | `None` | Links request to response |
| `ttl_seconds` | `float \| None` | `None` | Message expiry time |

### MessageType Enum

| Value | Usage |
|---|---|
| `QUERY` | User query requiring a response |
| `RESPONSE` | Reply to a query |
| `ERROR` | Error report |
| `EVENT` | Fire-and-forget notification |
| `ROUTE_REQUEST` | Intent routing request |
| `ROUTE_DECISION` | Routing decision result |
| `TASK_DECOMPOSE` | GoT task breakdown |
| `TASK_RESULT` | Task execution result |
| `TASK_AGGREGATE` | Aggregated task results |
| `MEMORY_STORE` | Store to memory |
| `MEMORY_SEARCH` | Search memory |
| `MEMORY_RESULT` | Memory search results |
| `HEARTBEAT` | Health check ping |
| `FEEDBACK` | User feedback on responses |
| `LEARNING_UPDATE` | Weight update notification |
| `SPAWN_REQUEST` | Create new domain module |
| `SPAWN_COMPLETE` | Module creation complete |
| `SYSTEM_IMPROVE` | Self-improvement trigger |
| `SALIENCE_SCORE` | Experience salience score |

### Priority Levels

Priority is an integer enum — higher values are processed first:

| Priority | Value | Use Case |
|---|---|---|
| `LOW` | 0 | Background tasks, learning |
| `NORMAL` | 1 | Standard processing |
| `HIGH` | 2 | User-facing queries |
| `CRITICAL` | 3 | Safety, shutdown commands |

### Creating Responses

Messages provide helper methods for creating correlated responses:

```python
# Create a response linked to the original message
response = original_msg.create_response(
    payload={"answer": "4"}
)

# Create an error response
error = original_msg.create_error(
    error="Division by zero",
    code="MATH_ERROR"
)
```

## Topic Patterns

Topics follow a hierarchical dot-notation:

```
query.route          — Intent routing
planning.create      — GoT DAG generation
memory.episodic.*    — All episodic memory ops
perception.*         — All perception events
*                    — Global wildcard (monitoring)
```

## CircuitBreaker

The `CircuitBreaker` (in `hbllm.network.circuit_breaker`) protects against cascading failures:

```python
from hbllm.network.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
)
```

States: `CLOSED` → `OPEN` → `HALF_OPEN` → `CLOSED`

A `CircuitBreakerRegistry` is also available for managing multiple breakers across nodes.
