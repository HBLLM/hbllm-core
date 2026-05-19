---
title: "Architecture — Adaptive Hybrid Network"
description: "Architecture deep-dive into HBLLM's layered distributed network — Transport Layer, Routing Intelligence Layer, NodeState Engine, Discovery, and Gossip."
---

# Adaptive Hybrid Network

The Adaptive Hybrid Network is a 5-layer distributed architecture that separates physical transport from intelligent routing, enabling HBLLM to operate across local, LAN, and cloud boundaries.

```
┌──────────────────────────────────────────┐
│  Cognition Layer (AI Reasoning)          │
├──────────────────────────────────────────┤
│  Execution Layer (Tools / Skills)        │
├──────────────────────────────────────────┤
│  Routing Intelligence Layer (RIL)        │
├──────────────────────────────────────────┤
│  Discovery Layer (Registry / Gossip)     │
├──────────────────────────────────────────┤
│  Transport Layer (Dumb Pipes)            │
└──────────────────────────────────────────┘
```

## Transport Layer

Transports are "dumb pipes" — they send and receive messages and report raw metrics. They do **not** understand routing, capabilities, or cognition.

All transports implement the `Transport` abstract base class:

```python
from hbllm.network.transports import Transport, TransportState
```

### Available Transports

| Transport | Module | Use Case |
|---|---|---|
| `InProcessTransport` | `transports.inprocess` | Local async dispatch (zero network overhead) |
| `WebSocketTransport` | `transports.websocket` | Hub ↔ Edge uplink (global spine) |
| `RedisTransport` | `transports.redis` | Redis Pub/Sub for backend clusters |
| `WebRTCTransport` | `transports.webrtc` | P2P data channels for edge-to-edge |

### Example

```python
from hbllm.network.transports import InProcessTransport

transport = InProcessTransport(transport_id="local")
await transport.start()

# Subscribe
sub = await transport.subscribe("perception.*", handler)

# Send
await transport.send("perception.temperature", message)

# Metrics
metrics = transport.get_metrics()
print(f"Latency: {metrics.avg_latency_ms:.1f}ms, Error rate: {metrics.error_rate:.2%}")

await transport.stop()
```

## Routing Intelligence Layer (RIL)

The RIL selects the best transport for each message using a scoring model:

```
score = local_subscribers(+100) + latency(+50) + reliability(+30)
      + type_bonus(+5 to +20) - load_penalty(-25)
```

```python
from hbllm.network.routing import RoutingIntelligenceLayer

ril = RoutingIntelligenceLayer(node_id="homeserver")
ril.register_transport(inprocess_transport)
ril.register_transport(websocket_transport)
ril.set_node_state(node_state_engine)        # Optional: load-aware scoring
ril.set_capability_registry(cap_registry)     # Optional: capability routing

await ril.start()

# Use like a MessageBus — routing is transparent
await ril.publish("query.route", message)
response = await ril.request("tool.search", message, timeout=10.0)
```

### Capability-Aware Routing

When a message payload contains `_target_capability`, the RIL queries the CapabilityRegistry to find the best transport:

```python
msg = Message(
    type=MessageType.QUERY,
    source_node_id="server",
    topic="tool.call",
    payload={"_target_capability": "gpu_inference", "prompt": "..."},
)
await ril.publish("tool.call", msg)  # Routes to best node with gpu_inference
```

### ExecutionContext

Every routed message carries an `ExecutionContext` for traceability:

```python
from hbllm.network.routing import ExecutionContext

ctx = ExecutionContext(origin_node="server", max_hops=5)
ctx.add_hop("ipc", "inprocess", "server", latency_ms=0.1)
ctx.add_hop("ws", "websocket", "mobile", latency_ms=45.0)

ctx.hop_count          # 2
ctx.total_latency_ms   # 45.1
ctx.visited_node("server")  # True (loop detection)
ctx.has_exceeded_max_hops   # False
```

## NodeState Engine

Tracks the dynamic state of the local node — its role, health, load, and peer graph:

```python
from hbllm.network.node_state import NodeStateEngine, NodeRole

engine = NodeStateEngine(
    node_id="homeserver",
    role=NodeRole.COORDINATOR,
    device_tier="server",
    authority_score=90,
)
engine.set_status(NodeStateStatus.HEALTHY)
engine.update_load(cpu_load=0.3, memory_load=0.5, task_queue_depth=12)

# Peer management
engine.register_peer(PeerInfo(node_id="mobile", capabilities=["gps", "camera"]))
peers_with_gps = engine.find_peer_by_capability("gps")

# Role shifting (audited)
engine.set_role(NodeRole.RELAY, reason="network_partition")

# Serializable snapshot
snapshot = engine.snapshot()
```

### Roles

| Role | Description |
|---|---|
| `STANDALONE` | No network participation |
| `EDGE` | Connects upstream to a Hub |
| `RELAY` | Forwards traffic between peers |
| `COORDINATOR` | Central Hub / authority node |

## Discovery Layer

### CapabilityRegistry

A read-optimized truth store answering "who can do X, and how do I reach them?":

```python
from hbllm.network.discovery import CapabilityRegistry

registry = CapabilityRegistry(default_ttl=120.0)
registry.register("homeserver", ["llm_inference", "search"], is_local=True)
registry.register("mobile", ["gps", "camera"], latency_ms=45.0)

# Queries
best = registry.find_best_for_capability("llm_inference")  # Scores by local+latency+load
all_gps = registry.find_by_capability("gps")
summary = registry.get_network_summary()
```

### mDNS Discovery

Discovers HBLLM nodes on the local network via ZeroConf:

```python
from hbllm.network.discovery import MDNSDiscovery

discovery = MDNSDiscovery(
    node_id="homeserver",
    role="coordinator",
    capabilities=["llm_inference"],
    api_port=8000,
)
discovery.on_peer_found = handle_new_peer   # async callback
discovery.on_peer_lost = handle_lost_peer

await discovery.start()   # Broadcasts _hbllm._tcp.local.
```

### Gossip Sync

Epidemic-style state synchronization with safety mechanisms:

```python
from hbllm.network.discovery import GossipSync

gossip = GossipSync(node_id="homeserver", max_hops=3, gossip_interval=10.0)
gossip.set_node_state(node_state_engine)
gossip.set_capability_registry(registry)
gossip.set_send_fn(send_to_peer)   # Your transport send function

await gossip.start()
```

**Safety mechanisms:**
- **TTL**: Max 3 hops to prevent infinite propagation
- **Seen-set**: Deduplicates messages (1000 entry cache)
- **Entry TTL**: Stale entries auto-prune after 120s
- **Version ordering**: Only newer entries are merged

## Authority Model

| Domain | Authority | Rule |
|---|---|---|
| Memory | Hub wins | Long-term truth from coordinator |
| Execution | Local wins | Local node decides tool execution |
| Routing | RIL wins | RIL dictates message pathing |
