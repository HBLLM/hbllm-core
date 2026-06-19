# Distributed Trust & Authority

HBLLM uses a decentralized **Zero-Trust** architecture to coordinate multiple nodes (Edge devices, Central hosts, Specialized models) without a central point of failure or security vulnerability.

## 1. The Trust Model (Identity)

Every node in the swarm is identified by its **Node ID** and its **Ed25519 Public Key**.

- **Registration**: On first boot, a node generates a persistent key pair. It registers its public key with the `ServiceRegistry`.
- **Signing**: Every `Message` sent via the `MessageBus` must contain a `signature`. This signature is a hash of the message's ID, type, and payload, signed by the node's private key.
- **Verification**: The `TrustInterceptor` on the bus verifies the signature of every incoming message against the registered public key. If a signature is present and invalid, the message is dropped — even from registered internal nodes.
- **Internal Node Trust**: Registered internal nodes (brain components like router, memory, audio_in, etc.) that send **unsigned** messages are trusted implicitly — they are part of the local brain and don't need cryptographic signatures for internal bus traffic. However, if a registered node sends a message **with** a signature, the signature is always verified to catch forged payloads.
- **Replay Protection (Vector Clocks)**: Valid signatures alone don't prevent replay attacks. Therefore, `ServiceRegistry` also verifies causal ordering via embedded `VectorClock` data. If a message attempts to replay an older clock or violates causality, it is flagged as a security violation and dropped.
- **Registration Order**: Nodes **must** be registered in the `ServiceRegistry` *before* `node.start()` is called. This ensures that the node's initial heartbeat and startup messages are verifiable by the trust model.


## 2. Capability-Based Access Control (CapBAC)

Nodes are restricted by their **Scopes**. A scope is a permission to interact with a specific topic group or memory class.

- **Example**: A `vehicle` node might have scopes `["transport", "navigation", "public"]`.
- **Enforcement**: When a node tries to store or retrieve data from the `MemoryNode`, its `source_node_id` is checked against the registry. If it doesn't have the required scope for that data (e.g., `SENSITIVE`), the request is rejected with a `FORBIDDEN` code.

## 3. Authority Hierarchy & Conflict Resolution

In a distributed system, concurrent updates to the same memory state are inevitable. HBLLM resolves this using:

- **Vector Clocks**: Every node maintains a logical clock. Messages carry their causal history, allowing the system to determine if one update happened *before*, *after*, or *concurrently* with another.
- **Authority Score**: Every node is assigned an authority score (0-100).
    - **Laptop/Central Host**: Authority 90-100
    - **Authenticated User Phone**: Authority 70-80
    - **Edge IOT/Car Node**: Authority 30-50
- **LWW (Last-Write-Wins) Resolution**: If two updates are concurrent, the update from the node with the higher **Authority Score** wins. If scores are equal, the system falls back to the highest wall-clock timestamp.

## 4. Node Lifecycle (Resilience)

- **Heartbeats**: Nodes send periodic health checks to the registry.
- **Dying Gasp**: When a node shuts down gracefully, it sends a `NODE_DEREGISTERED` message. This allows the registry to reclaim its resources immediately, preventing other nodes from routing traffic to it.
- **Implicit Recovery**: If a node crashes and re-registers, it broadcasts its new state. Other nodes update their routing tables and vector clocks to re-sync causal history.
- **Node Compromise & Key Revocation**: If a node is suspected of being compromised, its identity can be permanently banned via `ServiceRegistry.revoke_node()`. This broadcasts a global `system.security.revocation` event, triggering gateways (like `SynapseGateway`) to instantly sever active WebSockets and flush outbound queues for the compromised node, isolating it from the network.
