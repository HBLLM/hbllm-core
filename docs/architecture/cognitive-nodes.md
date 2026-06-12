---
title: "Cognitive Nodes — HBLLM Brain Components"
description: "Reference documentation for all 28+ cognitive nodes in the HBLLM architecture, including Router, Planner, Critic, Learner, and Spawner."
---

# Cognitive Nodes Reference

Every cognitive function in HBLLM is encapsulated in a **Node** — a self-contained, asynchronous unit that communicates exclusively via the message bus.

## Node Base Class

All nodes inherit from `hbllm.network.node.Node`:

```python
from hbllm.network.node import Node, NodeType

class MyNode(Node):
    def __init__(self):
        super().__init__(
            node_id="my-node",
            node_type=NodeType.CORE,
            capabilities=["custom-processing"]
        )

    async def on_start(self) -> None:
        """Subscribe to topics when launched."""
        ...

    async def on_stop(self) -> None:
        """Cleanup when shutting down."""
        ...

    async def handle_message(self, message):
        """Process incoming messages."""
        ...
```

## Node Hardening & Identity

In distributed environments, every node functions as an autonomous security principal with the following hardened attributes:

- **Cryptographic Identity**: Nodes generate an **Ed25519** key pair on startup. All outbound messages are signed using this private key to prevent spoofing.
- **Authority Score**: A numeric value (`0-100`) determining the node's relative trust level. High-trust devices (e.g., local workstations) carry higher authority than edge peripherals.
- **Scoped Permissions**: Nodes register with specific `scopes` (e.g., `["public", "navigation"]`). The `MemoryNode` enforces these scopes, ensuring nodes only access authorized data categories.
- **Dying Gasp**: Graceful shutdown notification (`NODE_DEREGISTERED`) that triggers immediate resource reclamation in the `ServiceRegistry`.

---

## Node Types

The `NodeType` enum defines the categories of nodes:

| Value           | Constant                 | Purpose                                      |
| --------------- | ------------------------ | -------------------------------------------- |
| `router`        | `NodeType.ROUTER`        | Intent classification and domain routing     |
| `core`          | `NodeType.CORE`          | Core reasoning (Critic, Decision, Workspace) |
| `domain_module` | `NodeType.DOMAIN_MODULE` | Domain-specific LoRA modules                 |
| `memory`        | `NodeType.MEMORY`        | Memory storage and retrieval                 |
| `planner`       | `NodeType.PLANNER`       | Task decomposition and GoT planning          |
| `learner`       | `NodeType.LEARNER`       | Continuous learning and DPO training         |
| `detector`      | `NodeType.DETECTOR`      | Perception and sensor input                  |
| `spawner`       | `NodeType.SPAWNER`       | Neurogenesis — creating new domain modules   |
| `meta`          | `NodeType.META`          | Meta-cognitive self-monitoring               |
| `perception`    | `NodeType.PERCEPTION`    | Audio/Vision input processing                |

---

## Core Reasoning Nodes

### RouterNode

- **Type:** `NodeType.ROUTER`
- **File:** `hbllm/brain/router_node.py`
- **Purpose:** Classifies user intent and selects the optimal domain expert(s). Uses the ONNX Vector Router for sub-millisecond routing decisions.

### PlannerNode

- **Type:** `NodeType.PLANNER`
- **File:** `hbllm/brain/planner_node.py`
- **Purpose:** Generates Graph-of-Thoughts (GoT) directed acyclic graphs for multi-step reasoning. Each node in the DAG represents a reasoning step.

### WorkspaceNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/workspace_node.py`
- **Purpose:** Blackboard-style consensus node. Aggregates outputs from multiple reasoning paths and resolves conflicts.

### CriticNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/critic_node.py`
- **Purpose:** Self-evaluation layer. Scores intermediate reasoning steps and provides constructive feedback.

### DecisionNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/decision_node.py`
- **Purpose:** Synthesizes the final output from workspace state, confidence scores, and critic feedback.
- **Upgrades:** Integrates a Bounded Rationality 3-tier validation path:
    1. **Level 1: Safety Gate**: Enforces risk-based filters (regular expression checks for low/medium risk, LLM-backed classifiers for high-risk tools).
    2. **Level 2: Policy Router Control Loop**: Regulates utility thresholds and routing behaviors using a 3-variable control loop (State Estimator $S_{\text{diag}}(t)$, Control Signal $S_{\text{ctrl}}(t)$, and Regulator Schmitt trigger). Includes stable-lock invariance checks to prevent threshold oscillations.
    3. **Level 3: Budget Controller**: Halves max tokens under high compute loads, monitors virtual memory usage via `psutil`, and applies a quadratic replanning penalty ($0.05 \cdot \text{depth}^2$) to suppress deep recursive routing.
- **Reference:** For a detailed breakdown, see [Decision & Control Plane](decision-gatekeeper.md).

### ProcessRewardNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/process_reward_node.py`
- **Purpose:** Provides continuous neural scoring `[0-1]` of intermediate reasoning steps, catching hallucinations before they compound.

---

## Meta-Cognitive Nodes

### LearnerNode

- **Type:** `NodeType.LEARNER`
- **File:** `hbllm/brain/learner_node.py`
- **Purpose:** Implements contrastive DPO using a persistent atomic JSON queue. Consolidates feedback into permanent LoRA weight updates during sleep cycles.

### CuriosityNode

- **Type:** `NodeType.DETECTOR`
- **File:** `hbllm/brain/curiosity_node.py`
- **Purpose:** Monitors conversation patterns for knowledge gaps and generates exploratory goals that trigger the SpawnerNode.

### SpawnerNode

- **Type:** `NodeType.SPAWNER`
- **File:** `hbllm/brain/spawner_node.py`
- **Purpose:** Artificial neurogenesis — creates new domain-specific LoRA adapters at runtime when the system encounters unfamiliar domains.

### SleepCycleNode

- **Type:** `NodeType.DOMAIN_MODULE`
- **File:** `hbllm/brain/sleep_node.py`
- **Purpose:** Multi-phase memory consolidation inspired by biological sleep cycles. Runs during idle periods.
- **Phases:** Memory Replay → Temporal Normalization → Contradiction Resolution → DPO Training → Curiosity Replay → Dream Journal
- **Triggers:** Idle timeout (auto), `system.sleep.force` (manual), REST API
- **Integration:** Queries `SelfModel` for weak domains to prioritize DPO training

### EvaluationNode

- **Type:** `NodeType.META`
- **File:** `hbllm/brain/evaluation_node.py`
- **Purpose:** Closes the intelligence feedback loop. Scores every interaction across 5 dimensions (task_success, plan_validity, tool_accuracy, memory_usage, confidence_error) and feeds results into GoalManager.
- **Micro-learning:** On negative user feedback, publishes `system.micro_learn` events so LearnerNode can queue DPO corrections in real-time.

### SkillInductionNode

- **Type:** `NodeType.META`
- **File:** `hbllm/brain/skill_induction_node.py`
- **Purpose:** Autonomous code generation for new tools. When ReflectionNode identifies a capability gap, this node prompts the LLM to generate a sandboxed Python function, validates it via AST security scanning, and registers it as a new tool.

### SchedulerNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/scheduler_node.py`
- **Purpose:** Proactive event scheduler backed by SQLite. Manages recurring and one-shot tasks with cron-style interval expressions, publishing events to the bus when they come due.
- **Supports:** `fire_and_forget` and `retry` policies for task execution.

### SelfModel

- **Type:** Utility (not a Node)
- **File:** `hbllm/brain/self_model.py`
- **Purpose:** SQLite-backed internal model of system capabilities. Tracks domain expertise levels, confidence calibration, performance trends (improving/stable/declining), and recommends model selection based on domain strength.
- **Integration:** Consulted by SleepCycleNode for targeted DPO training, by DecisionNode for model routing, and by GoalManager for self-improvement priorities.

### IdentityNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/identity_node.py`
- **Purpose:** Maintains ethical constraints, personality traits, and behavioral consistency across tenant profiles.

### WorldModelNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/world_model_node.py`
- **Purpose:** Sandboxed AST-level simulation for "what-if" reasoning. Validates code execution plans by analyzing imports and potential side effects.
- **Upgrades:** Extended to handle physical action dry-runs and simulated repository/code world modifications:
    - **AST Static Analysis**: Parses Python code to compile and check imports against `dangerous_imports` (blocking modules like `os`, `subprocess`, `sys`, `shutil`, `socket`).
    - **Heuristic & LLM Evaluation**: Uses Regex blocklists for bash command safety (blocking `rm -rf`, fork bombs, etc.) alongside optional LLM-based systems safety evaluations.
    - **Repository Mutation Simulation**: Simulates file writes and deletions in a virtual state (`_virtual_files`), parsing dependencies dynamically (`simulate_parse_imports`) and running dry-run compilations (`simulate_compilation`) to predict build success or failure before writing to disk.

### ExperienceNode

- **Type:** `NodeType.META`
- **File:** `hbllm/brain/experience_node.py`
- **Purpose:** Computes salience scores for interactions and triggers high-value experiences for consolidation during sleep.

### MetaReasoningNode

- **Type:** `NodeType.META`
- **File:** `hbllm/brain/meta_node.py`
- **Purpose:** Monitors the brain's own reasoning patterns to identify systematic biases or inefficiencies.

### WebResearchNode

- **Type:** `NodeType.META`
- **File:** `hbllm/brain/web_research_node.py`
- **Purpose:** Autonomous knowledge acquisition from the internet. Detects knowledge gaps, searches the web via BrowserNode, verifies source credibility, and ingests validated findings into episodic or semantic memory based on the 3-tier classification system (Information / Task Knowledge / Core Knowledge).

### CollectiveNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/collective_node.py`
- **Purpose:** Multi-agent coordination and consensus building.

### SentinelNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/sentinel_node.py`
- **Purpose:** Proactively scans async bus traffic for policy violations and governance constraints.

### RuleExtractorNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/rule_extractor.py`
- **Purpose:** Mines high-salience interactions for recurring *if→then* preferences, auto-promoting them to behavioral guardrails.

### RevisionNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/revision_node.py`
- **Purpose:** Manages a self-critique loop that iteratively refines the response based on confidence scores and critic feedback.

### HardwareHAL

- **File:** `hbllm/modules/hardware_hal.py`
- **Purpose:** Autonomous system introspection. Benchmarks disk latency, CPU threads, and VRAM bandwidth to recommend the optimal quantization policy (INT4 vs INT8) for the current device.

---

## Edge & Gateway Nodes

To support the hierarchical distributed architecture, HBLLM uses specialized gateway nodes to bridge the central MessageBus to remote edges securely.

### SynapseGateway

- **Type:** `NodeType.CORE`
- **File:** `hbllm/serving/synapse_gateway.py`
- **Purpose:** Centralized WebSocket hub that acts as the ingress point for Edge Nodes. It authenticates connections, subscribes to internal `edge.*` topics, and multiplexes JSON/msgpack traffic between the core brain and remote clients.
- **Security:** Actively listens to `system.security.revocation` events and instantly terminates WebSockets matching compromised edge IDs.

### UplinkNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/network/uplink_node.py`
- **Purpose:** Client-side proxy running on remote devices (like a laptop or IoT peripheral). It establishes a persistent WebSocket connection to the `SynapseGateway`, authenticates using its Ed25519 keys, and tunnels local tool invocations seamlessly to the upstream brain.

---

## Perception Nodes

### VisionNode

- **Type:** `NodeType.PERCEPTION`
- **File:** `hbllm/perception/vision_node.py`
- **Purpose:** Multimodal vision processing — image captioning, OCR text extraction, and dense embedding extraction.
- **Rust Acceleration:** Attempts to load `hbllm_perception_rs` (Rust-native ONNX engine) first, falling back to PyTorch `transformers.pipeline` if unavailable. The Rust path eliminates the `torch` dependency from the hot path and reduces inference latency by ~3–5×.
- **Frame Caching:** Integrates a `ChangeDetector` (perceptual hashing) to skip expensive inference on static/near-identical frames. Results are cached per entity/session key.
- **Embedding Extraction:** Produces 768-dim vision embeddings via `_embed_image()`, which are then projected to the LLM latent space (4096-dim) by `MultimodalProjector` for workspace thought publishing.
- **Topics:** `vision.process`, `vision.ocr`, `vision.caption`, `module.evaluate`

### AudioInputNode

- **Type:** `NodeType.PERCEPTION`
- **File:** `hbllm/perception/audio_in_node.py`
- **Purpose:** Speech-to-text streaming transcription.
- **Upgrades:** Implements high-fidelity cloud routing with robust local fallbacks:
    - **NVIDIA Cloud Whisper API**: When `NVIDIA_API_KEY` is active, routes requests to the NVIDIA Cloud Whisper API using the `openai/whisper-large-v3` model.
    - **Local Fallback**: Automatically falls back to a thread-safe local Whisper model (`whisper.load_model(model_size)`) if cloud API calls fail or timeout.
    - **Streaming Buffering**: Accumulates PCM chunks from `sensory.audio.stream` messages and flushes them when silence timeout is reached or final chunk is received.

### AudioOutputNode

- **Type:** `NodeType.PERCEPTION`
- **File:** `hbllm/perception/audio_out_node.py`
- **Purpose:** Text-to-speech with per-tenant voice configurations.
- **Upgrades:** Integrates dual-path cloud and local synthesis engines:
    - **NVIDIA Riva TTS Client**: Interfaces with the `riva.client` gRPC library for cloud or local Riva/NIM text-to-speech synthesis (using `Magpie-Multilingual.EN-US.Aria` by default).
    - **Local SpeechT5 Fallback**: If `riva.client` is missing or gRPC synthesis fails, falls back to local PyTorch `SpeechT5` (`microsoft/speecht5_tts` and HifiGan vocoder) with tenant-specific custom speaker xvectors.
    - **Text Chunking**: Sentence-splits long text (>450 chars) to maintain synthesize voice consistency and quality.

### Perception Infrastructure

#### RealityEventBus

- **File:** `hbllm/perception/reality_bus.py`
- **Purpose:** Unified ingestion pipeline for physical and digital reality events. Assigns logical clocks, ingest timestamps, and routes `PerceptionEvent`s to subscribers.
- **Pre-subscribers:** Synchronous `subscribe_pre()` hooks fire *before* async fan-out — used by `ReflexArc` for sub-millisecond fast-path routing.
- **Modality tiers:** `SYSTEM` (high trust), `APP` (throttled), `SENSOR` (sampled), `INFERRED` (rate-limited).

#### EventNormalizer

- **File:** `hbllm/perception/normalizer.py`
- **Purpose:** Noise filtering, deduplication, and budget enforcement between the raw `RealityEventBus` and the `WorldStateEngine`.
- **Embedding-aware dedup:** When events carry embeddings, cosine similarity is used alongside signature matching — semantically different events with the same type/subtype are preserved.

#### MultimodalProjector

- **File:** `hbllm/perception/vector_projector.py`
- **Purpose:** Projects modality-specific embeddings (vision 768-dim, audio, sensor readings) into the shared LLM latent space (default 4096-dim).
- **Weight loading:** Supports `.safetensors` trained projection weights with identity + zero-padding fallback.

#### ReflexArc

- **File:** `hbllm/perception/reflex_arc.py`
- **Purpose:** Sub-millisecond autonomic fast-path that routes critical sensor alerts (fire, collision, anomaly) directly to action executors, bypassing the Global Workspace / LLM reasoning loop entirely.
- **Rules:** Supports pattern-matching `ReflexRule`s and event-accumulating `SpikingReflexRule`s. Spiking rules leverage a Leaky Integrate-and-Fire (LIF) neuron model to trigger actions only when cumulative input currents from high-frequency events exceed spiking thresholds.
- **Auditability:** Every activation is logged to `EventLog`, including the recorded `spike_strength` indicating priority/urgency.

#### Rust Perception Engine

- **Crate:** `rust/perception/` (`hbllm_perception_rs`)
- **Build:** `cd rust/perception && maturin develop --release`
- **Exports:** `VisionEngine` (ONNX model loading, embedding, captioning, frame hashing) and `ChangeDetector` (perceptual hash change detection with configurable Hamming distance threshold).
- **Dependencies:** `ort` (ONNX Runtime), `image`, `ndarray`, `pyo3`


## Action Nodes

### ExecutionNode

- **File:** `hbllm/actions/execution_node.py`
- **Purpose:** Sandboxed Python code execution with strict compute and memory bounds.
- **Security:** Blocks dangerous imports (`os`, `subprocess`, `shutil`).

### McpClientNode

- **File:** `hbllm/actions/mcp_client_node.py`
- **Purpose:** Model Context Protocol client for external tool integration.

### BrowserNode

- **File:** `hbllm/actions/browser_node.py`
- **Purpose:** Web page interaction and scraping.

### LogicNode

- **File:** `hbllm/actions/logic_node.py`
- **Purpose:** Formal verification and constraint solving via Z3.

### FuzzyNode

- **File:** `hbllm/actions/fuzzy_node.py`
- **Purpose:** Approximate reasoning with scikit-fuzzy.

### MqttIoTNode

- **File:** `hbllm/actions/iot_mqtt_node.py`
- **Purpose:** MQTT-based IoT device control and sensor data ingestion.
- **Dependencies:** `paho-mqtt`

### Ros2Node

- **File:** `hbllm/actions/ros2_node.py`
- **Purpose:** ROS2 robotics integration for perception, navigation, and motor control.
- **Dependencies:** `rclpy` (ROS2 Python client)

### ToolRouterNode

- **File:** `hbllm/actions/tool_router.py`
- **Purpose:** Routes tool-use requests to the appropriate action node.

### ApiNode

- **File:** `hbllm/actions/api_node.py`
- **Purpose:** External HTTP API calls.

---

## Creating Custom Nodes

See the [Custom Nodes Guide](../guides/custom-nodes.md) for step-by-step instructions.
