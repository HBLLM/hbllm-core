---
title: "Cognitive Nodes — HBLLM Brain Components"
description: "Reference documentation for all 25+ cognitive nodes in the HBLLM architecture, including Router, Planner, Critic, Learner, and Spawner."
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

## Node Types

The `NodeType` enum defines the categories of nodes:

| Value | Constant | Purpose |
|---|---|---|
| `router` | `NodeType.ROUTER` | Intent classification and domain routing |
| `core` | `NodeType.CORE` | Core reasoning (Critic, Decision, Workspace) |
| `domain_module` | `NodeType.DOMAIN_MODULE` | Domain-specific LoRA modules |
| `memory` | `NodeType.MEMORY` | Memory storage and retrieval |
| `planner` | `NodeType.PLANNER` | Task decomposition and GoT planning |
| `learner` | `NodeType.LEARNER` | Continuous learning and DPO training |
| `detector` | `NodeType.DETECTOR` | Perception and sensor input |
| `spawner` | `NodeType.SPAWNER` | Neurogenesis — creating new domain modules |
| `meta` | `NodeType.META` | Meta-cognitive self-monitoring |
| `perception` | `NodeType.PERCEPTION` | Audio/Vision input processing |

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

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/sleep_node.py`
- **Purpose:** 3-phase memory consolidation (Replay → Prune → Strengthen) inspired by biological sleep cycles. Runs during idle periods.

### IdentityNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/identity_node.py`
- **Purpose:** Maintains ethical constraints, personality traits, and behavioral consistency across tenant profiles.

### WorldModelNode

- **Type:** `NodeType.CORE`
- **File:** `hbllm/brain/world_model_node.py`
- **Purpose:** Sandboxed AST-level simulation for "what-if" reasoning. Validates code execution plans by analyzing imports and potential side effects.

### ExperienceNode

- **Type:** `NodeType.META`
- **File:** `hbllm/brain/experience_node.py`
- **Purpose:** Computes salience scores for interactions and triggers high-value experiences for consolidation during sleep.

### MetaReasoningNode

- **Type:** `NodeType.META`
- **File:** `hbllm/brain/meta_node.py`
- **Purpose:** Monitors the brain's own reasoning patterns to identify systematic biases or inefficiencies.

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

---

## Perception Nodes

### VisionNode

- **Type:** `NodeType.PERCEPTION`
- **File:** `hbllm/perception/vision_node.py`
- **Purpose:** Image captioning, OCR text extraction, and object detection.

### AudioInputNode

- **Type:** `NodeType.PERCEPTION`
- **File:** `hbllm/perception/audio_in_node.py`
- **Purpose:** Speech-to-text streaming transcription.

### AudioOutputNode

- **Type:** `NodeType.PERCEPTION`
- **File:** `hbllm/perception/audio_out_node.py`
- **Purpose:** Text-to-speech with per-tenant voice configurations.

---

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
