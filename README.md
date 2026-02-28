<div align="center">
  <h1>🧠 HBLLM Core</h1>
  <p><b>Human-Brain Inspired Cognitive Architecture</b></p>
  <p><em>An open-source AGI framework that thinks, learns, and adapts — not just responds.</em></p>

  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
  [![Rust](https://img.shields.io/badge/Rust-Accelerated-orange.svg)](https://www.rust-lang.org/)
  [![Tests](https://img.shields.io/badge/Tests-390%2B%20passing-brightgreen.svg)](#)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

<br/>

## What Makes This Different

**Standard LLMs** are monolithic transformers: prompt → model → response. One path, one perspective, stateless.

**HBLLM Core** is a **modular cognitive architecture** with 22 specialized brain nodes that communicate over an asynchronous message bus — like a real brain:

```
                        ┌─────────────────────────────────────────┐
    Input ───────────►  │              HBLLM Core Brain           │
    (text, vision,      │                                         │
     audio, sensors)    │   Router ──► Planner ──► Decision       │
                        │     │          │            │           │
                        │   Memory    Learner      Critic        │
                        │   (5 types)    │            │           │
                        │              World       Identity      │
                        │              Model       (ethics)      │
                        │                │                        │
                        │           Curiosity ──► Spawner        │
                        │           (explores)    (creates new   │
                        │                          specialists)  │
                        └───────────────────────────┬─────────────┘
                                                    │
    Output ◄────────────────────────────────────────┘
    (actions, speech, motor control, API calls)
```

## Architecture

### 🧠 Brain Nodes (13 cognitive modules)

| Node               | Role                                       | Analog               |
| ------------------ | ------------------------------------------ | -------------------- |
| **Router**         | Routes inputs to the right cognitive path  | Thalamus             |
| **Planner**        | Breaks goals into multi-step plans         | Prefrontal cortex    |
| **Decision**       | Makes final decisions from evidence        | Executive function   |
| **Critic**         | Self-evaluates quality and correctness     | Error monitoring     |
| **Learner**        | Updates knowledge from outcomes            | Hippocampal learning |
| **Curiosity**      | Explores novel situations proactively      | Dopaminergic system  |
| **World Model**    | Builds internal model of the environment   | Predictive coding    |
| **Identity**       | Maintains values, ethics, and personality  | Self-model           |
| **Meta Reasoning** | Reasons about its own reasoning            | Metacognition        |
| **Workspace**      | Shared cognitive workspace for integration | Global workspace     |
| **Collective**     | Ensemble reasoning from multiple nodes     | Neural ensemble      |
| **Sleep Cycle**    | Consolidates learning during idle time     | Memory consolidation |
| **Spawner**        | Dynamically creates specialist sub-agents  | Neurogenesis         |

### 👁️ Perception (3 input channels)

| Node             | Capability                                |
| ---------------- | ----------------------------------------- |
| **Vision**       | Image understanding and visual processing |
| **Audio Input**  | Speech recognition and sound analysis     |
| **Audio Output** | Speech synthesis and audio generation     |

### 🧬 Memory Systems (5 types — like human memory)

| Memory         | What It Stores            | Example                                     |
| -------------- | ------------------------- | ------------------------------------------- |
| **Episodic**   | Events and experiences    | "User came home at 6:30pm on Tuesday"       |
| **Semantic**   | Facts and knowledge       | "Living room temperature preference = 23°C" |
| **Procedural** | Skills and how-to         | "To make coffee: fill water → grind → brew" |
| **Value**      | Preferences and judgments | "User prefers warm lighting over cool"      |
| **Working**    | Current task context      | Active conversation state                   |

### ⚡ Action Nodes (6 output channels)

| Node           | Capability                           |
| -------------- | ------------------------------------ |
| **Execution**  | Run tasks and commands               |
| **API**        | Call external APIs and services      |
| **Browser**    | Web interaction and scraping         |
| **Logic**      | Formal logical reasoning (Z3 solver) |
| **Fuzzy**      | Probabilistic/fuzzy reasoning        |
| **MCP Client** | Model Context Protocol integration   |

### 🔌 Infrastructure

| Component            | Purpose                                       |
| -------------------- | --------------------------------------------- |
| **MessageBus**       | Async pub/sub communication between all nodes |
| **Service Registry** | Dynamic node discovery and routing            |
| **Circuit Breaker**  | Fault tolerance and graceful degradation      |
| **Load Balancer**    | Distribute work across node replicas          |
| **Policy Engine**    | YAML-based governance rules                   |
| **Tracing**          | Full observability of cognitive processing    |

---

## Use Cases

### 🏠 Smart Home & Home Automation

HBLLM Core can power truly intelligent home systems that **learn and adapt** — not just follow rules:

```python
from hbllm.network.bus import InProcessBus
from hbllm.brain.router_node import RouterNode
from hbllm.brain.planner_node import PlannerNode
from hbllm.brain.decision_node import DecisionNode
from hbllm.brain.learner_node import LearnerNode
from hbllm.brain.world_model_node import WorldModelNode
from hbllm.memory.memory_node import MemoryNode

# The brain learns your patterns over time:
#
# Week 1:  "Turn on lights" → turns on lights
# Week 4:  Notices you always dim lights at 9pm → does it automatically  
# Month 2: Learns your wake-up routine → starts coffee before alarm
# Month 6: Predicts energy usage → optimizes heating/cooling schedule
```

**What makes it different from Google Home / Alexa:**

| Feature    | Alexa/Google            | HBLLM Core                              |
| ---------- | ----------------------- | --------------------------------------- |
| Learning   | Pre-programmed routines | Learns from observation                 |
| Memory     | Stateless commands      | Episodic + semantic + procedural memory |
| Planning   | Single-step actions     | Multi-step plans (Planner node)         |
| Adaptation | Manual rule updates     | Self-improving (Learner node)           |
| Privacy    | Cloud-dependent         | Runs 100% locally                       |
| Reasoning  | Pattern matching        | Logical + fuzzy + world model           |

### 🤖 Robotics

HBLLM Core provides the cognitive layer for autonomous robots:

```
Sensors ──► Perception Nodes ──► Router ──► Planner ──► Decision ──► Motors
  │                                │                        │
  │                          World Model              Critic
  │                        (understands               (checks
  │                         physics,                   safety)
  │                         obstacles)                   │
  └────── Memory ◄──────── Learner ◄────────────────────┘
         (remembers         (improves
          what worked)       from mistakes)
```

**Capabilities:**
- **Path planning** — Planner node breaks navigation into steps
- **Object manipulation** — World Model understands physical constraints  
- **Failure recovery** — Critic detects errors, Learner adapts
- **Task learning** — Procedural memory stores learned skills
- **Human interaction** — Audio + Vision perception for natural communication
- **Safety** — Policy engine enforces hard safety constraints

### 🏭 Industrial Automation

| Application                  | How HBLLM Core Helps                                     |
| ---------------------------- | -------------------------------------------------------- |
| **Predictive maintenance**   | World Model learns equipment patterns, predicts failures |
| **Quality control**          | Vision node + Critic node detect defects and anomalies   |
| **Process optimization**     | Learner node continuously improves production parameters |
| **Multi-robot coordination** | Message bus enables distributed swarm intelligence       |

### 🧪 Research & General AI

- **Cognitive science** — Experiment with different brain architectures
- **Reinforcement learning** — Built-in reward/feedback loops
- **Multi-agent systems** — Spawner creates specialized sub-agents
- **Embodied AI** — Connect perception and action for physical agents

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Run the API Server

```bash
# Full brain mode (requires model weights)
python -m hbllm.serving.api

# Provider mode (uses OpenAI/Anthropic as backend)
HBLLM_PROVIDER=openai OPENAI_API_KEY=sk-... python -m hbllm.serving.api
```

### Python API

```python
import asyncio
from hbllm.network.bus import InProcessBus
from hbllm.brain.router_node import RouterNode
from hbllm.brain.decision_node import DecisionNode
from hbllm.memory.memory_node import MemoryNode
from hbllm.network.messages import Message, MessageType

async def main():
    bus = InProcessBus()
    await bus.start()

    # Start cognitive nodes
    memory = MemoryNode(node_id="memory_01", db_path="brain.db")
    router = RouterNode(node_id="router_01")
    decision = DecisionNode(node_id="decision_01")

    for node in [memory, router, decision]:
        await node.start(bus)

    # Send a message through the cognitive pipeline
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="router.query",
        payload={"text": "What's the optimal temperature for my living room?"},
    )
    await bus.publish("router.query", msg)

asyncio.run(main())
```

---

## Project Structure

```
hbllm-core/
├── hbllm/                    # Python cognitive architecture
│   ├── brain/                # 13 cognitive nodes
│   │   ├── router_node.py    #   Input routing (thalamus)
│   │   ├── planner_node.py   #   Multi-step planning
│   │   ├── decision_node.py  #   Final decision making
│   │   ├── critic_node.py    #   Self-evaluation
│   │   ├── learner_node.py   #   Learning from outcomes
│   │   ├── curiosity_node.py #   Exploration drive
│   │   ├── world_model_node.py # Environment modeling
│   │   ├── identity_node.py  #   Values and ethics
│   │   ├── meta_node.py      #   Meta-reasoning
│   │   ├── workspace_node.py #   Cognitive workspace
│   │   ├── collective_node.py#   Ensemble reasoning
│   │   ├── sleep_node.py     #   Memory consolidation
│   │   ├── spawner_node.py   #   Dynamic agent creation
│   │   ├── policy_engine.py  #   Governance rules
│   │   └── llm_interface.py  #   Model abstraction
│   ├── memory/               # 5 memory systems
│   │   ├── episodic.py       #   Event memory
│   │   ├── semantic.py       #   Fact memory
│   │   ├── procedural.py     #   Skill memory
│   │   └── value_memory.py   #   Preference memory
│   ├── perception/           # Input channels
│   │   ├── vision_node.py
│   │   ├── audio_in_node.py
│   │   └── audio_out_node.py
│   ├── actions/              # Output channels
│   │   ├── execution_node.py
│   │   ├── api_node.py
│   │   ├── browser_node.py
│   │   ├── logic_node.py
│   │   ├── fuzzy_node.py
│   │   └── mcp_client_node.py
│   ├── network/              # Communication infrastructure
│   │   ├── bus.py            #   Message bus (pub/sub)
│   │   ├── node.py           #   Base node abstraction
│   │   ├── registry.py       #   Service discovery
│   │   ├── circuit_breaker.py#   Fault tolerance
│   │   └── ...
│   ├── model/                # Transformer model
│   ├── training/             # SFT, DPO, evaluation
│   └── serving/              # FastAPI server
├── rust/                     # Rust accelerators
│   ├── tokenizer/            #   High-performance tokenizer
│   └── data_tools/           #   Data cleaning & dedup
├── tests/                    # 390+ tests
└── pyproject.toml
```

---

## Extending for Your Use Case

### Adding a Custom Node

```python
from hbllm.network.node import Node, NodeType

class TemperatureSensorNode(Node):
    """Custom perception node for IoT temperature sensors."""

    def __init__(self, node_id: str, mqtt_topic: str):
        super().__init__(node_id, NodeType.DETECTOR, capabilities=["temperature"])
        self.mqtt_topic = mqtt_topic

    async def on_start(self):
        await self.bus.subscribe(self.mqtt_topic, self.handle_message)

    async def on_stop(self):
        pass

    async def handle_message(self, message):
        temp = message.payload.get("temperature")
        # Publish to the cognitive pipeline
        await self.publish("perception.temperature", Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            payload={"temperature": temp, "unit": "celsius"},
        ))
        return None
```

### Connecting to Hardware (Raspberry Pi / Jetson)

```python
# HBLLM runs on any Python 3.11+ system
# No GPU required in provider mode

# On Raspberry Pi:
pip install -e .
HBLLM_PROVIDER=openai python -m hbllm.serving.api --host 0.0.0.0 --port 8000
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where we need help:
- 🤖 **IoT/Robotics nodes** — MQTT, Zigbee, ROS2 integrations
- 🧠 **New cognitive nodes** — Emotion modeling, spatial reasoning
- 📊 **Benchmarks** — Comparing cognitive architecture vs monolithic LLMs
- 📱 **Edge optimization** — Running efficiently on Raspberry Pi / Jetson

## License

MIT License — free to use in personal, commercial, and research projects.

---

<div align="center">
  <p><b>HBLLM Core</b> — AI that thinks, not just responds.</p>
  <p>⭐ Star this repo if you believe AI should be more than a chatbot.</p>
</div>
