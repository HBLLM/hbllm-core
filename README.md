<!-- SEO Keywords: Sovereign Personal AI, Open Source AGI, Cognitive Architecture, Large Language Models, Multi-Agent Systems, Edge AI, Hybrid Quantization, LoRA Tuning, Privacy-First AI, On-Premise AI, Python 3.11 AI Framework, Autonomous Agents, Multi-Tenant AI, Human Modeling, Spiking Neural Networks -->

<div align="center">
  <h1>🧠 HBLLM Core</h1>
  <h3>An AI That Thinks Like a Person</h3>
  <p><b>A continuously thinking, goal-driven, memory-forming cognitive brain that models <em>you</em> — runs entirely on your own hardware, no cloud required.</b></p>
  <p><em>Local by default. Distributed when you want it. It learns who you are, what you're building, and who matters to you — your data never leaves your device unless you choose.</em></p>

  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
  [![Rust](https://img.shields.io/badge/Rust-Accelerated-orange.svg)](https://www.rust-lang.org/)
  [![Tests](https://img.shields.io/badge/Tests-4200%2B%20passing-brightgreen.svg)](#)
  [![Files](https://img.shields.io/badge/Source-421%20modules-purple.svg)](#)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.md)
</div>

<br/>

> [!NOTE]  
> **Quick Links:**
> - 📐 **[Architecture Overview](docs/architecture/overview.md)** — Full layered design with 50+ cognitive nodes
> - 👤 **[Cognitive Subsystems](docs/architecture/cognitive-subsystems.md)** — UserModel, ProjectGraph, ExecutiveCortex, RelationshipMemory, RealityGraph
> - 🛡️ **[Security Architecture](SECURITY.md)** — Identity Triplet, Tenant Guard, Audit Log, Encryption at Rest
> - ⚡ **[Benchmarks](docs/api/benchmarks.md)** — Memory Profiling, Fast-Path Latency

---

## What Makes HBLLM Different?

Most AI systems answer questions when you ask them. HBLLM Core does something fundamentally different — **it thinks all the time, entirely on your own machine, and it models you**.

### It Remembers

Nine memory subsystems — working, episodic, semantic, procedural, knowledge graph, value, spatial, temporal, and importance-scored — give it genuine recall. It knows what happened yesterday, what it learned last month, and what you care about most.

### It Models You

The **Human Modeling Layer** continuously learns who you are from every interaction:

- **UserModel** — Your expertise domains, communication preferences, beliefs, trust levels, and temporal work patterns
- **ProjectGraph** — What you're building: goals, blockers, open questions, decisions, milestones — auto-detected from conversation
- **ExecutiveCortex** — Cognitive focus management: what to prioritize, when to interrupt, how to allocate compute
- **RelationshipMemory** — Your social graph: who matters, their roles, sentiment trends, interaction history
- **RealityGraph** — Unified world state: merges sensor data, knowledge graph, and cognitive models into one view

### It Pursues Goals

The executive control system breaks high-level objectives into persistent DAG tasks, retries failures, verifies real-world outcomes, and picks up exactly where it left off after a reboot.

### It Scales Across Your Devices

Phone, laptop, and home server share knowledge via `SynapseGateway` with Ed25519 cryptographic trust — zero cloud required.

### It Knows When to Stop

A policy engine blocks harmful actions, every decision is audited, and the system slows itself down when overloaded.

```text
                 ┌───────────────────────────────────────────────────┐
 Input ────────► │                 HBLLM Core Brain                  │
 (text, vision,  │                                                   │
  audio, IoT)    │   Perception ──► Router ──► Planner ──► Critic    │
                 │       │            │            │          │       │
                 │  WorldState     Memory      Workspace  Decision   │
                 │  Tracker      (9 types)    (blackboard)   │       │
                 │       │                        │          │       │
                 │  SNN Stream                 Identity    Policy    │
                 │  (Comprehension              Ethics    Engine    │
                 │   + Expression)                │          │       │
                 │       │                        │          │       │
                 │  Human Modeling Layer ──────────────── Actions    │
                 │  (UserModel, ProjectGraph,              │       │
                 │   ExecutiveCortex, Social,    OS Adapter ─┘       │
                 │   RealityGraph)                                   │
                 └──────────────────────────────────┬────────────────┘
                                                    │
 Action / Output ◄──────────────────────────────────┘
```

> 📖 **[Architecture →](docs/architecture/overview.md)** · **[Cognitive Nodes →](docs/architecture/cognitive-nodes.md)** · **[Memory Systems →](docs/architecture/memory-systems.md)** · **[Cognitive Subsystems →](docs/architecture/cognitive-subsystems.md)**

---

## Key Capabilities

| Category | What it does | Docs |
|----------|-------------|------|
| **👤 Human Modeling** | Learns expertise, preferences, trust, and habits from every interaction — the system models *you* | [Cognitive Subsystems](docs/architecture/cognitive-subsystems.md) |
| **📋 Project Awareness** | Tracks goals, blockers, decisions, and milestones per project — auto-detects what you're working on | [Cognitive Subsystems](docs/architecture/cognitive-subsystems.md) |
| **🧠 Always-On Cognition** | Stays awake between queries — notices events, forms thoughts, and acts proactively | [Executive Brain](docs/architecture/executive-brain-layer.md) |
| **💾 9 Memory Subsystems** | Working, Episodic, Semantic, Procedural, Value, Knowledge Graph, Spatial, Temporal, Importance | [Memory Systems](docs/architecture/memory-systems.md) |
| **🎯 Goal Pursuit** | Decomposes objectives into persistent DAG tasks, retries failures, verifies outcomes | [Executive Brain](docs/architecture/executive-brain-layer.md) |
| **🤝 Social Intelligence** | Relationship memory with sentiment tracking, interaction history, and notification priority | [Cognitive Subsystems](docs/architecture/cognitive-subsystems.md) |
| **🌍 Unified World State** | RealityGraph merges KnowledgeGraph, BrainWorldState, and sensor data into one view | [Cognitive Subsystems](docs/architecture/cognitive-subsystems.md) |
| **⚙️ Fully Local** | Runs on CPU-only machines — 125M model needs ~500MB RAM, 1.5B fits in 4GB with INT4 | [Benchmarks](docs/api/benchmarks.md) |
| **⚡ SNN Cognitive Stream** | Spiking Neural Networks for concept extraction, content planning, and reward evaluation | [Cognitive Nodes](docs/architecture/cognitive-nodes.md) |
| **🧪 Self-Personalizing** | SpawnerNode auto-creates 2MB LoRA adapters — the brain grows new specialist regions at runtime | [Zoning](docs/zoning/how-it-works.md) |
| **🌐 Distributed** | Spans your devices via Ed25519 cryptographic trust — zero cloud required | [Adaptive Network](docs/architecture/adaptive-network.md) |
| **🛑 Safety & Control** | Policy engine, restraint engine, PII redaction, voice auth, rollback, audit trail | [Human Control](docs/architecture/human-control.md) |
| **🔌 Plugin SDK** | Declarative `@subscribe` plugins with auto-binding — extend any cognitive loop | [Plugin Guide](docs/guides/plugins.md) |

---

## Architecture at a Glance

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          HBLLM Core (421 files)                     │
│                                                                     │
│  ┌──────────┐  ┌─────────────────┐  ┌───────────────────────────┐  │
│  │Perception│  │   Message Bus   │  │     Security & Governance │  │
│  │ (23 files│  │   (43 files)    │  │       (14 files)          │  │
│  │ STT,TTS, │  │ InProcess/Redis │  │ PII, VoiceAuth, Policy,  │  │
│  │ Vision,  │  │ Trust,Registry  │  │ Audit, Tenant Guard      │  │
│  │ Gesture) │  │                 │  │                           │  │
│  └────┬─────┘  └───────┬─────┬──┘  └───────────────────────────┘  │
│       │                │     │                                      │
│       ▼                ▼     │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Brain Layer (166 files)                   │   │
│  │                                                             │   │
│  │  Cognitive Loop    SNN Stream       Autonomy & Executive    │   │
│  │  ├─ Router         ├─ Comprehension ├─ AutonomyCore         │   │
│  │  ├─ Planner        ├─ Expression    ├─ StateMachine         │   │
│  │  ├─ Workspace      ├─ TrainedPRM    ├─ TaskGraph            │   │
│  │  ├─ Critic         └─ ContentPlan   ├─ GoalDecomposition    │   │
│  │  └─ Decision                        ├─ ReflexLibrary        │   │
│  │                                     └─ RestraintEngine      │   │
│  │  Human Modeling          Meta-Cognitive                     │   │
│  │  ├─ UserModel            ├─ Learner (DPO)                  │   │
│  │  ├─ ProjectGraph         ├─ Curiosity                      │   │
│  │  ├─ ExecutiveCortex      ├─ Spawner (neurogenesis)         │   │
│  │  ├─ RelationshipMemory   ├─ Sleep Cycle                    │   │
│  │  └─ RealityGraph         └─ Identity & Ethics              │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌────────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────────┐  │
│  │  Memory    │  │ Actions  │  │ Serving │  │  Model/Training  │  │
│  │ (18 files) │  │(20 files)│  │(39 files│  │   (27 files)     │  │
│  │ 9 types   │  │ Sandbox, │  │ HTTP,WS,│  │ Transformer,     │  │
│  │ + search  │  │ IoT,MCP, │  │ Studio, │  │ Quant, DPO,      │  │
│  │ + compact │  │ Browser  │  │ SSE     │  │ Tokenizer        │  │
│  └───────────┘  └──────────┘  └─────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Platform-Agnostic Design

HBLLM Core is a **fully platform-independent cognitive library** with zero dependencies on any frontend, desktop, or server framework.

| Layer | Responsibility | Location |
|-------|---------------|----------|
| **Core** | Cognitive nodes, memory, bus, agent executor, plugins | `hbllm/` (421 files) |
| **Platform Bridge** | UI, config, persistence, platform-specific tools | External (e.g. [Sentra](../sentra/)) |
| **Plugins** | Modular cognitive extensions | `hbllm/plugins/` |

```python
# Extend the core with platform-specific tools
from hbllm.actions.agent_executor import AgentExecutor

class MyPlatformExecutor(AgentExecutor):
    def _register_platform_tools(self):
        self.tools.register("my_tool", ..., my_tool_fn, {})
```

> 📖 **[Plugin Development →](docs/guides/plugins.md)** · **[Custom Nodes →](docs/guides/custom-nodes.md)**

---

## Hardware Requirements

| Target | Model | RAM | GPU? |
|--------|-------|-----|------|
| **Raspberry Pi 5** (8GB) | 125M | ~1GB | ❌ |
| **Laptop** (no GPU) | 125M–500M | ~2–4GB | ❌ |
| **Desktop** (16GB) | 1.5B INT4 | ~4GB | Optional |
| **Desktop + GPU** (6GB VRAM) | 1.5B FP16 | ~6GB | ✅ Faster |
| **Cloud / API Mode** | Any (OpenAI/Anthropic/Ollama) | ~200MB | ❌ |

> Rust SIMD kernels (AVX2 on x86, NEON on ARM) accelerate INT4/INT8 quantized inference on CPU. **No CUDA required.**

---

## Quick Start

```bash
git clone https://github.com/hbllm/hbllm-core.git
cd HBLLM/core
pip install -e .

# Optional integrations
pip install paho-mqtt        # IoT / MQTT Home Automation
export HBLLM_ROS2_ENABLED=1  # ROS2 Robotics (requires rclpy)
```

```python
import asyncio
from hbllm.brain.factory import BrainFactory, BrainConfig

async def main():
    # Cloud-backed (easy start)
    brain = await BrainFactory.create("openai/gpt-4o")

    # Or fully local — no API keys needed
    # brain = await BrainFactory.create_local("./checkpoints/sft/my_domain")

    # Customize what's enabled
    # config = BrainConfig(
    #     inject_user_model=True,          # Learn who the user is
    #     inject_project_graph=True,       # Track project state
    #     inject_executive_cortex=True,    # Focus management
    #     inject_relationship_memory=True, # Social graph
    #     inject_reality_graph=True,       # Unified world state
    #     inject_perception=False,         # Skip heavy ML models
    # )
    # brain = await BrainFactory.create("openai/gpt-4o", config=config)

    result = await brain.process(
        "Analyze our server logs and design a firewall rule.",
        tenant_id="tenant-001",
    )
    print(f"Response: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Stages: {result.stages_completed}")
    print(f"Latency: {result.latency_ms:.0f}ms")

    await brain.shutdown()

asyncio.run(main())
```

### CLI

```bash
hbllm info                         # View active brain architecture
hbllm nodes                        # List all loaded cognitive nodes
hbllm serve --port 8000            # Start the FastAPI + MCP Server
hbllm plugin list                  # List installed plugins
hbllm plugin new my-plugin         # Scaffold a new custom plugin
hbllm train --model-size 125m      # Start local pre-training loop
hbllm data --dataset fineweb       # Run data preparation pipeline
```

> 📖 **[Quickstart Guide →](docs/guides/quickstart.md)** · **[Configuration →](docs/guides/configuration.md)** · **[Deployment →](docs/guides/deployment.md)**

---

## Project Stats

| Metric | Value |
|--------|-------|
| Source modules | **421** Python files across 16+ packages |
| Brain layer | **166** files — 16 cognitive subsystems |
| Test suite | **4200+** tests (unit, integration, e2e) |
| Cognitive nodes | **50+** async nodes on the message bus |
| Memory types | **9** (Working, Episodic, Semantic, Procedural, KG, Value, Spatial, Temporal, Importance) |
| Human modeling | **5** subsystems (UserModel, ProjectGraph, ExecutiveCortex, RelationshipMemory, RealityGraph) |
| SNN networks | **5** (Comprehension, Association, Reasoning, Expression, TrainedPRM) |
| Rust crates | **4** (compute, semantic_search, knowledge_graph, perception) |
| LoRA adapter size | **~2MB** each |

---

## Documentation

| Section | Contents |
|---------|----------|
| **[Architecture](docs/architecture/)** | [Overview](docs/architecture/overview.md), [Cognitive Nodes](docs/architecture/cognitive-nodes.md), [Memory](docs/architecture/memory-systems.md), [Message Bus](docs/architecture/message-bus.md), [Sleep Cycle](docs/architecture/sleep-cycle.md), [Executive Brain](docs/architecture/executive-brain-layer.md), [Cognitive Subsystems](docs/architecture/cognitive-subsystems.md) |
| **[Zoning](docs/zoning/)** | [LoRA Routing](docs/zoning/lora-routing.md), [Weighted Domains](docs/zoning/weighted-domains.md), [Hybrid Quantization](docs/zoning/hybrid-quantization.md) |
| **[Guides](docs/guides/)** | [Quickstart](docs/guides/quickstart.md), [Custom Nodes](docs/guides/custom-nodes.md), [Plugins](docs/guides/plugins.md), [Training](docs/guides/training.md), [Deployment](docs/guides/deployment.md), [IoT/Robotics](docs/guides/robotics-iot.md) |
| **[API Reference](docs/api/)** | [Brain Factory](docs/api/brain-factory.md), [Brain Subsystems](docs/api/brain-subsystems.md), [Network](docs/api/network.md), [Model](docs/api/model.md), [Rust Kernels](docs/api/rust-kernels.md) |

---

## Contributing

We welcome contributions! Key areas:

- 👤 **Human Modeling** — Improve UserModel learning, add new trust dimensions, project graph features
- 🧠 **Cognitive Plugins** — Extend emotion, temporal, swarm or build new ones
- 📱 **Edge Devices** — Optimization for Raspberry Pi 5 & Jetson Orin Nano
- 🌐 **Starter Zones** — Pre-trained LoRAs for Medicine, Law, Creative Writing

> 📖 **[Contributing Guide →](docs/contributing.md)**

---

## License

HBLLM Core is released under **GNU General Public License v3.0 (GPLv3)**.

<div align="center">
  <p><b>HBLLM Core</b> — An AI that thinks like a person, models who you are, runs on your hardware, and scales on your terms.</p>
  <p>⭐ Star this repository to support open-source, privacy-first cognitive architectures!</p>
</div>
