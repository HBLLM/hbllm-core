<!-- SEO Keywords: Sovereign Personal AI, Open Source AGI, Cognitive Architecture, Large Language Models, Multi-Agent Systems, Edge AI, Hybrid Quantization, LoRA Tuning, Privacy-First AI, On-Premise AI, Python 3.10 AI Framework, Autonomous Agents, Multi-Tenant AI -->

<div align="center">
  <h1>🧠 HBLLM Core: Your Sovereign Personal AI</h1>
  <p><b>A highly personalized, multi-tenant AI ecosystem that runs entirely on your own hardware.</b></p>
  <p><em>An advanced cognitive architecture that learns from you, protects your data, and manages your digital life through secure client and server plugins.</em></p>

  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
  [![Rust](https://img.shields.io/badge/Rust-Accelerated-orange.svg)](https://www.rust-lang.org/)
  [![Tests](https://img.shields.io/badge/Tests-1600%2B%20passing-brightgreen.svg)](#)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.md)
</div>

<br/>

> [!NOTE]  
> **Reviewer Quick Links:**
> - 🛡️ **[Security Architecture](SECURITY.md)** (Identity Triplet, Tenant Guard, Audit Log, Encryption at Rest)
> - 🔐 **[Governance & Policies](docs/api/governance.md)** (PolicyEngine, SentinelNode, Constitutional Principles)
> - ⚡ **[Reproducible Benchmarks](docs/api/benchmarks.md)** (Memory Profiling, Fast-Path Latency)

## Why HBLLM Core?

While the industry races to build massive, centralized models that ingest your private data, **HBLLM Core** is built for digital sovereignty. It is a **Personal AI Platform** engineered to give you complete ownership over a cognitive agent that truly knows you, without ever sending your life to the cloud.

It achieves this by replacing massive GPU clusters with an efficient, modular architecture: **28+ specialized brain nodes** orchestrated via an async Pub/Sub message bus. By running on hardware as accessible as a Raspberry Pi or a laptop, it ensures your data remains strictly yours.

```text
                    ┌─────────────────────────────────────────┐
    Input ────────► │              HBLLM Core Brain           │
    (text, vision,  │                                         │
     audio)         │   Router ──► Planner ──► Decision       │
                    │     │          │            │           │
                    │   Memory    Learner      Critic        │
                    │   (5 types)    │            │           │
                    │              World       Identity      │
                    │              Model       (ethics)      │
                    │                │                        │
                    │           Curiosity ──► Spawner        │
                    └───────────────────────────┬─────────────┘
                                                │
    Output ◄────────────────────────────────────┘
```

By decoupling reasoning, memory, evaluation, and action, HBLLM can maintain lifelong memories, securely execute multi-step tools across your local and cloud environments, and dynamically adapt to your personal domains using hot-swappable LoRA adapters.

> 📖 **[Full Architecture →](docs/architecture/overview.md)** · **[Cognitive Nodes →](docs/architecture/cognitive-nodes.md)** · **[Memory Systems →](docs/architecture/memory-systems.md)**

---

## Key Capabilities

| Category | Highlights | Docs |
|----------|-----------|------|
| **🧠 Agentic Reasoning** | GoT planning, PRM scoring, MoE domain blending | [Cognitive Nodes](docs/architecture/cognitive-nodes.md) |
| **🧪 Personalization** | Dynamically adapts to your knowledge via 2MB LoRA adapters | [How It Works](docs/zoning/how-it-works.md) |
| **💾 Memory Systems** | Working, Episodic, Semantic, Procedural, Knowledge Graph | [Memory Systems](docs/architecture/memory-systems.md) |
| **🌐 Swarm Architecture** | Edge devices act as synced limbs via WebSocket `UplinkNode` | [Message Bus](docs/architecture/message-bus.md) |
| **🔌 Plugin SDK** | Declarative `@subscribe` plugins with auto-binding | [Plugin Guide](docs/guides/plugins.md) |
| **🛡️ Governance** | PolicyEngine + SentinelNode + tenant isolation | [Governance](docs/api/governance.md) |
| **🔐 Enterprise Security** | Identity triplet, `@require_tenant`, audit log, encryption | [Security](SECURITY.md) |
| **⚙️ Infrastructure** | 128k+ context, Rust SIMD quantization, ONNX router | [Benchmarks](docs/api/benchmarks.md) |
| **🧬 Neurogenesis** | SpawnerNode auto-creates new domain specialists | [Zoning](docs/zoning/how-it-works.md) |

---

## Platform-Agnostic Design

HBLLM Core is a **fully platform-independent cognitive library**. It has zero dependencies on any frontend, desktop, or server framework.

| Layer | Responsibility | Location |
|-------|---------------|----------|
| **Core** | Cognitive nodes, memory, bus, agent executor, plugins | `hbllm/` |
| **Platform Bridge** | UI, config, persistence, platform-specific tools | External (e.g. [Sentra](../sentra/)) |
| **Plugins** | Modular cognitive extensions | `hbllm/plugins/` |

Platforms extend the core via thin subclass wrappers:
```python
from hbllm.actions.agent_executor import AgentExecutor

class MyPlatformExecutor(AgentExecutor):
    def _register_platform_tools(self):
        self.tools.register("my_tool", ..., my_tool_fn, {})
```

> 📖 **[Plugin Development →](docs/guides/plugins.md)** · **[Custom Nodes →](docs/guides/custom-nodes.md)**

---

## Cognitive Plugins

Three built-in plugins ship with the core, using the declarative `@subscribe` SDK:

| Plugin | Capabilities | Topics |
|--------|-------------|--------|
| **Emotion Modeling** | VAD tracking, tone adaptation | `system.experience` → `emotion.state` |
| **Temporal Reasoning** | Time references, deadline tracking | `system.experience` → `temporal.context` |
| **Swarm Orchestrator** | Task decomposition, parallel execution | `swarm.request` → `swarm.complete` |

> 📖 **[Plugin SDK Reference →](docs/guides/plugins.md)**

---

## Quick Start

```bash
git clone https://github.com/your-org/hbllm-core.git
cd hbllm-core
pip install -e .
```

```python
import asyncio
from hbllm.brain.factory import BrainFactory

async def main():
    brain = await BrainFactory.create("openai/gpt-4o")
    result = await brain.process("Analyze our server logs and design a firewall rule.")
    print(result.text)
    await brain.shutdown()

asyncio.run(main())
```

> 📖 **[Quickstart Guide →](docs/guides/quickstart.md)** · **[Configuration →](docs/guides/configuration.md)** · **[Deployment →](docs/guides/deployment.md)**

---

## Documentation

| Section | Contents |
|---------|----------|
| **[Architecture](docs/architecture/)** | System overview, cognitive nodes, memory, message bus, sleep cycle |
| **[Zoning](docs/zoning/)** | LoRA routing, weighted domains, hybrid quantization |
| **[Guides](docs/guides/)** | Quickstart, custom nodes, plugins, training, deployment, IoT/robotics |
| **[API Reference](docs/api/)** | Brain factory, subsystems, network, model, tokenizer, Rust kernels |

---

## Contributing

We welcome contributions! Key areas:
- 🧠 **Cognitive Plugins** — Extend emotion, temporal, swarm or build new ones
- 📱 **Edge Devices** — Optimization for Raspberry Pi 5 & Jetson Orin Nano
- 🌐 **Starter Zones** — Pre-trained LoRAs for Medicine, Law, Creative Writing

> 📖 **[Contributing Guide →](docs/contributing.md)**

## License

HBLLM Core is released under **GNU General Public License v3.0 (GPLv3)**.

<div align="center">
  <p><b>HBLLM Core</b> — Your Sovereign Personal AI.</p>
  <p>⭐ Star this repository to support open-source, privacy-first cognitive architectures!</p>
</div>
