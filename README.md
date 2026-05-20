<!-- SEO Keywords: Sovereign Personal AI, Open Source AGI, Cognitive Architecture, Large Language Models, Multi-Agent Systems, Edge AI, Hybrid Quantization, LoRA Tuning, Privacy-First AI, On-Premise AI, Python 3.10 AI Framework, Autonomous Agents, Multi-Tenant AI -->

<div align="center">
  <h1>🧠 HBLLM Core: Local-First Sovereign AI Runtime</h1>
  <p><b>A private, sovereign AI engine designed for standalone local utility with optional distributed cognition.</b></p>
  <p><em>An advanced cognitive architecture that runs entirely on your own hardware by default, scales to a hierarchical swarm when needed, and protects your data through cryptographic distributed trust.</em></p>

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

While the industry races to build massive, centralized models that ingest your private data, **HBLLM Core** is built for **digital sovereignty**. It is a **Local-First AI Runtime** engineered to provide absolute privacy by default. 

**Standalone by Default**: HBLLM Core runs completely on its own without needing to connect to the broader HBLLM network or any cloud service. It is designed to be fully functional entirely on your local hardware.

**Optional Distributed Cognition**: When requested, HBLLM replaces monolithic cloud dependencies with a modular **Hierarchical Swarm of Specialized Cognitive Nodes**. By securely connecting your own devices (phones, laptops, edge servers) via Ed25519 cryptographic trust, it creates a personal AI network. 

**Universal Connectivity**: The `SynapseGateway` is not just for AI-to-AI communication. It allows **non-AI applications**, traditional software, web backends, and IoT devices to seamlessly connect to the cognitive swarm and utilize its intelligence.

```text
                    ┌─────────────────────────────────────────┐
    Input ────────► │              HBLLM Core Brain           │
    (text, vision,  │                                         │
     audio)         │   EventBus ─► Router ──► Planner        │
                    │     │           │          │            │
                    │ EventLog     Memory    Learner/Critic   │
                    │     │       (5 types)      │            │
                    │ CausalGraph              World/Identity │
                    │     │                      │            │
                    │ Human Guard ─► Verifier ─► OS Adapter   │
                    └────────────────────────────┬────────────┘
                                                 │
    Action / Output ◄────────────────────────────┘
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
| **🌐 Swarm Architecture** | Hierarchical edge devices via `SynapseGateway` and `UplinkNode` | [Cognitive Nodes](docs/architecture/cognitive-nodes.md) |
| **🦾 Embodiment & Actuation**| OS/Device adapter, idempotency tracking, execution verification | [Embodiment](docs/architecture/embodiment.md) |
| **🛑 Human Control Layer** | Trust boundaries, intervention policies, explanation-first mode | [Human Control](docs/architecture/human-control.md) |
| **📦 Cognitive Compaction**| Causal graphs, attention-based memory folding, decision deltas | [Causality & Compaction](docs/architecture/causality-and-compaction.md) |
| **🔌 Plugin SDK** | Declarative `@subscribe` plugins with auto-binding | [Plugin Guide](docs/guides/plugins.md) |
| **🛡️ Governance & Trust** | Ed25519 Distributed Trust, Vector Clock Anti-Replay | [Distributed Trust](docs/architecture/distributed-trust.md) |
| **🔐 Enterprise Security** | Identity triplet, Node Revocation (`system.dlq`), audit log | [Security](SECURITY.md) |
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
