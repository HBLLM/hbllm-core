<!-- SEO Keywords: Sovereign Personal AI, Open Source AGI, Cognitive Architecture, Large Language Models, Multi-Agent Systems, Edge AI, Hybrid Quantization, LoRA Tuning, Privacy-First AI, On-Premise AI, Python 3.11 AI Framework, Autonomous Agents, Multi-Tenant AI -->

<div align="center">
  <h1>🧠 HBLLM Core: An AI That Thinks Like a Person</h1>
  <p><b>A continuously thinking, goal-driven, memory-forming cognitive brain — runs entirely on your own hardware, no cloud required.</b></p>
  <p><em>Local by default. Distributed when you want it. HBLLM stays awake, notices what's happening around it, sets its own goals, and learns from every interaction — your data never leaves your device unless you choose.</em></p>

  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
  [![Rust](https://img.shields.io/badge/Rust-Accelerated-orange.svg)](https://www.rust-lang.org/)
  [![Tests](https://img.shields.io/badge/Tests-2300%2B%20passing-brightgreen.svg)](#)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.md)
</div>

<br/>

> [!NOTE]  
> **Reviewer Quick Links:**
> - 🛡️ **[Security Architecture](SECURITY.md)** (Identity Triplet, Tenant Guard, Audit Log, Encryption at Rest)
> - 🔐 **[Governance & Policies](docs/api/governance.md)** (PolicyEngine, SentinelNode, Constitutional Principles)
> - ⚡ **[Reproducible Benchmarks](docs/api/benchmarks.md)** (Memory Profiling, Fast-Path Latency)

## What is HBLLM Core?

Most AI systems answer questions when you ask them. HBLLM Core does something fundamentally different — **it thinks all the time, entirely on your own machine**.

It has a genuine **memory** like a human does: short-term memory for what just happened, long-term memory for things it's learned, a knowledge graph of how concepts relate to each other, and even a sense of personal values built up from reward signals over time. Everything stored locally, everything private.

It can **set goals and pursue them in the background** — breaking big objectives into steps, retrying failed ones, checking that its actions actually worked in the real world, and picking up exactly where it left off after a reboot, just like a person resuming work after sleep.

When you're ready, it can **scale across your own devices** — your phone, laptop, and home server sharing knowledge and collaborating, all connected by cryptographic trust, with no cloud middleman involved.

And it has the **safety instincts of a responsible person** — it knows when it's overloaded and slows down, it has a conscience in the form of a policy engine that blocks harmful actions before they happen, and every decision it makes is logged to an immutable audit trail.

```text
                    ┌─────────────────────────────────────────┐
    Input ────────► │              HBLLM Core Brain           │
    (text, vision,  │                                         │
     audio, OS)     │   Perception ─► Router ──► Planner      │
                    │       │           │           │          │
                    │  WorldState    Memory    Critic/Eval     │
                    │       │       (5 types)       │          │
                    │  Comprehension              Identity     │
                    │  Stream (SNN)                  │          │
                    │       │                       │          │
                    │  Expression  ─► Policy ──► OS Adapter   │
                    │  Stream (SNN)                            │
                    └────────────────────────────┬────────────┘
                                                 │
    Action / Output ◄────────────────────────────┘
```

> 📖 **[Full Architecture →](docs/architecture/overview.md)** · **[Cognitive Nodes →](docs/architecture/cognitive-nodes.md)** · **[Memory Systems →](docs/architecture/memory-systems.md)**

---

## Key Capabilities

| Category | What it does | Docs |
|----------|-------------|------|
| **⚙️ Fully Local by Default** | Runs entirely on your own hardware — no cloud, no API keys, no data leaving your device | [Benchmarks](docs/api/benchmarks.md) |
| **🧠 Always-On Cognition** | Stays awake between queries — notices events, forms thoughts, and acts proactively without being asked | [Executive Brain](docs/architecture/executive-brain-layer.md) |
| **💾 Human-Like Memory** | Five memory types (Episodic, Semantic, Procedural, Value, Knowledge Graph) that persist across reboots and grow over time | [Memory Systems](docs/architecture/memory-systems.md) |
| **🎯 Goal Pursuit** | Decomposes objectives into persistent DAG tasks, retries failures, verifies real-world outcomes, and survives crashes | [Executive Brain](docs/architecture/executive-brain-layer.md) |
| **👁️ World Awareness** | Reads your OS, sensors, calendar, and apps — maintains a live probabilistic model of what's happening right now | [Embodiment](docs/architecture/embodiment.md) |
| **🧪 Self-Personalizing** | Dynamically adapts to your knowledge via 2MB LoRA adapters — grows new specialist regions at runtime | [Zoning](docs/zoning/how-it-works.md) |
| **🌐 Distributed When You Want It** | Optionally spans your phone, laptop, and edge servers via `SynapseGateway` with Ed25519 cryptographic trust — zero cloud required | [Adaptive Network](docs/architecture/adaptive-network.md) |
| **🛑 Human Control Layer** | Policy engine blocks harmful actions, every decision is audited, and the system slows itself down when overloaded | [Human Control](docs/architecture/human-control.md) |
| **📦 Memory Compaction** | Causal graphs, attention-based memory folding, and decision deltas keep the brain efficient over long lifetimes | [Causality & Compaction](docs/architecture/causality-and-compaction.md) |
| **🧬 SNN Cognitive Stream** | Spiking Neural Networks for concept extraction, content planning, and reward evaluation with STDP learning | [Pipeline](docs/architecture/pipeline.md) |
| **🔌 Plugin SDK** | Declarative `@subscribe` plugins with auto-binding — extend any part of the cognitive loop | [Plugin Guide](docs/guides/plugins.md) |
| **🔐 Enterprise Security** | Multi-tenant isolation, encrypted memory scopes, node revocation, and vector clock replay protection | [Security](SECURITY.md) |
| **🧬 Neurogenesis** | SpawnerNode auto-creates new domain specialist LoRA adapters — the brain literally grows new regions | [Zoning](docs/zoning/how-it-works.md) |

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
    # Cloud-backed
    brain = await BrainFactory.create("openai/gpt-4o")

    # Or fully local — no API keys needed
    # brain = await BrainFactory.create_local("./checkpoints/sft/my_domain")

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
| **[Architecture](docs/architecture/)** | System overview, cognitive nodes, memory, message bus, sleep cycle, executive brain |
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
  <p><b>HBLLM Core</b> — An AI that thinks like a person, runs on your hardware, and scales on your terms.</p>
  <p>⭐ Star this repository to support open-source, privacy-first cognitive architectures!</p>
</div>
