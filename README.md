<!-- SEO Keywords: Sovereign Cognitive OS, Neuro-Symbolic AI, Event-Sourced Graph Substrate, Active Inference, Symbol Grounding Engine, Pure-HCIR, Python 3.11 AI Framework, Rust Acceleration, Multi-Tenant AI, Spiking Neural Networks -->

<div align="center">
  <h1>🧠 HBLLM Core</h1>
  <h3>Pure-HCIR Event-Sourced Cognitive OS &amp; Neuro-Symbolic AI</h3>
  <p><b>A deterministic, LLM-free cognitive kernel and active-inference world model built over an event-sourced graph substrate (HCIR) — runs entirely on your own hardware.</b></p>
  <p><em>Nodes emit intent. Operators calculate. HCIR owns state. Zero cloud dependencies required in the core reasoning loop.</em></p>

  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
  [![Rust](https://img.shields.io/badge/Rust-13%20Crates-orange.svg)](https://www.rust-lang.org/)
  [![Tests](https://img.shields.io/badge/Tests-599%20Core%20%2B%2070%20Plugin%20Suites-brightgreen.svg)](#)
  [![Held-Out v2](https://img.shields.io/badge/Held--Out%20v2-92.5%25%20Zero--Shot%20(0%20LLM%20Tokens)-cyan.svg)](#verified-benchmarks)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.md)
</div>

<br/>

> [!NOTE]
> **Quick Links:**
> - 📐 **[Architecture Overview](docs/architecture/overview.md)** — Pure-HCIR Cognitive OS, 8 invariants, and single composition root (`build_cognitive_system`)
> - 🌍 **[Cortical World Model](docs/architecture/cortical-world-model.md)** — 5-stage `CorticalPipeline`, 17 world subpackages, and `TransitionKnowledgeBase`
> - 🗣️ **[Language & Reading](docs/architecture/language-and-reading.md)** — 100% LLM-free multilingual parser/realizer (`en`, `si`, `ta`) and Track T4 reading comprehension
> - 🛡️ **[Security & Sandboxing](docs/security.md)** — `ConstitutionalVerifier`, two-tier capability sandboxing, JWT auth, and multi-tenant isolation
> - ⚡ **[Benchmarks & Held-Out v2](docs/api/benchmarks.md)** — Frozen suites (`dev`, `novelty`, `widened_dev`, `heldout`), `eval-*` gate, and native simulators

---

## Why HBLLM Core?

Instead of relying on an autoregressive transformer as the primary brain, **HBLLM Core models cognition as formal operations over an event-sourced graph substrate (`hbllm.hcir`)**:

- **Event-Sourced State Ownership:** No cognitive subsystem mutates state directly. Proposals are submitted as `TransactionOperation`s, checked by `ConstitutionalVerifier`, and committed atomically by `TransactionManager` across a 4-tier workspace (`Working`, `Brain`, `Persistent`, `Meta`).
- **100% LLM-Free Cognitive Core:** World modeling, active inference, Pearl's *do(a)* causal induction (`TransitionKnowledgeBase`), symbol grounding (`hbllm.hcir.grounding`), 14 composable reasoning operators (`hbllm.hcir.reasoning`), neuromorphic SNN dynamics (`hbllm.hcir.snn`), and multilingual dialogue (`hbllm.hcir.language`) run deterministically with **0 LLM tokens**.
- **Domain-Agnostic General Intelligence:** All core mechanisms operate on universal spatial, causal, topological, and relational primitives — never task-specific heuristics. Environment adapters live cleanly outside `hbllm/` in `plugins/`.
- **Governed Real-World Capabilities:** File inspection, AST analysis, memory recall, graph explanation (`ConceptExplainer`), autonomous curiosity (`EpistemicCuriosityController`), and web fetch execute through `CommandDispatcher` and `CapabilitySandboxManager`.
- **Native Rust Acceleration:** 13 optional Rust/PyO3 crates (with automatic pure-Python fallbacks) accelerate structural-sharing graph snapshots, BLAKE3 canonical hashing, parallel rollout simulation, subgraph isomorphism, SIMD math, and vector search.

```text
  Observation / Dialogue ──► build_cognitive_system() [Single Composition Root]
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  Language & Reading       Cortical World Model          Composable Reasoning
  (hbllm.hcir.language)    (hbllm.hcir.world)           (hbllm.hcir.reasoning)
  en/si/ta, SymSpell,      5-Stage CorticalPipeline,    14 Formal Operators,
  Discourse & Track T4     TransitionKB, Grounding,     RelationalSchema &
  SVO Ingestion            PrefrontalWorkingMemory      SkillMacroChunker
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                 TransactionManager + ConstitutionalVerifier
                                    │
                                    ▼
         TieredWorkspace (Working | Brain | Persistent | Meta) + SQLiteEventStore
                                    │
                                    ▼
           CommandDispatcher + CapabilitySandboxManager ──► Action / Response
```

> 📖 **[Architecture →](docs/architecture/overview.md)** · **[HCIR Kernel OS →](docs/architecture/hcir-kernel-os.md)** · **[Entry Points →](docs/architecture/entry-points.md)** · **[Symbol Grounding →](docs/architecture/grounding.md)**

---

## Key Capabilities

| Subsystem | Role | Package | Docs |
|---|---|---|---|
| **HCIR Cognitive OS Kernel** | 8-opcode ISA, `TieredWorkspace`, `TransactionManager`, `ConstitutionalVerifier`, `ExecutiveRuntime` | `hbllm.hcir` / `hbllm.hcir.kernel` | [HCIR Kernel OS](docs/architecture/hcir-kernel-os.md) |
| **Cortical World Model** | 5-stage active inference (`Perception → Assimilation → Simulation → Deliberation → Motor`), `TransitionKnowledgeBase` | `hbllm.hcir.world` | [Cortical World Model](docs/architecture/cortical-world-model.md) |
| **Symbol Grounding Engine** | Perceptual feature extraction (`handcrafted`, `snn`, `learned`), bipartite tracking, concept induction | `hbllm.hcir.grounding` | [Grounding](docs/architecture/grounding.md) |
| **Language & Reading** | Non-LLM multilingual parser/realizer (`en`, `si`, `ta`), `HCIRGateway`, `ReadingCurriculumIngestor` | `hbllm.hcir.language` | [Language & Reading](docs/architecture/language-and-reading.md) |
| **Composable Reasoning** | 14 immutable-snapshot operators (`Deduction`, `Abduction`, `Causal`, `Spatial`, `Analogy`, `EmbodiedCausal`, …) | `hbllm.hcir.reasoning` | [Execution Runtime](docs/architecture/cognitive-execution-runtime.md) |
| **Neuromorphic SNN** | LIF neurons, STDP, STP, WTA, cortical oscillations, `NeuromodulationEngine`, SNN perception | `hbllm.hcir.snn` | [SNN Engine](docs/architecture/snn-cognitive-engine.md) |
| **Capabilities & Curiosity** | `CommandDispatcher`, `ConceptExplainer`, `EpistemicCuriosityController`, `BeliefConflictResolver`, sandboxed tools | `hbllm.hcir.capabilities` | [Capabilities](docs/architecture/capabilities-and-curiosity.md) |
| **Declarative Skills** | Composable `SymbolicPredicate` signatures & `SubgoalSequence` bytecode compilation | `hbllm.hcir.skills` | [World Model API](docs/api/world-model.md) |
| **Security & Serving** | FastAPI REST + MCP server, fail-closed `JWTAuthMiddleware`, `TenantGuard`, `AuditTrail`, `PIIRedactor` | `hbllm.serving` / `hbllm.security` | [Security](docs/security.md) |
| **Native Rust Substrate** | 13 Rust crates (`hbllm_hcir_graph`, `hbllm_simulation_engine`, `hbllm_structure_matcher`, `hbllm_compute`, …) | `rust/` / `hbllm.native` | [Rust Kernels](docs/api/rust-kernels.md) |

---

## Verified Benchmarks

### Sealed Held-Out v2 (`eval-phase2-r10`, 2026-11-21)

Executed under the anti-tamper `eval-*` git tag gate on `benchmarks/suites/heldout.yaml` (`n = 120` unseen trials across 5 regimes, commit `a26f09bd`, ledger in `reports/heldout/LEDGER.md`):

| Metric | Measured Result | Wilson 95% CI | Notes |
|---|---|---|---|
| **Zero-Shot Solve Rate** | **0.9250** (`111 / 120`) | `[0.8636, 0.9600]` | Generality `0.8236`, 4 exploration deaths (`3.33%`), **0 LLM tokens** |
| **Cross-Seed Transfer** | **0.9035** (`114` episodes) | `[0.8354, 0.9453]` | `skills_consolidated = 161` across distinct procedural seeds |
| **Same-Layout Replay** | **1.0000** | — | Generality `0.9021`, `mean_steps_to_stable_policy = 0.0` |

**Per-Family Breakdown (`eval-phase2-r10`):**
- `continuous`: **100.0%** zero-shot (`100.0%` cross-seed transfer)
- `symbolic`: **100.0%** zero-shot (`100.0%` cross-seed transfer)
- `grid`: **95.0%** zero-shot (`97.37%` cross-seed transfer)
- `stochastic_dynamic`: **85.0%** zero-shot (`78.95%` cross-seed transfer)
- `partial_observability`: **80.0%** zero-shot (`68.42%` cross-seed transfer)

> For native simulator benchmarks (`crafter`, `minigrid`, `gym_sokoban`, `overcooked_ai_py`, `minihack`, `ai2thor`, `safety_gymnasium`, `alfworld`, `arc_agi`), see **[Benchmarks & Profiling](docs/api/benchmarks.md)**.

---

## Quick Start

```bash
git clone https://github.com/hbllm/hbllm-core.git
cd HBLLM/core

# Core Pure-HCIR installation (no GPU / no PyTorch required)
pip install -e .

# Optional: install FastAPI/JWT server extras or build native Rust wheels
pip install -e ".[server]"
for crate in rust/*/; do [ -f "$crate/Cargo.toml" ] && maturin develop --manifest-path "$crate/Cargo.toml" --release; done
```

### Python SDK (Single Composition Root)

```python
from hbllm.hcir.world.percept import EmbodiedObservation
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

# 1. Boot the single canonical Cognitive OS runtime
system = build_cognitive_system(
    CognitiveSystemConfig(
        world_id="quickstart",
        memory_dir=None,  # In-memory SQLite / workspace (or path for persistence)
        enable_grounding=True,
        perception_mode="handcrafted",  # "handcrafted" | "snn" | "learned"
    )
)

# 2. LLM-free multilingual dialogue & knowledge graph ingestion
system.process_dialogue("A capacitor stores energy in an electric field.")
reply = system.process_dialogue("What does a capacitor store?")
print(reply.response_text)

# 3. Deterministic 5-stage cortical step on spatial observation
obs = EmbodiedObservation(
    lattice=[[0, 0, 0], [0, 1, 0], [0, 0, 2]],
    available_actions=[1, 2, 3, 4],
    context_id="quickstart",
    episode_id="0",
)
action, meta = system.engine.step(obs)
print(f"Action: {action}, Phase: {meta.get('phase')}")
```

### CLI

```bash
hbllm diagnostics                   # Hardware, Rust SIMD, and Cognitive OS health
hbllm info                          # Cognitive OS invariants and 5-stage cortical loop
hbllm kernel status                 # Inspect verified TransactionManager, stages & native graph
hbllm mind step --obs obs.json      # Single deterministic active-inference step
hbllm bench dev                     # Run frozen dev benchmark suite
hbllm bench profile                 # Profile substrate & 32x32 5-stage E2E latency
hbllm serve --port 8000             # Start FastAPI REST + MCP Server (requires .[server])
hbllm plugin list                   # List installed plugins in plugins/
```

> 📖 **[Quickstart Guide →](docs/guides/quickstart.md)** · **[CLI Manual →](docs/guides/cli.md)** · **[Configuration →](docs/guides/configuration.md)** · **[Deployment →](docs/guides/deployment.md)**

---

## Project Stats

| Metric | Value |
|---|---|
| Core source modules (`hbllm/`) | **988** Python files |
| HCIR Cognitive OS (`hbllm/hcir/`) | **413** Python files across **11** subpackages (budget `<= 150,000` LOC) |
| Cortical World Model (`hbllm/hcir/world/`) | **149** Python files across **17** domain subpackages |
| Reasoning operators (`hbllm/hcir/reasoning/`) | **14** default composable operators (0 LLM dependencies) |
| Workspace tiers | **4** (`Working`, `Brain`, `Persistent`, `Meta`) |
| Test suites | **599** core test files (`tests/`) + **70** plugin test files (`plugins/`) |
| Rust crates (`rust/`) | **13** PyO3 native acceleration crates |

---

## Documentation

| Section | Contents |
|---|---|
| **[Architecture](docs/architecture/overview.md)** | [Overview](docs/architecture/overview.md), [Entry Points](docs/architecture/entry-points.md), [HCIR Kernel OS](docs/architecture/hcir-kernel-os.md), [Cortical World Model](docs/architecture/cortical-world-model.md), [Symbol Grounding](docs/architecture/grounding.md), [Language & Reading](docs/architecture/language-and-reading.md), [Reasoning Runtime](docs/architecture/cognitive-execution-runtime.md), [SNN Engine](docs/architecture/snn-cognitive-engine.md), [Capabilities & Curiosity](docs/architecture/capabilities-and-curiosity.md), [Memory Systems](docs/architecture/memory-systems.md), [Graph Substrate](docs/architecture/graph_substrate.md), [Sandboxing](docs/architecture/sandboxing.md), [Innate Priors](docs/architecture/priors.md) |
| **[ADRs](docs/adr/README.md)** | [ADR 001](docs/adr/0001-hbllm-core-architecture.md), [ADR 002](docs/adr/0002-operational-architecture-and-governance.md), [ADR 003](docs/adr/0003-epistemic-runtime.md), [ADR 004 (Pure-HCIR)](docs/adr/0004-pure-hcir-cognitive-os.md) |
| **[Guides](docs/guides/quickstart.md)** | [Quickstart](docs/guides/quickstart.md), [CLI Reference](docs/guides/cli.md), [Configuration](docs/guides/configuration.md), [Deployment](docs/guides/deployment.md), [Plugin SDK](docs/guides/plugins.md), [Custom Nodes](docs/guides/custom-nodes.md), [Packages & Skills](docs/guides/hbpkg-and-skills.md), [Persistence & Backup](docs/guides/persistence-backup.md), [Robotics & IoT](docs/guides/robotics-iot.md) |
| **[API Reference](docs/api/hcir.md)** | [HCIR Kernel](docs/api/hcir.md), [World Model](docs/api/world-model.md), [Language](docs/api/language.md), [Capabilities](docs/api/capabilities.md), [Reasoning](docs/api/reasoning.md), [SNN](docs/api/snn.md), [Memory](docs/api/memory.md), [Governance](docs/api/governance.md), [Security](docs/api/security.md), [Serving](docs/api/serving.md), [Benchmarks](docs/api/benchmarks.md), [Rust Kernels](docs/api/rust-kernels.md) |

---

## Contributing & License

See **[Contributing Guide](docs/contributing.md)** and **[AGENTS.md](AGENTS.md)** for engineering invariants, architectural ratchets, and verification commands.

HBLLM Core is released under **GNU General Public License v3.0 (GPLv3)** — see [LICENSE.md](LICENSE.md).
