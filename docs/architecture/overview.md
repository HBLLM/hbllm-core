---
title: "Architecture Overview — Pure-HCIR Cognitive OS"
description: "System-wide architecture of HBLLM Core: 8 non-negotiable invariants, single composition root (build_cognitive_system), Pure-HCIR subsystems, Rust substrate, and external plugin boundaries."
---

# Architecture Overview

> [!NOTE]
> For formal architectural decisions, see **[ADR 004: Pure-HCIR Cognitive OS](../adr/0004-pure-hcir-cognitive-os.md)** alongside historical **[ADR 001](../adr/0001-hbllm-core-architecture.md)** and **[ADR 002](../adr/0002-operational-architecture-and-governance.md)**.

**HBLLM Core** is an event-sourced, neuro-symbolic **Cognitive OS** built around the **Human-Cognitive Intermediate Representation (`hbllm.hcir`)**. Rather than treating an autoregressive LLM as the runtime brain, HBLLM separates formal state ownership, cortical active inference, symbol grounding, multilingual parsing/realization, and governed execution into deterministic, verifiable subsystems.

---

## 1. Eight Non-Negotiable Core Invariants

1. **State Ownership Invariant:** *Nodes emit intent. HCIR owns state.* All state changes are proposed as `TransactionOperation`s and committed atomically via `TransactionManager`.
2. **Reasoning Invariant:** *Operators calculate. HCIR decides what becomes reality.* Reasoning operators receive immutable `FrozenGraphView` / `CognitiveContext` snapshots and return pure `CognitiveResult`s.
3. **Zero LLM Dependency in Core:** `hbllm.hcir` (`world`, `grounding`, `reasoning`, `language`, `snn`, `capabilities`, `kernel`) has **zero** LLM client or network imports (enforced by `tests/unit/architecture/test_loc_budget.py`).
4. **Constitutional Governance:** Every mutation passes through `ConstitutionalVerifier` and verification pipeline stages (`ScopeVerifier`, `SchemaVerifier`, `ResourceVerifier`, `PolicyVerifier`, `BranchModeVerifier`) before commit.
5. **Multi-Tenant Memory Isolation:** Episodic, semantic, procedural, and belief memories are partitioned by `Scope` / `tenant_id` across `TieredWorkspace`.
6. **Target State Evolution:** Changes upgrade toward declarative HCIR grammar, formal transactions, and dynamic causal predicates — never monkey-patches or task-specific branches.
7. **General Intelligence Directive:** The core remains strictly domain- and task-agnostic; environment adapters live externally in `plugins/` (see [Innate Priors](priors.md)).
8. **Internal Memory Governance:** Learned world ontology lives in `HCIRSymbolicWorldTheory` (`TransitionKnowledgeBase`), transient state lives in `PrefrontalWorkingMemory`, and behaviors live in `DeclarativeNeuroSymbolicSkill`.

---

## 2. Layered Pure-HCIR System Topology

```mermaid
graph TB
    subgraph ENTRY["Entry Points & Adapters"]
        CLI["Typer CLI (hbllm.cli)"]
        SERVE["FastAPI + MCP Server (hbllm.serving)"]
        PLUGINS["Environment & Device Adapters (plugins/*_adapter)"]
        BOOT["Single Composition Root<br/>build_cognitive_system() (hbllm.runtime.bootstrap)"]
        CLI & SERVE & PLUGINS --> BOOT
    end

    subgraph HCIR["Pure-HCIR Cognitive OS (hbllm.hcir)"]
        direction TB
        subgraph WORLD["Cortical World Model (hcir.world — 17 subpackages)"]
            AEE["AutonomousEpistemicEngine + 5-Stage CorticalPipeline"]
            TKB["HCIRSymbolicWorldTheory + TransitionKnowledgeBase"]
            PWM["PrefrontalWorkingMemory + Predictors (Physics/SNN/WholeGrid)"]
        end

        subgraph GROUND["Symbol Grounding (hcir.grounding)"]
            SGE["FeatureExtractors (handcrafted/snn/learned), Tracker, ConceptLearner, HCIRCommitter"]
        end

        subgraph LANG["Language & Reading (hcir.language)"]
            MLR["MultilingualLanguageRuntime (en/si/ta), SymSpell Normalizer, HCIRGateway"]
            READ["ExpositoryProseProcessor &amp; Track T4 ReadingCurriculumIngestor"]
        end

        subgraph REAS["Composable Reasoning &amp; SNN"]
            URR["UnifiedReasoningRuntime + 14 Operators (hcir.reasoning)"]
            SNN["LIF, STDP, WTA, Oscillations, Neuromodulation (hcir.snn)"]
        end

        subgraph KERNEL["HCIR Kernel &amp; Memory Tiers (hcir.kernel)"]
            TM["TransactionManager + ConstitutionalVerifier + CommitAuthorityRegistry"]
            CAP["CommandDispatcher + CapabilitySandboxManager + hcir.capabilities"]
            TW["TieredWorkspace (Working | Brain | Persistent | Meta) + SQLiteEventStore"]
        end

        AEE <--> TKB & PWM
        SGE <--> AEE
        LANG --> CAP
        WORLD & GROUND & LANG & REAS & CAP --> TM
        TM --> TW
        SNN -.-> WORLD & GROUND & REAS
    end

    subgraph RUST["Native Rust Substrate (13 PyO3 Crates)"]
        RG["hbllm_hcir_graph (Chunked CoW + BLAKE3)"]
        SIM["hbllm_simulation_engine &amp; hbllm_structure_matcher"]
        SIMD["hbllm_compute, hbllm_semantic_search, hbllm_perception_rs, ..."]
    end

    BOOT --> HCIR
    TW <-->|"Verified Accelerator"| RG
    HCIR -.-> SIM & SIMD
```

---

## 3. Subsystem Map & Documentation Guide

| Subsystem | Package | Architecture Doc | API Reference |
|---|---|---|---|
| **Composition Root & CLI** | `hbllm.runtime.bootstrap`, `hbllm.cli` | [System Entry Points](entry-points.md) | [CLI Reference](../guides/cli.md) |
| **HCIR Kernel & Transactions** | `hbllm.hcir.kernel`, `hbllm.hcir` | [HCIR Kernel OS](hcir-kernel-os.md) | [HCIR API](../api/hcir.md) |
| **Cortical World Model** | `hbllm.hcir.world`, `hbllm.hcir.skills` | [Cortical World Model](cortical-world-model.md) | [World Model API](../api/world-model.md) |
| **Symbol Grounding Engine** | `hbllm.hcir.grounding` | [Symbol Grounding](grounding.md) | — |
| **Language & Reading** | `hbllm.hcir.language`, `hbllm.hcir.stores` | [Language & Reading](language-and-reading.md) | [Language API](../api/language.md) |
| **Composable Reasoning** | `hbllm.hcir.reasoning` | [Cognitive Execution Runtime](cognitive-execution-runtime.md) | [Reasoning API](../api/reasoning.md) |
| **Neuromorphic SNN** | `hbllm.hcir.snn` | [SNN Cognitive Engine](snn-cognitive-engine.md) | [SNN API](../api/snn.md) |
| **Capabilities & Curiosity** | `hbllm.hcir.capabilities` | [Capabilities & Curiosity](capabilities-and-curiosity.md) | [Capabilities API](../api/capabilities.md) |
| **Memory & Event Store** | `hbllm.hcir.workspace_tiers`, `hbllm.hcir.stores` | [Memory Systems](memory-systems.md) | [Memory API](../api/memory.md) |
| **Graph Substrate (Rust)** | `rust/hcir_graph`, `hbllm.native` | [Graph Substrate](graph_substrate.md) | [Rust Kernels](../api/rust-kernels.md) |
| **Sandboxing & Security** | `hbllm.hcir.kernel.capability_sandboxing`, `hbllm.security` | [Sandboxing](sandboxing.md) | [Security](../security.md), [Governance API](../api/governance.md) |
| **Developmental Learning** | `plugins/developmental_adapter` | [Developmental Learning](developmental-learning.md) | [Benchmarks](../api/benchmarks.md) |
| **Boundaries & Telemetry** | `hbllm.telemetry`, `hbllm.observability` | [Packaging & Boundaries](packaging-and-boundaries.md), [Telemetry](telemetry-observability.md) | [Telemetry API](../api/telemetry.md) |
| **Message Bus & Federation** | `hbllm.network` | [Message Bus](message-bus.md), [Resilience](network-resilience.md), [Distributed Trust](distributed-trust.md) | [Network API](../api/network.md) |

---

## 4. Compatibility & Transition Layers

During the Pure-HCIR consolidation (see [ADR 004](../adr/0004-pure-hcir-cognitive-os.md)):
- `hbllm/brain/reasoning/`, `hbllm/brain/language/`, and core `hbllm/brain/snn/` act as compatibility re-export shims pointing into `hbllm.hcir.*`.
- Legacy message-bus nodes in `hbllm/brain/` and `hbllm/serving/provider.py` remain strictly isolated from `hbllm/hcir/**` (verified by `tests/unit/architecture/test_loc_budget.py` and `test_layering.py`).
- External environment integrations (`arc_agi_adapter`, `crafter_adapter`, `babyai_adapter`, `sokoban_adapter`, `overcooked_adapter`, `nethack_adapter`, `ai2thor_adapter`, `safety_gym_adapter`, `alfworld_adapter`, `developmental_adapter`) live in `plugins/` and never leak domain logic into `hbllm/`.
