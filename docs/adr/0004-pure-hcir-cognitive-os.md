---
title: "ADR 004: Pure-HCIR Cognitive OS & LLM-Free Core Consolidation"
description: "Architectural decision graduating HBLLM Core from legacy multi-node LLM MessageBus + LoRA Zoning to the unified Pure-HCIR Cognitive OS."
---

# ADR 004: Pure-HCIR Cognitive OS & LLM-Free Core Consolidation

- **Status**: Implemented
- **Deciders**: HBLLM Core Architecture Team
- **Date**: 2026-11-23
- **Technical Area**: Unified Cognitive Runtime (`hbllm/runtime/`, `hbllm/hcir/`)
- **Supersedes**: Legacy LoRA/Zoning execution path & `ADR 003` standalone epistemics

---

## Context and Problem Statement

Early HBLLM iterations distributed cognitive functions across an asynchronous `MessageBus` of LLM-prompted nodes, LoRA domain adapters ("Zoning"), and the standalone `ADR 003` epistemic layer. While flexible for experimenting with LLM-backed orchestration, that design hit four structural walls:

1. **Fragmented State Ownership:** Legacy brain nodes and custom drivers maintained separate SQLite databases and unverified mutable dictionaries, risking state drift across turns and tenants.
2. **Hallucination & Latency Overhead in Core Loops:** Spatial, causal, and symbolic planning cannot tolerate autoregressive LLM latency or uncalibrated token probabilities in real-time loops.
3. **Artificial Layering Silos:** Keeping `reasoning`, `language`, and `snn` outside `hbllm.hcir` prevented the cortical world model and language runtime from sharing immutable snapshots (`FrozenGraphView`), `TransactionManager` atomicity, and `TieredWorkspace` memory tiers directly.
4. **Exiled / Dead Code Sprawl:** Legacy causal-LM pretraining, Zoning/LoRA scaffolding, and zero-effect experimental hooks obscured the active architecture.

---

## Decision

Graduate HBLLM Core to the **Pure-HCIR Cognitive OS** built around a single composition root and a consolidated, 100% LLM-free cognitive substrate under `hbllm.hcir`:

1. **Single Composition Root (`hbllm.runtime.bootstrap.build_cognitive_system`):**
   - Instantiates `TieredWorkspace` (`Working`, `Brain`, `Persistent`, `Meta`), `CommitAuthorityRegistry`, `ConstitutionalVerifier`, `TransactionManager`, `KernelServices`, `AutonomousEpistemicEngine`, `CorticalGroundingBridge`, `ExecutiveRuntime`, and `UnifiedReasoningRuntime` as one verified runtime container (`CognitiveSystem`).
   - All runtime paths (`mind` CLI, `kernel` CLI, `bench` harness, dialogue processing, and embodied adapters) target `build_cognitive_system()`.

2. **Direct HCIR Consolidation (`hbllm/hcir/`):**
   - **`hbllm/hcir/reasoning/`**: Houses `UnifiedReasoningRuntime`, `OperatorRegistry`, and the 14 default composable operators (`Deduction`, `Induction`, `Abduction`, `Spatial`, `Temporal`, `Causal`, `Analogy`, `Counterfactual`, `Contradiction`, `Prediction`, `Simulation`, `ActiveInference`, `EmbodiedCausal`, `SNNReasoning`) plus auxiliary schema/macro engines (`RelationalSchemaInducer`, `SkillMacroChunker`). (`hbllm/brain/reasoning/` retains compatibility re-exports.)
   - **`hbllm/hcir/language/`**: Houses the 100% LLM-free multilingual Wernicke + Broca substrate (`MultilingualLanguageRuntime` for `en`, `si`, `ta`), `HCIRGateway`, SymSpell-indexed `BaseOrthographicNormalizer`, `DiscourseSegmenter`, `ExpositoryProseProcessor`, `ReadingCurriculumIngestor` (Track T4), and `LexiconAcquisitionLoop` (A17).
   - **`hbllm/hcir/snn/`**: Houses `LIFNeuron`, `SpikingNetwork`, `STDP`, `STPManager`, `WTACircuit`, `OscillationManager`, `NeuromodulationEngine`, `ComprehensionStream`, and visual/audio perception gating ensembles.
   - **`hbllm/hcir/capabilities/`**: Houses kernel-governed capability executors (`ConceptExplainer`, `EpistemicCuriosityController`, `BeliefConflictResolver`, `ToolCapabilityExecutor`, `WebCapabilityExecutor`, `MemoryCapabilityExecutor`, `PlanningCapabilityExecutor`, `RoutingCapabilityExecutor`, `WorldKernelCapabilityExecutor`), dispatched via `CommandDispatcher` and metered by `AutonomyTracker`.

3. **Modularized Cortical World Model & `TransitionKnowledgeBase`:**
   - Decomposed `hbllm/hcir/world/` into 17 cohesive domain subpackages (`assimilation`, `causal`, `deliberation`, `episodic`, `grounding`, `hypothesis`, `learning`, `macro`, `motor`, `objects`, `perception`, `planning`, `prediction`, `predictors`, `spatial`, `state`, `working_memory`) driven by the 5-stage `CorticalPipeline`.
   - Made `TransitionKnowledgeBase` (`hbllm.hcir.world.causal.transition_kb`) the HCIR-backed single source of truth inside `HCIRSymbolicWorldTheory`, storing empirical transition counts, locus-aware lethal beliefs, causal effect rules, and epoch-isolated context beliefs as graph nodes.

4. **Governance & Architectural Ratchets (`tests/unit/architecture/`):**
   - `test_loc_budget.py`: Enforces `<= 150,000` LOC for `hbllm/hcir/**`, `<= 300,000` LOC across `hbllm/**/*.py`, a decreasing ratchet on legacy `__path__.append` shims, and **zero** LLM client/scaffold imports (`openai`, `anthropic`, `litellm`, `transformers`, `plugins.llm_scaffold`, `hbllm.serving.provider`) inside `hbllm/hcir/**`.
   - `test_layering.py`, `test_no_direct_sqlite.py`, `test_single_composition_root.py`, `test_structural_debt_ratchet.py`, `test_rust_abi_version.py`, `test_no_duplicate_faculties.py`: Enforce dependency hygiene and latency/correctness invariants in CI.

---

## Measured Verification (`eval-phase2-r10`)

On the sealed **Held-Out v2** benchmark suite (`benchmarks/suites/heldout.yaml`, `n=120` trials across 5 task families, `eval-phase2-r10` on commit `a26f09bd`, `2026-11-21`, documented in `reports/heldout/LEDGER.md`):

- **Zero-Shot Solve Rate:** **0.9250** (`111/120`, Wilson 95% CI `[0.8636, 0.9600]`, generality score `0.8236`, 4 exploration deaths / `3.33%`).
  - `grid`: `0.9500`, `continuous`: `1.0000`, `symbolic`: `1.0000`, `partial_observability`: `0.8000`, `stochastic_dynamic`: `0.8500`.
- **Cross-Seed Transfer:** **0.9035** (`114` transfer episodes, Wilson 95% CI `[0.8354, 0.9453]`, `skills_consolidated=161`).
- **Same-Layout Replay:** **1.0000** solve rate (generality score `0.9021`).
- **LLM Token Cost:** **0 tokens**.

---

## Consequences

- **Positive:** Single verifiable state graph, deterministic sub-second cognitive execution, unified non-LLM language/reading/reasoning/grounding, and clean separation of optional external LLM scaffolds (`plugins/llm_scaffold/`) from the core.
- **Trade-offs & Ongoing Remediation:** Legacy `hbllm/brain/`, `hbllm/serving/`, and `hbllm/drivers/cognitive_blackbox.py` still coexist for backward compatibility and HTTP serving, and `docs/` have been pruned of obsolete Zoning/LLM/legacy-brain references to reflect Pure-HCIR as the primary architecture.

---

## References

- [System Entry Points](./../architecture/entry-points.md)
- [Cortical World Model](./../architecture/cortical-world-model.md)
- [Language & Reading Comprehension](./../architecture/language-and-reading.md)
- [Capabilities & Curiosity](./../architecture/capabilities-and-curiosity.md)
- [Architecture Overview](./../architecture/overview.md)

