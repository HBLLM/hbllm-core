# WP15: Perceptual Feature Extractor Empirical Comparison & Decision Report

> **Author:** Antigravity Autonomous Agent  
> **Date:** 2026-11-08  
> **Evaluation Harness:** WP0 Frozen Evaluation Harness (`benchmarks/suites/runner.py`)  
> **Commit SHA:** `05c5c34bd61d61b10034a7e9a3e9e5f67c0cbf18`  

---

## 1. Executive Summary & Objective

Property 1 of the AGI Gap Analysis (`reports/audit/2026-11-08_agi_gap_analysis.md`) identified that hand-crafted geometric and heuristic feature extractors create an artificial ceiling on agent generality. To address this gap without violating core neuro-symbolic principles, **WP15** implemented and empirically benchmarked three distinct perceptual feature extraction regimes within the Symbol Grounding Engine (`hbllm.hcir.grounding`):

1. **Handcrafted Baseline (`handcrafted`):** Geometric connected components, bounding-box aspect ratio, fill density, and color normalisation (`SpatialGridFeatureExtractor`).
2. **Spiking Neural Perceptual Stack (`snn`):** Leaky integrate-and-fire (LIF) membrane integration with deterministic synaptic weights and normalized potential vectors (`SNNPerceptualFeatureExtractor`).
3. **Learned Contrastive Projection (`learned`):** Offline-trained contrastive linear projection layer with LayerNorm and non-linear activation (`LearnedPerceptualFeatureExtractor`).

All three regimes were evaluated across the **Dev** and frozen **Held-out** benchmark splits using identical downstream cognitive kernels, active inference planners, and constitutional governance.

---

## 2. Invariant 3 Formal Re-Scoping & Architectural Boundary (PROVISIONAL)

### Prior Reading
> *"Zero LLM Dependency in Cognitive Core: The core reasoning runtime (`hbllm.brain.reasoning`, `hbllm.hcir.world`) is 100% LLM-independent. Core constraint solving, formal logic, A*/MCTS planning, kinematic simulation, and spatial transforms must run deterministically with sub-millisecond execution without requiring external API calls."*

### Formal Clarification (Provisional pending closed-loop evaluation harness)
To permit neural sensory processing without introducing non-determinism, API latency, or LLM hallucination into the cognitive loop:
- **Strictly Prohibited in Core Reasoning:**
  - External network calls, cloud API requests (OpenAI, Anthropic, Gemini, HuggingFace Inference API).
  - Unbounded stochastic sampling or online unverified backpropagation during decision rollouts.
  - LLM-generated code execution inside sensory-motor loops.
- **Permitted in Core Sensory-Motor Ingestion:**
  - Deterministic feed-forward neural encoders (e.g. LIF spiking dynamics, frozen contrastive encoders, compact convolutional feature projectors).
  - Encoders conforming to the formal `FeatureExtractor` protocol (`extractor_name`, `can_extract()`, `extract()`).
  - Model parameters versioned deterministically within `TieredWorkspace.persistent` or compiled native libraries.

---

## 3. Empirical Results: RETRACTED

> [!WARNING]
> **RETRACTED (2026-11-09 Post-Implementation Review §1.2 - §1.3):**  
> The previous evaluation suites (`benchmarks/suites/`) were scripted: the harness moved the avatar directly and returned unconditional `success = True` regardless of engine decisions. The 3-way tie (`0.9367`) across Handcrafted, SNN, and Learned modes was an artifact of the scripted harness rather than true downstream invariance. All previous empirical numbers and JSON reports in `reports/heldout/` and `reports/dev/` are retracted.
>
> Re-evaluation will be conducted strictly using the closed-loop, unscripted evaluation harness.

---

## 4. Next Step: Re-evaluation on Closed-Loop Harness

Re-evaluation will run on procedurally generated, closed-loop environments (`benchmarks/envs/`) with raw sensory lattices, zero oracle role labels, and baselines (`random`, `oracle`, `no_grounding`). Once executed on the closed-loop harness, empirical results will be recorded here.

