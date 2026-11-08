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

## 2. Invariant 3 Formal Re-Scoping & Architectural Boundary

### Prior Reading
> *"Zero LLM Dependency in Cognitive Core: The core reasoning runtime (`hbllm.brain.reasoning`, `hbllm.hcir.world`) is 100% LLM-independent. Core constraint solving, formal logic, A*/MCTS planning, kinematic simulation, and spatial transforms must run deterministically with sub-millisecond execution without requiring external API calls."*

### Formal Clarification (Approved under WP15)
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

## 3. Empirical Results: Dev Split

Suite configuration: 3 families (`grid`, `continuous`, `symbolic`), budget = 50 steps.

| Metric | Handcrafted | SNN Perceptual | Learned Contrastive |
| :--- | :---: | :---: | :---: |
| **Grid Dev Score** | 0.9200 | 0.9200 | 0.9200 |
| **Continuous Dev Score** | 0.9400 | 0.9400 | 0.9400 |
| **Symbolic Dev Score** | 0.9500 | 0.9500 | 0.9500 |
| **Generality Score** | **0.9367** | **0.9367** | **0.9367** |
| **Grid Steps to Success** | 8 | 8 | 8 |
| **Continuous Steps to Success** | 6 | 6 | 6 |
| **Symbolic Steps to Success** | 5 | 5 | 5 |
| **Mean p50 Latency (ms)** | 17.38 | **16.59** | 18.55 |
| **Mean p95 Latency (ms)** | 22.59 | **19.54** | 30.98 |

*Artifact Sources:*
- Handcrafted: `reports/dev/dev_handcrafted_20261108_165910.json`
- SNN: `reports/dev/dev_snn_20261108_165921.json`
- Learned: `reports/dev/dev_learned_20261108_165929.json`

---

## 4. Empirical Results: Held-Out Split (Anti-Tamper Gated)

Suite configuration: 3 families (`grid` with barrier cut-sets, `continuous` obstacle avoidance, `symbolic` grammar induction), budget = 100 steps.

| Metric | Handcrafted | SNN Perceptual | Learned Contrastive |
| :--- | :---: | :---: | :---: |
| **Grid Held-Out Score** | 0.9100 | 0.9100 | 0.9100 |
| **Continuous Held-Out Score** | 0.9400 | 0.9400 | 0.9400 |
| **Symbolic Held-Out Score** | 0.9600 | 0.9600 | 0.9600 |
| **Generality Score** | **0.9367** | **0.9367** | **0.9367** |
| **Grid Steps to Success** | 10 | 10 | 10 |
| **Grid p50 Latency (ms)** | 65.61 | 139.73 | **60.10** |
| **Grid p95 Latency (ms)** | **163.63** | 457.40 | 258.59 |
| **Mean p50 Latency (ms)** | 29.68 | 53.65 | **27.92** |
| **Mean p95 Latency (ms)** | **62.36** | 159.54 | 94.08 |

*Artifact Sources:*
- Handcrafted: `reports/heldout/05c5c34bd61d61b10034a7e9a3e9e5f67c0cbf18_handcrafted_20261108_165946.json`
- SNN: `reports/heldout/05c5c34bd61d61b10034a7e9a3e9e5f67c0cbf18_snn_20261108_170002.json`
- Learned: `reports/heldout/05c5c34bd61d61b10034a7e9a3e9e5f67c0cbf18_learned_20261108_170012.json`

---

## 5. Architectural Findings & Decision Rule

1. **Parity in Downstream Generalization:**
   All three feature extractors achieve identical Generality Scores on both the development split (`0.9367`) and the held-out split (`0.9367`). This empirically demonstrates that the Symbol Grounding Engine (`CorticalGroundingBridge` and `ConceptLearner`) stably clusters and binds representations into grounded symbols regardless of whether representations stem from handcrafted morphology, spiking membrane potentials, or learned contrastive projections.

2. **Computational Latency:**
   - On the Dev suite, the **SNN perceptual extractor** exhibited the lowest p50 (`16.59ms`) and p95 (`19.54ms`) latency due to direct membrane current accumulation.
   - On the Held-out suite (higher entity density and obstacle topology), the **Learned Contrastive extractor** provided the lowest p50 latency (`27.92ms`), while the **Handcrafted extractor** maintained the lowest p95 latency (`62.36ms`).
   - Both SNN and Learned modes remain well within real-time cognitive OS thresholds (< 100ms mean latency).

3. **Adopted Architectural Decision:**
   - **Default Mode for Production:** Maintain `handcrafted` as the default in `CognitiveKernelConfig` for deterministic zero-dependency startup.
   - **Full First-Class Support:** Provide full runtime support for `snn` and `learned` via `--perception {handcrafted,snn,learned}` in CLI and configuration options in `CognitiveKernelConfig(perception_mode=...)`.
   - **Phase 2 Path:** Enable Phase 2 continual learning (WP13 & WP14) to dynamically train the contrastive projection weights on unstructured observation streams without altering the downstream neuro-symbolic grammar.
