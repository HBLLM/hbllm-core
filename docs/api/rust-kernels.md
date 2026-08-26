---
title: "Rust Kernels — SIMD-Accelerated Native Extensions"
description: "API reference for all 13 HBLLM Rust-accelerated native crates — compute kernels, tokenizer, data tools, semantic search, knowledge graph, policy evaluation, perception, network utilities, concept extraction, confidence calibration, HCIR graph substrate, simulation engine, and structure matcher."
---

# Rust Kernels & SIMD Extensions

HBLLM includes **13 native Rust crates** that provide SIMD-accelerated performance for CPU inference, data processing, tokenization, vector search, graph algorithms, policy evaluation, persistent HCIR graph operations, parallel counterfactual simulation, and analogical structure mapping. These extensions are optional — pure Python fallbacks exist across all components, but the native Rust extensions provide 5× to 100× throughput gains on CPU deployments (workstations, Apple Silicon, edge servers).

**Location:** `rust/`

---

## 13 Native Crates Index

| Crate | Directory | Python Module | Purpose | Speedup vs Python |
|---|---|---|---|---|
| **`compute_kernel`** | `rust/compute_kernel/` | `hbllm_compute` | INT4/INT8 quantized MatMul, dynamically dispatched SIMD GEMV | 10–30× |
| **`tokenizer`** | `rust/tokenizer/` | `hbllm-tokenizer` | BPE tokenizer training and ultra-fast text encoding | 15–40× |
| **`data_tools`** | `rust/data_tools/` | `hbllm-data-tools` | MinHash LSH deduplication, fast UTF-8 cleaning | 10–25× |
| **`semantic_search`**| `rust/semantic_search/` | `hbllm_semantic_search` | SIMD-vectorized cosine & dot product similarity search | 20–50× |
| **`knowledge_graph`**| `rust/knowledge_graph/` | `hbllm_knowledge_graph` | High-speed graph traversal, neighbor queries, shortest paths | 15–35× |
| **`policy_eval`** | `rust/policy_eval/` | `hbllm_policy_eval` | Native AST policy evaluation and rule verification | 25–50× |
| **`confidence`** | `rust/confidence/` | `hbllm_confidence` | Native epistemic uncertainty and confidence scoring | 10–20× |
| **`concept_extract`**| `rust/concept_extract/` | `hbllm_concept_extract` | Fast episodic text pattern clustering & rule mining | 8–15× |
| **`perception`** | `rust/perception/` | `hbllm_perception_rs` | Audio frame preprocessing and vector projection | 12–25× |
| **`network_utils`** | `rust/network_utils/` | `hbllm_network_utils` | High-throughput binary message framing & serialization | 15–30× |
| **`hcir_graph`** | `rust/hcir_graph/` | `hbllm_hcir_graph` | Persistent chunked structural-sharing HCIR GraphState with BLAKE3 canonical hashing | 40–100× |
| **`simulation_engine`** | `rust/simulation_engine/` | `hbllm_simulation_engine` | Resident state multi-branch counterfactual simulation & DAG transition cache | 25–80× |
| **`structure_matcher`** | `rust/structure_matcher/` | `hbllm_structure_matcher` | Analogical constraint satisfaction mapping & systematicity scoring | 30–60× |

---

## 1. Compute Kernel (`hbllm_compute`)

The compute kernel provides SIMD-optimized matrix operations for quantized autoregressive decoding on CPU:

### Instruction Set Support

| Architecture | Instruction Set | Status |
|---|---|---|
| **x86_64** | AVX-512 / AVX2 / FMA | ✅ Auto-detected at boot |
| **ARM64** | NEON / ASIMD (Apple Silicon) | ✅ Auto-detected at boot |
| **Fallback** | Portable Scalar | ✅ Universal compatibility |

### Operations

- **`gemv_4bit_simd`**: Dynamically dispatched fused register-level weight unpacking and matrix-vector multiplication for single-token autoregressive decoding.
- **INT8 $\times$ FP16 MatMul**: Higher precision linear transformation for sensitive projection layers.
- **Dynamic Dequantization**: Per-channel scaling and zero-point alignment.

---

## 2. Tokenizer (`hbllm_tokenizer`)

High-performance Byte-Pair Encoding (BPE) engine:

```python
from hbllm_tokenizer import Trainer, Vocab

# Train a custom tokenizer directly from text files
trainer = Trainer(vocab_size=32768, min_frequency=2)
vocab = trainer.train_from_files(["./data/corpus.txt"])
vocab.save("vocab.json")

# Fast encoding & decoding
vocab = Vocab.load("vocab.json")
token_ids = vocab.encode("HBLLM native tokenization")
decoded_text = vocab.decode(token_ids)
```

---

## 3. Data Tools (`hbllm_data_tools`)

Provides industrial data curation algorithms:

- **`Deduplicator(num_perm, threshold, shingle_size)`**: MinHash Locality-Sensitive Hashing (LSH) for near-duplicate document removal.
- **`fast_clean_batch(docs)`**: Multi-threaded Unicode normalization, zero-width space removal, and control character stripping.

```python
from hbllm_data_tools import Deduplicator, fast_clean_batch

cleaned_docs = fast_clean_batch(["Doc 1\x00 dirty", "Doc 2 clean"])
dedup = Deduplicator(num_perm=128, threshold=0.8)
unique_docs = dedup.deduplicate(cleaned_docs)
```

---

## 4. Semantic Search (`hbllm_semantic_search`)

SIMD-accelerated vector index for `SemanticMemory`:

```python
import numpy as np
from hbllm_semantic_search import VectorIndex

# 1024-dim dense vector index
index = VectorIndex(dimension=1024, metric="cosine")
index.add_items(ids=["doc1", "doc2"], vectors=embeddings_array)

# Query nearest neighbors in < 1ms
results = index.search(query_vector, top_k=5)
```

---

## 5. Knowledge Graph (`hbllm_knowledge_graph`)

Native graph structures executing BFS shortest paths and multi-hop neighbor lookups:

```python
from hbllm_knowledge_graph import NativeGraph

graph = NativeGraph()
graph.add_edge("Python", "Rust", relation="interops_with", weight=1.0)
graph.add_edge("Rust", "SIMD", relation="utilizes", weight=1.0)

path = graph.shortest_path("Python", "SIMD")
# ["Python", "Rust", "SIMD"]
```

---

## 6. Policy Evaluator (`hbllm_policy_eval`)

Fast AST-based compliance evaluation for `PolicyEngine`:

```python
from hbllm_policy_eval import PolicyRuleSet

rules = PolicyRuleSet.from_yaml("policies/default_guardrails.yaml")
decision = rules.evaluate(
    action="file_write",
    context={"path": "/etc/passwd", "tenant": "user-01"},
)
# Returns: {"allowed": False, "violated_rule": "no_system_path_mutation"}
```

---

## 7. Epistemic Confidence (`hbllm_confidence`)

Native beta-distribution Bayesian updating and Expected Calibration Error (ECE) curves for the Epistemic Runtime.

---

## 8. Concept Extractor (`hbllm_concept_extract`)

Fast recurring n-gram and association graph clustering from raw episodic conversation turns.

---

## 9. Perception RS (`hbllm_perception_rs`)

Fast Fourier Transform (FFT), Mel-filterbank extraction, and voice audio framing.

---

## 10. Network Utils (`hbllm_network_utils`)

Zero-copy binary message framing, CRC32 checksumming, and protocol buffer serialization.

---

## 11. HCIR Graph Substrate (`hbllm_hcir_graph`)

Persistent, chunked structural-sharing graph state representation for HCIR. Provides $O(1)$ snapshots, sub-millisecond clone times, chunk-level copy-on-write isolation, and deterministic BLAKE3 canonical state hashing:

```python
from hbllm_hcir_graph import NativeGraph

graph = NativeGraph()
graph.add_node("node_1", "Entity", "ACTIVE", {"name": "ObjectA"}, 1000.0)
graph.add_node("node_2", "Entity", "ACTIVE", {"name": "ObjectB"}, 1000.0)
graph.add_edge("e1", "SUPPORTS", ["node_1"], ["node_2"], 1.0, {}, 1000.0)

# O(1) immutable snapshot with chunked structural sharing
snapshot = graph.snapshot()

# Canonical BLAKE3 hash across nodes and hyperedges
hash_val = graph.canonical_hash()
```

---

## 12. Simulation Engine (`hbllm_simulation_engine`)

Resident native cognitive runtime managing parallel multi-branch counterfactual simulation rollouts. Evaluates action sequences across isolated graph branches with zero Python FFI crossing per step, Rayon thread-pool parallelism, 3D bounding-box collision detection, and transition DAG memoization:

```python
from hbllm_simulation_engine import NativeCognitiveRuntime

runtime = NativeCognitiveRuntime()
# Run parallel branches across native CPU cores
results = runtime.evaluate_rollout_branches(
    branches=[
        {
            "branch_id": 1,
            "actions": [
                {
                    "operator": "move",
                    "subject": "block_a",
                    "target": "table",
                    "parameters": {"x": 1.0, "y": 2.0},
                }
            ],
            "initial_risk": 0.0,
            "initial_cost": 0.0,
            "max_steps": 5,
        }
    ]
)
```

---

## 13. Structure Matcher (`hbllm_structure_matcher`)

Analogical constraint-satisfaction graph isomorphism engine. Discovers systematic variable-to-entity mappings and structural alignment scores between source schemas and target domains with backtracking search:

```python
from hbllm_structure_matcher import match_structures

match_result = match_structures(
    pattern_edges=[{"rel_type": "REVOLVES_AROUND", "source_var": "planet", "target_var": "sun"}],
    target_nodes=["electron", "nucleus"],
    target_edges=[{"rel_type": "REVOLVES_AROUND", "source": "electron", "target": "nucleus"}],
)
# Returns mapping: {"planet": "electron", "sun": "nucleus"} with systematicity score
```

---

## Building All Crates

Build all 13 Rust extensions using `maturin`:

```bash
cd HBLLM/core

# Install maturin build tool
pip install maturin

# Build and install all crates in release mode
for crate in rust/*/; do
  if [ -f "$crate/Cargo.toml" ]; then
    echo "Building $crate..."
    maturin develop --manifest-path "$crate/Cargo.toml" --release
  fi
done
```
