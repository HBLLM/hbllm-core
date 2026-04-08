---
title: "Rust Kernels — SIMD-Accelerated Native Extensions"
description: "API reference for HBLLM's Rust-accelerated native crates — compute kernels (INT4/INT8 SIMD inference), data tools (MinHash dedup, text cleaning), and tokenizer (BPE training)."
---

# Rust Kernels

HBLLM includes three Rust crates that provide SIMD-accelerated performance for CPU inference, data processing, and tokenization. These are optional — the Python fallbacks work on all platforms, but Rust gives 5-50× speedups on CPU-only deployments.

**Location:** `rust/`

---

## Crate Index

| Crate | Directory | PyO3 Module | Purpose |
|---|---|---|---|
| `compute_kernel` | `rust/compute_kernel/` | `hbllm_compute_kernel` | INT4/INT8 quantized matmul, SIMD dot products |
| `data_tools` | `rust/data_tools/` | `hbllm_data_tools_rs` | MinHash deduplication, fast text cleaning |
| `tokenizer` | `rust/tokenizer/` | `hbllm_tokenizer_rs` | BPE tokenizer training and inference |

---

## Compute Kernel

The compute kernel provides SIMD-optimized matrix operations for quantized inference on CPU.

### Architecture Support

| Architecture | Instruction Set | Status |
|---|---|---|
| x86_64 | AVX2 / AVX-512 | ✅ Auto-detected at runtime |
| ARM64 | NEON | ✅ Auto-detected at runtime |
| Fallback | Scalar | ✅ Always available |

### Operations

- **INT4 × FP16 MatMul** — Base model weights stored in 4-bit, computed in FP16
- **INT8 × FP16 MatMul** — Higher precision variant for sensitive layers
- **SIMD Dot Product** — Vectorized inner product for attention scores
- **Dequantization** — INT4/INT8 → FP16 conversion with per-channel scales

### Integration

The compute kernel is automatically used by `hbllm.model.quantization` when the Rust extension is available:

```python
from hbllm.model.quantization import quantize_model

# Quantize model to INT4 (uses Rust SIMD if available)
quantize_model(model, bits=4)
```

---

## Data Tools

Rust-accelerated text processing for the data pipeline.

### Functions

| Function | Purpose | Speedup vs Python |
|---|---|---|
| `fast_clean_batch(docs)` | Batch text cleaning (Unicode normalization, whitespace, control chars) | ~10× |
| `Deduplicator(num_perm, threshold, shingle_size)` | MinHash LSH deduplication | ~20× |

### Usage

```python
from hbllm_data_tools_rs import fast_clean_batch, Deduplicator

# Clean a batch of documents
cleaned = fast_clean_batch(["  Hello   World  ", "Another\x00doc"])
# ["Hello World", "Another doc"]

# Deduplicate
dedup = Deduplicator(num_perm=128, threshold=0.8, shingle_size=5)
unique = dedup.deduplicate(cleaned)
```

---

## Tokenizer

Rust BPE tokenizer for training custom vocabularies.

### Classes

| Class | Purpose |
|---|---|
| `Trainer(vocab_size, min_frequency)` | Train a BPE vocabulary from text |
| `Vocab` | Encode/decode text with a trained vocabulary |

### Usage

```python
from hbllm_tokenizer_rs import Trainer, Vocab

# Train a tokenizer
trainer = Trainer(vocab_size=32768, min_frequency=2)
vocab = trainer.train_from_text(corpus_text)

# Save / load
vocab.save("vocab.json")
vocab = Vocab.load("vocab.json")

# Encode / decode
token_ids = vocab.encode("Hello, world!")
text = vocab.decode(token_ids)
```

---

## Building from Source

```bash
# Build all three crates
cd HBLLM/core

# Using maturin (recommended)
pip install maturin
maturin develop --manifest-path rust/compute_kernel/Cargo.toml --release
maturin develop --manifest-path rust/data_tools/Cargo.toml --release
maturin develop --manifest-path rust/tokenizer/Cargo.toml --release
```

### Development

```bash
# Format
cargo fmt --manifest-path rust/compute_kernel/Cargo.toml

# Lint
cargo clippy --manifest-path rust/compute_kernel/Cargo.toml --workspace -- -D warnings

# Check
cargo check --manifest-path rust/compute_kernel/Cargo.toml
```

!!! tip "Rust is Optional"
    All three crates have pure-Python fallbacks. The brain works on any platform without compiling Rust — you just get faster inference, data processing, and tokenization with it installed.
