---
title: "Data Pipeline — Download, Clean, Tokenize & Shard Training Data"
description: "API reference for HBLLM's data preparation pipeline. Rust-accelerated or pure-Python modes for downloading, cleaning, deduplicating, tokenizing, and sharding training data."
---

# Data Pipeline

HBLLM provides two data pipeline implementations for preparing training data — a Rust-accelerated production pipeline and a pure-Python experimental pipeline.

**Module:** `hbllm.data`

---

## Module Index

| File | Class / Function | Purpose |
|---|---|---|
| `pipeline.py` | `DataPipeline` | Rust-accelerated: download → clean → dedup → train BPE → shard |
| `pipeline.py` | `PurePythonPipeline` | tiktoken-based: stream → clean → tokenize → shard (no Rust) |
| `downloader.py` | `DatasetDownloader` | Stream data from HuggingFace datasets (FineWeb, StarCoder, OpenWebMath, etc.) |
| `sharder.py` | `ShardWriter` | Pack tokenized sequences into binary `.bin` shard files |
| `scorer.py` | `QualityScorer` | Filter low-quality documents by heuristics |
| `synthesizer.py` | `DataSynthesizer` | Generate synthetic training data for domain-specific LoRA training |
| `dataloader.py` | `ShardDataset` / `ShardDataLoader` | PyTorch DataLoader for reading binary shards |
| `interaction_miner.py` | `InteractionMiner` | Extract high-quality Q/A pairs from live interactions for DPO training |

---

## Rust-Accelerated Pipeline

Requires the Rust extensions (`hbllm_data_tools_rs`, `hbllm_tokenizer_rs`).

```python
from hbllm.data.pipeline import DataPipeline

pipeline = DataPipeline(work_dir="./data/training")
pipeline.run_all(
    dataset_name="fineweb",       # or "fineweb+starcoderdata+openwebmath"
    max_samples=100_000,
    vocab_size=32768,
    sequence_length=2048,
    data_weights=[0.6, 0.2, 0.2], # Per-dataset weights (with multi-dataset)
)
```

### Pipeline Steps

1. **Download** — Stream JSONL from HuggingFace via `DatasetDownloader`
2. **Clean** — Rust `fast_clean_batch()` for high-speed text normalization
3. **Dedup** — Rust `Deduplicator` using MinHash LSH (128 permutations, 0.8 threshold)
4. **Quality Filter** — `QualityScorer` removes short/noisy documents
5. **Train BPE** — Rust BPE tokenizer trainer
6. **Shard** — Pack into binary `.bin` files via `ShardWriter`

---

## Pure-Python Pipeline

No Rust required — uses tiktoken (cl100k_base, ~100K vocab).

```python
from hbllm.data.pipeline import PurePythonPipeline

pipeline = PurePythonPipeline(work_dir="./data/training")
stats = pipeline.run_all(
    dataset_name="fineweb",
    max_samples=50_000,
    sequence_length=2048,
    min_doc_tokens=64,
    shard_size_mb=64,
)
print(f"Processed {stats['total_docs']} docs, {stats['total_tokens']} tokens")
```

---

## CLI Usage

```bash
# Download and prepare data (auto-selects Rust or Python pipeline)
hbllm data --dataset fineweb --max-samples 100000

# Multi-dataset mixing
hbllm data --dataset "fineweb+starcoderdata" --weights 0.7,0.3
```

---

## Interaction Miner

**Module:** `hbllm.data.interaction_miner.InteractionMiner`

Automatically extracts high-quality query/response pairs from live brain interactions. These pairs feed the DPO training loop for continuous self-improvement.

```python
from hbllm.data.interaction_miner import InteractionMiner

miner = InteractionMiner(data_dir="data")
miner.record_interaction(
    query="What is quantum computing?",
    response="Quantum computing uses qubits...",
    reward=0.95,
    tenant_id="tenant-001",
)

# Export for DPO training
pairs = miner.export_dpo_pairs(min_reward=0.8, limit=1000)
```

---

## Data Synthesizer

**Module:** `hbllm.data.synthesizer.DataSynthesizer`

Generates synthetic training data for domain-specific LoRA training when real data is insufficient. Used by the `SpawnerNode` during artificial neurogenesis.
