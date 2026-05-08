---
title: "Benchmarks & Profiling — Measure HBLLM Performance Without Guessing"
description: "API reference for HBLLM's benchmark runner and performance profiler. Measure latency, memory efficiency, domain routing accuracy, and multi-tenant throughput on any hardware."
---

# Benchmarks & Profiling

HBLLM ships with a comprehensive benchmark suite and performance profiler to validate cognitive architecture performance on your hardware — no external tools needed.

---

## Benchmark Runner

**Module:** `hbllm.benchmarks.runner`

The benchmark runner compares HBLLM's zoning architecture against monolithic model baselines across 4 dimensions.

### Available Suites

| Suite | What it measures |
|---|---|
| `latency` | Message bus pub/sub p50/p99, node start overhead, bus throughput (msg/s), **Planner Early Convergence (MCTS breakout)** |
| `memory` | **Router ONNX Peak RAM (Measured via `tracemalloc`)**, LoRA zoning vs full-model memory |
| `specialization` | **Router Fast-Path Latency (ONNX)**, Domain routing accuracy, self-expansion capability |
| `multi_tenant` | 10-tenant concurrent throughput, tenant isolation verification |

### Recent Architectural Optimizations Tested

**1. Router Fast-Path Latency & Memory**
Instead of forcing every user query through a massive LLM, the `RouterNode` utilizes an ultra-fast Int8 ONNX Embedding Model (`paraphrase-MiniLM-L3-v2`). The benchmark runner instantiates the real node to prove:
- **Memory Footprint:** < 1 MB dynamically allocated RAM at runtime.
- **Latency:** Measured Fast-Path latency for classification before triggering base SLM fallbacks.

**2. Planner Early Convergence Exit**
The `PlannerNode`'s Graph-of-Thoughts loop has been optimized with an "Early Convergence Exit". The benchmark runner proves that if the reward score of an internal thought hits `> 0.90`, the execution loop terminates in **~18ms** instead of running for the full 15-second search budget.

### CLI Usage

```bash
# Run all benchmark suites
python -m hbllm.benchmarks.runner --suite all

# Run a specific suite
python -m hbllm.benchmarks.runner --suite latency

# Save results to JSON
python -m hbllm.benchmarks.runner --suite memory --output results.json
```

### Python API

```python
from hbllm.benchmarks.runner import run_suite, run_all

# Run a single suite
report = await run_suite("latency")
report.print_report()

# Run all suites
reports = await run_all()

# Save results
report.save("benchmark_results.json")
```

### Result Types

```python
@dataclass
class BenchmarkResult:
    name: str          # Human-readable metric name
    metric: str        # Machine-readable metric key
    value: float       # Measured value
    unit: str          # Unit (ms, MB, msg/s, %, etc.)
    metadata: dict     # Extra context

@dataclass
class BenchmarkReport:
    suite: str                          # Suite name
    results: list[BenchmarkResult]      # All measurements
    comparisons: list[dict]             # HBLLM vs baseline comparisons
```

---

## Performance Profiler

**Module:** `hbllm.benchmarks.profiler`

The profiler measures real resource usage under load — complementing the benchmark runner's architectural comparisons.

### Profile Suites

| Suite | What it profiles |
|---|---|
| `memory` | EpisodicMemory write speed & DB size at 100/1K/10K turns, SemanticMemory TF-IDF indexing, ProceduralMemory skill storage |
| `throughput` | Sustained bus throughput with varying payload sizes (10B → 100KB) |
| `startup` | Node start/stop times for RouterNode, DecisionNode, PlannerNode, CriticNode |
| `pipeline` | End-to-end message flow latency (p50, p99, mean, stdev) over 1000 messages |

### CLI Usage

```bash
# Run all profiler suites
python -m hbllm.benchmarks.profiler --suite all

# Profile memory systems
python -m hbllm.benchmarks.profiler --suite memory

# Save profile to JSON
python -m hbllm.benchmarks.profiler --output profile.json
```

### Python API

```python
from hbllm.benchmarks.profiler import run_profile

report = await run_profile("throughput")
```

---

## Evaluation Modules

**Module:** `hbllm.benchmarks`

| File | Purpose |
|---|---|
| `eval_prm.py` | Evaluate Process Reward Model scoring accuracy |
| `eval_speculative.py` | Benchmark speculative decoding speedup vs standard generation |
| `eval_tot.py` | Evaluate Graph-of-Thoughts planning quality |
