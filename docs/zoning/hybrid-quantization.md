---
title: "Hybrid Quantization — Run AI Models Without Expensive GPU"
description: "How HBLLM's Rust SIMD compute kernel enables 4-bit quantized inference on CPU, eliminating the need for expensive GPUs. AVX2/NEON acceleration for x86 and ARM."
---

# Hybrid Quantization

HBLLM's hybrid quantization system is what makes it possible to **run cognitive AI without an expensive GPU**. By quantizing the base model to INT4 while keeping LoRA experts at FP16, HBLLM reduces memory 4× with minimal quality loss — enabling deployment on CPU-only hardware.

!!! success "Memory Savings"
    | Model | FP16 (No Quant) | INT4 (Quantized) | Savings |
    |---|---|---|---|
    | 125M | ~500MB | ~150MB | **3.3×** |
    | 500M | ~2GB | ~600MB | **3.3×** |
    | 1.5B | ~6GB | ~1.8GB | **3.3×** |
    
    Add LoRA adapters at just ~2MB each — negligible overhead.

## Architecture

```
┌─────────────────────────────────────────┐
│           Shared Base Model             │
│         (4-bit quantized, ~750MB)       │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ LoRA A   │  │ LoRA B   │  │ LoRA C ││
│  │ (16-bit) │  │ (16-bit) │  │(16-bit)││
│  │  ~2MB    │  │  ~2MB    │  │ ~2MB   ││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
```

## Rust Compute Kernel

The `hbllm_compute` crate provides SIMD-optimized dequantization:

- **AVX2** (x86_64) — 256-bit vector operations
- **AVX-512** (server x86_64) — 512-bit vector operations  
- **NEON** (ARM) — For Raspberry Pi, Apple Silicon, Jetson

### Performance

| Operation | Scalar | AVX2 | Speedup |
|---|---|---|---|
| INT4 Dequant (1K elements) | 2.1μs | 0.3μs | **7x** |
| INT8 Dequant (1K elements) | 1.8μs | 0.25μs | **7.2x** |
| Group-scale correction | 0.9μs | 0.15μs | **6x** |

### Group-Size Quantization

Weights are quantized in groups (default `group_size=128`):

```
For each group of 128 weights:
  scale = max(abs(weights)) / 7  (for INT4)
  bias  = min(weights)
  quantized = round((weights - bias) / scale)
```

This preserves per-group dynamic range while achieving 4x memory compression.

## Usage

The quantization system is automatically invoked when loading models with the `quantize` flag:

```python
from hbllm.model.transformer import HBLLMForCausalLM

model = HBLLMForCausalLM.from_pretrained(
    "hbllm/base-500m",
    quantize="int4",       # Enable 4-bit base
    lora_precision="fp16", # Keep LoRA at 16-bit
)
```
