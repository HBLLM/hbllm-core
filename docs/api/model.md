---
title: "Model Internals — Transformer Architecture Reference"
description: "API reference for HBLLM's decoder-only transformer model — GQA attention, SwiGLU FFN, RoPE, speculative decoding, LoRA injection, and Process Reward Models. All running without massive GPU."
---

# Model Internals

HBLLM's transformer follows the LLaMA 3 / Mistral architecture with Grouped Query Attention, SwiGLU FFN, RMSNorm, and Rotary Position Embeddings.

**Module:** `hbllm.model`

---

## Module Index

| File | Class / Function | Purpose |
|---|---|---|
| `transformer.py` | `HBLLMForCausalLM` | Full model with LM head for training and generation |
| `transformer.py` | `HBLLMModel` | Base transformer (embeddings + N blocks + final norm) |
| `transformer.py` | `TransformerBlock` | Single decoder block (GQA + SwiGLU + residual) |
| `transformer.py` | `HBLLMForProcessReward` | Sequence classification head for step-level PRM scoring |
| `attention.py` | `GroupedQueryAttention` | GQA with RoPE, sliding window, attention sinks, and KV cache |
| `feedforward.py` | `SwiGLUFFN` | SwiGLU feed-forward network |
| `feedforward.py` | `MoEFFN` | Mixture-of-Experts FFN with load balancing loss |
| `embeddings.py` | `TokenEmbedding` | Token embedding layer |
| `embeddings.py` | `RotaryEmbedding` | RoPE position encoding |
| `normalization.py` | `RMSNorm` | Root Mean Square Layer Normalization |
| `config.py` | `ModelConfig` / `get_config()` | Model presets (125M, 500M, 1.5B) |
| `quantization.py` | Quantization utilities | INT4/INT8 quantization for CPU inference |
| `speculative.py` | Speculative decoding helpers | Draft-verify acceleration |
| `export.py` | ONNX export utilities | Export for edge deployment |
| `hf_adapter.py` | HuggingFace adapter conversion | PEFT → HBLLM state_dict |
| `model_loader.py` | Checkpoint loading utilities | Safe loading with `weights_only=True` |

---

## HBLLMForCausalLM

The main model class used for training and inference.

```python
from hbllm.model.config import get_config
from hbllm.model.transformer import HBLLMForCausalLM

config = get_config("125m")
model = HBLLMForCausalLM(config)
```

### Forward Pass

```python
output = model(
    input_ids=tokens,          # [batch, seq_len]
    labels=labels,             # Optional: for computing CE loss
    attention_mask=mask,       # Optional: causal mask
    past_key_values=kv_cache,  # Optional: KV cache
    use_cache=True,            # Return updated cache
)

logits = output["logits"]       # [batch, seq_len, vocab_size]
loss = output.get("loss")       # Cross-entropy + MoE aux loss
ce_loss = output.get("ce_loss") # Pure cross-entropy
lb_loss = output.get("lb_loss") # MoE load balancing loss
```

### Text Generation

```python
generated = model.generate(
    input_ids=prompt_ids,      # [batch, prompt_len]
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    eos_token_id=tokenizer.eos_id,
)
```

### Speculative Decoding

Accelerated generation using a smaller draft model:

```python
draft_config = get_config("125m")
draft_model = HBLLMForCausalLM(draft_config)

# 2-4× faster generation
output = model.generate_speculative(
    input_ids=prompt_ids,
    draft_model=draft_model,
    gamma=4,                   # Draft tokens per step
    adaptive_gamma=True,       # EWMA-based dynamic adjustment
    max_new_tokens=200,
)
```

### LoRA Adapter Loading

```python
import torch
state_dict = torch.load("adapter.pt", weights_only=True)
model.load_lora_adapter(state_dict, r=8, lora_alpha=16.0)
model.set_lora_active(True)
```

---

## Grouped Query Attention

**Module:** `hbllm.model.attention.GroupedQueryAttention`

Features:

- **GQA**: Multiple query heads share fewer KV heads → smaller KV cache
- **RoPE**: Rotary position embeddings for relative position encoding
- **Sliding Window**: Configurable window size for O(1) VRAM scaling with long contexts
- **Attention Sinks**: Keeps first N tokens for stability in long sequences
- **SDPA**: Uses PyTorch's `scaled_dot_product_attention` for automatic FlashAttention-2 dispatch

| Config Parameter | Default | Description |
|---|---|---|
| `num_attention_heads` | 12 (125M) | Query heads |
| `num_kv_heads` | 4 (125M) | Key-value heads (GQA ratio) |
| `head_dim` | 64 | Per-head dimension |
| `sliding_window` | 4096 | Sliding window size (0 = disabled) |
| `attention_sinks` | 4 | Number of sink tokens to always retain |

---

## HBLLMForProcessReward

Scores intermediate reasoning steps for MCTS-based planning:

```python
from hbllm.model.transformer import HBLLMForProcessReward

prm = HBLLMForProcessReward(config)
output = prm(input_ids=step_tokens, labels=quality_labels)

scores = output["scores"]  # [batch, 1] in [0.0, 1.0]
loss = output.get("loss")  # BCE + MoE aux loss
```

---

## Model Presets

| Preset | Params | Layers | Hidden | Heads | KV Heads | Est. Size (BF16) |
|---|---|---|---|---|---|---|
| `125m` | ~125M | 12 | 768 | 12 | 4 | ~0.25 GB |
| `500m` | ~500M | 24 | 1024 | 16 | 4 | ~1 GB |
| `1.5b` | ~1.5B | 32 | 2048 | 32 | 8 | ~3 GB |
