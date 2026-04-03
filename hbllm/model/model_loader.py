"""
Unified Model Loader — single entry point for loading any model into HBLLM.

Supports:
- Native HBLLM models by size preset: "125m", "500m", "1.5b", "7b", "13b"
- HuggingFace models by name: "meta-llama/Llama-3.2-8B", "mistralai/Mistral-7B-v0.1"
- Local HuggingFace models by path: "/path/to/my/model"

Usage::

    from hbllm.model.model_loader import load_model

    # Native HBLLM model
    model = load_model("125m")

    # HuggingFace model (auto-downloads)
    model = load_model("gpt2")
    model = load_model("meta-llama/Llama-3.2-8B", load_in_4bit=True)

    # Local model
    model = load_model("/models/my-finetuned-llama/", device="cuda:0")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

# Native HBLLM model presets
_NATIVE_PRESETS = {"125m", "500m", "1.5b", "7b", "13b"}


def load_model(
    source: str,
    device: str = "auto",
    dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    max_memory: dict[str, str] | None = None,
    attn_implementation: str | None = None,
) -> nn.Module:
    """
    Load a model for use with the HBLLM cognitive pipeline.

    Args:
        source: Model identifier. Can be:
            - A native preset: "125m", "500m", "1.5b", "7b", "13b"
            - A HuggingFace model name: "gpt2", "meta-llama/Llama-3.2-8B"
            - A local path: "/path/to/model"
        device: Device to load on ("auto", "cpu", "cuda:0", etc.)
        dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        trust_remote_code: Trust remote code for custom model architectures
        max_memory: Device memory limits, e.g. {"0": "6GiB", "cpu": "12GiB"}
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa")

    Returns:
        A model with the HBLLM interface:
            - forward(input_ids, ...) → dict with "logits"
            - generate(input_ids, ...) → Tensor
            - Works with LoRAManager, Trainer, DPO, etc.
    """
    source_lower = source.lower().strip()

    # Check if it's a native HBLLM preset
    if source_lower in _NATIVE_PRESETS:
        return _load_native(source_lower, device)

    # Check if it's a local path
    if Path(source).is_dir():
        return _load_huggingface(
            source, device, dtype, load_in_8bit, load_in_4bit,
            trust_remote_code, max_memory, attn_implementation
        )

    # Assume it's a HuggingFace model name
    return _load_huggingface(
        source, device, dtype, load_in_8bit, load_in_4bit,
        trust_remote_code, max_memory, attn_implementation
    )


def _load_native(size: str, device: str) -> nn.Module:
    """Load a native HBLLM model by preset size."""
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM

    config = get_config(size)
    model = HBLLMForCausalLM(config)

    if device != "auto" and device != "cpu":
        model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Loaded native HBLLM model '%s': %.1fM parameters", size, param_count / 1e6)

    return model


def _load_huggingface(
    model_name_or_path: str,
    device: str,
    dtype: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
    trust_remote_code: bool,
    max_memory: dict[str, str] | None,
    attn_implementation: str | None,
) -> nn.Module:
    """Load a HuggingFace model via the adapter."""
    from hbllm.model.hf_adapter import HuggingFaceModelAdapter

    return HuggingFaceModelAdapter(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
        max_memory=max_memory,
        attn_implementation=attn_implementation,
    )


def list_available_models() -> dict[str, dict[str, Any]]:
    """List all available native model presets with their estimated sizes."""
    from hbllm.model.config import CONFIGS

    result = {}
    for name, config in CONFIGS.items():
        result[name] = {
            "name": config.name,
            "params_estimate": f"{config.num_params_estimate / 1e6:.0f}M",
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
        }
    return result
