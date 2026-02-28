"""
Model configuration — defines architecture hyperparameters.

Supports loading from YAML config files and provides presets
for 125M, 500M, and 1.5B parameter configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for the HBLLM transformer model."""

    # Model identity
    name: str = "hbllm-125m"

    # Core architecture
    vocab_size: int = 32768
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 4  # For Grouped Query Attention
    intermediate_size: int = 3072  # SwiGLU FFN dimension
    max_position_embeddings: int = 2048

    # Normalization
    rms_norm_eps: float = 1e-5

    # Embeddings
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0

    # Dropout
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Initialization
    initializer_range: float = 0.02

    # MoE (Phase 3) — disabled by default
    use_moe: bool = False
    num_experts: int = 16
    num_active_experts: int = 2
    use_shared_expert: bool = True

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_size

        # Per layer: attention + FFN + norms
        attn_params = (
            self.hidden_size * self.hidden_size  # Q
            + self.hidden_size * (self.head_dim * self.num_kv_heads)  # K
            + self.hidden_size * (self.head_dim * self.num_kv_heads)  # V
            + self.hidden_size * self.hidden_size  # O
        )
        ffn_params = 3 * self.hidden_size * self.intermediate_size  # SwiGLU has 3 matrices
        norm_params = 2 * self.hidden_size  # 2 RMSNorms per layer
        layer_params = attn_params + ffn_params + norm_params

        total = embed_params + (self.num_layers * layer_params) + self.hidden_size  # final norm

        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_size  # output projection

        return total

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        model_cfg = raw.get("model", raw)
        return cls(**{k: v for k, v in model_cfg.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ----- Presets -----

CONFIGS = {
    "125m": ModelConfig(
        name="hbllm-125m",
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=4,
        intermediate_size=3072,
        max_position_embeddings=2048,
    ),
    "500m": ModelConfig(
        name="hbllm-500m",
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=4096,
        max_position_embeddings=4096,
    ),
    "1.5b": ModelConfig(
        name="hbllm-1.5b",
        hidden_size=2048,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=8192,
        max_position_embeddings=4096,
    ),
}


def get_config(size: str = "125m") -> ModelConfig:
    """Get a preset model config by size."""
    if size not in CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[size]
