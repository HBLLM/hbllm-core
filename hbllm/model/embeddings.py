"""
Token embeddings with Rotary Position Encoding (RoPE).

RoPE encodes position information directly into the attention computation
via rotation matrices, enabling length extrapolation beyond training context.
Used in LLaMA, Mistral, and most modern LLMs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: float = 0.02):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position as a rotation in 2D subspaces of the embedding dimension.
    Key properties:
    - Relative position encoding (captures distance between tokens)
    - Extrapolates to longer sequences than trained on
    - Applied at the attention level, not added to embeddings

    Reference: Su et al., "RoFormer" (2021)
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos and sin values for positions up to seq_len."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin values for the given positions.

        Args:
            x: Input tensor (used only for dtype/device)
            position_ids: Position indices [batch_size, seq_len]

        Returns:
            (cos, sin) each of shape [batch_size, seq_len, dim]
        """
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine values [batch, seq_len, head_dim] or broadcastable
        sin: Sine values [batch, seq_len, head_dim] or broadcastable

    Returns:
        Rotated (q, k) tensors
    """
    # Reshape cos/sin for broadcasting: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
