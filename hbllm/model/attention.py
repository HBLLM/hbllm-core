"""
Grouped Query Attention (GQA) with Rotary Position Embeddings.

GQA uses fewer key-value heads than query heads, reducing KV cache memory
while maintaining quality. This is the attention mechanism used by LLaMA 2/3,
Mistral, and other modern LLMs.

When num_kv_heads == num_attention_heads → standard Multi-Head Attention
When num_kv_heads == 1 → Multi-Query Attention
When 1 < num_kv_heads < num_attention_heads → Grouped Query Attention
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hbllm.model.config import ModelConfig
from hbllm.model.embeddings import RotaryEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Multiple query heads share a single key-value head group.
    This reduces KV cache size by a factor of (num_heads / num_kv_heads)
    with minimal quality loss.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_ids: [batch_size, seq_len]
            attention_mask: [batch_size, 1, seq_len, kv_seq_len]
            past_key_value: Cached (key, value) from previous forward passes
            use_cache: Whether to return updated key-value cache

        Returns:
            (output, past_key_value) where output is [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache for autoregressive generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        new_past_key_value = (key_states, value_states) if use_cache else None

        # Expand KV heads to match query heads (GQA)
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax + dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back to [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, new_past_key_value

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads.

        [batch, num_kv_heads, seq_len, head_dim]
        → [batch, num_heads, seq_len, head_dim]
        """
        if self.num_kv_groups == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(batch, self.num_heads, seq_len, head_dim)
