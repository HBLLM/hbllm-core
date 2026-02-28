"""
HBLLM Transformer — the full model assembled from components.

Decoder-only transformer with:
- Token embeddings
- N transformer decoder layers (GQA + SwiGLU + RMSNorm)
- Final RMSNorm + language model head

Architecture follows LLaMA 3 / Mistral design:
- Pre-norm (RMSNorm before attention and FFN)
- Rotary position embeddings (applied in attention)
- Grouped Query Attention
- SwiGLU feed-forward network
- Optional tied embeddings
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hbllm.model.attention import GroupedQueryAttention
from hbllm.model.config import ModelConfig
from hbllm.model.embeddings import TokenEmbedding
from hbllm.model.feedforward import SwiGLUFFN, MoEFFN
from hbllm.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Architecture (pre-norm):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention and FFN
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        if getattr(config, "use_moe", False):
            self.mlp = MoEFFN(config)
        else:
            self.mlp = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass through one transformer block.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_ids: [batch_size, seq_len]
            attention_mask: Causal mask
            past_key_value: KV cache for this layer
            use_cache: Whether to return updated cache

        Returns:
            (hidden_states, past_key_value, load_balancing_loss)
        """
        # Self-attention with residual connection (pre-norm)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # FFN with residual connection (pre-norm)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        load_balancing_loss = torch.tensor(0.0, device=hidden_states.device)
        if isinstance(self.mlp, MoEFFN):
            hidden_states, load_balancing_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            
        hidden_states = residual + hidden_states

        return hidden_states, new_kv, load_balancing_loss


class HBLLMModel(nn.Module):
    """
    HBLLM base model (no language model head).

    Produces hidden states from token IDs.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = TokenEmbedding(
            config.vocab_size, config.hidden_size, config.initializer_range
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        """
        Forward pass through the base model.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            position_ids: [batch_size, seq_len] position indices
            attention_mask: [batch_size, seq_len] or pre-built 4D mask
            past_key_values: KV cache from previous forward passes
            use_cache: Whether to return KV cache

        Returns:
            (hidden_states, past_key_values, total_load_balancing_loss)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Build causal attention mask
        if attention_mask is None:
            attention_mask = self._build_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
            if past_key_values is not None:
                past_len = past_key_values[0][0].shape[2]
                attention_mask = self._build_causal_mask(
                    seq_len, hidden_states.device, hidden_states.dtype,
                    past_key_values_length=past_len,
                )

        # Forward through all layers
        new_past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None
        total_load_balancing_loss = torch.tensor(0.0, device=hidden_states.device)

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv, layer_lb_loss = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            total_load_balancing_loss += layer_lb_loss
            if use_cache:
                new_past_key_values.append(new_kv)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_past_key_values, total_load_balancing_loss

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Build a causal attention mask.

        Returns a 4D mask [1, 1, seq_len, total_len] where masked positions
        are filled with -inf and unmasked positions are 0.
        """
        total_len = seq_len + past_key_values_length

        # Upper triangular mask (future tokens masked)
        mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=past_key_values_length + 1)

        # Reshape to [1, 1, seq_len, total_len] for broadcasting
        return mask.unsqueeze(0).unsqueeze(0)


class HBLLMForCausalLM(nn.Module):
    """
    HBLLM with causal language model head for text generation.

    This is the main model class used for training and inference.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = HBLLMModel(config)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"HBLLM [{config.name}] initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Estimated size: {total_params * 2 / 1e9:.2f} GB (bf16)")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using small init."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] for computing cross-entropy loss
            position_ids: [batch_size, seq_len]
            attention_mask: Causal mask
            past_key_values: KV cache
            use_cache: Whether to return KV cache

        Returns:
            Dict with 'logits', optionally 'loss' and 'past_key_values'
        """
        hidden_states, new_past_key_values, lb_loss = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        result = {"logits": logits}

        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Add aux loss: empirically alpha=0.01 or 0.02 is used
            loss = ce_loss + (0.01 * lb_loss)
            result["loss"] = loss
            result["ce_loss"] = ce_loss
            result["lb_loss"] = lb_loss

        if use_cache:
            result["past_key_values"] = new_past_key_values

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with top-k/top-p sampling.

        Args:
            input_ids: [batch_size, prompt_len] initial token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold
            eos_token_id: Stop generation at this token

        Returns:
            [batch_size, prompt_len + generated_len] full sequence
        """
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass (with KV cache for efficiency)
            if past_key_values is not None:
                # Only need to process the last token
                model_input = input_ids[:, -1:]
            else:
                model_input = input_ids

            outputs = self.forward(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]  # Last token logits
            past_key_values = outputs.get("past_key_values")

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
