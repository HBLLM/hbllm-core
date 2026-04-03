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

import torch
import torch.nn as nn
import torch.nn.functional as F

from hbllm.model.attention import GroupedQueryAttention
from hbllm.model.config import ModelConfig
from hbllm.model.embeddings import TokenEmbedding
from hbllm.model.feedforward import MoEFFN, SwiGLUFFN
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
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
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

        # Automatic Hybrid Quantization Injection (Phase 4)
        if getattr(config, "quantization_level", 16) in [4, 8]:
            from hbllm.modules.lora import LoRAManager
            LoRAManager.inject(
                self,
                quantization_level=config.quantization_level,
                r=getattr(config, "lora_rank", 8)
            )

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None, torch.Tensor]:
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
            if past_key_values:
                if hasattr(past_key_values[0], "seq_len"):
                    past_len = past_key_values[0].seq_len
                else:
                    past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0

            position_ids = torch.arange(
                past_len, past_len + seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Build causal attention mask
        if attention_mask is None:
            past_len = 0
            if past_key_values is not None:
                if hasattr(past_key_values[0], "seq_len"):
                    past_len = past_key_values[0].seq_len
                else:
                    past_len = past_key_values[0][0].shape[2]

            if past_len == 0:
                attention_mask = self._build_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
            else:
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

    def load_lora_adapter(
        self,
        state_dict: dict[str, torch.Tensor],
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        """
        Dynamically injects a LoRA adapter into the model and loads weights.

        Args:
            state_dict: The LoRA Weights
            r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability
            target_modules: List of module name suffixes to apply LoRA (e.g. ['q_proj', 'v_proj']).
                            If None, targets self-attention and FFN.
        """
        import logging

        from hbllm.modules.lora import LoRAManager
        logger = logging.getLogger(__name__)

        # Check if LoRA is already injected; if not, inject it.
        # This assumes we swap standard Linear with LoRALinear.
        has_lora = any("lora_A" in name for name, _ in self.named_parameters())
        if not has_lora:
            logger.info("Injecting new LoRA adapters (r=%d)...", r)
            LoRAManager.inject(
                self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules
            )

        # Load the state into the 'default' adapter slot so the LocalProvider can trigger it globally
        logger.info("Loading LoRA state dict into default Multi-LoRA slot...")
        LoRAManager.add_adapter(self, adapter_name="default", state_dict=state_dict)

    def set_lora_active(self, active: bool = True) -> None:
        """Toggle LoRA adapters on or off for inference."""
        from hbllm.modules.lora import LoRAManager
        LoRAManager.set_active_adapter(self, "default" if active else None)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
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

            # Sample using helper
            next_token = self._sample_logits(logits, temperature, top_k, top_p)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def _sample_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Helper to sample a single token from logits."""
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-7)

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

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

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    class AdaptiveGammaController:
        """EWMA-based controller that dynamically adjusts draft depth (gamma)."""

        def __init__(self, gamma_min: int = 1, gamma_max: int = 8, ewma_alpha: float = 0.3):
            self.gamma_min = gamma_min
            self.gamma_max = gamma_max
            self.ewma_alpha = ewma_alpha
            self.acceptance_rate: float = 0.5  # initial estimate

        def step(self, accepted: int, total: int) -> int:
            """Update EWMA and return the next gamma value."""
            if total > 0:
                batch_rate = accepted / total
                self.acceptance_rate = (
                    self.ewma_alpha * batch_rate + (1 - self.ewma_alpha) * self.acceptance_rate
                )
            return max(self.gamma_min, min(self.gamma_max, round(self.acceptance_rate * self.gamma_max)))

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        draft_model: HBLLMForCausalLM,
        gamma: int = 4,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        adaptive_gamma: bool = False,
        gamma_min: int = 1,
        gamma_max: int = 8,
        ewma_alpha: float = 0.3,
    ) -> torch.Tensor:
        """
        Accelerated autoregressive generation using speculative decoding.

        Args:
            input_ids: [batch_size, prompt_len] initial token IDs
            draft_model: Smaller HBLLMForCausalLM instance for drafting
            gamma: Number of tokens the draft model predicts per step (static mode)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop generation at this token
            adaptive_gamma: If True, dynamically adjust gamma based on acceptance rate
            gamma_min: Minimum gamma when adaptive (floor)
            gamma_max: Maximum gamma when adaptive (ceiling)
            ewma_alpha: EWMA smoothing factor for acceptance rate tracking

        Returns:
            [batch_size, prompt_len + generated_len] full sequence
        """
        self.eval()
        draft_model.eval()

        controller = None
        if adaptive_gamma:
            controller = self.AdaptiveGammaController(
                gamma_min=gamma_min, gamma_max=gamma_max, ewma_alpha=ewma_alpha
            )
            gamma = controller.step(0, 0)  # initial gamma from default acceptance_rate

        target_past_key_values = None
        draft_past_key_values = None

        n_generated = 0
        while n_generated < max_new_tokens:
            current_gamma = gamma
            # --- Drafting Phase ---
            draft_input_ids = input_ids
            draft_tokens = []

            for _ in range(current_gamma):
                if draft_past_key_values is not None:
                    model_input_draft = draft_input_ids[:, -1:]
                else:
                    model_input_draft = draft_input_ids

                draft_outputs = draft_model(
                    input_ids=model_input_draft,
                    past_key_values=draft_past_key_values,
                    use_cache=True,
                )
                draft_logits = draft_outputs["logits"][:, -1, :]
                draft_past_key_values = draft_outputs.get("past_key_values")

                next_draft_token = self._sample_logits(draft_logits, temperature, top_k, top_p)
                draft_tokens.append(next_draft_token)
                draft_input_ids = torch.cat([draft_input_ids, next_draft_token], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1) # [batch_size, current_gamma]

            # --- Verification Phase ---
            if target_past_key_values is not None:
                # Target model digests the last accepted token + the new draft tokens
                target_input = torch.cat([input_ids[:, -1:], draft_tokens_tensor], dim=1)
            else:
                target_input = torch.cat([input_ids, draft_tokens_tensor], dim=1)

            target_outputs = self.forward(
                input_ids=target_input,
                past_key_values=target_past_key_values,
                use_cache=True,
            )
            target_past_key_values = target_outputs.get("past_key_values")

            # Extract logits predicting the draft tokens and the bonus token
            eval_logits = target_outputs["logits"][:, -(current_gamma + 1):, :] # [batch_size, current_gamma + 1, vocab]

            # --- Rejection/Acceptance ---
            accepted_count = 0
            for i in range(current_gamma):
                target_token = self._sample_logits(eval_logits[:, i, :], temperature, top_k, top_p)

                # Check for exact match across batches
                if (target_token == draft_tokens_tensor[:, i:i+1]).all():
                    accepted_count += 1
                    input_ids = torch.cat([input_ids, target_token], dim=1)
                    n_generated += 1
                    if eos_token_id is not None and (target_token == eos_token_id).all():
                        return input_ids
                    if n_generated >= max_new_tokens:
                        return input_ids
                else:
                    # Mismatch! Reject this and all following draft tokens
                    # Append the correct target token
                    input_ids = torch.cat([input_ids, target_token], dim=1)
                    n_generated += 1
                    break
            else:
                # If all draft tokens were accepted, we get a bonus token from the last logit
                bonus_token = self._sample_logits(eval_logits[:, current_gamma, :], temperature, top_k, top_p)
                input_ids = torch.cat([input_ids, bonus_token], dim=1)
                n_generated += 1
                accepted_count += 1

            if eos_token_id is not None and (input_ids[:, -1:] == eos_token_id).all():
                return input_ids

            if n_generated >= max_new_tokens:
                return input_ids

            # --- KV Cache Synchronization ---
            # Truncate both caches to the correctly accepted length before the bonus/reject token
            # Since input_ids was extended by (accepted_count + 1) tokens, the accepted sequence
            # (excluding the final newly appended token to be processed next) has length:
            cache_keep_len = input_ids.shape[1] - 1

            def truncate_kv_cache(pkv, keep_len):
                if pkv is None: return None
                new_pkv = []
                for layer_idx in range(len(pkv)):
                    k, v = pkv[layer_idx]
                    new_pkv.append((k[:, :, :keep_len, :], v[:, :, :keep_len, :]))
                return new_pkv

            target_past_key_values = truncate_kv_cache(target_past_key_values, cache_keep_len)
            draft_past_key_values = truncate_kv_cache(draft_past_key_values, cache_keep_len)

            # --- Adaptive Gamma Update ---
            if controller is not None:
                gamma = controller.step(accepted_count, current_gamma)

        return input_ids

class HBLLMForProcessReward(nn.Module):
    """
    HBLLM with a sequence classification head for Process Reward Modeling.

    Scores an entire reasoning step. By default, it takes the representation of
    the last non-padding token and projects it to a continuous scalar.
    Used for step-level MCTS UCT scoring (0.0 means incorrect, 1.0 means correct).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = HBLLMModel(config)

        # Single logit output for binary correctness estimation
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

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
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the Process Reward Model.

        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size] continuous labels in [0.0, 1.0] to compute BCE loss

        Returns:
            Dict with 'logits', 'scores' (sigmoid of logits), optionally 'loss'
        """
        batch_size = input_ids.shape[0]

        # Run base transformer
        hidden_states, _, lb_loss = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # If user provides a 2D mask (1=valid, 0=pad), we extract the last valid token representation
        # Otherwise, default to the last token sequence index
        if attention_mask is not None and attention_mask.dim() == 2:
            sequence_lengths = attention_mask.sum(dim=1).long() - 1
            pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        else:
            pooled_hidden_states = hidden_states[:, -1, :]

        # Linear projection to single logit
        logits = self.score(pooled_hidden_states)

        # Map logit to [0, 1] probability
        scores = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "scores": scores,
        }

        # Compute BCE loss if labels are provided
        if labels is not None:
            # labels shape: [batch_size], logits shape: [batch_size, 1]
            loss_fct = nn.BCEWithLogitsLoss()

            # Ensure proper dtypes and shapes
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)

            bce_loss = loss_fct(logits.float(), labels.float())

            # Add MoE auxiliary loss if present
            loss = bce_loss + (0.01 * lb_loss)

            result["loss"] = loss
            result["bce_loss"] = bce_loss
            result["lb_loss"] = lb_loss

        return result

