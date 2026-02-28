"""
SwiGLU Feed-Forward Network.

SwiGLU replaces standard ReLU/GELU FFN with a gated mechanism:
  FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down

This gives ~1% improvement over standard FFN at the same parameter count.
Used in LLaMA, PaLM, and other modern LLMs.

Note: SwiGLU has 3 weight matrices instead of 2 (gate, up, down),
so intermediate_size should be 2/3 of what a standard FFN would use
for the same total parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hbllm.model.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Architecture:
        gate = Swish(x @ W_gate)
        up = x @ W_up
        output = (gate ⊙ up) @ W_down

    Where Swish(x) = x * sigmoid(x) = SiLU(x)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, seq_len, hidden_size]

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        output = self.down_proj(hidden)
        output = self.dropout(output)
        return output

class MoEFFN(nn.Module):
    """
    Mixture of Experts Feed-Forward Network.

    Routes each token to top-k experts out of N available.
    Also computes a load balancing loss to prevent expert collapse.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_active_experts
        
        # Gate network to predict routing probabilities
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # The experts (parallel SwiGLU FFNs)
        self.experts = nn.ModuleList([SwiGLUFFN(config) for _ in range(self.num_experts)])
        
        # Shared expert routing (always active for all tokens)
        self.use_shared_expert = getattr(config, "use_shared_expert", False)
        if self.use_shared_expert:
            self.shared_expert = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
            load_balancing_loss: scalar tensor
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # [batch * seq_len, hidden_size]

        # 1. Routing / Gating
        router_logits = self.gate(x_flat)  # [batch * seq_len, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts for each token
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_active_experts, dim=-1)
        
        # Normalize weights for the selected experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        # 2. Compute Load Balancing Loss
        # f_i (fraction of tokens routed to expert i)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1)  # [batch * seq_len, num_experts]
        tokens_per_expert = expert_mask.float().mean(dim=0)  # [num_experts]
        
        # P_i (average routing probability of expert i across all tokens)
        router_prob_per_expert = F.softmax(router_logits, dim=-1).mean(dim=0)  # [num_experts]
        
        # auxiliary load balancing loss (alpha * N * sum(f_i * P_i))
        load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

        # 3. Apply Experts
        final_hidden_states = torch.zeros_like(x_flat)

        # For efficiency, iterate through experts and process their assigned tokens
        for i, expert in enumerate(self.experts):
            # Find which tokens were routed to this expert
            idx, nth_expert = torch.where(selected_experts == i)

            if idx.numel() > 0:
                expert_inputs = x_flat[idx]
                expert_outputs = expert(expert_inputs)
                
                # Apply routing weight
                expert_weights = routing_weights[idx, nth_expert].unsqueeze(-1)
                weighted_outputs = expert_outputs * expert_weights
                
                # Accumulate back into final hidden states
                final_hidden_states.index_add_(0, idx, weighted_outputs)

        # 4. Optional Shared Expert
        if self.use_shared_expert:
            shared_output = self.shared_expert(x_flat)
            final_hidden_states = final_hidden_states + shared_output

        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)

        return final_hidden_states, load_balancing_loss
