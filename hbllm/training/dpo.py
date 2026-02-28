"""
Direct Preference Optimization (DPO) toolkit.

Provides utilities to compute the DPO loss for online continuous learning.
DPO optimizes the policy directly from preferences without an explicit reward model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the DPO loss for a batch of policy and reference log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the chosen responses from the policy model.
        policy_rejected_logps: Log probabilities of the rejected responses from the policy model.
        reference_chosen_logps: Log probabilities of the chosen responses from the reference model.
        reference_rejected_logps: Log probabilities of the rejected responses from the reference model.
        beta: Temperature parameter for the DPO loss (default 0.1).

    Returns:
        losses: The DPO loss for each example in the batch.
        chosen_rewards: Estimated rewards for chosen responses.
        rejected_rewards: Estimated rewards for rejected responses.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    # DPO loss is the negative log sigmoid of the scaled log ratio differences
    losses = -F.logsigmoid(beta * logits)

    # Optional: estimate implicit rewards for monitoring
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute log probabilities of the given labels under the logits.
    
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        
    Returns:
        logps: [batch_size]
    """
    # Shift so that tokens predict the next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities of the actual labels
    # We must mask out ignore_index BEFORE gather to prevent out-of-bounds error
    gather_indices = shift_labels.clone()
    loss_mask = shift_labels != ignore_index
    gather_indices[~loss_mask] = 0
    
    # [batch_size, seq_len_minus_1]
    per_token_logps = torch.gather(log_probs, dim=2, index=gather_indices.unsqueeze(2)).squeeze(2)
    
    # Mask out ignored tokens
    per_token_logps = per_token_logps * loss_mask
    
    # Sum over sequence length
    return per_token_logps.sum(dim=-1)
