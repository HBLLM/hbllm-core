import torch
import pytest

from hbllm.training.dpo import compute_dpo_loss, get_batch_logps

def test_dpo_loss_computation():
    batch_size = 4
    
    # Simulate log probabilities for chosen and rejected responses
    # Policy model (what we're training)
    pi_chosen = torch.randn(batch_size, requires_grad=True)
    pi_rejected = torch.randn(batch_size, requires_grad=True)
    
    # Reference model (frozen blueprint)
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)
    
    beta = 0.1
    
    losses, chosen_rewards, rejected_rewards = compute_dpo_loss(
        policy_chosen_logps=pi_chosen,
        policy_rejected_logps=pi_rejected,
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=beta
    )
    
    # Check shapes
    assert losses.shape == (batch_size,)
    assert chosen_rewards.shape == (batch_size,)
    assert rejected_rewards.shape == (batch_size,)
    
    # DPO loss should be positive (it's a negative log sigmoid)
    assert (losses >= 0).all()
    
    # Ensure gradients can flow back to the policy log probabilities
    loss = losses.mean()
    loss.backward()
    
    assert pi_chosen.grad is not None
    assert pi_rejected.grad is not None
    
    # According to DPO formulation, taking a gradient step should:
    # 1. Increase pi_chosen (gradient is negative w.r.t loss)
    # 2. Decrease pi_rejected (gradient is positive w.r.t loss)
    assert (pi_chosen.grad < 0).all()
    assert (pi_rejected.grad > 0).all()

def test_get_batch_logps():
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Set some labels to ignore index
    labels[0, 3:] = -100
    labels[1, 4:] = -100
    
    logps = get_batch_logps(logits, labels, ignore_index=-100)
    
    # Should return a scalar per element in batch
    assert logps.shape == (batch_size,)
    
    # We can't guarantee logps are positive/negative without specific values, 
    # but we can ensure they are finite real numbers
    assert torch.isfinite(logps).all()
