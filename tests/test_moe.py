import torch
import torch.nn as nn
import pytest

from hbllm.model.config import ModelConfig
from hbllm.model.feedforward import MoEFFN

@pytest.fixture
def config():
    cfg = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        num_active_experts=2,
        use_moe=True,
        use_shared_expert=True
    )
    return cfg

def test_moe_ffn_forward(config):
    # Initialize MoE FFN
    moe = MoEFFN(config)
    
    # Create dummy input [batch_size, seq_len, hidden_size]
    batch_size = 2
    seq_len = 5
    hidden_size = config.hidden_size
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Run forward pass
    output, load_balancing_loss = moe(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    # Check load balancing loss is scalar
    assert load_balancing_loss.numel() == 1
    assert load_balancing_loss.item() >= 0.0

def test_moe_ffn_routing(config):
    moe = MoEFFN(config)
    
    # Force the router to have predictable weights
    # Make expert 0 very likely for all tokens
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0, :] = 100.0  # Expert 0 dominates
        moe.gate.weight[1, :] = 50.0   # Expert 1 is second
        
    x = torch.randn(1, 10, config.hidden_size)
    
    output, lb_loss = moe(x)
    
    # With deterministic routing, the load balancing loss should be heavily skewed.
    # We just want to ensure it calculates without error.
    assert lb_loss.item() > 0.0

def test_moe_no_shared_expert(config):
    config.use_shared_expert = False
    moe = MoEFFN(config)
    
    assert not hasattr(moe, 'shared_expert')
    
    x = torch.randn(2, 5, config.hidden_size)
    output, lb_loss = moe(x)
    
    assert output.shape == (2, 5, config.hidden_size)
