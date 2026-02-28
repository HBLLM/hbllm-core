import torch
import torch.nn as nn

from hbllm.modules.lora import LoRALinear, LoRAManager


def test_lora_linear_forward():
    """Test that LoRALinear computes correctly."""
    batch_size, seq_len, in_dim = 2, 4, 16
    out_dim = 8
    
    # Base layer
    base = nn.Linear(in_dim, out_dim)
    
    # Input
    x = torch.randn(batch_size, seq_len, in_dim)
    
    # Base output
    with torch.no_grad():
        base_out = base(x)
        
    # Wrap in LoRA (r=4, alpha=8)
    lora = LoRALinear(base, r=4, lora_alpha=8.0, lora_dropout=0.0)
    
    # Check dimensions
    assert lora.lora_A.shape == (4, in_dim)
    assert lora.lora_B.shape == (out_dim, 4)
    assert lora.scaling == 2.0  # 8.0 / 4
    
    # Since B is initialized to 0, initial forward should equal base_out
    with torch.no_grad():
        lora_out = lora(x)
        
    assert torch.allclose(base_out, lora_out, atol=1e-6)
    
    # Modify B to ensure LoRA path takes effect
    nn.init.ones_(lora.lora_B)
    with torch.no_grad():
        lora_out_modified = lora(x)
        
    assert not torch.allclose(base_out, lora_out_modified)
    
    # Test disable
    lora.active = False
    with torch.no_grad():
        lora_out_disabled = lora(x)
        
    assert torch.allclose(base_out, lora_out_disabled, atol=1e-6)


def test_lora_manager():
    """Test injecting LoRA into a nested PyTorch module."""
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 16)
            self.q_proj = nn.Linear(16, 16)
            self.nested = nn.ModuleDict({
                "v_proj": nn.Linear(16, 16),
                "other": nn.Linear(16, 16)
            })
            
        def forward(self, x):
            x = self.embed(x)
            x = self.q_proj(x)
            x = self.nested["v_proj"](x)
            x = self.nested["other"](x)
            return x

    model = DummyModel()
    
    # Inject LoRA
    injected = LoRAManager.inject(
        model, 
        r=2, 
        target_modules=["q_proj", "v_proj"]
    )
    
    assert len(injected) == 2
    assert "q_proj" in injected
    assert "nested.v_proj" in injected
    
    # Check types
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.nested["v_proj"], LoRALinear)
    assert isinstance(model.nested["other"], nn.Linear)  # Should not be injected
    
    # Verify state dict extraction
    state = LoRAManager.get_lora_state_dict(model)
    assert len(state) == 4  # q_proj.lora_A, q_proj.lora_B, nested.v_proj.lora_A, nested.v_proj.lora_B
    assert "q_proj.lora_A" in state
    
    # Forward pass
    x = torch.randint(0, 10, (2, 4))
    out = model(x)
    assert out.shape == (2, 4, 16)
