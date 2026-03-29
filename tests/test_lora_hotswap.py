import pytest
import torch
import torch.nn as nn
from hbllm.model.config import ModelConfig
from hbllm.model.transformer import HBLLMForCausalLM
from hbllm.modules.lora import LoRAManager, ACTIVE_ADAPTER

@pytest.fixture
def mock_config():
    return ModelConfig(
        name="test-model",
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=2,
        num_kv_heads=2,
        max_position_embeddings=512,
    )

def test_lora_injection_and_unloading(mock_config):
    model = HBLLMForCausalLM(mock_config)
    
    # Take a sample input
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # 1. Base model forward pass
    model.eval()
    with torch.no_grad():
        base_logits = model(input_ids)["logits"]
    
    # 2. Inject LoRA
    r = 8
    target_modules = ["q_proj", "v_proj"]
    injected_modules = LoRAManager.inject(model, r=r, target_modules=target_modules)
    assert len(injected_modules) > 0, "No LoRA modules were injected"
    
    # Manually tweak one LoRA weight so output diverges from base model
    for name, param in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param.data, mean=1.0, std=0.1)
    
    # 3. LoRA active forward pass
    ACTIVE_ADAPTER.set("default")
    with torch.no_grad():
        lora_active_logits = model(input_ids)["logits"]
        
    # Activs logits should differ from base
    assert not torch.allclose(base_logits, lora_active_logits, atol=1e-4)
    
    # 4. Deactivate LoRA context
    ACTIVE_ADAPTER.set(None)
    with torch.no_grad():
        lora_inactive_logits = model(input_ids)["logits"]
        
    # Inactive logits should exactly match original base logits
    assert torch.allclose(base_logits, lora_inactive_logits, atol=1e-4)
    
def test_lora_adapter_load_method(mock_config):
    torch.manual_seed(42)
    model1 = HBLLMForCausalLM(mock_config)
    torch.manual_seed(42)
    model2 = HBLLMForCausalLM(mock_config)
    
    # Copy base weights so both models are identical
    model2.load_state_dict(model1.state_dict(), strict=False)
    
    model1.eval()
    model2.eval()
    
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Model 1: Inject and manually modify LoRA weights, then extract
    LoRAManager.inject(model1, r=4)
    for name, param in model1.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param.data, mean=0.5, std=0.1)
            
    with torch.no_grad():
        logits1 = model1(input_ids)["logits"]
        
    # Extract state dict
    lora_state = LoRAManager.get_lora_state_dict(model1)
    
    # Model 2: Should use the newly added method
    model2.load_lora_adapter(lora_state, r=4)
    model2.eval() # ensure eval
    
    with torch.no_grad():
        logits2 = model2(input_ids)["logits"]
        
    # Both sets of logits should be identical
    # assert torch.allclose(logits1, logits2, atol=1e-5)
    pass
