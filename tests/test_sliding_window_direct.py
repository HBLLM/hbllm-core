
import torch
from hbllm.model.config import ModelConfig
from hbllm.model.transformer import HBLLMForCausalLM

def test_sliding_window_vram_constant():
    # Use a small model for testing
    config = ModelConfig(
        name="test-swa-8",
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        sliding_window=8,  # Tiny window for testing
        attention_sinks=2,
    )
    
    model = HBLLMForCausalLM(config)
    model.eval()
    
    # Force mock tokenizer or just use IDs
    prompt = torch.randint(0, 32768, (1, 5))
    
    # 1. Initial forward pass
    print("Initial pass (5 tokens)...")
    outputs = model(prompt, use_cache=True)
    pkv = outputs["past_key_values"]
    
    # 2. Add tokens up to the window (8)
    print("Growing to window size (8 tokens)...")
    next_token = torch.randint(0, 32768, (1, 1))
    for _ in range(3):
        outputs = model(next_token, past_key_values=pkv, use_cache=True)
        pkv = outputs["past_key_values"]
        
    current_len = pkv[0][0].shape[2]
    print(f"Current KV cache length: {current_len}")
    assert current_len == 8
    
    # 3. Add tokens BEYOND the window
    print("Going beyond window (to 12 tokens)...")
    for _ in range(4):
        outputs = model(next_token, past_key_values=pkv, use_cache=True)
        pkv = outputs["past_key_values"]
        
    final_len = pkv[0][0].shape[2]
    print(f"Final KV cache length: {final_len}")
    
    # Check that length stayed at 8
    assert final_len == 8, f"Expected cache length 8, got {final_len}"
    print("SUCCESS: VRAM (KV Cache) stayed constant beyond sliding window.")

if __name__ == "__main__":
    test_sliding_window_vram_constant()
