import torch
from hbllm.model.config import ModelConfig
from hbllm.model.transformer import HBLLMModel
from hbllm.modules.hardware_hal import HardwareHAL
from hbllm.model.quantization import HybridLinear

def verify_hal():
    print("--- Testing Hardware HAL ---")
    profile = HardwareHAL.get_profile()
    print(f"Device: {profile.device_type}")
    print(f"Arch: {profile.arch}")
    print(f"RAM: {profile.total_ram_gb:.2f} GB")
    
    # Recommend for a 13B model
    recommendation = HardwareHAL.recommend_policy(13.0)
    print(f"Recommendation for 13B: {recommendation}")
    return recommendation

def verify_hybrid_model(policy):
    print("\n--- Testing Hybrid Model Initialization ---")
    # Small test config
    config = ModelConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        num_kv_heads=2,
        intermediate_size=512,
        quantization_level=policy["quantization"].value
    )
    
    model = HBLLMModel(config)
    
    # Check for HybridLinear
    hybrid_count = 0
    for name, module in model.named_modules():
        if isinstance(module, HybridLinear):
            hybrid_count += 1
            if hybrid_count == 1:
                print(f"Verified: Found HybridLinear at '{name}' with {module.base.bits}-bit base.")
                
    print(f"Total Hybrid layers injected: {hybrid_count}")
    
    if hybrid_count > 0:
        print("SUCCESS: Hybrid Quantization system is active.")
    else:
        print("FAILURE: No Hybrid layers found.")

if __name__ == "__main__":
    policy = verify_hal()
    verify_hybrid_model(policy)
