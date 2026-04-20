from hbllm.model.config import ModelConfig
from hbllm.model.quantization import HybridLinear
from hbllm.model.transformer import HBLLMModel
from hbllm.modules.hardware_hal import HardwareHAL


def test_verify_hal():
    profile = HardwareHAL.get_profile()
    assert profile.device_type is not None
    assert profile.arch is not None
    assert profile.total_ram_gb >= 0.0

    # Recommend for a 13B model
    recommendation = HardwareHAL.recommend_policy(13.0)
    assert recommendation is not None
    assert "quantization" in recommendation


def test_verify_hybrid_model():
    # Small test config
    config = ModelConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        num_kv_heads=2,
        intermediate_size=512,
        quantization_level=8,  # 8 corresponds to INT8 in hardware_hal QuantizationPolicy
    )

    model = HBLLMModel(config)

    # Check for HybridLinear
    hybrid_count = 0
    for name, module in model.named_modules():
        if isinstance(module, HybridLinear):
            hybrid_count += 1

    assert hybrid_count > 0, "No HybridLinear layers were injected by config"
