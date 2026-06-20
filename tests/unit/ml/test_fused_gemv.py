"""
Test for Rust Fused 4-bit GEMV Correctness vs PyTorch Linear.
"""

import torch

from hbllm.model.quantization import QuantizedLinear


def test_fused_gemv_correctness():
    # Set seed for reproducibility
    torch.manual_seed(42)

    in_features = 256
    out_features = 128
    group_size = 128

    # 1. Create a random full-precision weight matrix
    w_fp = torch.randn(out_features, in_features)

    # 2. Instantiate our QuantizedLinear layer
    layer = QuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        bits=4,
        bias=True,
        group_size=group_size,
    )

    # 3. Quantize and pack weights
    packed, scale, q_bias = QuantizedLinear.quantize_weight(w_fp, bits=4, group_size=group_size)
    layer.weight_shards.copy_(packed)
    layer.scale.copy_(scale)
    layer.q_bias.copy_(q_bias)
    layer.bias_param.data.copy_(torch.randn(out_features))

    # 4. Generate random input sequence of length 1 (decode sequence)
    x = torch.randn(2, 1, in_features)  # batch_size = 2, seq_len = 1

    # Evaluate using PyTorch fallback (standard two-pass dequantize + linear)
    # We temporarily set rust_engine to None to force fallback path
    import hbllm.model.quantization as q_mod

    original_engine = q_mod.rust_engine

    q_mod.rust_engine = None
    y_fallback = layer(x)

    # Evaluate using Rust fast-path fused kernel (if available)
    if original_engine is not None and hasattr(original_engine, "gemv_4bit_simd"):
        q_mod.rust_engine = original_engine
        y_rust = layer(x)

        # Assert mathematical equivalence between fast-path GEMV and standard fallback
        assert torch.allclose(y_rust, y_fallback, atol=1e-3), (
            "Fused GEMV outputs do not match fallback outputs"
        )
        print("Fused GEMV outputs match fallback outputs!")
    else:
        print("Rust engine or gemv_4bit_simd not available, skipping fast-path test")
