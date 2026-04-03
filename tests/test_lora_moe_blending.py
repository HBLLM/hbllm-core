import torch
import torch.nn as nn

from hbllm.modules.lora import ACTIVE_ADAPTER, LoRALinear


def test_lora_moe_blending():
    # Setup a standard PyTorch Linear layer (e.g. 16->32)
    base = nn.Linear(16, 32)

    # Inject LoRA
    lora_layer = LoRALinear(base, r=4, lora_alpha=16)
    lora_layer.add_adapter("coding")
    lora_layer.add_adapter("math")

    # Make deterministic weights for testing
    nn.init.ones_(lora_layer.lora_A["coding"])
    nn.init.ones_(lora_layer.lora_B["coding"])

    # math is 2.0 everywhere
    nn.init.constant_(lora_layer.lora_A["math"], 2.0)
    nn.init.constant_(lora_layer.lora_B["math"], 2.0)

    lora_layer.eval()

    # Test Data: batch of 1, 16 features, all ones
    x = torch.ones((1, 16))

    # Expected Coding Output:
    # A = 4x16 ones
    # B = 32x4 ones
    # x @ A = (1x16) @ (16x4) = 1x4 (values 16)
    # (x @ A) @ B = (1x4) @ (4x32) = 1x32 (values 16 * 4 = 64)
    # scaling = alpha/r = 16/4 = 4.0
    # Final LoRA coding component = 64 * 4.0 = 256.0

    # Base output
    with torch.no_grad():
        base_out = base(x)

        ACTIVE_ADAPTER.set("coding")
        coding_out = lora_layer(x) - base_out
        assert torch.allclose(coding_out, torch.full_like(coding_out, 256.0)), (
            f"Coding out {coding_out[0, 0]} != 256.0"
        )

        # Math component = x @ (2.0) @ (2.0) = 16 * 2.0 * 4 * 2.0 = 256
        # wait!
        # x @ A = (1x16) @ (16x4 of 2.0s) = values: 16 * 2.0 = 32.0. (1x4)
        # (x@A) @ B = (1x4 of 32s) @ (4x32 of 2.0s) = values: 4 * 32.0 * 2.0 = 256.0
        # scaled = 256.0 * 4.0 = 1024.0!
        ACTIVE_ADAPTER.set("math")
        math_out = lora_layer(x) - base_out
        assert torch.allclose(math_out, torch.full_like(math_out, 1024.0)), (
            f"Math out {math_out[0, 0]} != 1024.0"
        )

        # Now test Dynamic MoE Blending: 0.7 coding + 0.3 math
        # It should exactly equal 0.7 * 256.0 + 0.3 * 1024.0!
        # 0.7 * 256 = 179.2
        # 0.3 * 1024 = 307.2
        # Total = 486.4

        ACTIVE_ADAPTER.set({"coding": 0.7, "math": 0.3})
        blended_out = lora_layer(x) - base_out
        assert torch.allclose(blended_out, torch.full_like(blended_out, 486.4)), (
            f"Blended out {blended_out[0, 0]} != 486.4"
        )

    print("Dynamic MoE Blending mathematical equivalence verified perfectly!")
