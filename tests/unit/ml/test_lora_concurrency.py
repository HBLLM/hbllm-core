import asyncio

import pytest
import torch
import torch.nn as nn

from hbllm.modules.lora import ACTIVE_ADAPTER, LoRALinear


@pytest.mark.asyncio
async def test_lora_concurrency():
    """
    Simulates three separate concurrent requests processing on the EXACT SAME
    base neural network instance simultaneously, and verifies that the `ContextVar`
    isolates their active adapters perfectly without cross-contamination.
    """
    # 1. Setup global base model (mimicking the globally shared LLM)
    base = nn.Linear(16, 32)
    lora_layer = LoRALinear(base, r=4, lora_alpha=16)
    lora_layer.add_adapter("A")
    lora_layer.add_adapter("B")
    lora_layer.add_adapter("C")

    # Deterministic identifiable weights
    nn.init.constant_(lora_layer.lora_A["A"], 1.0)
    nn.init.constant_(lora_layer.lora_B["A"], 1.0)

    nn.init.constant_(lora_layer.lora_A["B"], 2.0)
    nn.init.constant_(lora_layer.lora_B["B"], 2.0)

    nn.init.constant_(lora_layer.lora_A["C"], 3.0)
    nn.init.constant_(lora_layer.lora_B["C"], 3.0)

    lora_layer.eval()

    # The baseline vector
    x = torch.ones((1, 16))

    # Pre-calculated mathematical expected delta (x@A@B * scaling)
    # A = 16 * 1.0 * 4 * 1.0 * 4.0 = 256.0
    # B = 16 * 2.0 * 4 * 2.0 * 4.0 = 1024.0
    # C = 16 * 3.0 * 4 * 3.0 * 4.0 = 2304.0
    EXPECTED = {"A": 256.0, "B": 1024.0, "C": 2304.0}

    results = {}

    async def run_concurrent_generation(adapter_name: str):
        # 1. Set the context var for THIS specific Task!
        ACTIVE_ADAPTER.set(adapter_name)

        # 2. Yield so another task jumps in and potentially overwrites the 'active' pointer if it were global
        await asyncio.sleep(0.01)

        # 3. Simulate sequential token loop (like generation)
        out_accumulator = 0.0

        for _ in range(5):
            with torch.no_grad():
                out = lora_layer(x) - base(x)

            # Double-check that during the loop, the ContextVar remained pure!
            assert ACTIVE_ADAPTER.get() == adapter_name

            out_accumulator += out[0, 0].item()

            # Context switch again to maximize race-condition chances
            await asyncio.sleep(0.01)

        # Average per token
        results[adapter_name] = out_accumulator / 5.0

    # Execute A, B, and C perfectly in parallel
    await asyncio.gather(
        run_concurrent_generation("A"),
        run_concurrent_generation("B"),
        run_concurrent_generation("C"),
    )

    print(f"Results: {results}")
    assert results["A"] == pytest.approx(EXPECTED["A"], rel=1e-4), f"A leaked: got {results['A']}"
    assert results["B"] == pytest.approx(EXPECTED["B"], rel=1e-4), f"B leaked: got {results['B']}"
    assert results["C"] == pytest.approx(EXPECTED["C"], rel=1e-4), f"C leaked: got {results['C']}"

    print("Lock-free LoRA concurrency via ContextVars is completely thread-safe!")
