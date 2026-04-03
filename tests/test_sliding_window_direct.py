"""
Tests for KV Cache Sliding Window Attention.

Validates that:
1. KVCache maintains constant buffer size when sliding_window is active.
2. Attention sink tokens (first N positions) are preserved during eviction.
3. Static mode still raises ValueError on overflow.
4. The full model pipeline works with KVCache objects end-to-end.
"""

import pytest
import torch
from hbllm.serving.kv_cache import KVCache
from hbllm.model.config import ModelConfig
from hbllm.model.transformer import HBLLMForCausalLM


# ─── Unit Tests for KVCache ──────────────────────────────────────────────────

class TestKVCacheStatic:
    """Tests for the non-sliding-window (static) mode."""

    def test_basic_update(self):
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
        )
        keys = torch.randn(1, 2, 4, 8)
        values = torch.randn(1, 2, 4, 8)

        k, v = cache.update(keys, values, seq_offset=0)
        assert k.shape == (1, 2, 4, 8)
        assert v.shape == (1, 2, 4, 8)
        assert cache.seq_len == 4

    def test_incremental_update(self):
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
        )
        k1 = torch.randn(1, 2, 4, 8)
        v1 = torch.randn(1, 2, 4, 8)
        cache.update(k1, v1, seq_offset=0)

        k2 = torch.randn(1, 2, 1, 8)
        v2 = torch.randn(1, 2, 1, 8)
        k, v = cache.update(k2, v2, seq_offset=4)

        assert k.shape == (1, 2, 5, 8)
        assert cache.seq_len == 5

    def test_overflow_raises_error(self):
        cache = KVCache(
            batch_size=1, max_seq_len=4, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
        )
        keys = torch.randn(1, 2, 5, 8)
        values = torch.randn(1, 2, 5, 8)

        with pytest.raises(ValueError, match="KV Cache exceeded"):
            cache.update(keys, values, seq_offset=0)


class TestKVCacheSlidingWindow:
    """Tests for the sliding window mode."""

    def test_within_window(self):
        """Tokens within the window should behave normally."""
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
            sliding_window=8, attention_sinks=2,
        )
        keys = torch.randn(1, 2, 5, 8)
        values = torch.randn(1, 2, 5, 8)
        k, v = cache.update(keys, values, seq_offset=0)

        assert k.shape == (1, 2, 5, 8)
        assert cache.seq_len == 5

    def test_at_window_boundary(self):
        """Filling exactly to the window size."""
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
            sliding_window=8, attention_sinks=2,
        )
        # Fill to exactly 8
        k1 = torch.randn(1, 2, 5, 8)
        v1 = torch.randn(1, 2, 5, 8)
        cache.update(k1, v1, seq_offset=0)

        k2 = torch.randn(1, 2, 3, 8)
        v2 = torch.randn(1, 2, 3, 8)
        k, v = cache.update(k2, v2, seq_offset=5)

        assert k.shape == (1, 2, 8, 8)
        assert cache.seq_len == 8

    def test_overflow_evicts_old_tokens(self):
        """Going beyond the window should evict old tokens and keep length constant."""
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
            sliding_window=8, attention_sinks=2,
        )
        # Fill to window
        k_init = torch.randn(1, 2, 8, 8)
        v_init = torch.randn(1, 2, 8, 8)
        cache.update(k_init, v_init, seq_offset=0)
        assert cache.seq_len == 8

        # Add one more token — should trigger eviction
        k_new = torch.randn(1, 2, 1, 8)
        v_new = torch.randn(1, 2, 1, 8)
        k, v = cache.update(k_new, v_new, seq_offset=8)

        assert cache.seq_len == 8, f"Expected 8, got {cache.seq_len}"
        assert k.shape == (1, 2, 8, 8)

    def test_sinks_preserved_during_eviction(self):
        """Attention sink tokens must survive eviction."""
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
            sliding_window=8, attention_sinks=2,
        )

        # Create identifiable tokens
        sink_keys = torch.ones(1, 2, 2, 8) * 42.0  # sinks (positions 0-1)
        other_keys = torch.randn(1, 2, 6, 8)        # positions 2-7
        all_keys = torch.cat([sink_keys, other_keys], dim=2)
        all_values = torch.randn(1, 2, 8, 8)

        cache.update(all_keys, all_values, seq_offset=0)

        # Push 4 more tokens to force heavy eviction
        for i in range(4):
            k_new = torch.randn(1, 2, 1, 8)
            v_new = torch.randn(1, 2, 1, 8)
            cache.update(k_new, v_new, seq_offset=8 + i)

        assert cache.seq_len == 8

        # Check sinks are still at positions 0-1
        k_out, _ = cache._read_cache()
        sink_result = k_out[:, :, :2, :]
        assert torch.allclose(sink_result, sink_keys), \
            "Attention sink tokens were corrupted during eviction!"

    def test_multiple_overflow_steps(self):
        """Repeatedly overflowing should maintain constant cache size."""
        cache = KVCache(
            batch_size=1, max_seq_len=32, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
            sliding_window=8, attention_sinks=2,
        )

        # Initial fill
        k_init = torch.randn(1, 2, 8, 8)
        v_init = torch.randn(1, 2, 8, 8)
        cache.update(k_init, v_init, seq_offset=0)

        # Add 20 more tokens one-by-one
        for i in range(20):
            k = torch.randn(1, 2, 1, 8)
            v = torch.randn(1, 2, 1, 8)
            cache.update(k, v, seq_offset=8 + i)

            assert cache.seq_len == 8, \
                f"Step {i}: expected seq_len=8, got {cache.seq_len}"

        assert cache.total_tokens_seen == 28

    def test_invalid_sinks_rejected(self):
        """attention_sinks >= sliding_window should raise ValueError."""
        with pytest.raises(ValueError, match="attention_sinks"):
            KVCache(
                batch_size=1, max_seq_len=16, num_kv_heads=2,
                head_dim=8, dtype=torch.float32,
                sliding_window=4, attention_sinks=4,
            )


class TestKVCacheQuantized:
    """Tests for 8-bit key quantization mode."""

    def test_quantized_basic(self):
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
            quantize_k=True,
        )
        keys = torch.randn(1, 2, 4, 8)
        values = torch.randn(1, 2, 4, 8)

        k, v = cache.update(keys, values, seq_offset=0)
        # Dequantized keys should be approximately equal (within quantization error)
        assert k.shape == (1, 2, 4, 8)
        assert torch.allclose(k, keys, atol=0.1), "Quantization error too large"

    def test_quantized_sliding_window(self):
        cache = KVCache(
            batch_size=1, max_seq_len=16, num_kv_heads=2,
            head_dim=8, dtype=torch.float32,
            quantize_k=True,
            sliding_window=8, attention_sinks=2,
        )

        k_init = torch.randn(1, 2, 8, 8)
        v_init = torch.randn(1, 2, 8, 8)
        cache.update(k_init, v_init, seq_offset=0)

        for i in range(4):
            k = torch.randn(1, 2, 1, 8)
            v = torch.randn(1, 2, 1, 8)
            cache.update(k, v, seq_offset=8 + i)

        assert cache.seq_len == 8


# ─── Integration Test (Model End-to-End) ─────────────────────────────────────

def test_sliding_window_vram_constant():
    """End-to-end: KV cache length stays constant when sliding_window is set."""
    config = ModelConfig(
        name="test-swa-8",
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        sliding_window=8,
        attention_sinks=2,
    )

    model = HBLLMForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, 32768, (1, 5))

    # 1. Initial forward pass
    outputs = model(prompt, use_cache=True)
    pkv = outputs["past_key_values"]

    # 2. Grow to window size (8 tokens)
    next_token = torch.randint(0, 32768, (1, 1))
    for _ in range(3):
        outputs = model(next_token, past_key_values=pkv, use_cache=True)
        pkv = outputs["past_key_values"]

    current_len = pkv[0][0].shape[2]
    assert current_len == 8

    # 3. Go BEYOND the window (to 12 tokens)
    for _ in range(4):
        outputs = model(next_token, past_key_values=pkv, use_cache=True)
        pkv = outputs["past_key_values"]

    final_len = pkv[0][0].shape[2]
    assert final_len == 8, f"Expected cache length 8, got {final_len}"


if __name__ == "__main__":
    test_sliding_window_vram_constant()
