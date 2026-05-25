"""
Test for KV Cache Serialization and Configuration Integrity Verification.
"""

import os

import pytest
import torch

from hbllm.model.config import ModelConfig
from hbllm.serving.kv_cache import KVCache


def test_kv_cache_integrity(tmp_path):
    device = torch.device("cpu")

    config = ModelConfig(
        name="test-model",
        num_layers=2,
        hidden_size=64,
        num_attention_heads=2,
        num_kv_heads=2,
        vocab_size=1000,
    )

    # Mock Tokenizer class
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

    tokenizer = MockTokenizer(1000)

    # 1. Instantiate KVCache and fill it
    cache = KVCache(
        batch_size=1,
        max_seq_len=64,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.float32,
        device=device,
    )

    k_val = torch.randn(1, 2, 10, 32)
    v_val = torch.randn(1, 2, 10, 32)

    # Fill cache
    cache.update(k_val, v_val, seq_offset=0)
    assert cache.seq_len == 10

    # Save cache
    cache_file = os.path.join(tmp_path, "kv_cache.kvc")
    cache.save_cache(cache_file, config, tokenizer)
    assert os.path.exists(cache_file)

    # 2. Reload into a new identical cache shape
    new_cache = KVCache(
        batch_size=1,
        max_seq_len=64,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.float32,
        device=device,
    )

    new_cache.load_cache(cache_file, config, tokenizer)
    assert new_cache.seq_len == 10
    assert torch.allclose(new_cache.key_cache, cache.key_cache)
    assert torch.allclose(new_cache.value_cache, cache.value_cache)
    print("Perfect matching on valid reload!")

    # 3. Mismatched Config reload should raise ValueError
    bad_config = ModelConfig(
        name="bad-model",
        num_layers=4,  # different num_layers!
        hidden_size=64,
        num_attention_heads=2,
        num_kv_heads=2,
        vocab_size=1000,
    )

    with pytest.raises(ValueError) as exc:
        new_cache.load_cache(cache_file, bad_config, tokenizer)
    assert "Strict integrity mismatch" in str(exc.value)
    print("Integrity checks correctly caught mismatch!")
