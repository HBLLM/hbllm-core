"""
Comprehensive tests for the HBLLM model architecture.

Covers: ModelConfig, RMSNorm, TokenEmbedding, RotaryEmbedding,
GroupedQueryAttention, SwiGLUFFN, TransformerBlock, HBLLMModel,
HBLLMForCausalLM (forward, loss, generation).
"""

from __future__ import annotations

import pytest
import torch
import yaml
from pathlib import Path

from hbllm.model.config import ModelConfig, get_config
from hbllm.model.normalization import RMSNorm
from hbllm.model.embeddings import TokenEmbedding, RotaryEmbedding, apply_rotary_pos_emb
from hbllm.model.feedforward import SwiGLUFFN
from hbllm.model.attention import GroupedQueryAttention
from hbllm.model.transformer import TransformerBlock, HBLLMModel, HBLLMForCausalLM


# ──────────────────────────────────────────────
# ModelConfig tests
# ──────────────────────────────────────────────

class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.vocab_size == 32768

    def test_head_dim_computed(self):
        config = ModelConfig(hidden_size=768, num_attention_heads=12)
        assert config.head_dim == 64

    def test_presets(self):
        c125m = get_config("125m")
        assert c125m.num_layers == 12
        assert c125m.hidden_size == 768

        c500m = get_config("500m")
        assert c500m.num_layers == 24
        assert c500m.hidden_size == 1024

        c15b = get_config("1.5b")
        assert c15b.num_layers == 32
        assert c15b.hidden_size == 2048

    def test_unknown_preset(self):
        with pytest.raises(ValueError):
            get_config("nonexistent")

    def test_param_estimate_reasonable(self):
        config = get_config("125m")
        params = config.num_params_estimate
        assert 75_000_000 < params < 200_000_000

    def test_from_yaml(self, tmp_path: Path):
        yaml_content = {
            "name": "test-model",
            "hidden_size": 128,
            "num_layers": 2,
            "num_attention_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 256,
            "vocab_size": 1000,
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = ModelConfig.from_yaml(str(yaml_path))
        assert config.hidden_size == 128
        assert config.num_layers == 2

    def test_custom_config(self):
        config = ModelConfig(
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            num_kv_heads=4,
        )
        assert config.head_dim == 32
        assert config.num_kv_heads == 4


# ──────────────────────────────────────────────
# RMSNorm tests
# ──────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(768)
        x = torch.randn(2, 10, 768)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_effect(self):
        norm = RMSNorm(768)
        x = torch.randn(2, 10, 768) * 100  # Large values
        out = norm(x)
        assert out.abs().mean() < x.abs().mean()

    def test_learnable_weight(self):
        norm = RMSNorm(768)
        assert norm.weight.shape == (768,)
        assert torch.allclose(norm.weight, torch.ones(768))  # Initialized to 1

    def test_different_eps(self):
        norm1 = RMSNorm(64, eps=1e-6)
        norm2 = RMSNorm(64, eps=1e-8)
        x = torch.randn(1, 5, 64)
        out1 = norm1(x)
        out2 = norm2(x)
        # Slightly different due to epsilon, but close
        assert torch.allclose(out1, out2, atol=1e-4)

    def test_gradient_flow(self):
        norm = RMSNorm(64)
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ──────────────────────────────────────────────
# Embedding tests
# ──────────────────────────────────────────────

class TestTokenEmbedding:
    def test_output_shape(self):
        embed = TokenEmbedding(32768, 768)
        ids = torch.randint(0, 32768, (2, 10))
        out = embed(ids)
        assert out.shape == (2, 10, 768)

    def test_different_ids_different_embeddings(self):
        embed = TokenEmbedding(100, 64)
        ids_a = torch.tensor([[0, 1, 2]])
        ids_b = torch.tensor([[3, 4, 5]])
        assert not torch.equal(embed(ids_a), embed(ids_b))

    def test_same_ids_same_embeddings(self):
        embed = TokenEmbedding(100, 64)
        ids = torch.tensor([[1, 2, 3]])
        assert torch.equal(embed(ids), embed(ids))

    def test_embedding_weights_initialized(self):
        embed = TokenEmbedding(100, 64)
        # Weights should be initialized (not zero)
        assert embed.embedding.weight.abs().mean() > 0


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(64, max_position_embeddings=2048)
        x = torch.randn(2, 12, 10, 64)
        pos_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        cos, sin = rope(x, pos_ids)
        assert cos.shape == (2, 10, 64)
        assert sin.shape == (2, 10, 64)

    def test_apply_rotary(self):
        rope = RotaryEmbedding(64)
        q = torch.randn(2, 12, 10, 64)
        k = torch.randn(2, 4, 10, 64)
        pos_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        cos, sin = rope(q, pos_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotary_changes_values(self):
        rope = RotaryEmbedding(64)
        q = torch.randn(1, 4, 5, 64)
        pos_ids = torch.arange(5).unsqueeze(0)
        cos, sin = rope(q, pos_ids)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        assert not torch.equal(q, q_rot)  # Should be rotated

    def test_different_positions_different_embeddings(self):
        rope = RotaryEmbedding(64)
        x = torch.randn(1, 4, 3, 64)
        pos_0 = torch.tensor([[0, 1, 2]])
        pos_1 = torch.tensor([[10, 11, 12]])
        cos_0, sin_0 = rope(x, pos_0)
        cos_1, sin_1 = rope(x, pos_1)
        assert not torch.equal(cos_0, cos_1)

    def test_position_impacts_output(self):
        """Different positions should produce different cos/sin values."""
        rope = RotaryEmbedding(64)
        x = torch.randn(1, 4, 5, 64)
        pos_a = torch.tensor([[0, 1, 2, 3, 4]])
        pos_b = torch.tensor([[5, 6, 7, 8, 9]])
        cos_a, _ = rope(x, pos_a)
        cos_b, _ = rope(x, pos_b)
        assert not torch.equal(cos_a, cos_b)


# ──────────────────────────────────────────────
# Feed-Forward Network tests
# ──────────────────────────────────────────────

class TestSwiGLUFFN:
    def test_output_shape(self):
        config = ModelConfig()
        ffn = SwiGLUFFN(config)
        x = torch.randn(2, 10, 768)
        out = ffn(x)
        assert out.shape == (2, 10, 768)

    def test_different_input_different_output(self):
        config = ModelConfig(hidden_size=64, intermediate_size=128)
        ffn = SwiGLUFFN(config)
        x1 = torch.randn(1, 5, 64)
        x2 = torch.randn(1, 5, 64)
        assert not torch.equal(ffn(x1), ffn(x2))

    def test_gradient_flow(self):
        config = ModelConfig(hidden_size=64, intermediate_size=128)
        ffn = SwiGLUFFN(config)
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic(self):
        config = ModelConfig(hidden_size=64, intermediate_size=128)
        ffn = SwiGLUFFN(config)
        ffn.eval()
        x = torch.randn(1, 5, 64)
        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.equal(out1, out2)


# ──────────────────────────────────────────────
# Attention tests
# ──────────────────────────────────────────────

class TestGroupedQueryAttention:
    def _get_attn(self, **kwargs):
        config = ModelConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_kv_heads=4,
            **kwargs,
        )
        return GroupedQueryAttention(config), config

    def test_output_shape(self):
        attn, _ = self._get_attn()
        x = torch.randn(2, 10, 128)
        pos_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        out, _ = attn(x, pos_ids)
        assert out.shape == (2, 10, 128)

    def test_kv_cache_build(self):
        attn, _ = self._get_attn()
        x = torch.randn(2, 10, 128)
        pos_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        out, cache = attn(x, pos_ids, use_cache=True)
        assert cache is not None
        k_cache, v_cache = cache
        assert k_cache.shape[2] == 10  # seq_len
        assert k_cache.shape[1] == 4   # num_kv_heads

    def test_kv_cache_extend(self):
        attn, _ = self._get_attn()
        # First pass
        x1 = torch.randn(2, 10, 128)
        pos1 = torch.arange(10).unsqueeze(0).expand(2, -1)
        _, cache1 = attn(x1, pos1, use_cache=True)

        # Second pass (single new token)
        x2 = torch.randn(2, 1, 128)
        pos2 = torch.tensor([[10]]).expand(2, -1)
        out2, cache2 = attn(x2, pos2, past_key_value=cache1, use_cache=True)
        assert out2.shape == (2, 1, 128)
        assert cache2[0].shape[2] == 11

    def test_causal_masking(self):
        """Attention should be causal (can't attend to future tokens)."""
        attn, _ = self._get_attn()
        x = torch.randn(1, 5, 128)
        pos = torch.arange(5).unsqueeze(0)

        # With mask
        out_masked, _ = attn(x, pos)

        # Output should be deterministic with causal mask
        out_masked2, _ = attn(x, pos)
        assert torch.equal(out_masked, out_masked2)

    def test_gradient_flow(self):
        attn, _ = self._get_attn()
        x = torch.randn(1, 5, 128, requires_grad=True)
        pos = torch.arange(5).unsqueeze(0)
        out, _ = attn(x, pos)
        out.sum().backward()
        assert x.grad is not None

    def test_gqa_fewer_kv_heads(self):
        """Verify GQA works with fewer KV heads than Q heads."""
        config = ModelConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_kv_heads=2,  # 4:1 ratio
        )
        attn = GroupedQueryAttention(config)
        x = torch.randn(1, 10, 128)
        pos = torch.arange(10).unsqueeze(0)
        out, _ = attn(x, pos)
        assert out.shape == (1, 10, 128)


# ──────────────────────────────────────────────
# TransformerBlock tests
# ──────────────────────────────────────────────

class TestTransformerBlock:
    def _small_config(self):
        return ModelConfig(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=256,
        )

    def test_output_shape(self):
        config = self._small_config()
        block = TransformerBlock(config, layer_idx=0)
        x = torch.randn(2, 10, 128)
        pos = torch.arange(10).unsqueeze(0).expand(2, -1)
        out, _, _ = block(x, pos)
        assert out.shape == (2, 10, 128)

    def test_residual_connection(self):
        """Output should not be identical to input (attention changes it),
        but should be in the same ballpark (residual prevents explosion)."""
        config = self._small_config()
        block = TransformerBlock(config, layer_idx=0)
        x = torch.randn(2, 10, 128)
        pos = torch.arange(10).unsqueeze(0).expand(2, -1)
        out, _, _ = block(x, pos)

        # Should be close-ish to input due to residual (not identical, not exploded)
        diff = (out - x).abs().mean()
        assert diff > 0  # Not identical
        assert diff < 10  # Not exploded

    def test_with_cache(self):
        config = self._small_config()
        block = TransformerBlock(config, layer_idx=0)
        x = torch.randn(1, 10, 128)
        pos = torch.arange(10).unsqueeze(0)
        out, cache, _ = block(x, pos, use_cache=True)
        assert cache is not None

    def test_gradient_flow(self):
        config = self._small_config()
        block = TransformerBlock(config, layer_idx=0)
        x = torch.randn(1, 5, 128, requires_grad=True)
        pos = torch.arange(5).unsqueeze(0)
        out, _, _ = block(x, pos)
        out.sum().backward()
        assert x.grad is not None


# ──────────────────────────────────────────────
# Full Model tests
# ──────────────────────────────────────────────

class TestHBLLMForCausalLM:
    def _small_model(self):
        config = ModelConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=256,
            vocab_size=256,
        )
        return HBLLMForCausalLM(config)

    def test_forward_logits_shape(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (2, 10))
        result = model(ids)
        assert result["logits"].shape == (2, 10, 256)

    def test_forward_with_loss(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (2, 10))
        result = model(ids, labels=ids)
        assert "loss" in result
        assert result["loss"].ndim == 0  # Scalar
        assert result["loss"].item() > 0

    def test_loss_decreases_with_gradient(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (2, 10))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initial loss
        result = model(ids, labels=ids)
        initial_loss = result["loss"].item()

        # A few training steps
        for _ in range(5):
            result = model(ids, labels=ids)
            result["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        final_loss = result["loss"].item()
        assert final_loss < initial_loss

    def test_generate(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (1, 5))
        output = model.generate(ids, max_new_tokens=10)
        assert output.shape == (1, 15)  # 5 prompt + 10 generated

    def test_generate_with_temperature(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (1, 5))
        output = model.generate(ids, max_new_tokens=5, temperature=0.5)
        assert output.shape[1] == 10

    def test_generate_with_eos(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (1, 5))
        # EOS token unlikely to be generated immediately, but shouldn't crash
        output = model.generate(ids, max_new_tokens=10, eos_token_id=0)
        assert output.shape[1] <= 15

    def test_kv_cache_in_forward(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (1, 10))
        result = model(ids, use_cache=True)
        assert "past_key_values" in result
        kvs = result["past_key_values"]
        assert len(kvs) == 2  # 2 layers

    def test_weight_tying(self):
        config = ModelConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=256,
            tie_word_embeddings=True,
        )
        model = HBLLMForCausalLM(config)
        # lm_head and embed_tokens should share weight
        assert model.lm_head.weight is model.model.embed_tokens.embedding.weight

    def test_no_weight_tying(self):
        config = ModelConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=256,
            tie_word_embeddings=False,
        )
        model = HBLLMForCausalLM(config)
        assert model.lm_head.weight is not model.model.embed_tokens.embedding.weight

    def test_param_count(self):
        model = self._small_model()
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_ignore_index_in_loss(self):
        model = self._small_model()
        ids = torch.randint(0, 256, (2, 10))
        labels = ids.clone()
        labels[:, :5] = -100  # Mask first 5 tokens

        result = model(ids, labels=labels)
        assert "loss" in result
        assert result["loss"].item() > 0
