"""
Tests for HuggingFace Model Adapter and Unified Model Loader.

Uses GPT-2 (124M) as a real HF model for integration testing,
and simple mocks for unit testing the adapter interface.
"""

import pytest
import torch
import torch.nn as nn

from hbllm.model.config import CONFIGS, ModelConfig, get_config
from hbllm.model.model_loader import list_available_models, load_model

# ── Config Presets ────────────────────────────────────────────────────────────


class TestConfigPresets:
    """Test model config presets."""

    def test_7b_config_exists(self):
        config = get_config("7b")
        assert config.name == "hbllm-7b"
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.num_attention_heads == 32
        assert config.num_kv_heads == 8
        assert config.intermediate_size == 11008
        assert config.max_position_embeddings == 1048576
        assert config.sliding_window == 4096

    def test_13b_config_exists(self):
        config = get_config("13b")
        assert config.name == "hbllm-13b"
        assert config.hidden_size == 5120
        assert config.num_layers == 40
        assert config.num_attention_heads == 40
        assert config.intermediate_size == 13824
        assert config.max_position_embeddings == 1048576
        assert config.sliding_window == 4096

    def test_7b_param_estimate(self):
        config = get_config("7b")
        # 7B model should estimate between 6B-8B params
        est = config.num_params_estimate
        assert 5_000_000_000 < est < 9_000_000_000, f"7B estimate out of range: {est}"

    def test_13b_param_estimate(self):
        config = get_config("13b")
        est = config.num_params_estimate
        assert 10_000_000_000 < est < 16_000_000_000, f"13B estimate out of range: {est}"

    def test_all_presets_have_valid_head_dim(self):
        for name, config in CONFIGS.items():
            assert config.hidden_size % config.num_attention_heads == 0, (
                f"Config '{name}': hidden_size={config.hidden_size} not divisible by "
                f"num_attention_heads={config.num_attention_heads}"
            )

    def test_unknown_size_raises(self):
        with pytest.raises(ValueError, match="Unknown model size"):
            get_config("999b")


# ── Unified Loader ────────────────────────────────────────────────────────────


class TestModelLoader:
    """Test unified model loader."""

    def test_load_native_125m(self):
        model = load_model("125m")
        assert hasattr(model, "forward")
        assert hasattr(model, "generate")
        assert hasattr(model, "config")

    def test_load_native_case_insensitive(self):
        model = load_model("125M")
        assert model.config.name == "hbllm-125m"

    def test_list_available_models(self):
        models = list_available_models()
        assert "125m" in models
        assert "7b" in models
        assert "13b" in models
        assert "params_estimate" in models["7b"]

    def test_native_model_forward(self):
        model = load_model("125m")
        input_ids = torch.randint(0, 100, (1, 10))
        output = model(input_ids)
        assert "logits" in output
        assert output["logits"].shape[0] == 1
        assert output["logits"].shape[1] == 10

    def test_native_model_generate(self):
        model = load_model("125m")
        input_ids = torch.randint(0, 100, (1, 5))
        output = model.generate(input_ids, max_new_tokens=3)
        assert output.shape[1] >= 5  # At least prompt length


# ── HF Adapter Interface ─────────────────────────────────────────────────────


class MockHFOutput:
    """Mock HuggingFace model output."""
    def __init__(self, logits, loss=None, past_key_values=None):
        self.logits = logits
        self.loss = loss
        self.past_key_values = past_key_values


class MockHFModel(nn.Module):
    """Minimal mock that simulates a HuggingFace causal LM."""

    def __init__(self, vocab_size=100, hidden_size=64, num_layers=2):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "_name_or_path": "mock-model",
        })()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return MockHFOutput(logits=logits, loss=loss)

    def generate(self, input_ids, **kwargs):
        max_new = kwargs.get("max_new_tokens", 5)
        for _ in range(max_new):
            hidden = self.embed(input_ids)
            logits = self.head(hidden)[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


class TestHFAdapterInterface:
    """Test HuggingFace adapter with mock model."""

    @pytest.fixture
    def adapter(self):
        from hbllm.model.hf_adapter import HuggingFaceModelAdapter
        mock_model = MockHFModel(vocab_size=100, hidden_size=64, num_layers=2)
        return HuggingFaceModelAdapter(_hf_model=mock_model, _hf_tokenizer=None)

    def test_forward_returns_dict(self, adapter):
        input_ids = torch.randint(0, 100, (1, 10))
        output = adapter(input_ids)
        assert isinstance(output, dict)
        assert "logits" in output
        assert output["logits"].shape == (1, 10, 100)

    def test_forward_with_labels(self, adapter):
        input_ids = torch.randint(0, 100, (1, 10))
        labels = torch.randint(0, 100, (1, 10))
        output = adapter(input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].dim() == 0  # Scalar

    def test_generate_returns_tensor(self, adapter):
        input_ids = torch.randint(0, 100, (1, 5))
        output = adapter.generate(input_ids, max_new_tokens=3)
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 8  # 5 prompt + 3 generated

    def test_config_property(self, adapter):
        config = adapter.config
        assert isinstance(config, ModelConfig)
        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.vocab_size == 100

    def test_named_parameters(self, adapter):
        params = list(adapter.named_parameters())
        assert len(params) > 0
        # Should have embedding and head parameters
        param_names = [n for n, _ in params]
        assert any("embed" in n for n in param_names)
        assert any("head" in n for n in param_names)

    def test_train_eval_modes(self, adapter):
        adapter.train()
        assert adapter._model.training is True
        adapter.eval()
        assert adapter._model.training is False

    def test_device_property(self, adapter):
        assert adapter.device == torch.device("cpu")

    def test_repr(self, adapter):
        r = repr(adapter)
        assert "HuggingFaceModelAdapter" in r
        assert "params=" in r

    def test_lora_compatible(self, adapter):
        """Adapter's linear layers can receive LoRA injection."""
        linear_modules = [
            (name, mod) for name, mod in adapter._model.named_modules()
            if isinstance(mod, nn.Linear)
        ]
        assert len(linear_modules) > 0, "No nn.Linear modules found for LoRA"
