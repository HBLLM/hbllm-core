"""
HuggingFace Model Adapter — wraps any AutoModelForCausalLM behind the
HBLLM interface so it can be used with the cognitive pipeline, LoRA
training, DPO, and all existing nodes.

Usage::

    from hbllm.model.hf_adapter import HuggingFaceModelAdapter

    # Load a 7B model in 4-bit quantization (fits in 8GB VRAM)
    model = HuggingFaceModelAdapter(
        "meta-llama/Llama-3.2-8B",
        load_in_4bit=True,
    )

    # Works exactly like HBLLMForCausalLM
    output = model(input_ids)  # → {"logits": ..., "loss": ...}
    tokens = model.generate(input_ids, max_new_tokens=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from hbllm.model.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class HFAdapterConfig:
    """Configuration for loading a HuggingFace model."""

    model_name_or_path: str = "gpt2"
    device: str = "auto"
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    max_memory: dict[str, str] | None = None  # e.g. {"0": "6GiB", "cpu": "12GiB"}
    attn_implementation: str | None = None  # "flash_attention_2", "sdpa", None


def _resolve_dtype(dtype_str: str) -> torch.dtype | str:
    """Convert string dtype to torch dtype."""
    mapping: dict[str, torch.dtype | str] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    return mapping.get(dtype_str, "auto")


class HuggingFaceModelAdapter(nn.Module):
    """
    Wraps a HuggingFace AutoModelForCausalLM to match the HBLLM interface.

    Returns dict with "logits" (and optionally "loss", "past_key_values")
    from forward(), and a Tensor from generate() — exactly like
    HBLLMForCausalLM.

    Supports:
    - Any HF causal LM (Llama, Mistral, Phi, Qwen, GPT-2, etc.)
    - 4-bit / 8-bit quantization via bitsandbytes
    - LoRA injection via HBLLM's existing LoRAManager
    - KV cache for efficient generation
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: str = "auto",
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        max_memory: dict[str, str] | None = None,
        attn_implementation: str | None = None,
        _hf_model: nn.Module | None = None,
        _hf_tokenizer: Any = None,
    ):
        super().__init__()
        self._model_name = model_name_or_path
        self._device_str = device
        self._dtype_str = dtype

        if _hf_model is not None:
            # Allow passing a pre-loaded model (for testing)
            self._model = _hf_model
            self._tokenizer = _hf_tokenizer
            self._config = self._build_config_from_model(_hf_model)
            return

        # Load model from HuggingFace
        self._model, self._tokenizer, self._config = self._load_model(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            attn_implementation=attn_implementation,
        )

    @staticmethod
    def _load_model(
        model_name_or_path: str,
        device: str,
        dtype: str,
        load_in_8bit: bool,
        load_in_4bit: bool,
        trust_remote_code: bool,
        max_memory: dict[str, str] | None,
        attn_implementation: str | None,
    ) -> tuple[Any, Any, ModelConfig]:
        """Load a HuggingFace model, tokenizer, and build HBLLM config."""
        try:
            from transformers import (  # type: ignore[attr-defined]
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "HuggingFace transformers is required. Install with:\n"
                "  pip install transformers accelerate\n"
                "For quantization:\n"
                "  pip install bitsandbytes"
            )

        logger.info(
            "Loading HF model: %s (device=%s, dtype=%s, 4bit=%s, 8bit=%s)",
            model_name_or_path,
            device,
            dtype,
            load_in_4bit,
            load_in_8bit,
        )

        # Build kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }

        resolved_dtype = _resolve_dtype(dtype)
        if resolved_dtype != "auto":
            model_kwargs["torch_dtype"] = resolved_dtype
        else:
            model_kwargs["torch_dtype"] = "auto"

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore[attr-defined]

                model_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                raise ImportError("4-bit quantization requires: pip install bitsandbytes")
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        if device != "auto" and not load_in_4bit and not load_in_8bit:
            model_kwargs["device_map"] = None
        else:
            model_kwargs["device_map"] = device if device != "auto" else "auto"

        if max_memory:
            model_kwargs["max_memory"] = max_memory

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Move to device if not using device_map
        if device != "auto" and not load_in_4bit and not load_in_8bit:
            model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Loaded %s: %.1fM parameters", model_name_or_path, param_count / 1e6)

        # Build HBLLM-compatible ModelConfig
        hbllm_config = ModelConfig(
            name=f"hf-{model_name_or_path.split('/')[-1]}",
            vocab_size=getattr(hf_config, "vocab_size", 32000),
            hidden_size=getattr(hf_config, "hidden_size", 4096),
            num_layers=getattr(hf_config, "num_hidden_layers", 32),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 32),
            num_kv_heads=getattr(
                hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", 32)
            ),
            intermediate_size=getattr(hf_config, "intermediate_size", 11008),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 4096),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
        )

        return model, tokenizer, hbllm_config

    @staticmethod
    def _build_config_from_model(model: nn.Module) -> ModelConfig:
        """Build a ModelConfig from a pre-loaded model."""
        hf_config = getattr(model, "config", None)
        if hf_config is None:
            # Fallback for models without a .config attribute
            param_count = sum(p.numel() for p in model.parameters())
            return ModelConfig(name=f"hf-custom-{param_count // 1_000_000}m")

        return ModelConfig(
            name=f"hf-{getattr(hf_config, '_name_or_path', 'custom').split('/')[-1]}",
            vocab_size=getattr(hf_config, "vocab_size", 32000),
            hidden_size=getattr(hf_config, "hidden_size", 768),
            num_layers=getattr(hf_config, "num_hidden_layers", 12),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 12),
            num_kv_heads=getattr(
                hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", 12)
            ),
            intermediate_size=getattr(hf_config, "intermediate_size", 3072),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 2048),
        )

    @property
    def config(self) -> ModelConfig:
        """HBLLM-compatible model config."""
        return self._config

    @property
    def tokenizer(self) -> Any:
        """The loaded tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass matching HBLLMForCausalLM interface.

        Returns dict with "logits", optionally "loss" and "past_key_values".
        """
        kwargs: dict[str, Any] = {"input_ids": input_ids}

        if labels is not None:
            kwargs["labels"] = labels
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if use_cache:
            kwargs["use_cache"] = True

        hf_output = self._model(**kwargs)

        # Convert HF output to HBLLM dict format
        result: dict[str, Any] = {"logits": hf_output.logits}

        if hasattr(hf_output, "loss") and hf_output.loss is not None:
            result["loss"] = hf_output.loss
            result["ce_loss"] = hf_output.loss
            result["lb_loss"] = torch.tensor(0.0)

        if hasattr(hf_output, "past_key_values") and hf_output.past_key_values is not None:
            result["past_key_values"] = hf_output.past_key_values

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Generate tokens using the HF model's generate method.

        Returns the full sequence (prompt + generated) as a Tensor,
        matching HBLLMForCausalLM.generate() signature.
        """
        self._model.eval()

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k if top_k > 0 else None
            gen_kwargs["top_p"] = top_p

        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        elif self._tokenizer and self._tokenizer.eos_token_id:
            gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

        output = self._model.generate(input_ids, **gen_kwargs)
        return cast(torch.Tensor, output)

    def named_parameters(self, **kwargs: Any) -> Any:  # type: ignore[override]
        """Delegate to inner model for LoRA compatibility."""
        return self._model.named_parameters(**kwargs)

    def parameters(self, **kwargs: Any) -> Any:  # type: ignore[override]
        """Delegate to inner model."""
        return self._model.parameters(**kwargs)

    def train(self, mode: bool = True) -> Any:
        """Set training mode."""
        self._model.train(mode)
        return self

    def eval(self) -> Any:
        """Set eval mode."""
        self._model.eval()
        return self

    def state_dict(self, **kwargs: Any) -> Any:
        """Delegate to inner model."""
        return self._model.state_dict(**kwargs)

    def load_state_dict(self, state_dict: dict[str, Any], **kwargs: Any) -> Any:  # type: ignore[override]
        """Delegate to inner model."""
        return self._model.load_state_dict(state_dict, **kwargs)

    def __repr__(self) -> str:
        param_count = sum(p.numel() for p in self._model.parameters())
        return (
            f"HuggingFaceModelAdapter(\n"
            f"  model={self._model_name},\n"
            f"  params={param_count / 1e6:.1f}M,\n"
            f"  device={self.device},\n"
            f"  config={self._config.name}\n"
            f")"
        )
