"""
Enterprise-Grade Quantization for HBLLM.

Implements high-performance hybrid layers that combine
quantized base weights with full-precision LoRA sidecars.

Quantization granularity: per-block (group_size=128).
This provides significantly better precision than per-tensor scaling
while maintaining near-identical inference speed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import our Rust accelerated kernel
try:
    from hbllm_compute import UniversalEngine

    rust_engine = UniversalEngine()
except ImportError:
    rust_engine = None

# Industry-standard block size for INT4/INT8 quantization (matches GGUF, bitsandbytes)
DEFAULT_GROUP_SIZE = 128


class QuantizedLinear(nn.Module):
    """
    Optimized 4/8-bit Linear Layer with per-block scaling.

    Each block of ``group_size`` input features has its own scale and bias,
    providing much finer control than per-tensor scaling while keeping
    overhead minimal (1 scale + 1 bias per block per output row).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        bias: bool = False,
        group_size: int = DEFAULT_GROUP_SIZE,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Number of quantization groups along the input dimension
        self.num_groups = math.ceil(in_features / group_size)

        # Reduced memory weight storage
        # 4-bit: 2 weights per byte, 8-bit: 1 weight per byte
        packed_in = in_features // (8 // bits)
        self.register_buffer(
            "weight_shards",
            torch.zeros((out_features, packed_in), dtype=torch.uint8),
        )

        # Per-block scale and bias: [out_features, num_groups]
        self.register_buffer(
            "scale",
            torch.ones((out_features, self.num_groups), dtype=torch.float32),
        )
        self.register_buffer(
            "q_bias",
            torch.zeros((out_features, self.num_groups), dtype=torch.float32),
        )

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hardware-agnostic dequantization with per-block scaling."""
        # Check for Rust SIMD acceleration
        if rust_engine is not None and not x.is_cuda and self.bits == 4:
            w_flat = rust_engine.dequantize_4bit_simd(
                self.weight_shards.numpy(),
                self.scale.numpy(),
                self.q_bias.numpy(),
                self.group_size,
            )
            w_float = (
                torch.from_numpy(w_flat)
                .reshape(self.out_features, self.in_features)
                .to(x.device, x.dtype)
            )
        else:
            # PyTorch fallback
            w_float = self._unpack_native()

        return F.linear(x, w_float, self.bias_param)

    def _unpack_native(self) -> torch.Tensor:
        """Software fallback for dequantization with per-block scaling."""
        if self.bits == 4:
            low = (self.weight_shards & 0x0F).to(torch.float32)
            high = (self.weight_shards >> 4).to(torch.float32)
            # Interleave low and high nibbles: [out_features, in_features]
            w = torch.stack([low, high], dim=-1).view(self.out_features, self.in_features)
        else:
            w = self.weight_shards.to(torch.float32)

        # Apply per-block scale and bias
        # w is [out_features, in_features]
        # scale, q_bias are [out_features, num_groups]
        # We need to expand scale/bias to match each group of input features
        scale_expanded = self.scale.repeat_interleave(self.group_size, dim=1)
        bias_expanded = self.q_bias.repeat_interleave(self.group_size, dim=1)

        # Trim to exact in_features (handles non-divisible case)
        scale_expanded = scale_expanded[:, : self.in_features]
        bias_expanded = bias_expanded[:, : self.in_features]

        return w * scale_expanded + bias_expanded

    @staticmethod
    def quantize_weight(
        weight: torch.Tensor,
        bits: int = 4,
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a full-precision weight matrix to packed format with per-block scaling.

        Args:
            weight: [out_features, in_features] float tensor
            bits: 4 or 8
            group_size: Number of elements per quantization group

        Returns:
            (weight_shards, scale, q_bias) — packed uint8, per-block scale, per-block bias
        """
        out_features, in_features = weight.shape
        num_groups = math.ceil(in_features / group_size)

        # Pad to group boundary if needed
        padded_in = num_groups * group_size
        if padded_in > in_features:
            weight = F.pad(weight, (0, padded_in - in_features))

        # Reshape to [out_features, num_groups, group_size]
        w_grouped = weight.view(out_features, num_groups, group_size)

        # Per-block min/max for symmetric quantization
        w_min = w_grouped.min(dim=-1, keepdim=True).values
        w_max = w_grouped.max(dim=-1, keepdim=True).values

        max_int = (1 << bits) - 1
        scale = (w_max - w_min).clamp(min=1e-6) / max_int  # [out_features, num_groups, 1]
        q_bias = w_min  # [out_features, num_groups, 1]

        # Quantize
        w_q = ((w_grouped - q_bias) / scale).round().clamp(0, max_int).to(torch.uint8)

        # Trim padding from quantized weights
        w_q = w_q.view(out_features, padded_in)[:, :in_features]

        # Pack 4-bit weights into bytes
        if bits == 4:
            assert in_features % 2 == 0, "in_features must be even for 4-bit packing"
            low = w_q[:, 0::2]
            high = w_q[:, 1::2]
            packed = (high << 4) | low
        else:
            packed = w_q

        return packed, scale.squeeze(-1), q_bias.squeeze(-1)


class HybridLinear(nn.Module):
    """
    Zero-Redundancy Hybrid Layer.

    Combines a quantized base layer with a LoRA sidecar.
    Delegates all LoRA computation to LoRALinear to avoid duplicating
    MoE blending, device paging, and dropout logic.
    """

    def __init__(self, base_layer: QuantizedLinear, r: int = 8):
        super().__init__()
        from hbllm.modules.lora import LoRALinear

        self.base = base_layer
        self.r = r

        # Create a LoRALinear that wraps our quantized base directly.
        # LoRALinear.forward() calls self.base_layer(x) + lora_delta,
        # so setting base_layer = our QuantizedLinear gives us the
        # quantized pass + LoRA sidecar in a single unified path.
        self.lora_sidecar = LoRALinear(base_layer, r=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single call: LoRALinear.forward() handles base pass + LoRA delta + MoE blending
        return self.lora_sidecar(x)

    # Expose adapter management for convenience
    def add_adapter(self, name: str) -> None:
        self.lora_sidecar.add_adapter(name)
