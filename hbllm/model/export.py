"""
Model Export Utilities.

Provides export functionality for trained HBLLM models:
  - GGUF export for llama.cpp / Ollama
  - ONNX export for edge/mobile deployment
  - Post-training quantization (INT8/INT4)

Usage:
    from hbllm.model.export import ModelExporter

    exporter = ModelExporter(model, tokenizer, config)
    exporter.export_onnx("./exports/model.onnx")
    exporter.quantize_dynamic("./exports/model_int8.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export HBLLM models to various formats."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    # ─── ONNX Export ─────────────────────────────────────────────────

    def export_onnx(
        self,
        output_path: str | Path,
        opset_version: int = 14,
        max_seq_len: int = 512,
    ) -> Path:
        """
        Export model to ONNX format for edge/mobile deployment.

        Args:
            output_path: Path for the output .onnx file.
            opset_version: ONNX opset version.
            max_seq_len: Max sequence length for dynamic axes.

        Returns:
            Path to the exported ONNX file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        device = next(self.model.parameters()).device

        # Create dummy input
        dummy_input = torch.randint(
            0, self.config.vocab_size, (1, max_seq_len), dtype=torch.long
        ).to(device)

        logger.info("Exporting to ONNX (opset=%d)...", opset_version)

        torch.onnx.export(
            self.model,
            (dummy_input,),
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX model exported to %s (%.1f MB)", output_path, size_mb)
        return output_path

    # ─── Dynamic Quantization (INT8) ─────────────────────────────────

    def quantize_dynamic(
        self,
        output_path: str | Path,
        dtype: torch.dtype = torch.qint8,
    ) -> Path:
        """
        Apply post-training dynamic quantization (INT8).

        Quantizes all Linear layers for ~2-4x speedup on CPU.

        Args:
            output_path: Path for the quantized model .pt file.
            dtype: Quantization dtype (torch.qint8 or torch.float16).

        Returns:
            Path to the quantized model.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Applying dynamic quantization (dtype=%s)...", dtype)

        # Move model to CPU for quantization
        cpu_model = self.model.cpu()
        cpu_model.eval()

        quantized = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
            cpu_model,
            {nn.Linear},
            dtype=dtype,
        )

        # Save the quantized model
        torch.save(
            {
                "model_state_dict": quantized.state_dict(),
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                "quantized": True,
                "dtype": str(dtype),
            },
            output_path,
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Quantized model saved to %s (%.1f MB)", output_path, size_mb)
        return output_path

    # ─── Float16 Export ──────────────────────────────────────────────

    def export_fp16(self, output_path: str | Path) -> Path:
        """
        Export model in float16 for ~50% size reduction.

        Args:
            output_path: Path for the output .pt file.

        Returns:
            Path to the fp16 model.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Converting model to float16...")

        fp16_model = self.model.half().cpu()

        torch.save(
            {
                "model_state_dict": fp16_model.state_dict(),
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                "dtype": "float16",
            },
            output_path,
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("FP16 model saved to %s (%.1f MB)", output_path, size_mb)
        return output_path

    # ─── GGUF Export ─────────────────────────────────────────────────

    def export_gguf(self, output_path: str | Path) -> Path:
        """
        Export model weights + metadata in a simplified GGUF-compatible format.

        This creates a file that can be used with llama.cpp or Ollama
        after conversion with their tooling.

        Args:
            output_path: Path for the output .gguf file.

        Returns:
            Path to the GGUF file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting to GGUF format...")

        metadata = {
            "general.architecture": "hbllm",
            "general.name": getattr(self.config, "name", "hbllm"),
            "hbllm.context_length": getattr(self.config, "max_position_embeddings", 2048),
            "hbllm.embedding_length": getattr(self.config, "hidden_size", 768),
            "hbllm.block_count": getattr(self.config, "num_layers", 12),
            "hbllm.attention.head_count": getattr(self.config, "num_attention_heads", 12),
            "hbllm.attention.head_count_kv": getattr(self.config, "num_kv_heads", 4),
            "hbllm.feed_forward_length": getattr(self.config, "intermediate_size", 3072),
            "hbllm.vocab_size": getattr(self.config, "vocab_size", 32768),
        }

        # Save as a combined archive with metadata + state dict
        state_dict = {k: v.cpu().float() for k, v in self.model.state_dict().items()}

        torch.save(
            {
                "gguf_metadata": metadata,
                "model_state_dict": state_dict,
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                "format": "gguf_prep",
            },
            output_path,
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            "GGUF-prep exported to %s (%.1f MB). "
            "Use llama.cpp's convert tool for final GGUF conversion.",
            output_path,
            size_mb,
        )
        return output_path

    # ─── Summary ─────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return model size statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Estimate sizes
        fp32_mb = total_params * 4 / (1024 * 1024)
        fp16_mb = total_params * 2 / (1024 * 1024)
        int8_mb = total_params * 1 / (1024 * 1024)

        return {
            "total_params": total_params,
            "trainable_params": trainable,
            "estimated_fp32_mb": round(fp32_mb, 1),
            "estimated_fp16_mb": round(fp16_mb, 1),
            "estimated_int8_mb": round(int8_mb, 1),
            "model_name": getattr(self.config, "name", "unknown"),
        }
