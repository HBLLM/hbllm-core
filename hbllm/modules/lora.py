"""
Low-Rank Adaptation (LoRA) for HBLLM.

Implements LoRA for linear layers (Attention weights, FFN weights),
allowing dynamic specialization of the base Model without updating
the full parameter set.
"""

from __future__ import annotations

import contextvars
import logging
import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Lock-Free Concurrency Context
ACTIVE_ADAPTER: contextvars.ContextVar[str | dict[str, float] | None] = contextvars.ContextVar("active_adapter", default=None)


def is_quantization_enabled() -> bool:
    """Helper to check if quantization is globally active."""
    # We could check a global context or config here
    return False


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer and applies low-rank updates.

    y = Wx + (B @ A)x * scaling

    Supports multiple adapter weights held in memory simultaneously,
    selectable via `active_adapter`. The base weight is kept frozen.
    """

    def __init__(
        self,
        base_layer: Any,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze base layer — handle both nn.Linear (.weight) and
        # QuantizedLinear (.weight_shards buffer, already frozen)
        if hasattr(self.base_layer, "weight") and self.base_layer.weight is not None:
            self.base_layer.weight.requires_grad = False
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.in_features = self.base_layer.in_features
        self.out_features = self.base_layer.out_features

        # Multi-LoRA Matrices
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialize default adapter for backward compatibility
        self.add_adapter("default")

    def add_adapter(self, adapter_name: str) -> None:
        """Initialize and register a new parameter set for this adapter."""
        if adapter_name not in self.lora_A:
            # Create on CPU and pin memory for fast paginated transfers later
            param_A = torch.zeros((self.r, self.in_features), device="cpu")
            param_B = torch.zeros((self.out_features, self.r), device="cpu")
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                param_A = param_A.pin_memory()
                param_B = param_B.pin_memory()

            self.lora_A[adapter_name] = nn.Parameter(param_A)
            self.lora_B[adapter_name] = nn.Parameter(param_B)

            # Kaiming uniform for A, Zeros for B
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute base output + LoRA output with MoE support."""
        result = cast(torch.Tensor, self.base_layer(x))

        active_adapter = ACTIVE_ADAPTER.get()

        if not active_adapter or self.r == 0:
            return result
        
        lora_out: Any = 0.0
        lora_x = self.dropout(x)

        if isinstance(active_adapter, dict):
            # Dynamic Mixture-of-Experts blending mapping: {adapter_name: weight}
            for adapt, weight in active_adapter.items():
                if adapt in self.lora_A:
                    A_w = self.lora_A[adapt]
                    B_w = self.lora_B[adapt]
                    # Ensure they are on exactly the same device as x (handles paging errors gracefully)
                    if A_w.device != x.device:
                        A_w = A_w.to(x.device)
                        B_w = B_w.to(x.device)
                    h = F.linear(lora_x, A_w)
                    out = F.linear(h, B_w)
                    lora_out += weight * out
        elif isinstance(active_adapter, str) and active_adapter in self.lora_A:
            # Single adapter fast path
            A_w = self.lora_A[active_adapter]
            B_w = self.lora_B[active_adapter]
            if A_w.device != x.device:
                A_w = A_w.to(x.device)
                B_w = B_w.to(x.device)
            h = F.linear(lora_x, A_w)
            lora_out = F.linear(h, B_w)
        else:
            return result

        return cast(torch.Tensor, result + (lora_out * self.scaling))


class LoRAManager:
    """
    Utility to inject and manage LoRA adapters within the HBLLM model.
    """

    @staticmethod
    def inject(
        model: nn.Module,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        quantization_level: int = 16,
    ) -> list[str]:
        """
        Recursively replaces matching Linear layers with LoRALinear or HybridLinear.
        """
        from hbllm.model.quantization import HybridLinear, QuantizedLinear

        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        injected = []

        # Recursively patch modules
        for name, module in list(model.named_modules()):
            for target in target_modules:
                if (
                    name.endswith(target)
                    and isinstance(module, nn.Linear)
                    and not isinstance(module, (LoRALinear, HybridLinear))
                ):
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    child_name = name.rsplit(".", 1)[-1] if "." in name else name

                    parent = model.get_submodule(parent_name) if parent_name else model

                    if quantization_level in [4, 8]:
                        # Wrap in Hybrid Shield
                        base_quant = QuantizedLinear(
                            module.in_features,
                            module.out_features,
                            bits=quantization_level,
                            bias=module.bias is not None,
                        )
                        # Copy optional bias from original linear
                        if module.bias is not None and base_quant.bias_param is not None:
                            base_quant.bias_param.data.copy_(module.bias.data)

                        lora_layer: nn.Module = HybridLinear(base_layer=base_quant, r=r)
                        logger.info(
                            "Injected HybridLinear (r=%d, bits=%d) into %s",
                            r,
                            quantization_level,
                            name,
                        )
                    else:
                        # Standard LoRA
                        lora_layer = LoRALinear(
                            base_layer=module,
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        )
                        logger.info("Injected LoRALinear (r=%d) into %s", r, name)

                    setattr(parent, child_name, lora_layer)
                    injected.append(name)
                    break

        return injected

    @staticmethod
    def add_adapter(
        model: nn.Module, adapter_name: str, state_dict: dict[str, torch.Tensor] | None = None
    ) -> None:
        """Create a new adapter inside all injected LoRALinear layers and optionally load its weights."""
        count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.add_adapter(adapter_name)
                count += 1

        if state_dict is not None:
            LoRAManager.load_lora_state_dict(model, state_dict, adapter_name=adapter_name)
            logger.info("Added and loaded LoRA adapter '%s' across %d layers.", adapter_name, count)
        else:
            logger.debug("Added empty LoRA adapter '%s' across %d layers.", adapter_name, count)

    @staticmethod
    def set_active(model: nn.Module, active: bool = True) -> None:
        """Legacy method to toggle the default adapter active/inactive."""
        LoRAManager.set_active_adapter(model, "default" if active else None)

    @staticmethod
    def set_active_adapter(model: nn.Module, adapter_name: str | dict[str, float] | None) -> None:
        """ContextVar-safe fast pointer matching for the active evaluation adapter."""
        ACTIVE_ADAPTER.set(adapter_name)
        logger.debug("Set ContextVar ACTIVE_ADAPTER=%s", adapter_name)

    @staticmethod
    def page_in(model: nn.Module, adapters: str | list[str] | dict[str, float]) -> None:
        """Asynchronously stream LoRA weights to GPU."""
        if isinstance(adapters, str):
            adapters = [adapters]
        elif isinstance(adapters, dict):
            adapters = list(adapters.keys())

        device = next(model.parameters()).device
        if device.type == "cpu":
            return

        count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                for adapt in adapters:
                    if adapt in module.lora_A and module.lora_A[adapt].device != device:
                        # Non-blocking transfer from pinned memory
                        module.lora_A[adapt].data = module.lora_A[adapt].data.to(
                            device, non_blocking=True
                        )
                        module.lora_B[adapt].data = module.lora_B[adapt].data.to(
                            device, non_blocking=True
                        )
                        count += 1
        if count > 0:
            logger.debug("Paged IN %d LoRA matrices to %s", count, device)

    @staticmethod
    def page_out(model: nn.Module, adapters: str | list[str] | dict[str, float]) -> None:
        """Stream LoRA weights back to CPU pinned memory to free VRAM."""
        if isinstance(adapters, str):
            adapters = [adapters]
        elif isinstance(adapters, dict):
            adapters = list(adapters.keys())

        device = next(model.parameters()).device
        if device.type == "cpu":
            return

        count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                for adapt in adapters:
                    if adapt in module.lora_A and module.lora_A[adapt].device != torch.device(
                        "cpu"
                    ):
                        cpu_data_A = module.lora_A[adapt].data.to("cpu", non_blocking=True)
                        cpu_data_B = module.lora_B[adapt].data.to("cpu", non_blocking=True)
                        if torch.cuda.is_available() or torch.backends.mps.is_available():
                            cpu_data_A = cpu_data_A.pin_memory()
                            cpu_data_B = cpu_data_B.pin_memory()
                        module.lora_A[adapt].data = cpu_data_A
                        module.lora_B[adapt].data = cpu_data_B
                        count += 1
        if count > 0:
            logger.debug("Paged OUT %d LoRA matrices back to CPU", count)

    @staticmethod
    def get_lora_state_dict(
        model: nn.Module, adapter_name: str = "default"
    ) -> dict[str, torch.Tensor]:
        """Extract only the specified LoRA adapter parameters, formatted as legacy flat tensors."""
        lora_state: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if f"lora_A.{adapter_name}" in name or f"lora_B.{adapter_name}" in name:
                # Strip the adapter_name mapping from the key to maintain legacy disk format mapping
                legacy_name = name.replace(f".{adapter_name}", "")
                lora_state[legacy_name] = param.data.cpu()
        return lora_state

    @staticmethod
    def load_lora_state_dict(
        model: nn.Module, state_dict: dict[str, torch.Tensor], adapter_name: str = "default"
    ) -> None:
        """
        Loads a flat legacy state dict (lora_A, lora_B) into a specific adapter's ParameterDict scope.
        Remaps keys on the fly so it maps specifically to the ParameterDict indexing.
        """
        mapped_dict = {}
        for k, v in state_dict.items():
            if "lora_A" in k and f"lora_A.{adapter_name}" not in k:
                mapped_dict[k.replace("lora_A", f"lora_A.{adapter_name}")] = v
            elif "lora_B" in k and f"lora_B.{adapter_name}" not in k:
                mapped_dict[k.replace("lora_B", f"lora_B.{adapter_name}")] = v
            else:
                mapped_dict[k] = v

        # Must load in 'strict=False' because the full model will have lots of other components
        model.load_state_dict(mapped_dict, strict=False)

    @staticmethod
    def save_adapter(
        model: nn.Module,
        adapter_name: str,
        path: str | Path,
        domain: str | None = None,
        rank: int = 8,
        source_repo: str = "local",
    ) -> str:
        """
        Extracts, wraps with metadata, saves, and computes SHA-256 for an adapter.
        """
        from hbllm.modules.adapter_registry import AdapterRegistry

        state_dict = LoRAManager.get_lora_state_dict(model, adapter_name=adapter_name)
        return AdapterRegistry.save_adapter(
            state_dict,
            Path(path),
            domain=domain or adapter_name,
            rank=rank,
            source_repo=source_repo,
        )

    @staticmethod
    def load_adapter(
        model: nn.Module,
        path: str | Path,
        adapter_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Safely loads a metadata-wrapped adapter into the model.
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)

        metadata = {}
        if isinstance(payload, dict) and "__hbllm_adapter_metadata__" in payload:
            metadata = payload["__hbllm_adapter_metadata__"]
            state_dict = payload["state_dict"]
        else:
            state_dict = payload

        target_name = adapter_name or metadata.get("domain", "default")

        # Ensure the adapter exist in the ParameterDicts first
        LoRAManager.add_adapter(model, target_name)
        LoRAManager.load_lora_state_dict(model, state_dict, adapter_name=target_name)

        logger.info("Loaded LoRA adapter '%s' from %s", target_name, path)
        return metadata
