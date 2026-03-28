"""
Low-Rank Adaptation (LoRA) for HBLLM.

Implements LoRA for linear layers (Attention weights, FFN weights),
allowing dynamic specialization of the base Model without updating 
the full parameter set.
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer and applies low-rank updates.

    y = Wx + (B @ A)x * scaling

    Supports multiple adapter weights held in memory simultaneously, 
    selectable via `active_adapter`. The base weight is kept frozen.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.in_features = self.base_layer.in_features
        self.out_features = self.base_layer.out_features

        # Multi-LoRA Matrices
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Flag to enable/disable specific adapter during forward pass
        self.active_adapter: str | None = "default"
        
        # Initialize default adapter for backward compatibility
        self.add_adapter("default")

    def add_adapter(self, adapter_name: str) -> None:
        """Initialize and register a new parameter set for this adapter."""
        if adapter_name not in self.lora_A:
            self.lora_A[adapter_name] = nn.Parameter(torch.zeros((self.r, self.in_features)))
            self.lora_B[adapter_name] = nn.Parameter(torch.zeros((self.out_features, self.r)))
            
            # Kaiming uniform for A, Zeros for B
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute base output + LoRA output."""
        result = self.base_layer(x)

        if not self.active_adapter or self.r == 0 or self.active_adapter not in self.lora_A:
            return result

        # LoRA path: dropout -> A -> B -> scaling
        lora_x = self.dropout(x)
        lora_h = F.linear(lora_x, self.lora_A[self.active_adapter])
        lora_out = F.linear(lora_h, self.lora_B[self.active_adapter])

        return result + (lora_out * self.scaling)


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
    ) -> list[str]:
        """
        Recursively replaces matching Linear layers with LoRALinear.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        injected = []

        # Recursively patch modules
        for name, module in list(model.named_modules()):
            for target in target_modules:
                if name.endswith(target) and isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    child_name = name.rsplit(".", 1)[-1] if "." in name else name

                    parent = model.get_submodule(parent_name) if parent_name else model
                    lora_layer = LoRALinear(
                        base_layer=module,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(parent, child_name, lora_layer)
                    injected.append(name)
                    break

        logger.info("Injected LoRA (r=%d) into %d modules", r, len(injected))
        return injected
        
    @staticmethod
    def add_adapter(model: nn.Module, adapter_name: str, state_dict: Dict[str, torch.Tensor] | None = None) -> None:
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
    def set_active_adapter(model: nn.Module, adapter_name: str | None) -> None:
        """O(1) fast pointer swap for the active evaluation adapter."""
        count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.active_adapter = adapter_name
                count += 1
        logger.debug("Set %d LoRA adapters active_adapter=%s", count, adapter_name)

    @staticmethod
    def get_lora_state_dict(model: nn.Module, adapter_name: str = "default") -> Dict[str, torch.Tensor]:
        """Extract only the specified LoRA adapter parameters, formatted as legacy flat tensors."""
        lora_state: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if f"lora_A.{adapter_name}" in name or f"lora_B.{adapter_name}" in name:
                # Strip the adapter_name mapping from the key to maintain legacy disk format mapping
                legacy_name = name.replace(f".{adapter_name}", "")
                lora_state[legacy_name] = param.data.cpu()
        return lora_state

    @staticmethod
    def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], adapter_name: str = "default") -> None:
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
