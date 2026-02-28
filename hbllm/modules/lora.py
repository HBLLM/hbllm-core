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

    The base weight is kept frozen. Only A and B are learned/swapped.
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

        # LoRA Matrices
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.reset_parameters()

        # Flag to enable/disable adapter during forward pass
        self.active = True

    def reset_parameters(self) -> None:
        """Initialize A with Kaiming uniform and B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute base output + LoRA output."""
        result = self.base_layer(x)

        if not self.active or self.r == 0:
            return result

        # LoRA path: dropout -> A -> B -> scaling
        lora_x = self.dropout(x)
        lora_h = F.linear(lora_x, self.lora_A)
        lora_out = F.linear(lora_h, self.lora_B)

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
        If target_modules is None, targets attention q/k/v/o and ffn gate/up/down.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        injected = []

        # Recursively patch modules
        for name, module in list(model.named_modules()):
            for target in target_modules:
                if name.endswith(target) and isinstance(module, nn.Linear):
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
    def set_active(model: nn.Module, active: bool = True) -> None:
        """Enable or disable all injected LoRA adapters in the model."""
        count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.active = active
                count += 1
        logger.debug("Set %d LoRA adapters active=%s", count, active)

    @staticmethod
    def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract only the LoRA adapter parameters for saving."""
        lora_state: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_state[name] = param.data.cpu()
        return lora_state

    @staticmethod
    def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load adapter parameters previously saved via get_lora_state_dict."""
        model.load_state_dict(state_dict, strict=False)
