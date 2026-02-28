"""
RMSNorm — Root Mean Square Layer Normalization.

Used in LLaMA, Mistral, and other modern LLMs instead of LayerNorm.
Simpler and faster: normalizes by RMS without centering (no mean subtraction).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x * (1 / RMS(x)) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)

    Compared to LayerNorm:
    - No mean subtraction (no centering)
    - No bias parameter
    - Faster computation
    - Empirically equivalent performance in transformers
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
