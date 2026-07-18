# training/model/rmsnorm.py
"""Root Mean Square Normalization (RMSNorm).

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """RMSNorm layer with optional fused CUDA kernel.

    Args:
        dim: Normalized dimension.
        eps: Epsilon for numerical stability.
        elementwise_affine: Whether to learn scale parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.dim = dim
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMSNorm."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for norm computation, then back
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class LayerNorm(nn.Module):
    """Standard LayerNorm for comparison / ablation."""

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def create_norm(dim: int, norm_type: str = "rmsnorm", **kwargs) -> nn.Module:
    """Factory for normalization layers."""
    if norm_type == "rmsnorm":
        return RMSNorm(dim, **kwargs)
    elif norm_type == "layernorm":
        return LayerNorm(dim, **kwargs)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
