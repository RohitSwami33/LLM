"""SwiGLU and standard MLP for the Transformer.

Reference: Shazeer, "GLU Variants Improve Transformer", 2020.

SwiGLU(x) = (xW_1) ⊙ Swish(xW_gate)   (no bias in modern LLMs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Args:
        d_model: Input/output dimension.
        d_ff: Inner dimension (will be rounded to multiple of 256).
        dropout: Dropout rate.
        bias: Whether to use bias.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        # SwiGLU uses two separate projections
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class GELUMLP(nn.Module):
    """Standard GELU MLP for ablation / comparison."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


def create_mlp(d_model: int, d_ff: int, activation: str = "swiglu", **kwargs) -> nn.Module:
    """Factory for MLP modules."""
    if activation == "swiglu":
        return SwiGLU(d_model, d_ff, **kwargs)
    elif activation == "gelu":
        return GELUMLP(d_model, d_ff, **kwargs)
    else:
        raise ValueError(f"Unknown activation: {activation}")
