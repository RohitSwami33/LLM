"""Transformer Block (decoder-only) with modular components.

Each block: RMSNorm -> Self-Attention -> Residual -> RMSNorm -> MLP -> Residual

Designed so MoE can replace the MLP, MLA can replace attention,
and mHC can wrap the residual connections.
"""

import torch
import torch.nn as nn
from typing import Optional

from .rmsnorm import create_norm
from .attention import CausalSelfAttention
from .mlp import create_mlp


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: Dropout rate.
        norm_type: Normalization type.
        activation: MLP activation type.
        rope: Use RoPE.
        rope_base: RoPE base frequency.
        max_seq_len: Maximum sequence length.
        flash: Use Flash Attention.
        bias: Use bias in linear layers.
        gradient_checkpointing: Enable activation checkpointing.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        activation: str = "swiglu",
        rope: bool = True,
        rope_base: float = 10000.0,
        max_seq_len: int = 2048,
        flash: bool = True,
        bias: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Pre-norm
        self.norm1 = create_norm(d_model, norm_type)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            rope=rope,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
            flash=flash,
            bias=bias,
        )

        # Pre-norm + MLP (MoE can replace this later)
        self.norm2 = create_norm(d_model, norm_type)
        self.mlp = create_mlp(d_model, d_ff, activation=activation, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return self._checkpointed_forward(x, mask)
        return self._forward(x, mask)

    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention block
        x = x + self.attn(self.norm1(x), mask=mask)
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x

    def _checkpointed_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Gradient checkpointing: recompute activations during backward pass."""
        def checkpoint_fn(x_):
            return self._forward(x_, mask)
        return torch.utils.checkpoint.checkpoint(
            checkpoint_fn, x, use_reentrant=False
        )
