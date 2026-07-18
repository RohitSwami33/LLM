"""MoE Transformer Block — replaces the MLP with a Mixture of Experts.

Each block: RMSNorm -> Self-Attention -> Residual -> RMSNorm -> MoE -> Residual

This is a drop-in replacement for TransformerBlock when using MoE architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .rmsnorm import create_norm
from .attention import CausalSelfAttention
from .moe import MoELayer


class MoETransformerBlock(nn.Module):
    """Transformer decoder block with MoE feed-forward layer.

    Identical to TransformerBlock but replaces the dense MLP with a
    Sparse Mixture of Experts layer.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension per expert.
        num_experts: Number of experts.
        top_k: Number of experts per token.
        capacity_factor: Token capacity factor (None = no dropping).
        shared_expert: Use a shared expert.
        dropout: Dropout rate.
        norm_type: Normalization type.
        rope: Use RoPE.
        rope_base: RoPE base frequency.
        max_seq_len: Maximum sequence length.
        flash: Use Flash Attention.
        bias: Use bias in linear layers.
        gradient_checkpointing: Enable activation checkpointing.
        load_balancing_weight: Weight for load balancing loss.
        router_z_loss_weight: Weight for router z-loss.
        router_temperature: Temperature for routing logits.
        router_noise: Noise std for training routing.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: Optional[float] = None,
        shared_expert: bool = False,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        rope: bool = True,
        rope_base: float = 10000.0,
        max_seq_len: int = 2048,
        flash: bool = True,
        bias: bool = False,
        gradient_checkpointing: bool = False,
        load_balancing_weight: float = 0.01,
        router_z_loss_weight: float = 0.001,
        router_temperature: float = 1.0,
        router_noise: float = 0.1,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Pre-norm + Attention (identical to TransformerBlock)
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

        # Pre-norm + MoE (replaces dense MLP)
        self.norm2 = create_norm(d_model, norm_type)
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            shared_expert=shared_expert,
            dropout=dropout,
            bias=bias,
            load_balancing_weight=load_balancing_weight,
            router_z_loss_weight=router_z_loss_weight,
            router_temperature=router_temperature,
            router_noise=router_noise,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: (B, T, d_model) input tensor.
            mask: (B, T) attention mask (1=attend, 0=ignore).

        Returns:
            output: (B, T, d_model) output tensor.
            moe_metrics: Dict of MoE-specific metrics.
        """
        if self.training and self.gradient_checkpointing:
            return self._checkpointed_forward(x, mask)
        return self._forward(x, mask)

    def _forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Attention block
        x = x + self.attn(self.norm1(x), mask=mask)

        # MoE block
        moe_output, moe_metrics = self.moe(self.norm2(x), training=self.training)
        x = x + moe_output

        return x, moe_metrics

    def _checkpointed_forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Gradient checkpointing: recompute activations during backward pass."""
        def checkpoint_fn(x_):
            return self._forward(x_, mask)
        return torch.utils.checkpoint.checkpoint(
            checkpoint_fn, x, use_reentrant=False
        )
