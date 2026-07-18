"""Multi-Head Attention with Flash Attention 2 and RoPE.

Designed for extensibility: MLA (Multi-head Latent Attention) can replace
this module by implementing the same interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .rope import RotaryEmbedding, apply_rotary_emb


def flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available."""
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask and optional Flash Attention.

    This module defines the interface that MLA will later implement.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout rate.
        rope: Whether to use RoPE.
        rope_base: RoPE base frequency.
        max_seq_len: Maximum sequence length.
        flash: Use Flash Attention when available.
        bias: Whether to use bias in Q/K/V projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        rope: bool = True,
        rope_base: float = 10000.0,
        max_seq_len: int = 2048,
        flash: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.flash = flash and flash_attention_available()

        # Q/K/V projections (fused into a single linear for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # RoPE
        self.rope = rope
        if rope:
            self.rotary = RotaryEmbedding(
                self.head_dim, max_seq_len=max_seq_len, base=rope_base
            )

        # Dropout for non-flash path
        self.attn_dropout = nn.Dropout(dropout) if not self.flash else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inference: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rope:
            freqs_cis = self.rotary(T)
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        # Attention computation
        if self.flash:
            # Flash Attention 2 (handles causal mask internally)
            from flash_attn import flash_attn_func
            # flash_attn_func expects (B, T, n_heads, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            dropout_p = self.dropout if self.training else 0.0
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)
            out = out.reshape(B, T, C)
        else:
            # Standard attention with causal mask
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Optional custom mask
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, T, C)

        return self.resid_dropout(self.out_proj(out))


class FlashCausalSelfAttention(nn.Module):
    """Lightweight wrapper that selects the best available attention implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = CausalSelfAttention(*args, **kwargs)

    def forward(self, x, mask=None, inference=False):
        return self.attn(x, mask=mask, inference=inference)
