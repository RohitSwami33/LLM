"""Rotary Positional Embeddings (RoPE).

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cos + i*sin)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor using complex multiplication."""
    # Reshape x to complex: (B, n_heads, seq_len, head_dim) -> (..., head_dim/2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: (seq_len, head_dim/2) -> (1, 1, seq_len, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_rot = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rot.type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary Embedding module with cached frequencies.

    Args:
        dim: Head dimension.
        max_seq_len: Maximum sequence length to precompute.
        base: RoPE base frequency.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tensors for the given sequence length."""
        if seq_len > self.max_seq_len:
            # Interpolate for longer sequences
            freqs_cis = precompute_freqs_cis(self.dim, seq_len, self.base)
        else:
            freqs_cis = self.freqs_cis
        return freqs_cis[:seq_len]

    def extend_seq_len(self, new_max: int):
        """Extend precomputed cache for longer sequences."""
        if new_max > self.max_seq_len:
            self.freqs_cis = precompute_freqs_cis(self.dim, new_max, self.base)
            self.max_seq_len = new_max
