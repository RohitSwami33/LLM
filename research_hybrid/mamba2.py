"""Mamba-2 hybrid block: chunked SSD layer in pure PyTorch (deferred, ablation).

The SSD core is the authors' minimal discrete algorithm (Dao & Gu, arXiv:2405.21060;
``mamba_ssm/modules/ssd_minimal.py``) — chunked quadratic form plus a small
within-chunk scan — re-expressed in plain torch (no einops). It is the portable,
correct reference: no fused selective-scan kernels required (runs on CPU, MPS,
T4, P100), at the cost of lower throughput than the CUDA kernels.

Why OFF by default: docs/research_report_32k.md section 3 — SSMs need ~2x
parameters for transformer parity (Mamba-3B ~= Transformer-6B), attention is
already near-linear via block-sparse masks at 32K, and the fused kernels that
make Mamba fast are CUDA-only. Jamba-style interleaving (1 attention : 7 Mamba)
is configurable via ``ModelConfig.hybrid.attn_layers``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_hybrid.config import HybridConfig


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum (official implementation).

    segsum[i, j] = sum_{k=j+1..i} x[k] for j <= i, else -inf. This is the
    difference of prefix sums; the fill keeps the -inf convention of the
    authors' einsums (exp(-inf) == 0).
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    This is the authors' algorithm (see module docstring), with einops
    ``rearrange`` expanded into reshape/permute calls.
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    b, t = X.shape[0], X.shape[1]
    n_chunks = t // block_len

    def chunks(x):
        # (b, (c l) ...) -> (b, c, l, ...)
        return x.reshape(b, n_chunks, block_len, *x.shape[2:])

    X, A, B, C = chunks(X), chunks(A), chunks(B), chunks(C)
    A = A.permute(0, 3, 1, 2)  # (b, h, c, l)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. intra-chunk state (right term of the off-diagonal factorization; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. inter-chunk recurrence (middle term; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. state -> output conversion (left term; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = Y_diag + Y_off
    Y = Y.reshape(b, t, *Y.shape[-2:])  # (b, c, l, h, p) -> (b, c*l, h, p)
    return Y, final_state


class Mamba2Block(nn.Module):
    """Mamba-2-style block: in-proj -> conv -> silu -> SSD -> z-gate -> out-proj.

    Structure follows ``mamba2_simple`` (state-spaces/mamba): linear in-projection
    split into x, z, B, C, dt; depthwise causal conv1d (kernel_size 4, groups=d);
    the SSD recurrence with A = -exp(A_log) scaled by softplus(dt); residual
    ``D * x`` skip and SiLU z-gating; out-projection. One state per head is shared
    across the batch sample (ngroups=1 broadcast, as in the authors' test).
    """

    def __init__(self, cfg: HybridConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.head_dim = cfg.head_dim
        self.n_heads = max(1, d_model // cfg.head_dim)
        self.d_state = cfg.d_state
        self.block_len = 256

        d_inner = self.n_heads * self.head_dim
        self.in_proj = nn.Linear(d_model, d_inner * 2 + 2 * self.d_state, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, 4, padding=3, groups=d_inner, bias=False)
        self.dt_proj = nn.Linear(d_inner, self.n_heads, bias=False)
        self.A_log = nn.Parameter(torch.randn(self.n_heads))
        self.D = nn.Parameter(torch.randn(self.n_heads))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, past_state: Optional[torch.Tensor] = None):
        """x: (B, T, d_model). Returns (y, aux={}, past_state).

        ``past_state`` ((B, n_heads, d_head, d_state)) carries the inter-chunk
        recurrence across calls (decode). In training it is None.
        """
        B, T, _ = x.shape
        d_inner = self.n_heads * self.head_dim
        split = self.in_proj(x).split([d_inner, d_inner, self.d_state, self.d_state], dim=-1)
        x_in, z, b, c = split

        x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)[:, :T, :]
        x_conv = F.silu(x_conv)
        x_h = x_conv.reshape(B, T, self.n_heads, self.head_dim)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log).view(1, 1, self.n_heads)

        pad = (-T) % self.block_len
        if pad:
            X = F.pad(x_h, (0, 0, 0, pad))
            Adt = F.pad(A * dt, (0, 0, 0, pad))
            B_ = F.pad(b, (0, 0, 0, pad))
            C_ = F.pad(c, (0, 0, 0, pad))
        else:
            X, Adt, B_, C_ = x_h, A * dt, b, c
        X2 = X * Adt.unsqueeze(-1)

        initial = None
        if past_state is not None:
            initial = past_state.unsqueeze(1)  # (b, 1, h, p, n)
        y, final_state = ssd_minimal_discrete(X2, Adt, B_.unsqueeze(2), C_.unsqueeze(2),
                                              self.block_len, initial_states=initial)
        y = y[:, :T]

        y = y + self.D.view(1, 1, self.n_heads, 1) * x_h
        y = y.reshape(B, T, d_inner)
        y = y * F.silu(z)
        return self.out_proj(y), {}, final_state
