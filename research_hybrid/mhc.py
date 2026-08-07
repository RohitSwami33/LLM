"""Manifold-Constrained Hyper-Connections (mHC) — DeepSeek-AI, arXiv:2512.24880.

**Status: implemented from the official paper, default OFF.** The research report
(section 5.8) postpones mHC for the 70M/6-layer model: the paper's wins are for deep
(100+ residual streams... in practice 3B/9B/27B) models where unconstrained
Hyper-Connections (HC) break the identity-mapping property of the residual stream.
With 6 layers the plain residual is already well-conditioned and n-stream expansion
costs n× activation memory/I/O. This module implements the exact published math so the
ablation is faithful:

  Layer update (Eq. 3):
    x_{l+1} = H^res_l · x_l + (H^post_l)^T · F(H^pre_l · x_l, W_l)

  Coefficients (Eq. 7), with vec(x) the flattened n·C stream:
    H~^pre  = α^pre · (RMSNorm(vec(x)) · φ^pre)  + b^pre    (1 x n)
    H~^post = α^post · (RMSNorm(vec(x)) · φ^post) + b^post   (1 x n)
    H~^res  = α^res · mat(RMSNorm(vec(x)) · φ^res) + b^res   (n x n)

  Projection (Eq. 8):  H^pre = σ(H~^pre), H^post = 2σ(H~^post),
                       H^res = Sinkhorn-Knopp(H~^res)

  Sinkhorn-Knopp (Eq. 9):  M^(0) = exp(H~^res); M^(t) = T_r(T_c(M^(t-1))), t_max = 20

Doubly-stochastic H^res ⇒ spectral norm ≤ 1 (non-expansive), closed under
multiplication, and a convex combination of features (Birkhoff polytope) — restoring
the identity-mapping stability HC lost. ``sinkhorn_iters`` = 20 as in the paper.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinkhorn_knopp(m: torch.Tensor, iters: int) -> torch.Tensor:
    """Entropic projection onto the Birkhoff polytope (doubly stochastic matrices).

    Eq. 9: start from exp(H), alternately normalize columns then rows. We subtract the
    row max before exponentiating for numerical stability; this is a per-row positive
    rescaling that the row normalization absorbs, so the projection is unchanged.
    """
    m = m - m.amax(dim=-1, keepdim=True)
    m = torch.exp(m)
    for _ in range(iters):
        m = m / (m.sum(dim=-1, keepdim=True) + 1e-9)
        m = m / (m.sum(dim=-2, keepdim=True) + 1e-9)
    return m


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * self.weight


class MHCLayer(nn.Module):
    """mHC wrapper around one block function F: (B, T, C) -> (B, T, C).

    Operates on an n-stream residual R of shape (B, T, n, C).
    """

    def __init__(self, d_model: int, n_streams: int, sinkhorn_iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        nC = n_streams * d_model

        self.norm = RMSNorm(nC)
        self.phi_pre = nn.Parameter(torch.empty(nC, n_streams))
        self.phi_post = nn.Parameter(torch.empty(nC, n_streams))
        self.phi_res = nn.Parameter(torch.empty(nC, n_streams * n_streams))
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.phi_pre, std=0.02)
        nn.init.normal_(self.phi_post, std=0.02)
        nn.init.normal_(self.phi_res, std=0.02)

    def _coefficients(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # vec: (B, T, n*C); Eq. 7 (flattened) then Eq. 8.
        z = self.norm(vec)
        h_pre = self.alpha_pre * torch.einsum("btc,cn->btn", z, self.phi_pre) + self.b_pre
        h_post = self.alpha_post * torch.einsum("btc,cn->btn", z, self.phi_post) + self.b_post
        h_res = self.alpha_res * torch.einsum(
            "btc,cab->btab", z, self.phi_res.view(self.n * self.d_model, self.n, self.n)
        ) + self.b_res
        return torch.sigmoid(h_pre), 2.0 * torch.sigmoid(h_post), _sinkhorn_knopp(h_res, self.sinkhorn_iters)

    def forward(self, block: Callable[[torch.Tensor], torch.Tensor], r: torch.Tensor) -> torch.Tensor:
        """r: (B, T, n, C). Returns updated stream R' (Eq. 3)."""
        B, T, n, C = r.shape
        vec = r.view(B, T, n * C)
        h_pre, h_post, h_res = self._coefficients(vec)

        block_in = torch.einsum("btn,btc->btc", h_pre, r)          # H^pre · R -> (B,T,C)
        block_out = block(block_in)                                 # F(·)
        stream_add = h_post.unsqueeze(-1) * block_out.unsqueeze(2)  # H^post^T · F -> (B,T,n,C)
        mixed = torch.einsum("btab,btc->btac", h_res, r)            # H^res · R
        return mixed + stream_add

    @staticmethod
    def expand(x: torch.Tensor, n: int) -> torch.Tensor:
        """Initialize the n-stream residual by tiling the input (HC convention)."""
        return x.unsqueeze(2).expand(-1, -1, n, -1)

    @staticmethod
    def readout(r: torch.Tensor) -> torch.Tensor:
        """Compress the stream back to a single vector (uniform readout)."""
        return r.mean(dim=2)
