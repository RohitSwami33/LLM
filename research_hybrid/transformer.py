"""Transformer block: pre-norm, GQA (or Mamba-2 SSD) attention, DeepSeekMoE FFN,
optional mHC wrapper.

The block is a standard pre-norm residual unit:
    x = x + Attn(LN1(x))
    x = x + MoE(LN2(x))
When ``cfg.hybrid.enabled``, layers listed in ``cfg.hybrid.attn_layers`` use the
chunked Mamba-2 SSD layer instead of attention (Jamba-style interleaving; see
docs/research_report_32k.md section 3). When ``cfg.mhc.enabled``, the block
operates on an n-stream residual (B, T, n, C) with the two sub-layers fused
inside the mHC residual update (Eq. 3 of arXiv:2512.24880); see ``mhc.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from research_hybrid.config import ModelConfig
from research_hybrid.attention import CausalGQA, precompute_rope
from research_hybrid.mla import MLAConfig, MLAAttention
from research_hybrid.moe import DeepSeekMoE
from research_hybrid.mhc import MHCLayer
from research_hybrid.mamba2 import Mamba2Block


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)

        self.use_mamba = cfg.hybrid.enabled and layer_idx not in cfg.hybrid.attn_layers
        if self.use_mamba:
            self.mamba = Mamba2Block(cfg.hybrid, cfg.d_model)
            self.attn = None
        elif getattr(cfg, "use_mla", False):
            mla_cfg = getattr(cfg, "mla", MLAConfig())
            self.attn = MLAAttention(cfg.d_model, cfg.n_q_heads, cfg.head_dim,
                                     cfg.rope_theta, mla_cfg)
        else:
            self.attn = CausalGQA(cfg.attention, cfg.d_model, cfg.n_q_heads,
                                  cfg.n_kv_heads, cfg.head_dim, cfg.rope_theta,
                                  qk_clip_tau=cfg.qk_clip_tau)
        self.moe = DeepSeekMoE(cfg.ff, cfg.d_model)

        if cfg.mhc.enabled:
            self.mhc = MHCLayer(cfg.d_model, cfg.mhc.n_streams, cfg.mhc.sinkhorn_iters)
        else:
            self.mhc = None

    def _inner(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
               use_cache: bool, training: bool):
        if self.use_mamba:
            h, mstate = self.mamba(self.norm1(x), past_state=past_kv)
            kv = mstate if use_cache else None
        else:
            h, kv = self.attn(self.norm1(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        h = h + x
        h, aux = self.moe(self.norm2(h), training=training)
        return h + x, kv, aux

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False, training: bool = True):
        if self.mhc is None:
            h, kv, aux = self._inner(x, cos, sin, past_kv, use_cache, training)
            return h, kv, aux
        if past_kv is not None:
            raise ValueError("mHC + KV-cache decoding is not implemented (mHC is off by default)")

        stream = MHCLayer.expand(x, self.cfg.mhc.n_streams)

        def block(flat_in: torch.Tensor) -> torch.Tensor:
            h, _, aux = self._inner(flat_in, cos, sin, None, False, training)
            self._aux = aux
            return h

        stream = self.mhc(block, stream)
        aux = getattr(self, "_aux", {})
        return MHCLayer.readout(stream), None, aux
