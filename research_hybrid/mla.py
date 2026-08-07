"""Multi-Head Latent Attention (MLA), training form — DeepSeek-V2 (arXiv:2405.04434).

**Status: implemented, default OFF.** The research report (section 5.2) postpones MLA
for the 70M/2048-context model: at this scale GQA-3 already yields an ~19 MB total KV
cache, while MLA pays its complexity budget (decoupled RoPE, latent compression, and
absorbed-projection inference) only at long context. This module implements the full
training-time forward faithfully, so a future long-context v2 can enable it without a
rewrite. The inference-time *absorption* trick (folding W^O and W^UK into the output
projection so the cache stores only ``[c_t; k_t^R]``) is documented in the docstring but
not implemented, per the "no invented math" rule: absorption requires custom kernels
(change of basis in the KV up-projection) and is not exercised by training.

Math (DeepSeek-V2, Eqs. 1-2 + 9):
  c_t^Q  = W_DQ  h_t          c_t^KV = W_DKV h_t          c_t^R = W_DKR h_t
  q_t^C  = W_UQ  c_t^Q        k_t^C  = W_UK  c_t^KV       v_t = W_UV c_t^KV
  q_t^R  = W_QR  c_t^R        k_t^R  = W_UR  c_t^R
  q_t    = [q_t^C; RoPE(q_t^R)]   k_t = [k_t^C; RoPE(k_t^R)]   v_t
  out    = softmax(q·k^T/√d)·v  (multi-head, single shared KV latent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_hybrid.attention import apply_rope


@dataclass
class MLAConfig:
    d_c: int = 128
    d_c_q: int = 256
    d_c_rope: int = 16


class MLAAttention(nn.Module):
    """Training-form MLA. KV cache (when used) stores the *unfused* key/value tensors."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope_theta: float,
                 cfg: MLAConfig = MLAConfig()):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.cfg = cfg

        self.w_dq = nn.Linear(d_model, cfg.d_c_q, bias=False)
        self.w_dkv = nn.Linear(d_model, cfg.d_c, bias=False)
        self.w_dkr = nn.Linear(d_model, cfg.d_c_rope, bias=False)

        self.w_uq = nn.Linear(cfg.d_c_q, n_heads * (head_dim - cfg.d_c_rope), bias=False)
        self.w_qr = nn.Linear(cfg.d_c_rope, n_heads * cfg.d_c_rope, bias=False)
        self.w_uk = nn.Linear(cfg.d_c, n_heads * (head_dim - cfg.d_c_rope), bias=False)
        self.w_ur = nn.Linear(cfg.d_c_rope, n_heads * cfg.d_c_rope, bias=False)
        self.w_uv = nn.Linear(cfg.d_c, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        nn.init.zeros_(self.wo.weight)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        c_q, c_kv, c_r = self.w_dq(x), self.w_dkv(x), self.w_dkr(x)

        q_c = self.w_uq(c_q).view(B, T, self.n_heads, self.head_dim - self.cfg.d_c_rope)
        q_r = apply_rope(
            self.w_qr(c_r).view(B, T, self.n_heads, self.cfg.d_c_rope),
            cos[:, :, :T], sin[:, :, :T],
        )
        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)

        k_c = self.w_uk(c_kv).view(B, T, self.n_heads, self.head_dim - self.cfg.d_c_rope)
        k_r = apply_rope(
            self.w_ur(c_r).view(B, T, self.n_heads, self.cfg.d_c_rope),
            cos[:, :, :T], sin[:, :, :T],
        )
        k = torch.cat([k_c, k_r], dim=-1).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        present_kv = None
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        if use_cache:
            present_kv = (k, v)

        if k.shape[2] > 1 and T == 1:
            attn_mask = torch.ones(1, 1, T, k.shape[2], dtype=x.dtype, device=x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out), present_kv
