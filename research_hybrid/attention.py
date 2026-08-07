"""Grouped-Query Attention with RoPE, sparse masks, and kernel dispatch.

Design (v2, 32K context — docs/research_report_32k.md section 2):
  - pattern "causal"          -> dense causal (stage A curriculum, 8K context)
  - pattern "sliding_window"  -> causal window W (Mistral-style, ablation)
  - pattern "block_sparse"    -> causal ∩ {|q-kv| <= W} ∩ {kv < anchor}
                               (window + sink/anchor block: StreamingLLM, SWAA) — DEFAULT
  - pattern "moba"            -> learned top-k block routing (MoBA, arXiv:2502.13189,
                               ablation; data-dependent selection, chunked path only)

Kernel dispatch ("auto"):
  - FlexAttention (Triton backend) on CUDA sm70+/MPS (torch >= 2.5/2.13),
  - SDPA (flash/mem-efficient) for dense causal on CUDA,
  - generic chunked sparse attention everywhere else (P100 sm60, CPU; bounded memory).

Per-head max-logit capture (for MuonClip's QK-Clip, arXiv:2507.20534 Algorithm 1)
is computed over the same allowed KV positions and stored in ``self.max_logits``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_hybrid.config import AttentionConfig, YaRNConfig


def _yarn_ntk_inv_freq(head_dim: int, theta: float, factor: float, seq_len: int,
                       low_ratio: float = 1.0, high_ratio: float = 0.1) -> torch.Tensor:
    """YaRN NTK-by-parts inverse frequencies (Peng et al., arXiv:2309.00071).

    Faithful to the canonical implementations (jquesnelle/yarn, HF transformers
    ``_compute_yarn_parameters``): dimensions whose wavelength exceeds
    ``low_ratio * seq_len`` keep the original frequency; dimensions with wavelength
    below ``high_ratio * seq_len`` use the extended (scaled) base; in between the
    two are linearly ramped. The attention-logit temperature of full YaRN is not
    applied (consistent with HF's yarn path); the rope_theta replacement alone is
    the v2 report's selected future-extension mechanism.
    """
    n = head_dim // 2
    dims = torch.arange(0, n, dtype=torch.float32)
    base_inv = 1.0 / (theta ** (dims / head_dim))
    ext_theta = theta * factor ** (head_dim / (head_dim - 2))
    ext_inv = 1.0 / (ext_theta ** (dims / head_dim))
    wavelength = 2.0 * math.pi * (theta ** (dims / head_dim))
    lo, hi = seq_len * low_ratio, seq_len * high_ratio
    ramp = torch.clamp((wavelength - hi) / (lo - hi), 0.0, 1.0)
    inv_freq = base_inv * (1.0 - ramp) + ext_inv * ramp
    return inv_freq


def precompute_rope(seq_len: int, head_dim: int, theta: float = 1_000_000.0,
                    device=None, dtype=torch.float32, yarn: Optional[YaRNConfig] = None):
    """Rotary position embeddings (RoPE), precomputed cos/sin tables.

    theta=1e6 is the Mistral-7B recipe for a native 32K context (no interpolation);
    theta=1e4 was the v1 (2K) setting. With ``yarn.enabled`` the inverse frequencies
    are rescaled by the NTK-by-parts scheme for extension beyond the trained length.
    """
    dims = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    if yarn is not None and yarn.enabled:
        inv_freq = _yarn_ntk_inv_freq(head_dim, yarn.original_theta, yarn.factor, seq_len)
        inv_freq = inv_freq.to(device)
    else:
        inv_freq = 1.0 / (theta ** (dims / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos[None, None, :, :], sin[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the last axis of ``x`` (B, H, T, head_dim).

    cos/sin have half the head_dim (one entry per frequency pair); they are
    duplicated so they broadcast elementwise against the full head_dim.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).reshape(x.shape)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    return x * cos + rotated * sin


def build_mask_mod(cfg: AttentionConfig) -> Callable:
    """Return a FlexAttention-style ``mask_mod(b, h, q_idx, kv_idx) -> bool``."""
    if cfg.pattern == "causal":

        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx

        return mask_mod
    if cfg.pattern == "sliding_window":
        w = cfg.window_size

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx <= w) & (kv_idx <= q_idx)

        return mask_mod
    if cfg.pattern == "block_sparse":
        w, anchor = cfg.window_size, cfg.anchor_size

        def mask_mod(b, h, q_idx, kv_idx):
            local = (q_idx - kv_idx <= w) & (kv_idx <= q_idx)
            sink = kv_idx < anchor
            return local | sink

        return mask_mod
    raise ValueError(f"pattern {cfg.pattern!r} has no static mask_mod (moba is chunked-only)")


def _flex_capable(device: torch.device) -> bool:
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        return cap is not None and cap[0] >= 7
    if device.type == "mps":
        return True
    return False


def _kv_slice(q0: int, qe: int, tk: int, cfg: AttentionConfig) -> Optional[torch.Tensor]:
    """KV index set allowed by the static mask for query positions in [q0, qe).

    Returns sorted long indices (contiguous slices for causal/SWA/block-sparse).
    """
    if cfg.pattern == "causal":
        return torch.arange(0, qe)
    if cfg.pattern == "sliding_window":
        lo = max(0, q0 - cfg.window_size)
        if lo >= qe:
            return None
        return torch.arange(lo, qe)
    if cfg.pattern == "block_sparse":
        lo = max(cfg.anchor_size, q0 - cfg.window_size)
        parts = []
        if cfg.anchor_size > 0 and 0 < qe:
            parts.append(torch.arange(0, min(cfg.anchor_size, qe)))
        if lo < qe:
            parts.append(torch.arange(lo, qe))
        return torch.cat(parts) if parts else None
    raise ValueError(f"pattern {cfg.pattern!r} is data-dependent; use MoBA selection instead")


def _moba_selection(q: torch.Tensor, k: torch.Tensor, cfg: AttentionConfig) -> torch.Tensor:
    """MoBA block selection (arXiv:2502.13189): top-k blocks by pooled affinity.

    Per query block, affinity = mean-pooled q · mean-pooled k^T (causal over
    blocks); the current block plus the top-k previous blocks are attended.
    Mean-pooled hidden-state affinity matches public reproductions of the paper's
    coarse selection. Returns (B, H, n_blocks, n_sel) sorted kv index tensors.
    """
    B, H, T, hd = q.shape
    bs = cfg.block_size
    n_blocks = math.ceil(T / bs)
    if k.shape[1] != H:  # GQA: broadcast KV heads to the query head count
        k = k.repeat_interleave(H // k.shape[1], dim=1)
    qp = q.reshape(B, H, n_blocks, bs, hd).mean(dim=3)
    kp = k.reshape(B, H, n_blocks, bs, hd).mean(dim=3)
    aff = torch.matmul(qp, kp.transpose(-2, -1)) / math.sqrt(hd)
    tri = torch.triu(torch.ones(n_blocks, n_blocks, device=q.device, dtype=torch.bool), diagonal=1)
    aff = aff.masked_fill(tri, float("-inf"))
    sel = aff.topk(cfg.top_blocks, dim=-1).indices
    current = torch.arange(n_blocks, device=q.device).view(1, 1, -1).expand(B, H, n_blocks)
    sel = torch.cat([sel, current.unsqueeze(-1)], dim=-1)
    sel = torch.sort(torch.unique(sel, dim=-1), dim=-1).values
    positions = torch.arange(T, device=q.device).reshape(n_blocks, bs)
    return positions[sel]


def _kv_indexer(q: torch.Tensor, k: torch.Tensor, tk: int, cfg: AttentionConfig) -> Callable:
    """Return ``index_fn(q0, qe) -> Optional[Tensor]``: allowed KV indices per chunk."""
    if cfg.pattern == "moba":
        sel = _moba_selection(q, k, cfg)
        bs = cfg.block_size

        def index_fn(q0: int, qe: int):
            b0, b1 = q0 // bs, (qe - 1) // bs + 1
            rows = sel[:, :, b0:b1].reshape(sel.shape[0], sel.shape[1], -1)
            flat = torch.unique(torch.sort(rows, dim=-1).values, dim=-1)
            return flat.reshape(-1)

        return index_fn
    slices = [_kv_slice(q0, min(q0 + cfg.chunk_size, tk), tk, cfg)
              for q0 in range(0, tk, cfg.chunk_size)]

    def index_fn(q0: int, qe: int):
        i = q0 // cfg.chunk_size
        s = slices[i]
        return None if s is None else s.to(q.device)

    return index_fn


class ChunkedSparseAttention:
    """Generic chunked attention over an arbitrary allowed-KV set.

    The single reference implementation used by every sparse pattern on devices
    without FlexAttention (P100/sm60, CPU) and by MoBA everywhere. Memory per
    chunk is bounded by ``chunk_size * (window + anchor)``.
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg

    @staticmethod
    def _attend(qc, kc, vc, causal_mask: torch.Tensor) -> torch.Tensor:
        B, H, C, hd = qc.shape
        if kc.shape[1] != H:  # GQA: broadcast KV heads to the query head count
            factor = H // kc.shape[1]
            kc = kc.repeat_interleave(factor, dim=1)
            vc = vc.repeat_interleave(factor, dim=1)
        out = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=causal_mask.unsqueeze(0).unsqueeze(0))
        return out

    def _allowed(self, idx: torch.Tensor, q: torch.Tensor, c: int) -> torch.Tensor:
        """Static-mask equivalent over the chunk: returns (C, Lk) bool.

        idx is the chunk's allowed KV set (sorted); rows are query positions
        q0..q0+c-1. The window boundary q - kv <= w must be enforced here
        (the slice alone only bounds the start of the KV range). MoBA's
        block selection already is the mask: all selected KV are allowed.
        """
        if self.cfg.pattern == "moba":
            return torch.ones(c, len(idx), dtype=torch.bool, device=q.device)
        if self.cfg.pattern == "causal":
            return idx[None, :] <= q[:, None]
        if self.cfg.pattern == "sliding_window":
            return (idx[None, :] <= q[:, None]) & (idx[None, :] >= q[:, None] - self.cfg.window_size)
        if self.cfg.pattern == "block_sparse":
            local = (idx[None, :] <= q[:, None]) & (idx[None, :] >= q[:, None] - self.cfg.window_size)
            return local | (idx[None, :] < self.cfg.anchor_size)
        raise ValueError(f"pattern {self.cfg.pattern!r} is data-dependent; use MoBA selection instead")

    def run(self, q, k, v, index_fn: Callable) -> torch.Tensor:
        B, H, T, hd = q.shape
        C = self.cfg.chunk_size
        out = torch.zeros_like(q)
        for q0 in range(0, T, C):
            qe = min(q0 + C, T)
            c = qe - q0
            idx = index_fn(q0, qe)
            if idx is not None and len(idx) > 0:
                kc = k[:, :, idx]
                vc = v[:, :, idx]
                qq = q0 + torch.arange(c, device=q.device)
                allowed = self._allowed(idx, qq, c)
                out[:, :, q0:qe] = self._attend(q[:, :, q0:qe], kc, vc, allowed)
        return out

    def max_logits(self, q, k, v, index_fn: Callable, scale: float) -> Optional[torch.Tensor]:
        """Per-head max of the (scaled) allowed logits: S_max^h = max_{B,i,j} q_i·k_j/√d.

        Used by MuonClip's QK-Clip (arXiv:2507.20534, Algorithm 1, step 2).
        """
        B, H, T, hd = q.shape
        C = self.cfg.chunk_size
        smax = None
        for q0 in range(0, T, C):
            qe = min(q0 + C, T)
            c = qe - q0
            idx = index_fn(q0, qe)
            if idx is None or len(idx) == 0:
                continue
            kc = k[:, :, idx]
            if kc.shape[1] != H:  # GQA: broadcast KV heads to the query head count
                kc = kc.repeat_interleave(H // kc.shape[1], dim=1)
            scores = torch.matmul(q[:, :, q0:qe], kc.transpose(-2, -1)) * scale
            qq = q0 + torch.arange(c, device=q.device)
            scores = scores.masked_fill(~self._allowed(idx, qq, c), float("-inf"))
            cur = scores.amax(dim=(0, 2, 3))
            smax = cur if smax is None else torch.maximum(smax, cur)
        return smax



class CausalGQA(nn.Module):
    """Grouped-Query Attention (GQA; Ainslie et al., EMNLP 2023) with v2 masks.

    Head layout: 9 query heads share 3 key/value heads; head_dim=64 keeps
    FlashAttention-2 / FlexAttention kernels in their fastest configuration.
    """

    def __init__(self, cfg: AttentionConfig, d_model: int, n_q_heads: int, n_kv_heads: int,
                 head_dim: int, rope_theta: float = 1_000_000.0,
                 qk_clip_tau: Optional[float] = None):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.qk_clip_tau = qk_clip_tau
        self.max_logits: Optional[torch.Tensor] = None
        self.kv_dim = n_kv_heads * head_dim

        self.wq = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.kv_dim, bias=False)
        self.wv = nn.Linear(d_model, self.kv_dim, bias=False)
        self.wo = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        nn.init.zeros_(self.wo.weight)

    def _reshape_gqa(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, heads, self.head_dim).transpose(1, 2)

    def _decode_mask(self, tk: int, device, dtype) -> torch.Tensor:
        """Additive decode mask matching the training pattern (window + anchor)."""
        if self.cfg.pattern in ("causal", "moba"):
            mask = torch.zeros(1, 1, 1, tk, dtype=dtype, device=device)
            return mask
        pos = torch.arange(tk, device=device)
        if self.cfg.pattern == "sliding_window":
            allowed = (pos >= tk - 1 - self.cfg.window_size)
        else:
            allowed = (pos < self.cfg.anchor_size) | (pos >= tk - 1 - self.cfg.window_size)
        mask = torch.zeros(1, 1, 1, tk, dtype=dtype, device=device)
        mask.masked_fill_(~allowed.view(1, 1, 1, -1), float("-inf"))
        return mask

    def _scale(self, head_dim: int) -> float:
        return head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        q = self._reshape_gqa(self.wq(x), self.n_q_heads)
        k = self._reshape_gqa(self.wk(x), self.n_kv_heads)
        v = self._reshape_gqa(self.wv(x), self.n_kv_heads)
        q, k = apply_rope(q, cos[:, :, :T], sin[:, :, :T]), apply_rope(k, cos[:, :, :T], sin[:, :, :T])

        present_kv = None
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        if use_cache:
            present_kv = (k, v)

        Tk = k.shape[2]
        decode = T == 1 and Tk > 1

        if decode:
            if k.shape[1] != q.shape[1]:  # GQA decode: broadcast KV heads
                factor = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(factor, dim=1)
                v = v.repeat_interleave(factor, dim=1)
            attn_mask = self._decode_mask(Tk, x.device, x.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        elif self.cfg.pattern == "causal" and self.cfg.kernel in ("auto", "sdpa"):
            if x.is_cuda or _flex_capable(x.device):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                out = self._chunked(q, k, v, Tk)
        elif self.cfg.kernel == "flex" or (self.cfg.kernel == "auto" and self.cfg.pattern != "causal"):
            try:
                out = self._flex(q, k, v)
            except Exception:
                out = self._chunked(q, k, v, Tk)
        else:
            out = self._chunked(q, k, v, Tk)

        self.max_logits = None
        if self.qk_clip_tau is not None and not decode:
            self.max_logits = self._compute_max_logits(q, k, v, Tk)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_q_heads * self.head_dim)
        return self.wo(out), present_kv

    def _chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tk: int) -> torch.Tensor:
        return ChunkedSparseAttention(self.cfg).run(q, k, v, _kv_indexer(q, k, tk, self.cfg))

    def _compute_max_logits(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tk: int) -> torch.Tensor:
        return ChunkedSparseAttention(self.cfg).max_logits(
            q, k, v, _kv_indexer(q, k, tk, self.cfg), self._scale(self.head_dim))

    def _flex(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("kernel='flex' requires torch >= 2.5") from exc
        B, H, T, hd = q.shape
        pad = (-T) % 128
        if pad:
            q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
        mask_mod = build_mask_mod(self.cfg)
        block_mask = create_block_mask(mask_mod, B, H, q.shape[2], k.shape[2], device=q.device)
        out = flex_attention(q, k, v, block_mask=block_mask)
        return out[:, :, :T, :]
