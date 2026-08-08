"""HybridLM: the Hybrid-MoE-70M decoder, and an EMA wrapper.

Layout: embedding (tied with the LM head) -> L x TransformerBlock -> final
RMSNorm -> tied LM head. The embedding weight is scaled by ``input_scale`` once
at init so the input embeddings have RMS ~ 1/sqrt(d_model) *AND* the tied head
outputs logits of O(1) (otherwise logits RMS ~ sqrt(d_model) at init, e.g. ~24
at d_model=576 -> CE ~300). All block aux losses (MoE balance / z) are collected
and summed with their configured coefficients.

``EMAWrapper`` keeps an fp32 exponential moving average of the parameters, updated
after every optimizer step and evaluated in place of the live weights at eval time
(documented in the design doc section 7, training recipe).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_hybrid.config import ModelConfig
from research_hybrid.attention import precompute_rope
from research_hybrid.transformer import RMSNorm, TransformerBlock


@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    aux: Dict[str, torch.Tensor]
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    hidden: Optional[torch.Tensor] = None  # final normed h (pre LM-head)


class HybridLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        with torch.no_grad():
            self.embed.weight.mul_(cfg.input_scale)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.input_scale = cfg.input_scale

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        training: bool = True,
        rope_theta: Optional[float] = None,
    ) -> ModelOutput:
        B, T = x.shape
        h = self.embed(x)  # input_scale baked into the weight at init
        theta = self.cfg.rope_theta if rope_theta is None else rope_theta

        past_len = past_kvs[0][0].shape[2] if past_kvs else 0
        cos, sin = precompute_rope(T + past_len, self.cfg.head_dim, theta,
                                   device=x.device, dtype=h.dtype, yarn=self.cfg.yarn)

        aux: Dict[str, torch.Tensor] = {}
        kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        ckpt = self.cfg.use_gradient_checkpointing and training and h.requires_grad
        for i, block in enumerate(self.blocks):
            if ckpt:
                h, block_aux = torch.utils.checkpoint.checkpoint(
                    self._block_fwd, i, h, cos, sin, training, use_reentrant=False)
                kv = None
            else:
                h, kv, block_aux = block(h, cos, sin, past_kvs[i] if past_kvs else None,
                                         use_cache=use_cache, training=training)
            if use_cache:
                kvs.append(kv)
            for k, v in block_aux.items():
                aux[k] = aux.get(k, 0.0) + v

        h = self.norm_f(h)
        if os.environ.get("PP_MEM_DEBUG") and x.is_cuda:
            print(f"[mem] after blocks: {torch.cuda.memory_allocated() // 2**20} MB "
                  f"(reserved {torch.cuda.memory_reserved() // 2**20} MB)", flush=True)

        loss = None
        logits = None
        if labels is not None:
            # Causal LM: logits[t] predicts labels[t+1]; labels are the full sequence
            # (self-supervised: labels == input ids), so logits[:-1] vs labels[1:].
            # Chunked over the final projection: a full (B, T, V) logits tensor is
            # 4.3 GB fp16 at B=8x8K / 2x32K (vocab 32768) and OOMs the 16 GB T4;
            # per-chunk it stays <~1.5 GB peak.
            loss = self._chunked_loss(h, labels, aux)
        else:
            logits = F.linear(h, self.embed.weight)

        return ModelOutput(logits=logits, loss=loss, aux=aux,
                           kv_cache=kvs if use_cache else None, hidden=h)

    def _chunked_loss(self, h: torch.Tensor, labels: torch.Tensor,
                      aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Cross-entropy over the tied head in chunks of ``cfg.loss_chunk``.

        Every chunk's CE leaves its fp32 input/softmax temps in the autograd
        graph (~768 MB x 16 chunks = ~12 GB) — they are ancestors of the
        accumulated loss. Each chunk is therefore gradient-checkpointed: only
        the chunk inputs are saved, and backward recomputes the temps one
        chunk at a time (peak ~1 GB). Checkpointing the whole loop instead
        defeats this (backward recomputes all chunks before backprop).
        """
        if h.is_cuda:
            torch.cuda.empty_cache()  # return the fragmented pool to the driver
        dbg = os.environ.get("PP_MEM_DEBUG")
        if dbg and h.is_cuda:
            print(f"[loss] h {tuple(h.shape)} {h.dtype} live {torch.cuda.memory_allocated() // 2**20} MB "
                  f"(reserved {torch.cuda.memory_reserved() // 2**20} MB)", flush=True)
        B, T, _ = h.shape
        V = self.cfg.vocab_size
        C = self.cfg.loss_chunk
        loss = torch.zeros((), device=h.device, dtype=torch.float32)
        n = 0
        for c0 in range(0, T - 1, C):
            c1 = min(c0 + C, T - 1)
            l = torch.utils.checkpoint.checkpoint(
                self._chunk_ce, h[:, c0:c1], labels[:, c0 + 1:c1 + 1],
                use_reentrant=False)
            if dbg and h.is_cuda and c0 < 2 * C:
                print(f"[loss] chunk {c0}: live {torch.cuda.memory_allocated() // 2**20} MB", flush=True)
            loss = loss + l.float() * (c1 - c0)
            n += c1 - c0
        loss = loss / max(n, 1)
        loss = loss + self.cfg.ff.balance_coef * aux.get("balance", 0.0)
        if self.cfg.ff.z_loss_coef > 0:
            loss = loss + self.cfg.ff.z_loss_coef * aux.get("z", 0.0)
        return loss

    def _chunk_ce(self, h_chunk: torch.Tensor, y_chunk: torch.Tensor) -> torch.Tensor:
        lg = F.linear(h_chunk, self.embed.weight)
        return F.cross_entropy(lg.reshape(-1, self.cfg.vocab_size), y_chunk.reshape(-1))

    def _block_fwd(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   training: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single block forward returning (h, aux) — used by gradient checkpointing."""
        h, kv, aux = self.blocks[i](x, cos, sin, None, False, training)
        return h, aux

    def num_parameters(self, active: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not active:
            return total
        per_expert = 3 * self.cfg.d_model * self.cfg.ff.routed_d_ff
        inactive = (self.cfg.ff.n_routed - self.cfg.ff.top_k) * per_expert * self.cfg.n_layers
        return total - inactive


class EMAWrapper:
    """fp32 EMA of the model parameters (decay 0.999), evaluated in place of live weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().float().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
