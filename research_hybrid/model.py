"""HybridLM: the Hybrid-MoE-70M decoder, and an EMA wrapper.

Layout: embedding (input-scaled, tied with the LM head) -> L x TransformerBlock ->
final RMSNorm -> tied LM head. All block aux losses (MoE balance / z) are collected and
summed with their configured coefficients.

``EMAWrapper`` keeps an fp32 exponential moving average of the parameters, updated
after every optimizer step and evaluated in place of the live weights at eval time
(documented in the design doc section 7, training recipe).
"""

from __future__ import annotations

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
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    aux: Dict[str, torch.Tensor]
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]


class HybridLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
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
        h = self.embed(x) * self.input_scale
        theta = self.cfg.rope_theta if rope_theta is None else rope_theta

        past_len = past_kvs[0][0].shape[2] if past_kvs else 0
        cos, sin = precompute_rope(T + past_len, self.cfg.head_dim, theta,
                                   device=x.device, dtype=h.dtype, yarn=self.cfg.yarn)

        aux: Dict[str, torch.Tensor] = {}
        kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            h, kv, block_aux = block(h, cos, sin, past_kvs[i] if past_kvs else None,
                                     use_cache=use_cache, training=training)
            if use_cache:
                kvs.append(kv)
            for k, v in block_aux.items():
                aux[k] = aux.get(k, 0.0) + v

        h = self.norm_f(h)
        logits = F.linear(h, self.embed.weight)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), labels.view(-1))
            loss = loss + self.cfg.ff.balance_coef * aux.get("balance", 0.0)
            if self.cfg.ff.z_loss_coef > 0:
                loss = loss + self.cfg.ff.z_loss_coef * aux.get("z", 0.0)

        return ModelOutput(logits=logits, loss=loss, aux=aux, kv_cache=kvs if use_cache else None)

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
