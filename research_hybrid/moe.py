"""DeepSeekMoE feed-forward block (arXiv:2401.06066).

Assembles: token-choice router (top-k) + shared experts (always on) + fine-grained
routed experts, with an optional per-expert capacity cap. Tokens that overflow an
expert are dropped (their routed contribution is skipped; the residual connection
carries them forward, per the Switch Transformer convention).

Auxiliary losses returned as a dict for the training loop to scale and sum:
  - ``balance``:  DeepSeekMoE load-balancing loss (alpha-weighted by the caller)
  - ``z``:        optional router z-loss
  - ``dropped``:  fraction of routing slots dropped (logged, not a loss)

Design: docs/research_report.md sections 3.6 and 6.3.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from research_hybrid.config import MoEConfig
from research_hybrid.experts import SharedExperts, RoutedExperts
from research_hybrid.router import Router, expert_capacity, load_balancing_loss, router_z_loss


class DeepSeekMoE(nn.Module):
    def __init__(self, cfg: MoEConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.router = Router(d_model, cfg.n_routed, cfg.top_k,
                             jitter_noise=cfg.jitter_noise, routing_fn=cfg.routing_fn)
        self.shared = SharedExperts(cfg.n_shared, d_model, cfg.shared_d_ff)
        self.routed = RoutedExperts(cfg.n_routed, d_model, cfg.routed_d_ff)

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, _ = x.shape
        num_tokens = B * T

        route = self.router(x, training=training)
        cap = expert_capacity(num_tokens, self.cfg.n_routed, self.cfg.capacity_factor)

        token_flat = x.view(num_tokens, -1)
        ids = route.indices.view(-1)
        weights = route.weights.view(-1)
        slot_tokens = (torch.arange(num_tokens, device=x.device)
                       .unsqueeze(1).expand(num_tokens, self.cfg.top_k).reshape(-1))

        dropped = 0
        if cap is not None and cap < num_tokens:
            keep_slot = torch.zeros(ids.numel(), dtype=torch.bool, device=x.device)
            for i in range(self.cfg.n_routed):
                mask = ids == i
                n = mask.sum().item()
                if n > cap:
                    keep_slot |= mask & (torch.cumsum(mask, dim=0) <= cap)
                    dropped += n - cap
                else:
                    keep_slot |= mask
            token_flat = token_flat[slot_tokens[keep_slot]]
            ids = ids[keep_slot]
            weights = weights[keep_slot]
            slot_tokens = slot_tokens[keep_slot]
        else:
            token_flat = token_flat[slot_tokens]

        routed_out, _ = self.routed(token_flat, ids, weights)
        out = torch.zeros_like(x.view(num_tokens, -1))
        out.index_add_(0, slot_tokens, routed_out)

        out = out.view(B, T, self.d_model) + self.shared(x)

        aux: Dict[str, torch.Tensor] = {}
        aux["balance"] = load_balancing_loss(route.logits, route.indices, self.cfg.n_routed, num_tokens)
        if self.cfg.z_loss_coef > 0:
            aux["z"] = router_z_loss(route.logits)
        aux["dropped"] = torch.tensor(dropped / max(route.indices.numel(), 1), device=x.device)
        return out, aux
