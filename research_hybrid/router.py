"""Token-choice router with auxiliary losses (Switch/DeepSeek lineage).

- ``Router``: logits -> softmax (or sigmoid) -> top-k -> renormalized weights.
  Optional training-time jitter noise on the logits (Mixtral, arXiv:2401.04088).
- ``load_balancing_loss``: DeepSeekMoE formulation L_bal = N * mean_i(f_i * P_i)
  (Dai et al., arXiv:2401.06066, Eq. 2; equivalent to the Switch Transformer loss
  scaled by N, Fedus et al. Eq. 4).
- ``router_z_loss``: logsumexp^2 penalty (Zoph et al., arXiv:2202.08906), optional.
- ``expert_capacity``: fixed per-expert batch (Switch, Eq. 3); ``None`` disables
  capacity (DeepSeek-V2 no-drop group routing).

Design: docs/research_report.md sections 3.5-3.7 and 6.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouteOutput:
    weights: torch.Tensor
    indices: torch.Tensor
    logits: torch.Tensor
    tokens_per_expert: torch.Tensor
    dropped: int


class Router(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 jitter_noise: float = 0.0,
                 routing_fn: str = "softmax_topk"):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.routing_fn = routing_fn
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor, training: bool) -> RouteOutput:
        B, T, _ = x.shape
        flat = x.view(B * T, -1)

        if training and self.jitter_noise > 0.0:
            flat = flat * torch.empty_like(flat).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        logits = self.gate(flat)
        if self.routing_fn == "softmax_topk":
            probs = F.softmax(logits, dim=-1)
        elif self.routing_fn == "sigmoid_topk":
            probs = torch.sigmoid(logits)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unknown routing_fn: {self.routing_fn}")

        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return RouteOutput(
            weights=weights, indices=indices, logits=logits,
            tokens_per_expert=None, dropped=0,
        )


def load_balancing_loss(logits: torch.Tensor, indices: torch.Tensor,
                        n_experts: int, num_tokens: int) -> torch.Tensor:
    """L_bal = N * sum_i f_i * P_i, computed over the batch (DeepSeekMoE Eq. 2)."""
    probs = F.softmax(logits, dim=-1)
    f = torch.zeros(n_experts, device=logits.device, dtype=logits.dtype)
    f.scatter_add_(0, indices.flatten(), torch.ones(indices.numel(), device=logits.device, dtype=logits.dtype))
    f = f / max(num_tokens, 1)
    p = probs.mean(dim=0)
    return n_experts * torch.sum(f * p)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Z-loss: mean over tokens of log(Σ exp(logits))^2."""
    return torch.logsumexp(logits, dim=-1).square().mean()


def expert_capacity(num_tokens: int, n_experts: int, capacity_factor: Optional[float]) -> int:
    """Switch Transformer Eq. 3: ceil(tokens/N * capacity_factor); None => no cap."""
    if capacity_factor is None:
        return num_tokens
    return math.ceil(num_tokens / n_experts * capacity_factor)
