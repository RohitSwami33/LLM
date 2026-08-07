"""Expert modules for the DeepSeekMoE feed-forward.

- ``SwiGLU``: the expert body (gated linear unit), shared by shared and routed experts.
- ``SharedExperts``: Ks always-on experts capturing common knowledge
  (DeepSeekMoE, arXiv:2401.06066, shared expert isolation).
- ``RoutedExperts``: N fine-grained experts with token-grouped dispatch. Tokens are
  sorted by their assigned expert so each expert processes one contiguous, vectorized
  batch (no per-token Python loops), then scatter-adds the weighted outputs.

Design: docs/research_report.md sections 3.6 and 6.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU expert: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, d_ff: int, zero_down: bool = True):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        if zero_down:
            nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SharedExperts(nn.Module):
    """All shared experts applied to all tokens."""

    def __init__(self, n_shared: int, d_model: int, d_ff: int):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for expert in self.experts:
            out = out + expert(x)
        return out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RoutedExperts(nn.Module):
    """N fine-grained experts with grouped dispatch.

    ``forward(token_flat, expert_ids, weights)``:
      - ``token_flat``: (T, d) — tokens replicated per routing slot (T = B*S*top_k).
      - ``expert_ids``:  (T,) int64.
      - ``weights``:     (T,) float — normalized routing weights.
    Tokens are sorted by expert id; each expert is applied to its contiguous slice and
    the weighted result scatter-added back into the output buffer. Returns
    (output (T, d), n_tokens_per_expert).
    """

    def __init__(self, n_experts: int, d_model: int, d_ff: int):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, token_flat: torch.Tensor, expert_ids: torch.Tensor,
                weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, d = token_flat.shape
        if T == 0:
            return token_flat, torch.zeros(self.n_experts, dtype=torch.long, device=token_flat.device)

        sorted_ids, order = torch.sort(expert_ids)
        sorted_weights = weights[order]
        sorted_tokens = token_flat[order]

        boundaries = torch.bincount(sorted_ids, minlength=self.n_experts)
        offsets = torch.cumsum(boundaries, dim=0) - boundaries
        out = torch.zeros_like(sorted_tokens)

        for i in range(self.n_experts):
            n = boundaries[i].item()
            if n == 0:
                continue
            start = offsets[i].item()
            slice_tokens = sorted_tokens[start : start + n]
            slice_weights = sorted_weights[start : start + n].unsqueeze(-1)
            out[start : start + n] = slice_weights * self.experts[i](slice_tokens)

        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(T, device=order.device)
        return out[inverse], boundaries

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
