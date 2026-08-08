"""Optimizers for the research model: MuonClip (default) and AdamW (baseline).

MuonClip is Algorithm 1 of Kimi K2 (arXiv:2507.20534), integrating:
  1. Muon momentum with Nesterov and Newton-Schulz orthogonalization,
  2. consistent update RMS scaling 0.2 * sqrt(max(n, m)) matching AdamW's update
     RMS (Moonlight, arXiv:2502.16982) — this allows transferring the AdamW LR,
  3. decoupled weight decay, and
  4. per-head QK-Clip: after the step, query/key projections are rescaled so that
     the per-head maximum attention logit S_max^h (captured by CausalGQA during
     the forward pass) stays below the threshold tau (default 100).

1D parameters (embeddings, norms, gains, biases) use AdamW per the Moonlight
recipe; 2D matrices use Muon with a distinct learning rate.

Evidence and hyper-parameters: docs/research_report_32k.md section 5.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_hybrid.config import TrainingConfig

_NS_COEFS = (0.3416, 0.3416, 0.6825)  # (a, b, c) Newton-Schulz polynomial (Keller, 2024)


def zeropower_via_newtonschulz(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize ``g`` via ``steps`` Newton-Schulz iterations (a=b=0.3416, c=0.6825).

    For non-square matrices the orthogonalization is applied to the squared side
    and re-expanded (as in the public Muon implementations). Returns the update
    in ``g``'s orientation.
    """
    a, b, c = _NS_COEFS
    if g.dim() != 2:
        raise ValueError("Muon applies to 2D parameters only")
    if g.shape[0] > g.shape[1]:
        g_t = g.transpose(0, 1)
        transposed = True
    else:
        g_t = g
        transposed = False
    x = g_t
    for _ in range(steps):
        x = a * x + b * (x @ x.transpose(-2, -1) @ x)
    x = x + eps * g_t
    if transposed:
        x = x.transpose(0, 1)
    return x


def build_param_groups(model: nn.Module, cfg: TrainingConfig):
    """Split parameters into Muon (2D) and AdamW (1D) groups per the Moonlight recipe."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon if p.ndim == 2 else adamw).append((name, p))
    return muon, adamw


class MuonClip(torch.optim.Optimizer):
    """MuonClip optimizer (Kimi K2, arXiv:2507.20534 Algorithm 1).

    Groups:
      - ``muon``: 2D parameters. Update = NS5(G + mu*M) * 0.2*sqrt(max(n,m)),
        applied with decoupled weight decay and lr ``muon_lr``.
      - ``adamw``: 1D parameters (embedding, norms, gains). Standard AdamW with
        lr ``adamw_lr`` and the same weight decay.

    QK-Clip: ``qk_clip_tau`` enables the post-step per-head rescaling of the
    query/key projection weights. ``S_max`` values are read from the model's
    attention modules (``attn.max_logits``), captured during the forward pass.
    For GQA the shared KV head is scaled by the minimum gamma over the q-heads
    that share it (approximation to per-head K, documented in the design doc).
    """

    def __init__(self, model: nn.Module, cfg: TrainingConfig):
        defaults = dict(muon_lr=cfg.lr, adamw_lr=cfg.lr_1d,
                        weight_decay=cfg.weight_decay, momentum=cfg.muon_momentum,
                        nesterov=cfg.muon_nesterov, ns_steps=cfg.muon_ns_steps,
                        betas=cfg.adam_betas, eps=cfg.adam_eps)
        super().__init__([{"params": []}], defaults)
        self.cfg = cfg
        self.model = model
        self.momentum_bufs: dict = {}
        self.adam_state: dict = {}
        self.qk_clip_tau = cfg.qk_clip_tau

        muon, adamw = build_param_groups(model, cfg)
        self.muon_params_list = list(muon)
        for name, p in self.muon_params_list:
            self.momentum_bufs[name] = torch.zeros_like(p, dtype=torch.float32)
        self.muon_params = [p for _, p in self.muon_params_list]
        self.adamw_params = [p for _, p in adamw]
        for p in self.adamw_params:
            self.adam_state[p] = {
                "step": 0,
                "exp_avg": torch.zeros_like(p, dtype=torch.float32),
                "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32),
            }

    @torch.no_grad()
    def step(self):
        muon_lr = self.param_groups[0]["muon_lr"] if self.param_groups else self.defaults["muon_lr"]
        wd = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        nesterov = self.defaults["nesterov"]
        ns_steps = self.defaults["ns_steps"]

        for name, p in self.muon_params_list:
            g = p.grad
            if g is None:
                continue
            buf = self.momentum_bufs[name]
            buf.mul_(momentum).add_(g.float())
            update = g.float().lerp(buf, momentum) if nesterov else buf
            # Muon is scale-invariant only after unit normalization: NS(x) has the
            # fixed point x @ x^T ~= I, but x @ x^T @ x grows with ||x||, so a
            # non-unit update diverges across the 5 NS iterations (seen as 1e10x
            # parameter blowups). Normalize first, per K2 Algorithm 1 (G_Normalize).
            o = zeropower_via_newtonschulz(update / (update.norm() + 1e-8), ns_steps)
            o.mul_(0.2 * max(p.shape[0], p.shape[1]) ** 0.5)
            p.mul_(1.0 - muon_lr * wd)
            p.add_(o.to(p.dtype), alpha=-muon_lr)

        for p in self.adamw_params:
            g = p.grad
            if g is None:
                continue
            st = self.adam_state[p]
            st["step"] += 1
            b1, b2 = self.defaults["betas"]
            st["exp_avg"].mul_(b1).add_(g.float(), alpha=1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(g.float(), g.float(), value=1 - b2)
            denom = st["exp_avg_sq"].sqrt().add_(self.defaults["eps"])
            step_size = self.defaults["adamw_lr"] * (1 - b2 ** st["step"]) ** 0.5 / (1 - b1 ** st["step"])
            p.mul_(1.0 - self.defaults["adamw_lr"] * wd)
            p.addcdiv_(st["exp_avg"].to(p.dtype), denom.to(p.dtype), value=-step_size)

        if self.qk_clip_tau is not None:
            self._qk_clip()

    @torch.no_grad()
    def _qk_clip(self):
        tau = self.qk_clip_tau
        for name, mod in self.model.named_modules():
            if not isinstance(mod, torch.nn.Module) or not hasattr(mod, "max_logits"):
                continue
            if mod.max_logits is None:
                continue
            smax = mod.max_logits  # (n_q_heads,)
            # clamp_min guards against all-negative score blocks (gamma would be
            # negative -> NaN sqrt); a non-positive S_max needs no clipping.
            gamma = (tau / smax.clamp_min(1e-8)).clamp(max=1.0)  # per q-head gamma
            sq = gamma.sqrt()
            wq = mod.wq.weight.view(mod.n_q_heads, -1)
            wq.mul_(sq[:, None].to(wq.dtype))
            groups = mod.n_q_heads // mod.n_kv_heads
            gamma_kv = gamma.view(mod.n_kv_heads, groups).min(dim=1).values
            wk = mod.wk.weight.view(mod.n_kv_heads, -1)
            wk.mul_(gamma_kv[:, None].to(wk.dtype).sqrt())
            mod.max_logits = None

    def state_dict(self):
        return {
            "momentum_bufs": {k: v.clone() for k, v in self.momentum_bufs.items()},
            "adam_state": {id(p): st.copy() for p, st in self.adam_state.items()},
        }

    def load_state_dict(self, state):
        for k, v in state["momentum_bufs"].items():
            if k in self.momentum_bufs:
                self.momentum_bufs[k].copy_(v)
        for p, st in self.adam_state.items():
            if id(p) in state["adam_state"]:
                for key, val in state["adam_state"][id(p)].items():
                    st[key] = val


class AdamW(torch.optim.AdamW):
    """AdamW with the v1 training recipe (fallback baseline optimizer)."""

    def __init__(self, model: nn.Module, cfg: TrainingConfig):
        params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, lr=cfg.adamw_lr, betas=cfg.adam_betas, eps=cfg.adam_eps,
                         weight_decay=cfg.weight_decay)


def make_optimizer(model: nn.Module, cfg: TrainingConfig):
    if cfg.optimizer == "muon_clip":
        return MuonClip(model, cfg)
    if cfg.optimizer == "adamw":
        return AdamW(model, cfg)
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")
