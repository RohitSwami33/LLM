"""Optimizer builder with support for Muon, AdamW, Lion, Adafactor, 8-bit AdamW."""

import torch
from torch.optim import AdamW
from typing import List, Dict, Any, Optional


def get_param_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.1,
    lr: float = 3e-4,
) -> List[Dict[str, Any]]:
    """Split parameters into weight-decay and no-weight-decay groups.

    Returns two param groups:
        [0]: 2D params (weights) -> apply weight decay
        [1]: 1D params (biases, norms, embeddings) -> no weight decay
    """
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D params: biases, layer norms, embeddings
        if param.ndim <= 1 or "bias" in name or "norm" in name or "emb" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


def build_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Supported optimizers:
        - adamw: Standard AdamW
        - muon: Muon optimizer (2D -> Muon, 1D -> Adam)
        - lion: Lion optimizer
        - adafactor: Adafactor (memory-efficient)
        - 8bit_adamw: 8-bit AdamW (requires bitsandbytes)
        - sophia: Sophia optimizer

    Args:
        model: The model.
        config: Training config dict with keys: optimizer, learning_rate, weight_decay, etc.

    Returns:
        Configured optimizer.
    """
    opt_name = config.get("optimizer", "adamw").lower()
    lr = config.get("learning_rate", 3e-4)
    weight_decay = config.get("weight_decay", 0.1)

    param_groups = get_param_groups(model, weight_decay=weight_decay, lr=lr)

    if opt_name == "adamw":
        return AdamW(
            param_groups,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
            eps=config.get("eps", 1e-8),
        )

    elif opt_name == "muon":
        from .muon import MuonWithAuxAdam

        # Split param groups: 2D for Muon, 1D for Adam
        muon_params = []
        adam_params = []
        for group in param_groups:
            for p in group["params"]:
                if p.ndim >= 2:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        return MuonWithAuxAdam(
            muon_params=muon_params,
            adam_params=adam_params,
            lr=lr,
            adam_lr=lr,
            momentum=config.get("momentum", 0.95),
            weight_decay=weight_decay,
        )

    elif opt_name == "lion":
        return Lion(
            param_groups,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.99)),
            weight_decay=weight_decay,
        )

    elif opt_name == "adafactor":
        from torch.optim import Adafactor
        return Adafactor(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=True,
            relative_step=False,
        )

    elif opt_name == "8bit_adamw":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                param_groups,
                betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
                eps=config.get("eps", 1e-8),
            )
        except ImportError:
            raise ImportError("bitsandbytes is required for 8-bit AdamW: pip install bitsandbytes")

    elif opt_name == "sophia":
        from .sophia import Sophia
        return Sophia(
            param_groups,
            lr=lr,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


class Lion(torch.optim.Optimizer):
    """Lion optimizer (EvoLved Sign Momentum).

    Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023.

    Args:
        params: Iterable of parameters.
        lr: Learning rate.
        betas: Coefficients for momentum.
        weight_decay: Weight decay.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(update.sign(), alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss
