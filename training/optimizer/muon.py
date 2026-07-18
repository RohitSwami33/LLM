"""Muon Optimizer.

Reference: Jordan et al., "Muon: An optimizer for hidden layers in neural networks", 2024.
Uses momentum-based updates with RMS-normalized updates for matrix parameters,
and AdamW-style updates for 1D parameters (biases, norms).
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple


class Muon(Optimizer):
    """Muon optimizer for transformer training.

    For 2D (matrix) parameters: uses momentum + RMS normalization.
    For 1D (vector) parameters: uses AdamW-style updates.

    Args:
        params: Iterable of parameters.
        lr: Learning rate.
        momentum: Momentum coefficient.
        weight_decay: Weight decay (L2 regularization).
        nesterov: Whether to use Nesterov momentum.
        backend: Backend for ortho update ("newton_schulz" or "none").
        backend_lr: Learning rate for the backend.
        backend_steps: Number of backend steps.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        backend: str = "newton_schulz",
        backend_lr: float = 0.02,
        backend_steps: int = 10,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            backend=backend,
            backend_lr=backend_lr,
            backend_steps=backend_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Update momentum
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # RMS normalization for 2D params (matrix weights)
                if p.ndim == 2:
                    update = self._ortho_update(update)

                # Apply update
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss

    def _ortho_update(self, update: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to the update (2D tensors only)."""
        # RMS normalize: divide by RMS of the update
        rms = update.pow(2).mean().sqrt()
        if rms > 0:
            update = update / rms
        return update


class MuonWithAuxAdam(Optimizer):
    """Muon for main params + Adam for embeddings/norms (combined optimizer).

    This is the recommended way to use Muon: handle 1D params with Adam,
    2D params with Muon, all in a single optimizer for single-step training.

    Args:
        muon_params: Parameters for Muon updates (2D weights).
        adam_params: Parameters for Adam updates (1D: embeddings, norms, biases).
        lr: Learning rate for Muon.
        adam_lr: Learning rate for Adam.
        momentum: Momentum for Muon.
        weight_decay: Weight decay.
        betas: Adam betas.
        eps: Adam epsilon.
    """

    def __init__(
        self,
        muon_params: List,
        adam_params: List,
        lr: float = 1e-3,
        adam_lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(muon_params + adam_params, defaults)

        # Create separate state for Muon and Adam groups
        self.muon_group = {"lr": lr, "momentum": momentum, "params": list(muon_params)}
        self.adam_group = {"lr": adam_lr, "betas": betas, "eps": eps,
                           "weight_decay": weight_decay, "params": list(adam_params)}

        # Initialize Adam state
        for p in adam_params:
            if p.grad is not None:
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

        # Initialize Muon state
        for p in muon_params:
            if p.grad is not None:
                state = self.state[p]
                state["momentum_buffer"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_lr = self.muon_group["lr"]
        adam_lr = self.adam_group["lr"]
        momentum = self.muon_group["momentum"]
        weight_decay = self.defaults["weight_decay"]
        beta1, beta2 = self.adam_group["betas"]
        eps = self.adam_group["eps"]

        # Muon updates for 2D params
        for p in self.muon_group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update = grad + momentum * buf

            # RMS normalization
            if p.ndim == 2:
                rms = update.pow(2).mean().sqrt()
                if rms > 0:
                    update = update / rms

            if weight_decay > 0:
                p.mul_(1 - muon_lr * weight_decay)
            p.add_(update, alpha=-muon_lr)

        # Adam updates for 1D params
        for p in self.adam_group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            beta_correction1 = 1 - beta1
            beta_correction2 = 1 - beta2
            step_size = adam_lr / beta_correction1
            bias_correction2_sqrt = beta_correction2 ** 0.5

            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)

            if weight_decay > 0:
                p.add_(p, alpha=-adam_lr * weight_decay)

        return loss
