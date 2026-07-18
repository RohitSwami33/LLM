"""Exponential Moving Average (EMA) of model parameters.

Reference: Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging", 1992.

EMA provides smoother model weights that often generalize better.
"""

import copy
import torch
from torch.optim.optimizer import Optimizer
from typing import Optional


class EMA:
    """Exponential Moving Average of model parameters.

    Args:
        model: The model to track.
        decay: EMA decay rate (typically 0.9999).
        warmup_steps: Number of steps before EMA kicks in.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, warmup_steps: int = 0):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.backup = None
        self.steps = 0

    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        self.steps += 1
        if self.steps <= self.warmup_steps:
            return

        effective_decay = self.decay
        model_params = dict(self.model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name in model_params:
            if name in shadow_params:
                model_param = model_params[name]
                shadow_param = shadow_params[name]
                shadow_param.mul_(effective_decay).add_(model_param, alpha=1 - effective_decay)

        # Also update buffers (running stats etc.)
        model_buffers = dict(self.model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())
        for name in model_buffers:
            if name in shadow_buffers:
                shadow_buffers[name].copy_(model_buffers[name])

    def store(self):
        """Store current model parameters for later restoration."""
        self.backup = copy.deepcopy(self.model.state_dict())

    def restore(self):
        """Restore model parameters from backup."""
        if self.backup is not None:
            self.model.load_state_dict(self.backup)
            self.backup = None

    def get_decay(self) -> float:
        """Get current effective decay based on step count."""
        if self.steps <= self.warmup_steps:
            return 0.0
        return self.decay

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "steps": self.steps,
            "shadow": self.shadow.state_dict(),
        }

    def load_state_dict(self, state: dict):
        """Restore EMA state from checkpoint."""
        self.decay = state["decay"]
        self.warmup_steps = state["warmup_steps"]
        self.steps = state["steps"]
        self.shadow.load_state_dict(state["shadow"])

    def copy_to(self, model: Optional[torch.nn.Module] = None):
        """Copy EMA parameters to model (for evaluation / inference)."""
        target = model or self.model
        target.load_state_dict(self.shadow.state_dict())
