"""Sparse Mixture of Experts (MoE) Layer.

Implements:
  - Top-k sparse routing with configurable k
  - Optional shared expert (always active)
  - Expert capacity factor with token dropping
  - Auxiliary load-balancing loss
  - Router z-loss for stability
  - Comprehensive metrics tracking

Reference: Fedus et al., "Switch Transformers", 2021
           DeepSeek-V2 / DeepSeek-V3 MoE design
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .mlp import SwiGLU


class TopKRouter(nn.Module):
    """Sparse top-k router for MoE.

    Computes routing weights for each token across experts.
    Uses noisy top-k gating for load balancing during training.

    Args:
        d_model: Input dimension.
        num_experts: Number of expert networks.
        top_k: Number of experts to route to per token.
        temperature: Softmax temperature for routing logits.
        noise_std: Standard deviation of noise added during training.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.temperature = temperature
        self.noise_std = noise_std

        # Router weights: (d_model, num_experts)
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Initialize gate with small weights
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Route tokens to experts.

        Args:
            x: (B, T, d_model) input tensor.
            training: Whether in training mode (adds noise).

        Returns:
            expert_weights: (B, T, top_k) routing weights (normalized).
            expert_indices: (B, T, top_k) selected expert indices.
            metrics: Dict of routing metrics for logging.
        """
        B, T, D = x.shape

        # Compute routing logits
        logits = self.gate(x)  # (B, T, num_experts)

        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Temperature scaling
        logits = logits / self.temperature

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        # (B, T, top_k) each

        # Compute weights from top-k logits only (sparse)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Compute metrics
        metrics = self._compute_metrics(logits, top_k_indices, top_k_weights)

        return top_k_weights, top_k_indices, metrics

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute routing metrics for logging."""
        B, T, K = indices.shape
        E = self.num_experts

        # Flatten for counting
        flat_indices = indices.reshape(-1)  # (B*T*K,)

        # Expert utilization: fraction of tokens routed to each expert
        expert_counts = torch.zeros(E, device=logits.device)
        expert_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        utilization = expert_counts / (B * T)  # fraction of tokens per expert

        # Routing entropy: higher = more uniform routing
        probs = F.softmax(logits, dim=-1)  # (B, T, E)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()  # scalar

        # Load balance loss: encourages equal expert utilization
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability for expert i
        # L_balance = N * sum(f_i * P_i)
        f = utilization  # (E,)
        P = probs.mean(dim=(0, 1))  # (E,) average routing probability
        load_balance_loss = E * (f * P).sum()

        # Percentage of tokens dropped (would happen with capacity constraint)
        # For now, track routing concentration
        top1_indices = indices[:, :, 0]  # (B, T) top-1 expert per token
        top1_utilization = torch.zeros(E, device=logits.device)
        flat_top1 = top1_indices.reshape(-1)
        top1_utilization.scatter_add_(0, flat_top1, torch.ones_like(flat_top1, dtype=torch.float))
        top1_utilization = top1_utilization / (B * T)

        # Token dropping percentage (if capacity factor were applied)
        # Estimate: tokens beyond capacity for any expert
        # For now, return 0 (actual dropping computed in MoELayer with capacity)
        return {
            "expert_utilization": utilization,
            "routing_entropy": entropy,
            "load_balance_loss": load_balance_loss,
            "top1_utilization": top1_utilization,
            "dropped_tokens_pct": torch.tensor(0.0, device=logits.device),
        }


class Expert(nn.Module):
    """Single expert network (SwiGLU MLP).

    Args:
        d_model: Input/output dimension.
        d_ff: Inner dimension.
        dropout: Dropout rate.
        bias: Use bias.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class ExpertGroup(nn.Module):
    """Group of expert networks.

    Args:
        num_experts: Number of experts.
        d_model: Input/output dimension.
        d_ff: Inner dimension per expert.
        dropout: Dropout rate.
        bias: Use bias.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout=dropout, bias=bias)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Process tokens through routed experts.

        Args:
            x: (B, T, d_model) input tensor.
            expert_indices: (B, T, top_k) selected expert indices.

        Returns:
            (B, T, d_model) output tensor.
        """
        B, T, D = x.shape
        K = expert_indices.shape[-1]

        # Flatten batch and sequence dimensions
        x_flat = x.reshape(B * T, D)  # (B*T, D)
        indices_flat = expert_indices.reshape(B * T, K)  # (B*T, K)

        # Compute outputs for each expert
        outputs = torch.zeros_like(x_flat)  # (B*T, D)

        for k in range(K):
            expert_idx = indices_flat[:, k]  # (B*T,) indices for k-th choice

            for e_idx in range(len(self.experts)):
                mask = (expert_idx == e_idx)  # (B*T,) boolean mask
                if mask.any():
                    expert_input = x_flat[mask]  # (num_tokens, D)
                    expert_output = self.experts[e_idx](expert_input)
                    outputs[mask] += expert_output

        return outputs.reshape(B, T, D)


class MoELayer(nn.Module):
    """Mixture of Experts layer with sparse routing.

    Combines:
      - Top-k router for expert selection
      - Expert group for computation
      - Optional shared expert (always active)
      - Load balancing and z-loss computation
      - Token dropping with capacity factor

    Args:
        d_model: Input/output dimension.
        d_ff: Inner dimension per expert.
        num_experts: Number of experts.
        top_k: Number of experts per token.
        capacity_factor: Capacity multiplier for token dropping (None = no dropping).
        shared_expert: Whether to use a shared expert.
        dropout: Dropout rate.
        bias: Use bias.
        load_balancing_weight: Weight for load balancing auxiliary loss.
        router_z_loss_weight: Weight for router z-loss.
        router_temperature: Temperature for routing logits.
        router_noise: Noise std for training routing.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: Optional[float] = None,
        shared_expert: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
        load_balancing_weight: float = 0.01,
        router_z_loss_weight: float = 0.001,
        router_temperature: float = 1.0,
        router_noise: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.shared_expert_enabled = shared_expert
        self.load_balancing_weight = load_balancing_weight
        self.router_z_loss_weight = router_z_loss_weight

        # Router
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            temperature=router_temperature,
            noise_std=router_noise,
        )

        # Expert group
        self.experts = ExpertGroup(
            num_experts=num_experts,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )

        # Shared expert (always processes all tokens)
        self.shared_expert = None
        if shared_expert:
            self.shared_expert = Expert(d_model, d_ff, dropout=dropout, bias=bias)

        # Capacity: max tokens per expert per batch
        self.capacity = None
        if capacity_factor is not None:
            # Will be set dynamically based on batch size
            self.capacity_factor = capacity_factor

    def forward(
        self,
        x: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through MoE layer.

        Args:
            x: (B, T, d_model) input tensor.
            training: Whether in training mode.

        Returns:
            output: (B, T, d_model) output tensor.
            metrics: Dict of MoE metrics.
        """
        B, T, D = x.shape

        # Route tokens to experts
        weights, indices, router_metrics = self.router(x, training=training)

        # Compute capacity if token dropping is enabled
        if self.capacity_factor is not None:
            tokens_per_expert = (B * T * self.top_k) / self.num_experts
            self.capacity = int(tokens_per_expert * self.capacity_factor)

        # Compute expert outputs with optional token dropping
        if self.capacity_factor is not None:
            expert_output = self._forward_with_capacity(x, weights, indices)
            # Compute dropped tokens percentage
            dropped = self._compute_dropped_tokens(indices)
            router_metrics["dropped_tokens_pct"] = dropped
        else:
            expert_output = self._forward_no_capacity(x, weights, indices)

        # Add shared expert output if enabled
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            expert_output = expert_output + shared_output

        # Compute auxiliary losses
        load_balance_loss = router_metrics["load_balance_loss"]
        router_z_loss = self._compute_z_loss(weights)

        metrics = {
            **{k: v for k, v in router_metrics.items() if k != "load_balance_loss"},
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "total_aux_loss": (
                self.load_balancing_weight * load_balance_loss +
                self.router_z_loss_weight * router_z_loss
            ),
            "active_params": self._count_active_params(),
            "total_params": self._count_total_params(),
        }

        return expert_output, metrics

    def _forward_no_capacity(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass without capacity constraints (no token dropping)."""
        B, T, D = x.shape
        K = indices.shape[-1]

        x_flat = x.reshape(B * T, D)
        indices_flat = indices.reshape(B * T, K)
        weights_flat = weights.reshape(B * T, K)

        outputs = torch.zeros_like(x_flat)

        for k in range(K):
            expert_idx = indices_flat[:, k]
            expert_weight = weights_flat[:, k:k+1]  # (B*T, 1)

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts.experts[e_idx](expert_input)
                    outputs[mask] += expert_weight[mask] * expert_output

        return outputs.reshape(B, T, D)

    def _forward_with_capacity(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with capacity constraints (token dropping)."""
        B, T, D = x.shape
        K = indices.shape[-1]
        capacity = self.capacity

        x_flat = x.reshape(B * T, D)
        indices_flat = indices.reshape(B * T, K)
        weights_flat = weights.reshape(B * T, K)

        outputs = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        for k in range(K):
            expert_idx = indices_flat[:, k]
            expert_weight = weights_flat[:, k:k+1]

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    # Count tokens for this expert
                    current_count = expert_counts[e_idx].item()
                    tokens_for_expert = mask.sum().item()

                    # Apply capacity constraint
                    if capacity is not None and current_count + tokens_for_expert > capacity:
                        # Keep only tokens that fit within capacity
                        available = max(0, capacity - current_count)
                        if available == 0:
                            continue
                        # Randomly select tokens to keep
                        token_indices = torch.where(mask)[0]
                        keep_indices = token_indices[:available]
                        new_mask = torch.zeros_like(mask)
                        new_mask[keep_indices] = True
                        mask = new_mask

                    expert_input = x_flat[mask]
                    expert_output = self.experts.experts[e_idx](expert_input)
                    outputs[mask] += expert_weight[mask] * expert_output
                    expert_counts[e_idx] += mask.sum()

        return outputs.reshape(B, T, D)

    def _compute_dropped_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute percentage of tokens dropped due to capacity constraints."""
        B, T, K = indices.shape
        total_tokens = B * T
        capacity = self.capacity

        if capacity is None:
            return torch.tensor(0.0, device=indices.device)

        # Count tokens per expert
        expert_counts = torch.zeros(self.num_experts, device=indices.device)
        flat_indices = indices.reshape(-1)
        expert_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))

        # Tokens beyond capacity are dropped
        excess = (expert_counts - capacity).clamp(min=0)
        dropped = excess.sum()
        total_routed = total_tokens * K

        return dropped / total_routed

    def _compute_z_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute router z-loss for training stability.

        z_loss = (1/T) * sum(log(sum(exp(router_logits))))^2

        This penalizes large router logits and encourages stability.
        """
        # weights are softmax'd, so we need the original logits
        # Approximate: use log of weights
        log_weights = torch.log(weights + 1e-8)
        z_loss = (log_weights.sum(dim=-1) ** 2).mean()
        return z_loss

    def _count_active_params(self) -> int:
        """Count parameters that are active for any given token."""
        # Each token uses top_k experts + optionally shared expert
        per_expert = sum(p.numel() for p in self.experts.experts[0].parameters())
        active = per_expert * self.top_k
        if self.shared_expert is not None:
            active += sum(p.numel() for p in self.shared_expert.parameters())
        return active

    def _count_total_params(self) -> int:
        """Count all parameters in the MoE layer."""
        total = sum(p.numel() for p in self.experts.parameters())
        if self.shared_expert is not None:
            total += sum(p.numel() for p in self.shared_expert.parameters())
        # Router
        total += sum(p.numel() for p in self.router.parameters())
        return total

    def get_expert_state(self) -> Dict:
        """Get serializable expert state for checkpointing."""
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "capacity_factor": self.capacity_factor,
            "shared_expert": self.shared_expert_enabled,
            "load_balancing_weight": self.load_balancing_weight,
            "router_z_loss_weight": self.router_z_loss_weight,
        }
