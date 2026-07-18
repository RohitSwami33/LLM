"""Full Transformer Language Model (decoder-only).

Architecture:
    Token Embedding -> [TransformerBlock x N] -> RMSNorm -> LM Head

Supports weight tying between embedding and LM head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import ModelConfig
from .rmsnorm import create_norm
from .transformer import TransformerBlock
from .moe_transformer import MoETransformerBlock


class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model.

    Args:
        config: ModelConfig instance.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.is_moe = config.architecture == "moe"

        # Token embedding (no positional embedding — RoPE handles positions)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks — architecture switching
        if self.is_moe:
            self.blocks = nn.ModuleList([
                MoETransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    num_experts=config.moe_num_experts,
                    top_k=config.moe_top_k,
                    capacity_factor=config.moe_capacity_factor if config.moe_capacity_factor > 0 else None,
                    shared_expert=config.moe_shared_expert,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                    rope=config.rope,
                    rope_base=config.rope_base,
                    max_seq_len=config.max_seq_len,
                    flash=config.flash_attention,
                    bias=config.bias,
                    gradient_checkpointing=config.gradient_checkpointing,
                    load_balancing_weight=config.moe_load_balancing_weight,
                    router_z_loss_weight=config.moe_router_z_loss_weight,
                    router_temperature=config.moe_router_temperature,
                    router_noise=config.moe_router_noise,
                )
                for _ in range(config.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                    activation=config.activation,
                    rope=config.rope,
                    rope_base=config.rope_base,
                    max_seq_len=config.max_seq_len,
                    flash=config.flash_attention,
                    bias=config.bias,
                    gradient_checkpointing=config.gradient_checkpointing,
                )
                for _ in range(config.n_layers)
            ])

        # Final normalization
        self.norm = create_norm(config.d_model, config.norm_type)

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # MoE metrics accumulator (populated during forward)
        self._moe_metrics = {}

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        if config.tie_weights:
            n_params -= self.tok_emb.weight.numel()  # Don't double-count tied weights
        arch_str = f"MoE({config.moe_num_experts}E, {config.moe_top_k}K)" if self.is_moe else "Transformer"
        print(f"TransformerLM [{arch_str}]: {n_params:,} parameters ({n_params/1e6:.1f}M)")
        if self.is_moe:
            active_params = self._count_active_params()
            print(f"  Active params per token: {active_params:,} ({active_params/1e6:.1f}M)")

    def _init_weights(self, module: nn.Module):
        """Initialize weights using scaled normal for residual projections."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: (B, T) token indices.
            labels: (B, T) target indices (shifted by 1). If None, returns logits only.
            mask: (B, T) attention mask (1=attend, 0=ignore).

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if labels provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} > max_seq_len {self.config.max_seq_len}"

        # Embed tokens
        x = self.emb_dropout(self.tok_emb(input_ids))

        # Transformer blocks — collect MoE metrics if applicable
        self._moe_metrics = {}
        for block in self.blocks:
            if self.is_moe:
                x, moe_metrics = block(x, mask=mask)
                # Accumulate MoE metrics across layers
                for k, v in moe_metrics.items():
                    if k not in self._moe_metrics:
                        self._moe_metrics[k] = []
                    self._moe_metrics[k].append(v)
            else:
                x = block(x, mask=mask)

        # Average MoE metrics across layers
        if self.is_moe and self._moe_metrics:
            for k in self._moe_metrics:
                vals = self._moe_metrics[k]
                if isinstance(vals[0], torch.Tensor):
                    self._moe_metrics[k] = torch.stack(vals).mean(dim=0)
                else:
                    self._moe_metrics[k] = sum(vals) / len(vals)

        # Final norm + LM head
        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            input_ids: (B, T) prompt token indices.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            top_p: Nucleus sampling.

        Returns:
            Generated token indices (B, T + max_new_tokens).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else \
                input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get next token logits
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        self.train()
        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return parameter count (optionally excluding embedding)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def get_moe_metrics(self) -> dict:
        """Return accumulated MoE metrics from last forward pass."""
        return dict(self._moe_metrics)

    def _count_active_params(self) -> int:
        """Count active parameters per token for MoE models."""
        if not self.is_moe:
            return self.get_num_params(non_embedding=True)

        # Count attention + routing params (always active)
        attn_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "attn" in name or "norm" in name or "tok_emb" in name or "lm_head" in name
        )

        # Count per-expert params * top_k
        expert_params = 0
        for name, p in self.named_parameters():
            if "experts." in name:
                expert_params += p.numel()

        # Each expert has the same param count
        if self.config.moe_num_experts > 0:
            per_expert = expert_params // self.config.moe_num_experts
            active_expert = per_expert * self.config.moe_top_k
        else:
            active_expert = 0

        # Shared expert
        shared_params = 0
        if self.config.moe_shared_expert:
            for name, p in self.named_parameters():
                if "shared_expert" in name:
                    shared_params += p.numel()

        return attn_params + active_expert + shared_params

    def get_total_and_active_params(self) -> Tuple[int, int]:
        """Return (total_params, active_params_per_token)."""
        total = self.get_num_params(non_embedding=True)
        active = self._count_active_params() if self.is_moe else total
        return total, active
