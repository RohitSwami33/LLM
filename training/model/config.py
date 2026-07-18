# training/model/config.py
"""Model configuration with YAML loading support."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import yaml


@dataclass
class ModelConfig:
    """Transformer model configuration.

    Designed for extensibility: future modules (MoE, MLA, mHC, MTP)
    can be added as new fields without breaking existing checkpoints.
    """

    # ── Architecture ────────────────────────────────────────────────────────
    vocab_size: int = 32000
    d_model: int = 576          # Hidden dimension  (~60 M params)
    n_heads: int = 8            # Attention heads   (72 per head)
    n_layers: int = 6           # Transformer blocks
    d_ff: int = 2304            # FFN hidden dim    (SwiGLU: 2/3 * 4d)
    dropout: float = 0.0        # Embedding / attention dropout
    max_seq_len: int = 2048

    # ── Components ──────────────────────────────────────────────────────────
    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    activation: str = "swiglu"  # "swiglu"  | "gelu"
    rope: bool = True
    rope_base: float = 10000.0
    flash_attention: bool = True
    bias: bool = False          # Linear layer bias
    tie_weights: bool = False   # Tie embed / LM-head

    # ── Training ────────────────────────────────────────────────────────────
    gradient_checkpointing: bool = True

    # ── Architecture variant ─────────────────────────────────────────────────
    architecture: str = "transformer"  # "transformer" | "moe"

    # ── MoE configuration (used when architecture="moe") ───────────────────
    moe_num_experts: int = 8    # Number of expert networks
    moe_top_k: int = 2          # Experts per token
    moe_capacity_factor: float = 1.25  # Capacity multiplier (None = no dropping)
    moe_shared_expert: bool = False    # Shared expert always active
    moe_load_balancing_weight: float = 0.01  # Auxiliary load balancing loss weight
    moe_router_z_loss_weight: float = 0.001  # Router z-loss weight
    moe_router_temperature: float = 1.0      # Routing softmax temperature
    moe_router_noise: float = 0.1            # Noise std for training routing

    # ── Future extensibility (unused by baseline) ───────────────────────────
    mla_latent_dim: int = 0     # >0 enables MLA
    mtp_heads: int = 1          # Multi-Token Prediction heads
    mhc: bool = False           # Manifold-Constrained Hyper Connections

    def __post_init__(self):
        # SwiGLU needs d_ff as 2/3 of 4*d_model, rounded to multiple of 256
        if self.activation == "swiglu":
            self.d_ff = int(self.d_ff / 256) * 256
            if self.d_ff < 256:
                self.d_ff = 256

        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        model_data = data.get("model", {})
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in model_data.items() if k in known})

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
