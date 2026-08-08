"""Configuration dataclasses for the Hybrid-MoE-70M research model."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional


@dataclass
class MoEConfig:
    type: Literal["dense", "deepseek_moe"] = "deepseek_moe"
    n_shared: int = 1
    shared_d_ff: int = 1152
    n_routed: int = 12
    routed_d_ff: int = 832
    top_k: int = 4
    capacity_factor: Optional[float] = 1.25
    balance_coef: float = 0.01
    z_loss_coef: float = 0.001
    jitter_noise: float = 0.01
    routing_fn: Literal["softmax_topk", "sigmoid_topk"] = "softmax_topk"


@dataclass
class AttentionConfig:
    pattern: Literal["causal", "sliding_window", "block_sparse", "moba"] = "block_sparse"
    kernel: Literal["auto", "sdpa", "flex", "eager"] = "auto"
    window_size: int = 8192
    anchor_size: int = 128
    block_size: int = 256
    top_blocks: int = 8
    chunk_size: int = 2048


@dataclass
class HybridConfig:
    """Mamba-2 (chunked SSD) hybrid layers. Off by default at 70M scale
    (see docs/research_report_32k.md section 3: 2x parameter tax, attention
    already near-linear via block-sparse masks)."""
    enabled: bool = False
    attn_layers: list = field(default_factory=list)
    d_state: int = 64
    head_dim: int = 64


@dataclass
class YaRNConfig:
    """Future context extension beyond the trained 32K (v2 report section 1)."""
    enabled: bool = False
    factor: float = 1.0
    original_theta: float = 1_000_000.0


@dataclass
class MHCConfig:
    enabled: bool = False
    n_streams: int = 4
    sinkhorn_iters: int = 20


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 576
    n_layers: int = 6
    n_q_heads: int = 9
    n_kv_heads: int = 3
    head_dim: int = 64
    rope_theta: float = 1_000_000.0
    context_len: int = 32768
    input_scale: float = 1.0 / 576.0**0.5
    ff: MoEConfig = field(default_factory=MoEConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    yarn: YaRNConfig = field(default_factory=YaRNConfig)
    mhc: MHCConfig = field(default_factory=MHCConfig)
    use_gradient_checkpointing: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    qk_clip_tau: Optional[float] = None
    loss_chunk: int = 512

    def __post_init__(self) -> None:
        if self.n_q_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_q_heads*head_dim ({self.n_q_heads}*{self.head_dim}) must equal d_model ({self.d_model})"
            )
        if self.n_kv_heads > self.n_q_heads:
            raise ValueError("n_kv_heads must be <= n_q_heads")
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be a multiple of n_kv_heads (GQA)")
        if self.hybrid.enabled:
            if any(i < 0 or i >= self.n_layers for i in self.hybrid.attn_layers):
                raise ValueError("hybrid.attn_layers indexes must be within [0, n_layers)")
        if self.yarn.enabled and self.yarn.factor <= 1.0:
            raise ValueError("yarn.factor must be > 1.0 when yarn is enabled")
        if self.ff.type == "deepseek_moe":
            if self.ff.top_k > self.ff.n_routed:
                raise ValueError("top_k must be <= n_routed")
            if self.ff.routed_d_ff % 16 != 0:
                raise ValueError("routed_d_ff should be a multiple of 16 for kernel efficiency")
            if self.ff.shared_d_ff % 16 != 0:
                raise ValueError("shared_d_ff should be a multiple of 16 for kernel efficiency")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        d = dict(d)
        d["ff"] = MoEConfig(**d.get("ff", {}))
        d["attention"] = AttentionConfig(**d.get("attention", {}))
        d["hybrid"] = HybridConfig(**d.get("hybrid", {}))
        d["yarn"] = YaRNConfig(**d.get("yarn", {}))
        d["mhc"] = MHCConfig(**d.get("mhc", {}))
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumStage:
    context_len: int = 32768
    fraction: float = 1.0
    attention_pattern: Literal["causal", "block_sparse"] = "causal"


@dataclass
class TrainingConfig:
    optimizer: Literal["muon_clip", "adamw"] = "muon_clip"
    lr: float = 0.02
    lr_1d: float = 0.01
    lr_min: float = 3.0e-5
    warmup_steps: int = 375
    total_steps: int = 4870
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    qk_clip_tau: Optional[float] = 100.0
    adamw_lr: float = 3.0e-4
    adamw_lr_min: float = 3.0e-5
    grad_clip: float = 1.0
    batch_tokens: int = 65536
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"
    grad_accum: int = 8
    adam_betas: tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    curriculum: list = field(default_factory=lambda: [
        CurriculumStage(context_len=8192, fraction=0.5, attention_pattern="causal"),
        CurriculumStage(context_len=32768, fraction=0.5, attention_pattern="block_sparse"),
    ])
    seed: int = 1337

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        d = dict(d)
        d["curriculum"] = [CurriculumStage(**s) for s in d.get("curriculum", [])]
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def stage_for_step(self, step: int) -> "CurriculumStage":
        """Return the curriculum stage active at ``step`` (fractional, in step order)."""
        if not self.curriculum:
            return CurriculumStage()
        total = self.total_steps
        boundary = 0.0
        for stage in self.curriculum:
            end = boundary + stage.fraction * total
            if step < end:
                return stage
            boundary = end
        return self.curriculum[-1]
