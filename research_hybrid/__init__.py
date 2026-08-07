"""Research hybrid architecture package.

Hybrid-MoE-70M (v2, 32K context): a decoder-only transformer combining
  - GQA attention (9 q-heads / 3 kv-heads, head_dim 64) with block-sparse
    window+anchor masks (FlexAttention on sm70+/MPS, chunked fallback elsewhere),
  - a DeepSeekMoE-style feed-forward (1 shared + 12 routed experts, top-4 routing),
  - a MuonClip (Muon + QK-Clip) default optimizer,
  - optional, evidence-deferred components: MLA (`mla.py`), mHC (`mhc.py`),
    Mamba-2 SSD layers (`mamba2.py`), MoBA attention, YaRN extension.

Design rationale: docs/research_report_32k.md (v2), docs/research_report.md (v1),
docs/design.md.
"""

from research_hybrid.config import ModelConfig, TrainingConfig, CurriculumStage, MoEConfig, AttentionConfig
from research_hybrid.attention import CausalGQA, precompute_rope, build_mask_mod, ChunkedSparseAttention
from research_hybrid.mla import MLAAttention
from research_hybrid.experts import SwiGLU, SharedExperts, RoutedExperts
from research_hybrid.router import Router, RouteOutput, load_balancing_loss, router_z_loss, expert_capacity
from research_hybrid.moe import DeepSeekMoE
from research_hybrid.mhc import MHCLayer
from research_hybrid.mamba2 import Mamba2Block, ssd_minimal_discrete
from research_hybrid.optim import MuonClip, AdamW, make_optimizer
from research_hybrid.transformer import TransformerBlock
from research_hybrid.model import HybridLM, EMAWrapper

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "CurriculumStage",
    "MoEConfig",
    "AttentionConfig",
    "CausalGQA",
    "precompute_rope",
    "build_mask_mod",
    "ChunkedSparseAttention",
    "MLAAttention",
    "SwiGLU",
    "SharedExperts",
    "RoutedExperts",
    "Router",
    "RouteOutput",
    "load_balancing_loss",
    "router_z_loss",
    "expert_capacity",
    "DeepSeekMoE",
    "MHCLayer",
    "Mamba2Block",
    "ssd_minimal_discrete",
    "MuonClip",
    "AdamW",
    "make_optimizer",
    "TransformerBlock",
    "HybridLM",
    "EMAWrapper",
]

__version__ = "0.2.0"
