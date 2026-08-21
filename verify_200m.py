"""Verify the chosen 200M/100M config by instantiating the real HybridLM."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from research_hybrid.config import ModelConfig, MoEConfig, AttentionConfig, MHCConfig, HybridConfig
from research_hybrid.model import HybridLM

cfg = ModelConfig(
    vocab_size=32768,
    d_model=768,
    n_layers=8,
    n_q_heads=12,
    n_kv_heads=6,
    head_dim=64,
    rope_theta=1_000_000.0,
    context_len=32768,
    ff=MoEConfig(
        type="deepseek_moe",
        n_shared=1,
        shared_d_ff=1536,
        n_routed=12,
        routed_d_ff=512,
        top_k=4,
        capacity_factor=1.25,
        balance_coef=0.01,
        z_loss_coef=0.001,
        jitter_noise=0.01,
        routing_fn="softmax_topk",
    ),
    attention=AttentionConfig(pattern="block_sparse", kernel="auto",
                              window_size=8192, anchor_size=128,
                              block_size=256, top_blocks=8, chunk_size=2048),
    hybrid=HybridConfig(enabled=False),
    mhc=MHCConfig(enabled=False, n_streams=4, sinkhorn_iters=20),
    use_ema=False,
)

m = HybridLM(cfg)
total = sum(p.numel() for p in m.parameters())
per_expert = 3 * cfg.d_model * cfg.ff.routed_d_ff
inactive = (cfg.ff.n_routed - cfg.ff.top_k) * per_expert * cfg.n_layers
active = total - inactive
print(f"HybridLM instantiated: total={total:,} ({total/1e6:.1f}M) "
      f"active={active:,} ({active/1e6:.1f}M) inactive={inactive:,} ({inactive/1e6:.1f}M)")

# quick forward smoke: (B=1, T=64)
import torch
x = torch.randint(0, 32768, (1, 64))
cos, sin = torch.zeros(1, 64, 64), torch.zeros(1, 64, 64)  # dummy rope (not used for param count)
with torch.no_grad():
    out = m(x)
print("forward OK, logits shape:", out.logits.shape if out.logits is not None else None)
