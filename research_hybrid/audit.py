"""Parameter / FLOPs / memory / cost audit for the Hybrid-MoE-70M (32K) design.

Run:  .venv/bin/python -m research_hybrid.audit

The audit asserts the 70M-active budget and prints the tables of the design doc
(section 4/5/6). FLOPs are exact MACs using the v2 report's formulas; the LM head
uses the tied embedding weight.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch

from research_hybrid.config import ModelConfig, TrainingConfig
from research_hybrid.model import HybridLM

VOCAB = 32768


def params(cfg: ModelConfig) -> Dict[str, int]:
    model = HybridLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    per_expert = 3 * cfg.d_model * cfg.ff.routed_d_ff
    inactive = (cfg.ff.n_routed - cfg.ff.top_k) * per_expert * cfg.n_layers
    active = total - inactive
    return {"total": total, "active": active, "inactive": inactive}


def flops_per_token(cfg: ModelConfig, T: int, pattern: str) -> Dict[str, float]:
    """MACs/token for one forward pass (exact formulas, design doc section 4)."""
    d, hq, hd = cfg.d_model, cfg.n_q_heads, cfg.head_dim
    hkv, m = cfg.n_kv_heads, cfg.ff.top_k
    d_s, d_r = cfg.ff.shared_d_ff, cfg.ff.routed_d_ff
    attn_avg = T / 2 if pattern == "causal" else (cfg.attention.window_size / 2 + cfg.attention.anchor_size)
    terms = {
        "Q proj": d * d,
        "KV proj": d * (hkv * hd) * 2,
        "QK^T+PV": 2 * attn_avg * hq * hd,
        "O proj": d * d,
        "router": d * cfg.ff.n_routed,
        "shared expert": 3 * d * d_s,
        "routed (top-k)": m * 3 * d * d_r,
    }
    layer = sum(terms.values())
    head = d * VOCAB
    fwd = layer * cfg.n_layers + head
    return {"terms": terms, "layer": layer, "forward": fwd, "head": head}


def memory(cfg: ModelConfig, train_cfg: TrainingConfig, B: int, T: int) -> Dict[str, float]:
    total = params(cfg)["total"]
    gb = 2 ** 30
    fp16_w = total * 2
    fp32_master = total * 4
    momentum = total * 4 if train_cfg.optimizer == "muon_clip" else 2 * total * 4
    grads = total * 2
    opt = fp16_w + fp32_master + momentum + grads
    ema = total * 4 if cfg.use_ema else 0
    act_block = B * T * cfg.d_model * 2 * cfg.n_layers
    kv = 6 * 2 * hkv(cfg) * cfg.head_dim * (cfg.attention.window_size + cfg.attention.anchor_size) * 2
    peak = opt + ema + act_block * 0.5 + 0.5 * 2 ** 30
    return {"weights_opt_gb": (opt + ema) / gb, "activations_gb": (act_block * 0.5) / gb,
            "kv_cache_gb": kv / gb, "peak_gb": peak / gb}


def hkv(cfg: ModelConfig) -> int:
    return cfg.n_kv_heads


def cost_table() -> str:
    rows = [
        ("Kaggle T4", "17-22k", "4.0-5.2 h", "8-10 h"),
        ("Kaggle P100", "6-9k", "9.8-14.8 h", "20-30 h"),
        ("RTX 3090", "50-65k", "1.4-1.8 h", "2.8-3.6 h"),
        ("RTX 4090", "65-90k", "1.0-1.4 h", "2.0-2.8 h"),
        ("M-series (MPS)", "20-30k", "3.0-4.4 h", "6-9 h"),
    ]
    out = "| Device | tok/s (est.) | 1 epoch (319M tok) | 2 epochs (~4.9k steps) |\n"
    out += "|---|---|---|---|\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out


def main() -> None:
    cfg = ModelConfig()
    train_cfg = TrainingConfig()

    p = params(cfg)
    print(f"params total={p['total']:,} active={p['active']:,} "
          f"(inactive routed experts: {p['inactive']:,})")
    assert 69_000_000 <= p["active"] <= 71_000_000, "active budget violated"
    assert 69_000_000 <= p["total"] <= 145_000_000
    print(f"active budget OK (69-71M active, total {p['total']:,})")

    for T, pat in [(8192, "causal"), (32768, "block_sparse")]:
        f = flops_per_token(cfg, T, pat)
        mflops = f["forward"] * 2 / 1e6
        print(f"\nFLOPs @ T={T} pattern={pat}: layer={f['layer']:,} MACs/token, "
              f"forward={f['forward']:,} MACs/token (~{mflops:.0f} MFLOPs/token), "
              f"train ~{3 * mflops:.0f} MFLOPs/token")
        for k, v in f["terms"].items():
            print(f"  {k:>16}: {v:>12,}")

    B, T = 2, 32768
    m = memory(cfg, train_cfg, B, T)
    print(f"\nMemory @ B={B} T={T}: weights+opt={m['weights_opt_gb']:.2f} GB, "
          f"activations~={m['activations_gb']:.2f} GB, kv_cache={m['kv_cache_gb']:.3f} GB, "
          f"peak≈{m['peak_gb']:.1f} GB (T4 has 16 GB)")

    print("\nTraining cost (from design doc):")
    print(cost_table())
    print("\nCurriculum stages:")
    for i, s in enumerate(train_cfg.curriculum):
        print(f"  stage {i}: context={s.context_len} pattern={s.attention_pattern} "
              f"fraction={s.fraction} (steps "
              f"{int(sum(x.fraction for x in train_cfg.curriculum[:i]) * train_cfg.total_steps)}.."
              f"{int(sum(x.fraction for x in train_cfg.curriculum[:i + 1]) * train_cfg.total_steps)})")


if __name__ == "__main__":
    main()
