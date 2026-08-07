# Design Document v2: "Hybrid-MoE-70M" — 32,768-token context research decoder

**Companions:** `docs/research_report_32k.md` (v2 report — supersedes v1 verdicts for the 32K upgrade), `docs/research_report.md` (v1, unchanged components: GQA, DeepSeekMoE, mHC, FA2).
**Status:** design frozen for implementation; every deferred component (MLA, mHC, Mamba-2 hybrid, MoBA, YaRN) remains available behind config flags for ablation.

---

## 1. Goals and Constraints

- 70M **active** parameters (target 69–71M, enforced by the audit script) — unchanged from v1.
- Context **32,768 tokens** (v1: 2048). 319M pretokenized tokens, SentencePiece 32K vocab.
- Trainable on Kaggle T4 (16GB, sm75), Kaggle P100 (sm60), RTX 3090/4090, Apple M-series (MPS) — kernel dispatch per device (report §2.1).
- FP16 + GradScaler, gradient checkpointing, EMA, cosine LR decay, warmup, gradient clipping — unchanged.
- Optimizer: **MuonClip-style** by default (evidence: Moonlight arXiv:2502.16982, IMU-1 arXiv:2602.02522, Kimi K2 arXiv:2507.20534); AdamW retained as the v1 baseline.
- Long-context training curriculum: 50% steps @ 8K dense attention → 50% @ 32K block-sparse (DeepSeek-V3 staged-training evidence, report §1).
- Modular, reproducible, benchmarkable; every decision traceable to the reports.

---

## 2. Architecture Diagram

```
                      ┌──────────────────────────────────────────────────┐
                      │              input ids (B, 32768)                │
                      └───────────────────────┬──────────────────────────┘
                                              ▼
                              ┌─────────────────────────────┐
                              │  Embedding 32768 × 576      │  ← tied with LM head
                              │  × (1/√576) input scale     │
                              └─────────────────────┬───────┘
                                                    ▼
        ┌───────────────────────── L × (6) TransformerBlock ────────────────────────┐
        │                                                                            │
        │   x ──► RMSNorm ──► GQA Self-Attention  (9 q-heads, 3 kv-heads, hd=64)     │
        │   │                  │  RoPE(θ=1e6)                                       │
        │   │                  │  mask = block_sparse: causal ∩ {|q−kv| ≤ 8192}     │
        │   │                  │         ∩ {kv < 128}  (sink/anchor block)          │
        │   │                  │  kernel = auto: flex (sm70+/MPS) → chunked eager   │
        │   │                  │  [pattern=moba: learned top-k block routing]       │
        │   │                  │  W_o zero-init                                     │
        │   │                  ▼                                                   │
        │   └────────────────► (+ residual)                                         │
        │                                                                            │
        │   x ──► RMSNorm ──► DeepSeekMoE FFN                                       │
        │   │                  │  Router: Linear(576→12) → softmax → top-4 → renorm  │
        │   │                  │    + jitter (train) + balance loss + z-loss (opt)   │
        │   │                  │  Shared expert (always on)  SwiGLU 576→1152→576     │
        │   │                  │  12 routed experts          SwiGLU 576→832→576     │
        │   │                  │  top-4 active → grouped dispatch (per-expert BMM)  │
        │   │                  │  capacity = ceil(T/12 × 1.25); overflow → passthrough│
        │   │                  │  down-proj zero-init                               │
        │   │                  ▼                                                   │
        │   └────────────────► (+ residual)                                         │
        │                                                                            │
        │   [optional mamba2 SSD layer replacing attention — default OFF]           │
        │   [optional mHC wrapper around the whole block — default OFF]             │
        └────────────────────────────────────────────────────────────────────────────┘
                                              ▼
                                        Final RMSNorm
                                              ▼
                             ┌─────────────────────────────┐
                             │  LM head (= Embeddingᵀ)     │
                             └─────────────────────┬───────┘
                                                  ▼
                                  logits (B, 32768, 32768)
                                  └─ loss = CE + α_bal·L_bal + α_z·L_z
```

**Deliverables/modules:** `config.py`, `attention.py` (GQA + 4 patterns + kernel dispatch), `mla.py` (off), `moe.py`, `router.py`, `experts.py`, `mamba2.py` (new — chunked SSD, off), `mhc.py` (off), `transformer.py`, `model.py`, `optim.py` (new — MuonClip + AdamW), `audit.py`, `smoke_test.py`.

---

## 3. Model Configuration

```yaml
# research_hybrid/configs/hybrid_70m_32k.yaml
model:
  vocab_size: 32768
  d_model: 576
  n_layers: 6
  n_q_heads: 9            # 9 × 64 = 576
  n_kv_heads: 3           # GQA-3
  head_dim: 64            # flash/flex-friendly
  rope_theta: 1_000_000.0 # Mistral-7B recipe: native 32K, no interpolation (v2 report §1)
  context_len: 32768
  ff:
    type: deepseek_moe    # unchanged from v1
    n_shared: 1
    shared_d_ff: 1152
    n_routed: 12
    routed_d_ff: 832
    top_k: 4
    capacity_factor: 1.25
    balance_coef: 0.01
    z_loss_coef: 0.001
    jitter_noise: 0.01
    routing_fn: softmax_topk
  attention:
    pattern: block_sparse # causal | sliding_window(W) | block_sparse | moba
    kernel: auto          # auto | sdpa | flex | eager   (auto: flex→chunked-eager)
    window_size: 8192     # SWA window; receptive field 6×8192 = 49K > 32K ✓
    anchor_size: 128      # sink/anchor tokens always attended (StreamingLLM, SWAA)
    block_size: 256       # block granularity for flex BlockMask and MoBA
    top_blocks: 8         # MoBA pattern only (learned top-k block selection)
  hybrid:                 # Mamba-2 SSD (v2 report §3: deferred at 70M, ablation only)
    enabled: false
    attn_layers: []       # e.g. [0, 4] for Jamba-style 2 attn / 4 mamba
    d_state: 64
    head_dim: 64
  mla:
    enabled: false        # unchanged; revisit at ≥128K
  mhc:                    # unchanged; deep-model benefit, 6 layers don't need it
    enabled: false
    n_streams: 4
    sinkhorn_iters: 20
  yarn:                   # future extension beyond 32K (v2 report §1)
    enabled: false
    factor: 1.0
    original_theta: 1_000_000.0
  use_gradient_checkpointing: true
  use_ema: true
  ema_decay: 0.999
training:
  optimizer: muon_clip    # muon_clip | adamw
  lr: 0.02                # muon: 2D matrices (Moonlight transfer recipe)
  lr_1d: 0.01             # muon: 1D params (embedding, norms, gains) via AdamW
  weight_decay: 0.1
  muon_momentum: 0.95
  muon_nesterov: true
  muon_ns_steps: 5
  qk_clip_tau: 100.0      # QK-Clip threshold (Kimi K2); null = disabled
  adamw_lr: 3.0e-4        # fallback baseline optimizer (v1 recipe)
  adamw_lr_min: 3.0e-5
  warmup_steps: 375
  total_steps: 4870       # 319M / 65536
  grad_clip: 1.0
  batch_tokens: 65536     # 2 × 32768 in stage B; 8 × 8192 in stage A
  precision: fp16
  curriculum:             # staged training (DeepSeek-V3; v2 report §1)
    stages:
      - context_len: 8192      # dense causal attention; 50% of steps
        fraction: 0.5
      - context_len: 32768     # block-sparse; 50% of steps
        fraction: 0.5
```

---

## 4. FLOPs Analysis (per token, MACs)

Let `d = 576`, `Hq = 9`, `hd = 64`, `W = 8192`, `d_shared = 1152`, `d_routed = 832`, `mK = 4`. Attention cost is averaged over uniform query positions: average window = W/2 = 4096 plus the 128-token anchor.

| Term | MACs/token | Formula |
|---|---|---|
| Q projection | 331,776 | d·d |
| KV projections | 221,184 | d·(Hkv·hd)·2 |
| QK^T (block-sparse, avg (W/2 + anchor)) | 2,433,024 | (4096+128)·Hq·hd |
| PV | 2,433,024 | (4096+128)·Hq·hd |
| O projection | 331,776 | d·d |
| Router | 6,912 | d·N |
| Shared expert (SwiGLU) | 1,990,656 | 3·d·d_shared |
| Routed experts (top-4) | 5,750,784 | mK·3·d·d_routed |
| **Layer total (transformer)** | **≈ 13.50M** | |
| **6 layers** | **≈ 81.0M** | |
| LM head (tied) | 18,874,368 | d·vocab |
| **Model total (forward)** | **≈ 99.9M MACs ≈ 200 MFLOPs/token** | |

Wait — attention at W=8192 is **4.9M MACs/token**, i.e. attention is now ~36% of the layer (dense at 32K would be 18.9M = 57%). **Model forward ≈ 99.9M MACs ≈ 200 MFLOPs/token; training (fwd+bwd) ≈ 600 MFLOPs/token with gradient-checkpointing recompute.**

Stage A (8K dense): attention = 2·4096·576 = 4.7M MACs/token — comparable to stage B, so the curriculum is compute-homogeneous.

**Per step (65,536 tokens, stage B):** ≈ 13.5 TFLOPs forward / ≈ 40 TFLOPs train.

**Hardware (fp16):**

| Device | Effective TFLOPs/s (est.) | tok/s (est.) | 1 epoch (319M tok) | Full run (2 epochs) |
|---|---|---|---|---|
| Kaggle T4 | 10–13 | 17–22k | 4.0–5.2 h | 8–10 h |
| Kaggle P100 (chunked eager) | 4–5 | 6–9k | 9.8–14.8 h | 20–30 h |
| RTX 3090 | 30–40 | 50–65k | 1.4–1.8 h | 2.8–3.6 h |
| RTX 4090 | 40–55 | 65–90k | 1.0–1.4 h | 2.0–2.8 h |
| M-series (MPS flex) | 12–18 | 20–30k | 3.0–4.4 h | 6–9 h |

Budget guidance: 2 epochs ≈ 4,870 steps (the corpus is 319M tokens; the dense run's 50k-step trajectory ≈ 3.3 epochs at batch 64K/2K).

---

## 5. Memory Analysis

**Weights + optimizer (fp16 master-fp32 training, MuonClip):**
- Total params ≈ 139.7M (emb 18.9M + 6×20.13M layers + 576).
- fp16 weights 278 MB; fp32 master 556 MB; **Muon momentum fp32 556 MB (no v-moment — halves AdamW's 1.11 GB)**; fp16 grads 278 MB → **≈ 1.7 GB**. AdamW baseline ≈ 2.2 GB.
- EMA shadow fp32 (optional, on during training) + 556 MB → ≈ 2.3 GB MuonClip w/ EMA.

**Activations (stage B: T=32768, B=2, checkpointed):**
- Per-block activation for the Q chunk (chunked attention) is bounded: Q-chunk 2048 → scores chunk (2048 × W+anchor) ≈ 2,048 × 8,320 × 2B ≈ 34 MB peak inside a block.
- Block inputs held for checkpointing: 6 × 2 × 2 × 32768 × 576 × 2B ≈ 905 MB (fp16, B=2). Stage A (B=8 × 8192) holds the same total.
- **Peak total (weights + opt + activations + CUDA context): ≈ 2.8 GB on T4** (16 GB) — batch 2 at 32K is safe; batch 4 at 32K fits with EMA deferred.

**KV cache (decode, rolling window + anchor, 32K ctx):**
- GQA-3: 6 × 2 × 3 × 64 × (8192 + 128) × 2 B ≈ **37 MB/seq** (dense 32K would be 147 MB). Anchor tokens are pinned in the cache (StreamingLLM).

---

## 6. Training Cost Estimate

| Device | tok/s (est.) | 1 epoch (319M tok) | 2 epochs (≈4.9k steps) | 12h Kaggle session |
|---|---|---|---|---|
| Kaggle T4 | 17–22k | 4.0–5.2 h | 8–10 h | ✓ 2 epochs |
| Kaggle P100 | 6–9k | 9.8–14.8 h | 20–30 h | ✓ 1 epoch |
| RTX 3090 | 50–65k | 1.4–1.8 h | 2.8–3.6 h | — |
| RTX 4090 | 65–90k | 1.0–1.4 h | 2.0–2.8 h | — |
| M-series (MPS) | 20–30k | 3.0–4.4 h | 6–9 h | — |

---

## 7. Implementation Plan (order of work)

1. **`config.py`** — v2 fields: `rope_theta=1e6`, `attention.{pattern=block_sparse, window_size=8192, anchor_size=128, top_blocks}`, `hybrid` (Mamba-2), `yarn`, `training.optimizer=muon_clip`, `training.curriculum`; validation extended.
2. **`attention.py`** — RoPE (θ param), GQA forward; **mask builders**: causal / SWA / **block_sparse (window + anchor)** / MoBA (learned top-k block selection); **kernel dispatch**: `auto` → FlexAttention (sm70+/MPS, `create_block_mask` from the same `mask_mod`) → **chunked eager fallback** (P100/CPU, bounded memory, same mask); decode path with rolling-window + anchor KV cache; optional per-head max-logit capture for QK-Clip.
3. **`mamba2.py`** (new) — chunked SSD layer (Mamba-2 state-space duality, pure PyTorch, ~linear in T): chunked quadratic form + small within-chunk scan; `d_state=64`; off by default.
4. **`experts.py`, `router.py`, `moe.py`** — unchanged from v1.
5. **`mla.py`, `mhc.py`** — unchanged, flags off.
6. **`transformer.py`** — block assembly; optional Mamba-2 layer in place of attention per `hybrid.attn_layers`.
7. **`optim.py`** (new) — `MuonClip` (Algorithm 1 of arXiv:2507.20534: NS5 on momentum, 0.2·√max(n,m) RMS scaling, decoupled wd, per-head QK-Clip on wq/wk after step, GQA shared-kv min-γ approximation) + `AdamW` (v1 recipe) + LR schedules (cosine with warmup, two-stage curriculum).
8. **`model.py`** — HybridLM updated: rope_theta, curriculum context (dynamic T at forward), rolling-window decode cache; EMAWrapper unchanged.
9. **`audit.py`** — exact params (active/total), per-layer FLOP table at both stages, memory estimate, cost table; asserts 70M-active budget and the attention mask density.
10. **`smoke_test.py`** — CPU correctness: all 4 patterns vs dense causal reference, chunked fallback == flex/eager equivalence, MoBA, Mamba-2 layer, MuonClip step sanity (with/without QK-Clip), AdamW path, curriculum forward, KV-cache decode == full forward, aux losses, EMA.

## 8. Integration & Benchmarks (next phase, after this implementation)

- Wire into the Kaggle pipeline (`training/kaggle/kaggle_pipeline.py`): model swap, MuonClip + GradScaler, EMA checkpoints, resume format → new dataset slug `research-v3-32k-...`.
- Benchmark vs the 68.7M dense model and the 69.4M v1 (2K) hybrid: log-loss / PPL at matched tokens and matched wall-clock on T4; long-range recall probes at 32K (passkey / needle-in-haystack style) to validate the sparse attention.
- Ablations: {block_sparse vs causal-32K vs SWA-only}, {anchor on/off}, {MoBA}, {Mamba-2 ratio 2:4}, {MuonClip vs AdamW}, {curriculum on/off}, {YaRN beyond 32K}.
