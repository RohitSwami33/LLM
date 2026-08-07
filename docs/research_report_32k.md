# Research Report v2: Upgrading Hybrid-MoE-70M to 32,768-token Context

**Companion:** `docs/research_report.md` (v1, 2048-token design, still authoritative for unchanged verdicts: GQA, DeepSeekMoE, mHC math, FA2).
**Status:** Phase 1 (research) complete; verdicts below supersede the v1 report for the 32K design. 2026-08-08.

---

## 0. Why 32K changes the design

At context `T`, per-token attention cost grows as `T` (2·(T/2)·Hq·hd MACs). For the v1 model at T=2048 attention was 1.18M MACs/token; at T=32768 **dense attention becomes 18.9M MACs/token ≈ 57% of transformer FLOPs** — larger than the entire MoE FFN (8.4M). Three consequences:

1. **Attention must go sub-dense** (SWA / block-sparse / learned selection), otherwise context 16× longer makes the model 2.5× more expensive per token at equal quality per head.
2. **The long-context literature becomes load-bearing**: SWA (Mistral), SWAT/SWAN/SWAA (training stability and adaptation), sink tokens (StreamingLLM), learned block routing (MoBA, NSA).
3. **The optimizer question reopens**: 2025–2026 results (Moonlight, MuonClip/Kimi K2, IMU-1) give Muon-family optimizers strong small- and large-scale evidence, which v1 treated as "opt-in only".

---

## 1. Context-length method: train from scratch at 32K with RoPE θ = 1e6

| Option | Evidence | Verdict at 32K |
|---|---|---|
| **RoPE θ=1e6, train from scratch** | Mistral-7B (arXiv 2310.06825) trains natively at 32K ctx with θ=1e6 and SWA; no interpolation artifacts; simplest, no extra data or search | **SELECTED** |
| YaRN (arXiv 2309.00071) | NTK-by-parts + temperature; 10× fewer tokens / 2.5× fewer steps than PI; reproducible 128K extension; used by Kimi K2 (32K→128K) | Not needed for a from-scratch 32K run; keep as the extension path (beyond 32K) |
| PI / linear (arXiv 2306.15595) | Simple but loses high-frequency information; needs 10×+ data | Rejected |
| NTK-aware / dynamic NTK | Halves the extension cost vs PI; still worse than YaRN | Rejected (YaRN dominates) |
| LongRoPE (arXiv 2402.13753) | Non-uniform per-dimension rescale via evolutionary search; 2048K extension, 8× without fine-tuning; used in Phi-3 | Overkill at 32K (needs search + eval corpus); document as future work |

**Decision:** train from scratch, fixed `rope_theta = 1e6`, no interpolation. YaRN (params: `scaling.factor`, `scaling.original_theta`) is implemented as a config knob for future extension. Rationale: interpolation methods optimize for *extending a short-trained model*; a from-scratch 32K run has no reason to pay their approximation cost, and θ=1e6 is the exact recipe validated by Mistral at this context length.

**Long-context training curriculum (evidence: DeepSeek-V3 tech report, staged 4K→32K→128K with ~45% of tokens at 4K):**
- Stage A: ~50% of steps at 8K context, **dense causal attention** (cost ≈ 2.4M MACs/token — cheaper than stage B).
- Stage B: ~50% of steps at 32K context, block-sparse attention (below).
- Same corpus order; one linear-ramp step between stages. This is cheaper *and* better-conditioned than 32K-only training, and the two attention costs are balanced (≈4.7M vs ≈4.9M MACs/token), so the schedule is compute-homogeneous.

---

## 2. Attention at 32K: block-sparse SWA + anchor block (FlexAttention)

FLOPs at T=32768, d=576, Hq=9, hd=64:

| Pattern | QK^T + PV MACs/token | Notes / evidence |
|---|---|---|
| Dense causal | 18.9M | 57% of transformer FLOPs; KV cache 147 MB/seq |
| SWA W=8192 | 4.7M | Mistral-7B (W=4096 @32K, arXiv 2310.06825); receptive field 6×8192 = 49K ≥ 32K ✓ |
| **SWA W=8192 + anchor block (128)** | **≈ 4.9M** | + sink tokens: StreamingLLM (arXiv 2309.17453), SWAA (arXiv 2512.10411) "keep first k tokens" |
| MoBA (arXiv 2502.13189) | ~sub-quadratic, data-dependent | Learned top-k *block* routing; in production at Kimi (see §2.2) |
| NSA (arXiv 2502.11089) | compression+selection+window, ~sub-quadratic | ACL 2025 best paper; needs custom Triton kernels (fla-org) |

**Evidence for SWA at scale:**
- **Mistral-7B**: 32K with pure SWA; quality holds for *in-domain* tasks where tokens outside W don't matter; long-range recall drops (documented by Hymba and SWAA).
- **StreamingLLM / SWAA**: attention sink (first tokens) recovers the long-range memory that pure SWA lacks; SWAA (arXiv 2512.10411) shows FA-trained models adapt to SWA at inference by (a) keeping the first `k` tokens, (b) interleaving FA/SWA layers, (c) using FA during decode — recovering ~90% of full-attention quality with 30–100% speedups.
- **Gemma-2 (arXiv 2408.00118)**: interleaves local (W=1024) and global layers at 7B–27B — global layers at *sparse* positions restore long-range behavior.
- **Hymba (arXiv 2411.13676)**: replacing all global attention with SWA degrades recall >20%; restoring global attention in only 3 layers (first/middle/last) recovers it.
- **SWAT (arXiv 2502.18845)**: SWA trains stably from scratch with sigmoid + RoPE/ALiBi; **SWAN (EMNLP 2025)**: NoPE + SWA-RoPE interleaving extrapolates beyond training length without long-context data.

**Decision:** default pattern `block_sparse` = causal ∩ {local window W=8192} ∩ {anchor: first 128 tokens}. All 6 layers use the same pattern (uniform — simpler than Gemma-2 interleaving; SWAA shows anchors + uniform SWA already recover most quality). Window 8192 is chosen so that the receptive field (n_layers·W = 49K) exceeds the 32K context (Mistral's own design rule), and 128-token anchor matches the sink-token evidence.

### 2.1 Kernel dispatch (device support, verified 2026)

| Kernel | T4 (sm75) | P100 (sm60) | 3090/4090 (sm86/89) | Apple M-series |
|---|---|---|---|---|
| FlexAttention (Triton backend) | ✓ | ✗ (Triton needs sm70+) | ✓ | ✓ (native since PyTorch 2.13; ~12× over SDPA on sparse patterns, PyTorch 2.13 release notes) |
| FlexAttention (FLASH/FA4 backend) | ✗ | ✗ | ✗ | ✗ (Hopper+ only) |
| SDPA (mem-efficient) | ✓ | ✓ | ✓ | ✓ |
| Chunked eager fallback (our `_eager_chunked`) | ✓ | ✓ | ✓ | ✓ |

Dispatch: `flex` on CUDA sm70+ and MPS; chunked eager elsewhere. Notes: Triton path is the default FlexAttention backend and performs ~90%/85% of FA2 fwd/bwd (FlexAttention blog); FA4 backend brings 1.6–3.2× fwd over Triton on Hopper+/Blackwell only (PyTorch blog, 2026-03). Deterministic backward available via `torch.use_deterministic_algorithms(True)`. The v1 concern about MPS support is resolved in torch ≥ 2.13.

### 2.2 Why not MoBA / NSA / SSA as the default?

- **MoBA** (arXiv 2502.13189, deployed in Kimi): selects top-k *blocks* per query block via pooled affinity scores, then attends over selected blocks; sub-quadratic; strong results at 1M. Implemented here as an optional pattern (`pattern="moba"`) — pure-PyTorch block selection works with the same chunked kernel — because it is the *research-grade* sparse attention with production evidence, and the cleanest data-dependent alternative to our static mask.
- **NSA** (arXiv 2502.11089): higher complexity — three branches (compression/selection/window) with learned gates and a two-pass backward; real speedups require its custom Triton kernels. Its quality parity at 64K is earned at engineering cost we don't need at 32K. **Deferred** (documented).
- **SSA** (arXiv 2507.15441): inference-oriented selective state-space attention; no training benefit at this scale. **Rejected.**
- **Mamba-family hybrid** is addressed in §3.

### 2.3 KV cache (decode)

GQA-3 with a rolling window: 6 × 2 × 3 × 64 × 8192 × 2B ≈ **37 MB/seq** (full 32K dense would be 147 MB). Rolling-buffer cache is implemented for generation; anchor tokens are always retained in the cache (StreamingLLM-style).

---

## 3. Hybrid Transformer–Mamba: verdict — defer at 70M, provide as ablation

| Work | Result | Relevance at 70M/32K |
|---|---|---|
| Mamba (arXiv 2312.00752) | Selective SSM; linear in T; 5× decode throughput; Mamba-3B ≈ Transformer-6B | **2× parameter tax** for parity |
| Mamba-2 (arXiv 2405.21060) | SSD theory; chunked algorithm ~25 lines of PyTorch; 2–8× faster than Mamba-1 scan; crosses FA2 at ~2K seq, 6× faster at 16K | Pure-PyTorch hybrid is implementable; the 2×-params tax persists |
| Jamba (arXiv 2403.19887) | Transformer+Mamba interleaved (1 attention : 7 Mamba), MoE every other layer, 256K ctx | Win at 52B/256K where KV cache dominates |
| Hymba (arXiv 2411.13676) | Small LMs: attention+SSM *heads in parallel*, meta tokens; 1.5B beats Llama-3.2-3B with 11.7× cache cut | Hybrid helps small models — but via parallel heads, not layer replacement |
| Nautile-370M (arXiv 2604.24809) | 370M: 2 SCA : 1 transformer layers; linear-time spectral attention (SeqCond); trained on one TPU slice | SCA ≠ Mamba; still marginal evidence at 70M |

**Verdict:** **defer Mamba layers in the default config.** Reasons: (a) SSM quality needs ~2× params (Mamba-3B ≈ Transformer-6B, and Mamba-2 inherits the scaling); at a hard 70M *active* budget that tax is fatal; (b) at 32K with block-sparse attention (≈4.9M MACs/token) attention is already near-linear — the SSM's linear advantage is spent; (c) fused selective-scan kernels are CUDA-only; the portable chunked SSD is correct but slower than FlexAttention on every target device; (d) Jamba's win case (52B/256K KV bottleneck) does not apply here. Nautile-370M is the only *small-scale* hybrid counter-evidence, and its SCA layer is not a Mamba layer.

**Provided for ablation:** `mamba2.py` — a chunked SSD layer (Mamba-2 style, pure PyTorch, no custom kernels) with Jamba-style ratio `hybrid.attn_layers` (e.g. `[0, 4]` = 2 attention / 4 Mamba layers). Default: 6/0. If a future 70M hybrid experiment runs, the Jamba interleaving study (interleaving order matters; 1:7 optimal at scale) governs the ratio.

---

## 4. Unchanged verdicts (re-confirmed at 32K)

| Component | v1 verdict | Re-check at 32K |
|---|---|---|
| **GQA** (9 q / 3 kv, hd=64) | IN | KEEP. KV 37 MB/seq with the rolling window; GQA is still the right KV-compute tradeoff (GQA ≈ MHA quality at ≈MQA speed, arXiv 2305.13245) |
| **MLA** (arXiv 2405.04434) | POSTPONED | STILL POSTPONED. MLA's 93.3% KV cut pays at ≥128K (DeepSeek-V2/V3, Kimi K2 all use MLA at scale); at 32K the windowed GQA cache (37 MB) is not a bottleneck and training FLOPs are unchanged. Revisit for a ≥128K v3. Implementation exists (`mla.py`) behind a flag |
| **DeepSeekMoE** (1×1152 shared + 12×928 top-4) | IN | KEEP. With attention cheaper, MoE is now ~55% of transformer FLOPs — the FLOP-efficiency case is stronger, not weaker (arXiv 2401.06066: 2B ≈ GShard 2.9B) |
| **mHC** (arXiv 2512.24880) | POSTPONED-but-implemented | STILL POSTPONED: the paper's identity-mapping benefit targets deep models (Table 1: 3B+); 6 layers don't need it. Flag stays |
| **FA2 / SDPA** | IN | KEEP for causal paths (8K stage, decode); FlexAttention replaces it for the 32K block-sparse path |

---

## 5. Optimizer: Muon (MuonClip-style) becomes the default — evidence review

| Work | Result |
|---|---|
| Muon (K. Jordan et al. 2024) | Momentum orthogonalized via Newton–Schulz; full-rank updates; beats AdamW at small scale |
| **Moonlight** (arXiv 2502.16982, "Muon is Scalable") | + weight decay + consistent update RMS (`0.2·√max(n,m)` scale) → reuses AdamW hyper-parameters; **~2× compute efficiency** (Muon needs ≈52% of AdamW FLOPs to match) |
| **IMU-1** (arXiv 2602.02522) | 430M model / 72B tokens: NorMuon achieves **−2.9% relative loss vs AdamW** — the small-model regime directly matching ours; recipe includes QK-norm, per-head gating, value residuals |
| **MuonClip / Kimi K2** (arXiv 2507.20534) | Muon + per-head **QK-Clip** (γ = τ/S_max, τ=100; W_q·√γ, W_k·√γ); **15.5T tokens with zero loss spikes**; used WSD + staged 4K→32K + YaRN→128K |
| torch.optim.Muon | Shipped in PyTorch ≥ 2.12 with `adjust_lr_fn="match_rms_adamw"` |

**Why Muon now, when v1 said "AdamW default, Muon opt-in":** v1 predated the small-model evidence. IMU-1's 430M/72B result is exactly our regime (70M/319M), Moonlight gives the hyper-parameter transfer recipe (lr 0.02 for 2D matrices, AdamW for 1D params), and QK-Clip removes the failure mode (max-logit explosion) that made raw Muon risky. The per-step cost is *lower* than AdamW (no second moment), and momentum is kept fp32.

**Decision — default optimizer: MuonClip-style.**
- Update rule (Algorithm 1, arXiv 2507.20534):
  - `M_t = μ·M_{t−1} + G_t` (μ = 0.95, Nesterov: NS applied to `G_t + μ·M_t`)
  - `O_t = NS5(M_t) · 0.2·√max(n, m)` (match Adam update RMS)
  - `W_t = W_{t−1} − η·(O_t + λ·W_{t−1})` (decoupled weight decay λ = 0.1)
  - QK-Clip per head after the step: `γ_h = min(1, τ/S_max^h)`; `W_q^h ← W_q^h·√γ_h`, `W_k^h ← W_k^h·√γ_h` (τ = 100). For GQA the shared KV head is scaled by the min γ over its q-heads (approximation, documented).
- Hyper-parameters (Moonlight transfer + NanoGPT speedrun): 2D lr 0.02, 1D (embeddings/bias/norm/gains) AdamW lr 0.01 wd 0.1, momentum 0.95, NS 5 iters, gradient clipping 1.0, fp32 momentum (fp16 params + GradScaler unchanged).
- AdamW remains available (`optimizer: adamw`, the v1 hyper-parameters) as the baseline for the ablation.

Schedule: cosine to 3e-5 per the user constraint (K2 uses WSD; cosine is equivalent-quality for our run length — noted as future option).

---

## 6. 2026 literature sweep (what else changed)

| Work | Finding | Action |
|---|---|---|
| SWAA (arXiv 2512.10411) | SWA adaptation recipes for inference | Incorporated (§2) |
| SWAN (EMNLP 2025) | NoPE + SWA-RoPE extrapolation | Supports SWA; not adopted (RoPE θ=1e6 simpler) |
| AllMem (arXiv 2602.13680) | SWA + TTT memory blocks | Inference-memory trick; N/A training |
| TTT-E2E (arXiv 2512.23675) | SWA + test-time training; constant inference latency; training 3.4× slower at 8K | Rejected (training cost) |
| SparDA (arXiv 2606.04511) | Decoupled sparse attention + forecast KV offload | Inference-only; N/A |
| FlexAttention FA4 backend (PyTorch blog 2026-03) | 1.6–3.2× over Triton on Hopper+; deterministic block-sparse backward | Not on our targets; Triton/MPS paths suffice |
| FlexAttention MPS (PyTorch 2.13) | Native Metal kernels, ~12× vs SDPA on sparse patterns | Adopted on M-series (§2.1) |
| torch.optim.Muon (PyTorch 2.12+) | Official Muon implementation | Documented; we ship our own for version-independence |
| Kimi K2 (arXiv 2507.20534) | MuonClip + ultra-sparse MoE + MLA + YaRN staging at 1T | Evidence for §1, §2.2, §5 |

---

## 7. Component comparison at 32K

| Criterion | Dense causal | SWA W=8192 | SWA+anchor (selected) | MoBA top-k | NSA | Mamba-2 hybrid |
|---|---|---|---|---|---|---|
| MACs/token (attention) | 18.9M | 4.7M | ≈4.9M | ~sub-quadratic | ~sub-quadratic | ~linear (SSM) |
| Long-range recall | best | weak | **strong (sink)** | strong | best | weak→ok |
| Trainable on P100/MPS | ✓ | ✓ | ✓ (chunked) | ✓ (chunked) | ✗ (kernels) | ✓ (chunked SSD) |
| Extra params | 0 | 0 | 0 | router 0.5K | gates 0.3K | (per-layer MLP) |
| Implementation risk | low | low | low | med | high | med |
| Production evidence | — | Mistral, Gemma-2 | StreamingLLM, SWAA | Kimi | DeepSeek | Jamba, Hymba |
| **Verdict** | stage A only | ablation | **default** | optional | deferred | optional ablation |

## 8. Final configuration (summary)

```
context_len 32768, rope_theta 1e6
attention.pattern = block_sparse (window 8192 + anchor 128, all 6 layers)
attention.kernel  = auto → flex (sm70+/MPS) → chunked eager (sm60/CPU)
MoE: unchanged (1×1152 shared + 12×928 routed top-4, cap 1.25)
MLA off, mHC off, mamba2 layers [] (ablations behind flags)
optimizer = muon_clip (lr 2d 0.02 / 1d 0.01, wd 0.1, μ 0.95, NS5, QK-Clip τ=100)
training curriculum: 50% steps @ 8K dense → 50% @ 32K block-sparse
```

## 9. References (new in v2)

- Kimi Team. *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534 (MuonClip Algorithm 1).
- Liu, J. et al. *Muon is Scalable for LLM Training.* arXiv:2502.16982 (Moonlight).
- Su, J. *QK-Clip: Taking Muon one step further.* frontier.soket.ai, 2025-07.
- IMU-1 team. arXiv:2602.02522 (NorMuon, small-model recipe).
- Dao, T. & Gu, A. *Transformers are SSMs: Mamba-2.* arXiv:2405.21060.
- Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling.* arXiv:2312.00752.
- Jamba (AI21). arXiv:2403.19887. Hymba (NVIDIA). arXiv:2411.13676. Nautile-370M. arXiv:2604.24809.
- Peng, B. et al. *YaRN.* arXiv:2309.00071. Chen, S. et al. *LongRoPE.* arXiv:2402.13753. arXiv:2306.15595 (PI). arXiv:2309.08710 (NTK-aware).
- Yuan, J. et al. *NSA.* arXiv:2502.11089 (ACL 2025 best paper). MoBA. arXiv:2502.13189. SSA. arXiv:2507.15441.
- SWAA. arXiv:2512.10411. SWAT. arXiv:2502.18845. SWAN (EMNLP 2025). AllMem. arXiv:2602.13680. TTT-E2E. arXiv:2512.23675. SparDA. arXiv:2606.04511. StreamingLLM. arXiv:2309.17453.
- PyTorch: *FlexAttention* (pytorch.org/blog/flexattention), *FlexAttention + FlashAttention-4* (2026-03), PyTorch 2.13 release notes (MPS FlexAttention, deterministic backward), `torch/optim/_muon.py` (GitHub main).
