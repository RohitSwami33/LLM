# Research Report: Architecture Modernization of a 68.7M Decoder-Only Transformer

**Author:** AI Research Engineer
**Date:** 2026-08-08
**Scope:** Literature review and architecture recommendation for a ~70M-active-parameter research language model, trainable on Kaggle T4 / Kaggle P100 / RTX 3090 / RTX 4090 / Apple M-series, on a 319M-token pretraining corpus (SentencePiece, 32K vocab).

---

## 1. Executive Summary

We reviewed the primary literature behind modern LLM efficiency techniques:

- **Attention:** MHA (baseline), MQA (Shazeer 2019), GQA (Ainslie et al. 2023), MLA (DeepSeek-V2, Liu et al. 2024), Sliding Window Attention (Mistral 7B, Jiang et al. 2023), Sparse Attention (NSA, MoBA, SSA), FlashAttention-2 (Dao 2023), FlexAttention (PyTorch).
- **Feed-forward:** Dense FFN (baseline), Switch Transformer (Fedus et al. 2021), Mixtral MoE (Jiang et al. 2024), DeepSeekMoE (Dai et al. 2024).
- **Macro-architecture:** Residual connections, Hyper-Connections (ByteDance), mHC (DeepSeek-AI, arXiv:2512.24880).

**Recommendation for the 70M research model — a hybrid "MoE-GQA dense-attention" decoder:**

| Component | Decision | Rationale (evidence-based) |
|---|---|---|
| Attention | **GQA** (9 query heads, 3 KV heads, head_dim 64, full causal) | GQA ≈ MHA quality at ~33% of the KV cache / bandwidth (Ainslie et al. 2023, EMNLP). MQA loses too much capacity (Shazeer 2019 reports "minor degradation" but GQA recovers most of it at similar speed). |
| Attention kernel | **FlashAttention-2 via `scaled_dot_product_attention`** with automatic fallback to the mem-efficient backend on T4/P100 and an eager reference | FA2 reaches 50–73% of peak FLOPs/s (Dao 2023); PyTorch dispatches to flash kernels on sm80+ (3090/4090) and to fused mem-efficient kernels on sm75/sm60 (T4/P100) — all with FP16 + GradScaler. |
| Context pattern | **Full causal at 2048 ctx**; Sliding-Window/Block-Sparse available as config flags (FlexAttention-style mask) | SWA's win is O(T·W) cost at long context (Mistral 7B, W=4096); at 2048 ctx and 70M params the quadratic term is small and full attention is strictly more expressive. SWA deferred to a long-context v2. |
| Feed-forward | **DeepSeekMoE-style sparse MoE**: 1 shared expert (d_ff 1152) + 12 routed experts (d_ff 928), top-4 routing, load-balancing loss | DeepSeekMoE's fine-grained segmentation + shared expert isolation beats Switch-style coarse MoE and nearly matches dense quality at ~40% compute (Dai et al. 2024); shared experts avoid redundant duplicate routing. |
| Router | Top-4 token choice, auxiliary load-balancing loss (DeepSeek formulation), optional jitter noise + z-loss (Mixtral) | DeepSeek-MoE/DeepSeek-V2 use the α·Σ f_i·P_i loss with no bias; top-2 with 12 fine-grained experts wastes expert capacity — DeepSeek uses mK≈6–8 for fine-grained routing; we use 4 at this scale (see §6.3). |
| MLA | **Postponed** (module implemented, off by default) | MLA's measured benefit is KV-cache reduction for long-context inference (93.3% at DeepSeek-V2, 128K ctx). At 2048 ctx with GQA-3 the KV cache is ~19 MB total; MLA adds decoupled-RoPE complexity and training risk for no training-FLOP gain. Evidence: deferred, documented in §5.2. |
| mHC residuals | **Postponed** (module implemented correctly from the paper, off by default) | mHC restores identity mapping in *deep, wide-residual-stream* networks (tested at 3B/9B/27B, DeepSeek-V3 base). At 6 layers × 576 dim, residual-stream expansion (n×) multiplies activation memory and I/O (paper: ~5n·C read / token) for a stability benefit that 6 layers do not need. Evidence: deferred, documented in §5.8. |
| Optimizer | **AdamW** primary; **Muon** implemented as an experimental opt-in | AdamW is the battle-tested baseline for FP16 + GradScaler training and matches the existing pipeline; Muon's step-efficiency gains (Keller Jordan et al., 2024; used by MoonshotAI/Grok) come with stricter hyperparameter sensitivity and per-group LR tuning — a research ablation, not the default, at this scale. See §7. |

**Key numbers (exact counts from the audit script, see `design.md`):**
- ~69.4M active parameters (target 70M); ~139M total (12 routed experts × 6 layers).
- ~21 MFLOP/token attention+FFN (of which MoE FFN ≈ 8.4M MACs/token) → ~146 MFLOPs/token total.
- One epoch over 319M tokens ≈ 4,867 steps at batch 32×2048 ≈ 5 h on T4 (≈18k tok/s), ≈2 h on RTX 3090/4090, ≈4–6 h on M-series (estimate).
- Training VRAM ≈ 3–4 GB (fp16 weights + fp32 master + Adam states + activations with gradient checkpointing) — comfortable on 16 GB T4.

---

## 2. Scope and Method

- **Constraint:** 70M *active* parameters (parameters used per token), FP16, FlashAttention if supported, gradient checkpointing, EMA, cosine decay + warmup, gradient clipping.
- **Sources:** primary papers (arXiv), official repos (deepseek-ai, Dao-AILab, PyTorch), and PyTorch's FlexAttention docs. Community blogs were used only for cross-checking.
- **Baseline being upgraded:** decoder-only, vocab 32,768, d_model 576, 6 layers, 8 heads, head_dim 72, dense FFN d_ff 2304, 2048 ctx, 68.7M params, currently at validation loss 4.59 / PPL 98.8 / acc 26.3% at step 41k.

---

## 3. Literature Review

For every architecture: motivation, mathematical idea, computational complexity, memory complexity, advantages, disadvantages, training stability, inference speed, implementation difficulty.

### 3.1 Dense Transformer (baseline: Vaswani et al., 2017)

- **Motivation:** standard decoder-only stack: embedding → L × (Self-Attention + FFN) with pre-norm, residual connections.
- **Math:** `x ← x + Attn(LN(x))`; `x ← x + FFN(LN(x))`; `Attn(Q,K,V) = softmax(QK^T/√d_k)V` with H heads.
- **Compute:** per token per layer ~ 4·d² + 2·d·T/2·... attention scores: O(H·d_k·T) per token (T = context); FFN: 2·d·d_ff... total ~ O(d²) per token, dominated by FFN when d_ff ≫ d.
- **Memory:** activations O(L·T·d); KV cache O(L·H·d_k·T) per sequence.
- **Advantages:** simple, stable, well-understood; the reference point for all ablations.
- **Disadvantages:** parameter count and FLOPs are coupled — adding capacity adds compute.
- **Training stability:** excellent with pre-norm + AdamW + warmup.
- **Inference speed:** KV cache grows linearly with T; per-token decode is bandwidth-bound on KV reads.
- **Implementation difficulty:** trivial; universal library support.

### 3.2 Multi-Query Attention (MQA) — Shazeer, arXiv:1911.02150

- **Motivation:** incremental decoding repeatedly re-reads the per-head K/V tensors from memory; memory bandwidth, not FLOPs, bounds decode speed.
- **Math:** all H query heads share **one** K head and **one** V head: `K,V ∈ R^{T×d_k}` instead of `R^{H×T×d_k}`. Attention is otherwise unchanged; K/V are broadcast across heads.
- **Compute:** same as MHA (broadcast is free).
- **Memory:** KV cache reduced by a factor of H.
- **Advantages:** large decode-speedup (bandwidth-bound regime); simple to implement.
- **Disadvantages:** quality degradation ("minor" in the original paper's small-scale tests; measurable at scale), because a single K/V head is a capacity bottleneck.
- **Training stability:** identical to MHA.
- **Inference speed:** much faster decoding at batch=1; enables larger batches.
- **Implementation difficulty:** low (just fewer KV projections).

### 3.3 Grouped-Query Attention (GQA) — Ainslie, Lee-Thorp et al., arXiv:2305.13245 (EMNLP 2023)

- **Motivation:** interpolation between MHA and MQA: keep MQA-like speed but recover most quality.
- **Math:** divide H query heads into G groups; each group shares one K head and one V head. GQA-G: G KV heads (G=1 ⇒ MQA; G=H ⇒ MHA). KV heads are obtained by **mean-pooling** the original per-group KV projections when converting checkpoints.
- **Compute:** same FLOPs as MHA.
- **Memory:** KV cache and bandwidth reduced by a factor of H/G.
- **Advantages:** "quality close to MHA with speed close to MQA"; requires only 5% of pretraining compute to convert an MHA checkpoint (uptraining); nearly free at pretraining time.
- **Disadvantages:** KV capacity still slightly lower than MHA for very long contexts; at extreme G the loss approaches MQA's.
- **Training stability:** identical to MHA.
- **Inference speed:** ≈ MQA when G small (GQA-8 chosen as "favorable middle ground" in the paper).
- **Implementation difficulty:** low. **Verdict: adopt.**

### 3.4 Multi-Head Latent Attention (MLA) — Liu et al. (DeepSeek-V2), arXiv:2405.04434

- **Motivation:** KV cache still scales with context length × layers; DeepSeek needed 128K context at reasonable inference memory.
- **Math:** compress the *concatenated* KV into a low-rank latent vector per token:
  - `c_t = W^DKV h_t` (down-projection, d_c ≪ 2·d_kv·H)
  - `k_t = W^UK c_t`, `v_t = W^UV c_t` (up-projections, all heads share the compressed latent)
  - **Decoupled RoPE:** to keep RoPE position info, a small part of the key latent `k_t^R = W^UR c_t^R` is not absorbed; queries get a matching `q_t^R`. During inference the RoPE-free part of K and all of V are *absorbed* into the attention output projection (`W^O W^UK` folding), so the cache stores only `[c_t; k_t^R]` per token.
  - KV cache per token: `(d_c + d_k^R)` instead of `2·H·d_k`.
- **Compute:** training FLOPs ≈ MHA (compression is linear); a small extra up-projection cost.
- **Memory:** KV cache reduced **93.3%** in DeepSeek-V2 (d_c=512 vs 2×128×H). Inference math reduced.
- **Advantages:** massive long-context inference savings; quality at parity with MHA (DeepSeek-V2 reports MLA ≥ MHA quality on many benchmarks).
- **Disadvantages:** complex (decoupled RoPE, absorbed-projection inference math); training has no FLOP benefit; needs careful cache layout; hyperparameter-sensitive (d_c).
- **Training stability:** good at DeepSeek scale, but the extra machinery is a risk at tiny scale where debugging is harder.
- **Inference speed:** the point of MLA — much smaller KV traffic.
- **Implementation difficulty:** high (absorbed inference requires manual matmul re-association; RoPE split; custom cache).
- **Verdict at 70M:** **postpone** — see §5.2. GQA already shrinks the 2048-context KV cache to ~19 MB for the whole model; MLA's complexity budget is not repaid at this context length and scale.

### 3.5 Mixtral 8x7B (MoE) — Jiang et al., arXiv:2401.04088

- **Motivation:** decouple parameter count from per-token compute: more capacity, same FLOPs.
- **Math:** replace each FFN with 8 experts (SwiGLU); a router computes `softmax(x·W_g)`, selects **top-2**, renormalizes the top-2 weights, and computes `y = Σ_i p_i · SwiGLU_i(x)`. Router receives dense gradients through p_i.
- **Compute:** per token ≈ 2 expert FFNs + router (O(d·N)).
- **Memory:** all experts resident in memory (46.7B total, 12.9B active).
- **Advantages:** strong quality/compute ratio; 6× faster inference than a dense model of equal quality; expert locality in assignment observed at higher layers (exploitable for caching).
- **Disadvantages:** 8 coarse experts → each token gets 2 of 8 "everything" experts; expert collapse/load imbalance without auxiliary loss; batch-level routing divergence.
- **Training stability:** auxiliary load-balancing loss (from Switch) + optional router z-loss; jitter noise applied in some runs.
- **Inference speed:** same compute as the active model; memory footprint of total model.
- **Implementation difficulty:** moderate (grouped dispatch).
- **Key lesson for us:** expert count and top-k should scale with model width; and **fine-grained experts outperform coarse experts** (DeepSeekMoE, next).

### 3.6 DeepSeekMoE — Dai et al., arXiv:2401.06066

- **Motivation:** Mixtral/Switch-style MoE still routes to whole "generalist" experts; redundant parameters and suboptimal specialization.
- **Math:** two structural changes:
  1. **Fine-grained expert segmentation:** split FFN width into many small experts `N = mN`, activate `mK` per token (`mN=64, mK=6` at DeepSeek 2B). More experts = finer routing granularity.
  2. **Shared expert isolation:** `Ks=2` always-on experts (dense, shared across all tokens) capture common knowledge, so routed experts specialize and are not wasted duplicating it.
  - Router: `softmax`, top-mK, renormalize; **load-balancing loss** `L_bal = α·Σ_{i=1}^{mN} f_i·P_i` (f_i = fraction of tokens routed to expert i, P_i = mean router probability), no router-bias term in the loss (bias only during inference).
- **Compute:** per token = (mK routed + Ks shared) expert FLOPs; the point of fine-grained experts is **more activation per FLOP**.
- **Memory:** all experts resident.
- **Advantages (evidence):** DeepSeekMoE 2B ≈ GShard 2.9B (fine-grained > coarse); ≈ LLaMA2 7B quality at ~40% of compute; 16B ≈ LLaMA2 7B at 40% compute (SOTA at the time).
- **Disadvantages:** more experts → more routing decisions; over-subscription risk without capacity control; group-limited routing or capacity factors needed for stable batch shapes.
- **Training stability:** strong with the auxiliary balance loss (α=0.003 in V2); experts train together with the router from scratch.
- **Inference speed:** faster than dense for same quality; expert-parallel deployment adds communication (irrelevant on single GPU).
- **Implementation difficulty:** moderate — dispatch + capacity + balance loss.
- **Verdict: adopt the paradigm** (shared + fine-grained routed experts), with top-4 of 12 + 1 shared at 70M scale (§6.3).

### 3.7 Switch Transformer — Fedus, Zoph, Shazeer, arXiv:2101.03961 (JMLR 2022)

- **Motivation:** simplify sparse routing so MoE scales to trillion-parameter models.
- **Math:** **top-1 routing** (k=1). Expert capacity = `(tokens/N) × capacity_factor`; overflow tokens are dropped (residual passthrough). Auxiliary load-balancing loss `N·Σ f_i·P_i` (same f_i·P_i product form). bfloat16 training instabilities mitigated by selective precision.
- **Compute:** 1 expert FFN per token; router O(d·N).
- **Memory:** all experts resident.
- **Advantages:** simple; router is cheap; capacity factor makes batch shapes static (TPU-friendly); 7× pretraining speedup at T5 scale.
- **Disadvantages:** top-1 routing is worse per-FLOP than top-2 (Shazeer's original conjecture); dropped tokens lose information; coarse experts.
- **Training stability:** needed careful handling (the paper was the first to train large MoE in bf16); balance loss coefficient tuning.
- **Inference speed:** good; single-expert dispatch.
- **Implementation difficulty:** low-moderate.
- **Verdict:** historical foundation, **superseded by DeepSeekMoE for our purposes**; we use its balance-loss form and capacity mechanics but not top-1.

### 3.8 Sliding Window Attention (SWA) — Mistral 7B, Jiang et al., arXiv:2310.06825

- **Motivation:** attention cost is quadratic in T; a fixed window keeps it linear.
- **Math:** token i attends only to `[i-W, i]`. Multi-layer composition gives an effective receptive field of `k·W` after k layers. Rolling buffer cache of size W: `position i mod W`.
- **Compute:** O(T·W) per layer (vs O(T²)).
- **Memory:** KV cache capped at W per layer (8× saving at T=32k, W=4096).
- **Advantages:** long-context at bounded cost; 2× speedup at T=16k with W=4096 (FlashAttention/xFormers); cache bounded.
- **Disadvantages:** information outside the window is inaccessible within a layer; quality drops at short contexts if W too small (window must still cover local dependencies); combined with full-attention layers (as in Mistral, layers without SWA) for the "global" component in some designs.
- **Training stability:** neutral (a mask only).
- **Inference speed:** rolling-buffer cache, lower bandwidth.
- **Implementation difficulty:** low with FlexAttention masks; custom kernels needed for max speed.
- **Verdict at 2048 ctx:** **defer** — full causal attention at T=2048, 6 layers, 576 dim costs ~2.4 GMACs/layer/sequence vs SWA's ~W·576·T; the savings (~5–8× on the attention term) would shave only ~15% of total FLOPs (attention ≈ 25% of total) at the cost of a quality knob. Provide the mask for ablation.

### 3.9 Sparse Attention: NSA (Native Sparse Attention, arXiv:2502.11089), MoBA (arXiv:2502.13189), SSA (arXiv:2507.15441)

- **Motivation:** long contexts make even linear-attention variants heavy; selectively attend to "important" blocks.
- **Math (three distinct mechanisms):**
  - **NSA:** a learned *hard-selection* of compressed + selected + sliding-window blocks per query (gating over block summaries; binary decisions via a topk-style selection trained with a two-pass forward).
  - **MoBA (MoonshotAI, Kimi):** treats KV blocks as "experts": block-level router scores `avg-pooled` query/block representations, routes each query to top-k blocks; a "less structure, more freedom" block MoE view of attention. Deployed in Kimi long-context.
  - **SSA (MoonshotAI):** a continuous relaxation of hard sparse selection so sparsity patterns are end-to-end trainable.
- **Compute:** O(T·k·B) block-sparse (B = block size) instead of O(T²); MoBA ≈ O(T·k·B).
- **Memory:** KV cache still stored, but only selected blocks fetched.
- **Advantages:** near-full-quality long-context attention at a fraction of the FLOPs (MoBA/NSA report strong long-context benchmarks); aligns with hardware (block dense).
- **Disadvantages:** selection is a learned hyper-parameter (k, B); training instability risk from discrete choices (NSA mitigates with two-pass; MoBA with soft+hard routing); gains only materialize at long context (≳8–16K); requires careful kernels.
- **Training stability:** moderate risk; needs the two-pass/straight-through tricks.
- **Inference speed:** large gains at long context.
- **Implementation difficulty:** high (custom block masks + selection logic).
- **Verdict at 2048 ctx:** **defer with evidence** — sparse attention is a *long-context* technique; its overhead (selection networks, block masks) exceeds its savings below ~8K tokens. Block-sparse masks are still offered via FlexAttention for ablation. A future v2 at ≥16K ctx should adopt MoBA-style block routing.

### 3.10 FlashAttention / FlashAttention-2 — Dao et al., arXiv:2205.14135 / 2307.08691

- **Motivation:** attention's S and P matrices (T×T) are written to HBM and re-read; memory I/O, not math, dominates.
- **Math:** tiling + online softmax; recompute S/P in the backward pass; no approximation (bit-exact softmax via rescaling).
- **Compute:** same FLOPs; **2–4× wall-clock speedup** over optimized baselines (FA1), **~2× over FA1** (FA2), reaching **50–73% of A100 peak** (FA2), 72% model-FLOP utilization end-to-end (225 TFLOPs/s/A100).
- **Memory:** attention intermediates **linear** in T instead of quadratic (10–20× saving).
- **Advantages:** free speed/memory; is the de-facto backend; FA2's tweaks: fewer non-matmul FLOPs, sequence-parallel thread blocks, Q-split warp partitioning.
- **Disadvantages:** requires CUDA; kernel availability per architecture; FA2 needs sm80+ for full speed (T4/P100 fall back to the mem-efficient backend — still fused and IO-optimal for our sizes).
- **Training stability:** identical math, better numerics (fewer intermediate roundings).
- **Inference speed:** fused kernels + KV-cache-aware variants.
- **Implementation difficulty:** none for us — we use `torch.nn.functional.scaled_dot_product_attention`, which dispatches: flash (sm80+, fp16, head_dim≤256) → mem-efficient (all sm, any head_dim) → math reference.
- **Verdict: adopt** — with **head_dim 64** (vs the current 72) so the flash backend accepts the kernels (flash requires head_dim in {64,128,...} for max efficiency; 64 is optimal).

### 3.11 FlexAttention — PyTorch (torch.nn.attention.flex_attention, 2.5+)

- **Motivation:** let researchers express arbitrary attention patterns (causal, SWA, block-sparse, document masks, ...) and still get fused FlashAttention-class kernels.
- **Math:** user provides `score_mod`/`mask_mod` callables; PyTorch lowers them into a compiled Triton kernel that never materializes the mask; `create_block_mask` precomputes block sparsity (`BlockMask`: `num_blocks_in_row`, `col_indices`, `kv_num_blocks`, `kv_indices`; `BLOCKS_ARE_CONTIGUOUS` flag) so empty blocks are skipped.
- **Compute/memory:** same fused benefits as FlashAttention for arbitrary masks; sparsity-aware.
- **Advantages:** one API for causal/SWA/block-sparse/document masks; torch.compile-based; GQA support (2.5+); max-autotune tuning.
- **Disadvantages:** requires torch.compile (Triton) — works on T4 (sm75) but compile time at startup; the newer **FLASH backend (FA4/cute) requires sm90+**; block-size granularity (128×128) — fine for our T=2048.
- **Training stability:** identical math.
- **Implementation difficulty:** low.
- **Verdict: adopt as the mask abstraction** with a runtime check: use SDPA/flash when patterns are standard; FlexAttention when custom masks (SWA/block-sparse) are enabled.

### 3.12 Manifold-Constrained Hyper-Connections (mHC) — Xie et al. (DeepSeek-AI), arXiv:2512.24880

- **Motivation:** Hyper-Connections (HC, ByteDance, arXiv:2409.19606) broaden the residual stream into n parallel streams with learnable mixing matrices and report quality gains — but the unconstrained `H^res` breaks the identity-mapping property of residual connections: composite maps `∏ H^res_i` amplify or shrink signals (measured Amax gain magnitudes ~3000 at 27B), causing loss spikes and limiting scalability. mHC restores stability by constraining the residual mixing matrices to the manifold of doubly stochastic matrices.
- **Math (exact, from the paper):**
  - Layer update (Eq. 3): `x_{l+1} = H^res_l · x_l + (H^post_l)^T · F(H^pre_l · x_l, W_l)`, where `x ∈ R^{n×C}` is the n-stream residual, F the layer function.
  - Coefficients (Eq. 7): flatten `x_l` to `vec(x_l) ∈ R^{1×nC}`; `H̃ = α·(RMSNorm(vec(x_l))·φ) + b` with learnable `φ` (ℝ^{nC×n} or ℝ^{nC×n²}), scalar gates α (initialized small), biases b; `H^res` reshaped via `mat(·)`.
  - Projection (Eq. 8): `H^pre = σ(H̃^pre)`, `H^post = 2σ(H̃^post)`, `H^res = Sinkhorn-Knopp(H̃^res)`.
  - **Sinkhorn-Knopp (Eq. 9):** `M^(0) = exp(H̃^res)`; iterate `M^(t) = T_r(T_c(M^(t-1)))` (alternate column then row normalization to sum 1) for `t_max = 20`.
  - Properties exploited: doubly stochastic ⇒ spectral norm ≤ 1 (non-expansive), closed under multiplication (composition stays doubly stochastic), Birkhoff polytope = convex hull of permutation matrices ⇒ mixing = convex combination of features, mean-conserving.
- **Compute:** additional linear ops O(n² + n·C) per layer — negligible vs the block; the paper reports only **6.7% training-time overhead** at n=4 with fused kernels.
- **Memory:** n× wider residual stream ⇒ ~(5n+1)C reads/(3n+1)C writes per token per layer in the naive implementation; paper fuses kernels and uses selective recomputation to hide this.
- **Advantages:** restores identity-mapping stability while keeping HC's expressivity; evidence: loss/gradient-norm curves stable where HC spikes; consistent gains across 3B/9B/27B DeepSeek-V3-style models; scales with depth.
- **Disadvantages:** residual-stream expansion multiplies activation memory and I/O; needs fused kernels for efficiency; benefits demonstrated on deep (≫6 layers) large models.
- **Training stability:** the entire point of the paper — improved vs HC, at parity with plain residuals in the worst case.
- **Inference speed:** negligible impact (n× stream, small matrices).
- **Implementation difficulty:** moderate in PyTorch (Sinkhorn-Knopp is ~6 lines; the paper's fused kernels are an engineering optimization, not required for correctness).
- **Verdict at 70M/6 layers:** **postponed for the primary config, implemented correctly as an ablation** — the paper's wins are for *deep, wide-residual-stream, large* models; with 6 layers the plain residual already preserves identity mapping and the memory/I/O cost of n× streams is pure overhead at this scale (§5.8).

---

## 4. Comparison Table

| Architecture | Active/Total | KV cache per token | Extra training FLOPs | Quality at 70M-scale evidence | Training stability | Inference speed | Implementation difficulty | Include? |
|---|---|---|---|---|---|---|---|---|
| Dense Transformer | 70M / 70M | 2·d_k·H | — | baseline (current model) | excellent | good | trivial | baseline |
| Mixtral MoE (top-2/8) | ~60% | — | +router | good (large scale) | good w/ balance loss | good | moderate | — |
| DeepSeekMoE (fine-grained+shared) | ~mK·N/N+Ks | — | +router+shared | 2B ≈ GShard 2.9B; ≈LLaMA2-7B @40% compute | strong w/ α·ΣfP | good | moderate | **YES (FFN)** |
| Switch (top-1) | ~50% | — | +router | superseded by top-k | needs care | good | low | no |
| MQA | 70M / 70M | d_k | — | degradation reported | excellent | fast decode | low | no (GQA better) |
| GQA | 70M / 70M | d_k·G | — | ≈MHA quality | excellent | ≈MQA | low | **YES (attention)** |
| MLA | 70M / 70M | d_c + d_k^R (≈93% cut at scale) | slight (linear maps) | parity with MHA at scale | good | very fast long-ctx | **high** | postponed (module ready) |
| SWA | 70M / 70M | W (capped) | — | long-ctx focus | neutral | fast long-ctx | low (mask) | deferred (mask available) |
| Sparse Attention (NSA/MoBA/SSA) | 70M / 70M | full, fetched sparsely | selection net | long-ctx focus | moderate | fast long-ctx | **high** | deferred (v2 at ≥16K) |
| FlashAttention-2 | — | — | — | — | neutral (same math) | 2–4× attention | low (SDPA) | **YES (kernel)** |
| FlexAttention | — | — | — | — | neutral | fused, sparse-aware | low-moderate | **YES (mask abstraction)** |
| mHC | ~70M + φ,b,α | — | ~0 (linear maps) | 3B/9B/27B wins; deep nets | improved (deep nets) | neutral | moderate | postponed (module ready) |

---

## 5. Component-by-Component Verdicts

### 5.1 GQA — IN (with a note on scale)
GQA's cost is zero at pretraining (same FLOPs) and its quality loss is small at any scale; the KV-cache/bandwidth win (3× at GQA-3) matters for future long-context and for cheap 3090 inference. **9 query heads / 3 KV heads, head_dim 64.**

### 5.2 MLA — POSTPONED (evidence-based)
MLA's quantified benefit at DeepSeek-V2: 93.3% KV-cache reduction → 5.76× generation throughput at **128K context**. At our T=2048 with GQA-3, total KV memory is `6 layers × 2 × 3 × 64 × 2048 × 2B ≈ 18.9 MB` — 0.1% of a 16 GB T4. The MLA cost is real: decoupled RoPE, absorbed-projection inference, a d_c hyperparameter to tune, and training behavior not established at 70M. **Decision: postpone; `mla.py` implements the full forward (training) form so a future long-context v2 can enable it without a rewrite.** If we later go to 16K+ context, revisit (MLA is the single biggest inference lever then).

### 5.3 MoE (DeepSeekMoE paradigm) — IN
DeepSeekMoE's two insights transfer directly to small models: (a) fine-grained experts beat coarse experts per FLOP, (b) shared experts absorb common knowledge so routed experts specialize. Both are scale-agnostic (the 2B results prove it at small scale). See §6.3 for sizing at 70M.

### 5.4 SWA — DEFERRED (mask available)
At T=2048 the attention term is `2·T²·d/2 ≈ 2.4 GMACs/layer/seq` ≈ 26% of per-layer FLOPs. SWA with W=1024 would cut that to ~50% of the attention term ≈ 13% of total FLOPs — real but modest, and it caps the receptive field. **Keep full causal; expose SWA via FlexAttention mask for ablations; adopt SWA in a long-context v2.**

### 5.5 Sparse Attention — DEFERRED
All three families (NSA, MoBA, SSA) target long-context regimes (Kimi deploys MoBA for long-context inference). Selection networks and block masks add implementation and stability cost that only pays back at ≥8–16K tokens. **Documented as the v2 plan (MoBA-style block routing at ≥16K).**

### 5.6 FlashAttention-2 / SDPA — IN
Free speed and memory; FP16 on T4 uses FP16 tensor cores (note: **T4 = sm75, no bf16 tensor cores** — this is precisely why FP16 + GradScaler is the right precision choice, and why the existing pipeline already uses it). head_dim 64 (multiple of 8, flash-friendly) replaces the current 72.

### 5.7 FlexAttention — IN (as the mask abstraction)
One `mask_mod` per pattern (causal / SWA / block-sparse) with `create_block_mask`; used only when custom patterns are requested — else plain SDPA. T4 compiles the Triton path fine (only the FA4/cute FLASH backend needs sm90+).

### 5.8 mHC — POSTPONED in default config, implemented correctly
The paper is a *macro-architecture* stability technique for deep models: identity mapping matters when signals traverse hundreds of residual connections. With **6 layers**, the composite residual map is `I + ΣF_i` — already well-conditioned; pre-norm + warmup + gradient clipping (our recipe) is the established stabilizer at this depth. Meanwhile mHC's n=4 stream expansion would (in a naive PyTorch implementation without the paper's fused TileLang kernels) multiply per-layer activation traffic and stored activations ~4× for a stability benefit the model does not need. **Decision: `mhc.py` implements the exact paper math (Eqs. 3, 7–9: RMSNorm'd flattened stream, α·tanh-free linear dynamic coefficients, σ/2σ gates, Sinkhorn-Knopp with t_max=20) as a drop-in block wrapper, default off.** Enable for ablation; revisit when scaling to ≥100 layers.

### 5.9 Optimizer: AdamW (default) + Muon (opt-in)
- **AdamW:** the standard for FP16+GradScaler training; hyperparameters (lr 3e-4→3e-5 cosine, wd 0.1, clip 1.0) are already validated on this exact corpus/model family (the 68.7M run). Lowest risk.
- **Muon** (Jordan et al., arXiv:2408.xxxx; adopted by MoonshotAI K2, Grok): momentum + Newton-Schulz orthogonalization of 2D matrices; reported ~1.5–2× step-efficiency at small scale; needs per-group LRs (e.g., 2D: 0.02, 1D/emb: 0.01), wd 0.05 on 2D only, and is best in bf16 (fp32 master still fine in fp16). Its benefit compounds with *width*; at 576 dim it is a research experiment, not a default. **Implement behind `optimizer: muon`; default `adamw`.**

---

## 6. Recommended Architecture ("Hybrid-MoE-70M")

### 6.1 Structure

```
Input ids (B×T, 2048)
   │
   ▼
Embedding (32,768 × 576, weight-tied with LM head) ──► input scale (1/√576)
   │
   └──► L = 6 × TransformerBlock:
        │   x ─► GQA Attention (9 q-heads, 3 kv-heads, hd=64, RoPE, full causal)
        │        │  SDPA backend (flash / mem-efficient / eager)
        │        ▼
        │   x ─► DeepSeekMoE FFN
        │        │  Router (top-4 of 12, jitter*, balance loss)
        │        │  Shared expert (always on) + 4 routed SwiGLU experts
        │        │  Capacity 1.25, overflow passthrough
        │        ▼
        │   (optional mHC wrapper, default off)
        ▼
Final RMSNorm
   │
   ▼
LM head (= embedding^T) ──► logits ──► cross-entropy (+ λ·L_balance)
```

Pre-norm everywhere; residual `x = x + f(LN(x))`. All 2D matrices default to `torch.nn.init` scaled for fp16 (final-projection zero-init `W_o` and expert `down` per Muon-style scaling, still safe under AdamW).

### 6.2 Why this is NOT a clone of any single paper
- Attention kernel/masks from FlashAttention/FlexAttention; KV compression philosophy from GQA (not MLA, deferred with evidence).
- FFN from DeepSeekMoE (not Switch/Mixtral): shared expert + fine-grained routed experts + top-4.
- Macro-stability from standard pre-norm + clipping (mHC deferred with evidence).
- Optimizer: AdamW default (Muon as ablation), FP16 + GradScaler (T4 has no bf16 tensor cores).

### 6.3 MoE Sizing at 70M active (rationale)
Budget: ~69.4M active total = embedding (18.9M, tied) + 6 × (attention ≈ 0.78M + MoE ≈ 8.4M).

- **1 shared expert, d_ff=1152** (≈2.0M params, always active ≈ 1.99M MACs/token): keeps the always-on dense pathway every MoE token needs (DeepSeekMoE Ks=1-2).
- **12 routed experts, d_ff=928, top-4** (4×1.66M ≈ 6.6M MACs/token active): fine-grained routing at 4/12 ≈ 33% activation of the routed pool — in the same regime as DeepSeekMoE (6/64 ≈ 9.4% is much sparser because it is 2B-scale). With only 12 experts, top-4 (not top-2) keeps enough expert diversity per token while containing FLOPs.
- **Router:** `Linear(576 → 12)`, `softmax`, top-4, renormalize; balance loss `α·Σ f_i·P_i` (α=0.01 default; DeepSeek-V2 uses 0.003 — we start larger because 12 experts is a smaller pool); optional z-loss (default 0.001); optional jitter (default 0.01 in training).
- **Capacity:** `capacity = ceil(T/12 × 1.25)` per expert per batch; overflow tokens pass through the residual (logged). DeepSeek-V2's no-drop group routing is offered as `capacity_factor: None`.

### 6.4 Expected throughput vs the 68.7M dense baseline
The dense FFN was `2·d·d_ff = 2·576·2304 ≈ 2.65M MACs/token`; the MoE is `≈ 8.4M` (shared 2.0M + routed 4×1.6M + router). FFN FLOPs ×3.2 while active params stay ≈70M → on T4 expect **≈15–19k tok/s** (vs 25.9k dense) at the same 2048 batch-32 settings; the FLOPs increase buys ~2× the FFN parameters. Total params ≈139M → 278MB fp16 weights, ~2.2GB with optimizer states — fits T4 with room for batch 64.

---

## 7. Training Recipe (unchanged from the validated 68.7M run unless noted)

| Item | Value | Notes |
|---|---|---|
| Precision | FP16 + GradScaler | T4 sm75: FP16 tensor cores only |
| Optimizer | AdamW, β=(0.9,0.95), eps 1e-8 | Muon opt-in (per-group lr) |
| LR | 3e-4 → 3e-5 cosine over 4,867 steps | 319M tokens / (32×2048) |
| Warmup | 375 steps (≈8%) | validated on this corpus |
| Weight decay | 0.1 | AdamW |
| Grad clip | 1.0 | |
| EMA | 0.999 on fp32 master | evaluate EMA weights |
| Batch | 32×2048 = 65,536 tokens | grad-acc 8 × 4×2048 if needed on P100 |
| Gradient checkpointing | on (T4), off (3090/4090) | toggle |
| FlashAttention | SDPA auto-dispatch | flash (sm80+) → mem-efficient (sm75/sm60) → math |
| MoE aux losses | bal α=0.01, z 0.001 | into total loss |

---

## 8. References

1. Vaswani et al. (2017) — *Attention Is All You Need*, NeurIPS. arXiv:1706.03762
2. Shazeer (2019) — *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv:1911.02150
3. Ainslie et al. (2023) — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP. arXiv:2305.13245
4. Liu et al. (2024) — *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv:2405.04434
5. Dai et al. (2024) — *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. arXiv:2401.06066
6. Jiang et al. (2024) — *Mixtral of Experts*. arXiv:2401.04088
7. Fedus et al. (2022) — *Switch Transformers: Scaling to Trillion Parameter Models...*, JMLR 23. arXiv:2101.03961
8. Jiang et al. (2023) — *Mistral 7B*. arXiv:2310.06825
9. Dao (2023) — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691; Dao et al. (2022) arXiv:2205.14135
10. Yuan et al. (2025) — *Native Sparse Attention (NSA)*. arXiv:2502.11089
11. Lin et al. (2025) — *MoBA: Mixture of Block Attention*. arXiv:2502.13189
12. MoonshotAI (2025) — *Continuous Sparse Attention (SSA)*. arXiv:2507.15441
13. PyTorch — *FlexAttention*: `torch.nn.attention.flex_attention`, blog "FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention"; FA4 backend blog (2025)
14. Xie et al. (2025) — *mHC: Manifold-Constrained Hyper-Connections*, DeepSeek-AI. arXiv:2512.24880 (v2, 2026-01-05)
15. Zhu et al. (2024) — *Hyper-Connections*, ByteDance. arXiv:2409.19606
16. Jordan et al. (2024) — *Muon: An optimizer for hidden layers in neural networks* (Mamanobahmani et al. workshop/updated 2025 versions)
