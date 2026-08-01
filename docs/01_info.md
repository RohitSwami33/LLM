# 01 — Why Muon Was Dropped & Why Training Crashed Multiple Times

A log of two hard-won lessons from pretraining the 68.7M TransformerLM on Kaggle T4s.

---

## 1. Why We Do NOT Use the Muon Optimizer

Muon (Newton-Schulz orthogonalization + momentum) was the original optimizer for this project. After extensive review and testing it was **removed in favor of plain AdamW**. Reasons below.

### 1.1 Muon is hard to implement correctly

`MuonWithAuxAdam` went through a full correctness review which found **6 bugs** before it even ran:

| # | Bug | Fix |
|---|-----|-----|
| 1 | Missing step counter for bias correction | Added explicit counter |
| 2 | First-moment bias correction frozen at `step=1` (`1-b1` instead of `1-b1**step`) | Real bias correction |
| 3 | Second-moment bias correction frozen at `step=1` (`sqrt(1-b2)` instead of `sqrt(1-b2**step)`) | Real bias correction |
| 4 | Weight decay applied *after* the update instead of before | Reordered |
| 5 | Closure was never evaluated | `with torch.enable_grad()` |
| 6 | Variable named `step` shadowed the step counter | Renamed |

Even after all 6 fixes the implementation was only ever *numerically identical to AdamW* (max diff 1.19e-7) — i.e., the complexity bought nothing on this model.

### 1.2 Muon destabilized training (NaN)

On Kaggle, the Muon run **diverged to NaN by step ~100** (`17.05 → 18.61 → 21.45 → NaN`). Root causes found:

- **Muon was applied to `tok_emb` and `lm_head`** — the two largest matrices. Orthogonalizing embedding/head matrices destabilizes training; Muon should only touch "hidden" (2D) parameters that stay near orthogonality, not the embedding/head pair.
- **Muon LR was set equal to Adam LR (3e-4)**. Muon needs a much higher LR (roughly **20x**, i.e. ~0.02) to make progress, because the Newton-Schulz step is a near-isometry, not a gradient descent step.
- Loss was computed inside FP16 autocast (fixed by computing loss in FP32 with `.float()` before `cross_entropy`).
- `grad_clip=1.0` was too aggressive for the combined optimizer states.

### 1.3 Conclusion

For a 68.7M model, AdamW is:
- **Stable** — no NaN, no LR gymnastics
- **Simple** — `torch.optim.AdamW(..., lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)`
- **Proven** — the current run trains cleanly from loss 10.43 down with it

Muon's theoretical benefit (better conditioning) matters at scale (e.g. 1B+ params) where the AdamW LR can't be tuned finely; at 68.7M it is pure risk with no measured gain. **If Muon is ever reintroduced: keep it off `tok_emb`/`lm_head`, use `muon_lr ≈ 20 × adam_lr`, and verify with a short CPU run against AdamW first.**

---

## 2. Why Training Crashed (Chronological Postmortem)

Seven kernel versions were pushed. The first six failed for six different reasons.

### 2.1 NaN loss (kernel v1 era, pre-AdamW)

- **Symptom:** loss `17.05 → 18.61 → 21.45 → NaN` around step 100
- **Causes:** Muon on embeddings/head, Muon LR = Adam LR, loss in FP16 autocast, `grad_clip=1.0`, `warmup=500` too short
- **Fixes:** switched to pure AdamW, FP32 loss, `grad_clip=5.0`, `warmup=2000`, token-ID range validation

### 2.2 `tokens.bin` missing (kernel v1)

- **Symptom:** `ERROR: tokens.bin not found at /kaggle/working/tokenized/tokens.bin`
- **Cause:** the training script expected a *pre-tokenized* file that nobody had produced in the session. Pretokenization must run **inside** the same kernel run.
- **Fix:** rewrote the kernel as a two-stage pipeline (`kaggle_pipeline.py`): STEP 1 pretokenize → STEP 2 train, in one session.

### 2.3 Wrong dataset mount path (kernel v2)

- **Symptom:** `FileNotFoundError: Corpus not found: /kaggle/input/research-v2-corpus/corpus.jsonl`
- **Cause:** assumed the classic `/kaggle/input/<dataset-slug>/` mount path. The CLI-mounted dataset actually lives at `/kaggle/input/datasets/tomiokasan/research-v2-corpus/` (verified with a debug kernel that dumped `rglob` of `/kaggle/input`).
- **Fix:** use the full `/kaggle/input/datasets/<owner>/<slug>/` path.

### 2.4 Wrong GPU via invalid `machine_shape` (kernel v2)

- **Symptom:** kernel ran on a **Tesla P100 (sm_60)** — Kaggle's PyTorch build needs sm_70+, so the run died; also much slower
- **Cause:** `kernel-metadata.json` used `machine_shape: "T4"` which is not a valid enum value. Valid values are `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`.
- **Fix:** `machine_shape: "NvidiaTeslaT4"` → kernel reliably gets a T4.

### 2.5 RoPE crash under DataParallel + gradient checkpointing (kernel v4)

- **Symptom:** `RuntimeError: The size of tensor a (36) must match the size of tensor b (2) at non-singleton dimension 4` inside `apply_rotary_emb`
- **Cause:** RoPE was implemented with complex numbers (`torch.view_as_complex` + complex multiply). This path breaks when the checkpointing wrapper re-invokes the block on a replicated (DataParallel) module.
- **Fix:** replaced the complex RoPE with the **pure-real rotate-half form** (cos/sin on `x[..., 0::2]`, `x[..., 1::2]`). Verified numerically identical to the complex version (max diff 4.8e-7), so checkpoints stay compatible. This form is also faster and robust under checkpointing.

### 2.6 `DataParallel` attribute error (kernel v5)

- **Symptom:** `AttributeError: 'DataParallel' object has no attribute 'config'`
- **Cause:** after wrapping the model in `nn.DataParallel`, the loss line still read `model.config.vocab_size` — but `config` lives on the inner module, not the wrapper.
- **Fix:** keep a `raw_model` reference; use `raw_model.config.vocab_size` (and `raw_model` for EMA, checkpoint save/load, eval, and generation).

### 2.7 CUDA OOM on GPU 0 (kernel v6)

- **Symptom:** `torch.OutOfMemoryError: Tried to allocate 1.95 GiB` during `backward()`
- **Cause:** `shift_logits = logits[:, :-1, :].float().contiguous()` materialized a **2.1 GB fp32 copy** of the (8, 2048, 32000) logits *every microbatch*, and DataParallel gathers outputs onto GPU 0.
- **Fix:** compute `F.cross_entropy` **inside the autocast block** on fp16 logits (PyTorch internally upcasts to fp32 for the softmax, so numerical safety is preserved) and drop the `.float().contiguous()` copy. Re-enabled gradient checkpointing now that RoPE is fixed. GPU usage dropped from 12.97 GB to **2.2 GB** per GPU.

### 2.8 (Fixed but not a crash) tok/s was under-reported 8x

- The token counter incremented once per optimizer step but only counted one micro-batch's tokens. Real throughput was ~11.2k tok/s, not 1400. Counter moved into the micro-batch loop.

---

## 3. Final Working Configuration (kernel v7)

| Setting | Value |
|---|---|
| Optimizer | `AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)` |
| GPUs | 2x Tesla T4 via `nn.DataParallel` (batch 4 per GPU) |
| Effective batch | 32 (batch 8, grad accum 4) |
| Mixed precision | FP16 autocast + GradScaler; loss computed inside autocast |
| Attention | `F.scaled_dot_product_attention(is_causal=True)` — the old code accepted `flash=True` but never used it |
| RoPE | pure-real rotate-half (checkpoint-safe) |
| Gradient checkpointing | ON (safe again after RoPE fix) |
| Throughput | ~26.4k tok/s (2.4x the single-GPU run) |
| GPU memory | ~2.2 GB / 15.6 GB per GPU |

### Lessons that would have saved 6 failed pushes

1. **Test the exact kernel path** (mount path, GPU enum, DataParallel wrapper) with a tiny debug kernel *before* the real run.
2. **Beware `.float()` on the full logits tensor** — it's a hidden 2 GB allocation per micro-batch.
3. **`nn.DataParallel` hides the inner module** — always keep `raw_model` for attributes, EMA, checkpoints, and generation.
4. **Complex-number RoPE is fragile under checkpointing/DP** — prefer the real rotate-half form.
5. **Validate kernel metadata enums** — an invalid `machine_shape` silently picks the wrong GPU.

---

## 4. Project Biography — Everything at a Glance

### 4.1 What we are building

A from-scratch **68.7M parameter TransformerLM** (dense, decoder-only, RoPE + SwiGLU + RMSNorm) pretrained on a custom research/coding corpus, trained on **free Kaggle GPU kernels** and managed entirely from the CLI.

### 4.2 The corpus

| Item | Value |
|---|---|
| Source | `datasets/research_v2/corpus.jsonl` (mirrored to Kaggle dataset `tomiokasan/research-v2-corpus`) |
| Docs | **384,996** |
| Raw text | 1.28 GB |
| Tokenized tokens | **319,003,339 (~319M)** |
| Tokenizer | SentencePiece, **32,000** vocab (BPE), `tokenizer/tokenizer.model` |
| Token storage | `tokens.bin`, uint16 (638 MB) |
| Split | 95% train (147,974 seq) / 5% val (7,789 seq) |

### 4.3 The model

| Setting | Value |
|---|---|
| Params | 68,721,984 (68.7M) |
| d_model / heads / layers | 576 / 8 / 6 |
| d_ff | 2304 (SwiGLU) |
| Context | **2048 tokens** |
| Vocab | 32,000 |
| Norm | RMSNorm (pre-norm) |
| Positional | RoPE, base 10000 |
| Tie weights | No |
| Dropout | 0.0 (pretraining) |

### 4.4 Training budget — the tokens question

| Item | Value |
|---|---|
| **Max steps** | 50,000 |
| Effective batch | 32 sequences (batch 8 across 2 GPUs, grad accum 4) |
| Tokens per step | 32 × 2048 = **65,536** |
| **Total training tokens** | 50,000 × 65,536 = **3,276,800,000 (~3.28B)** |
| Epochs over corpus | 3.28B / 319M ≈ **10.3 epochs** |
| Warmup | 2000 steps (0.4% of run), linear → 3e-4 |
| Schedule | Cosine → min_lr 3e-5 |
| Optimizer | AdamW (lr 3e-4, betas 0.9/0.95, wd 0.1, eps 1e-8) |
| Grad clip | 5.0 |
| Precision | FP16 autocast + GradScaler (loss inside autocast) |
| EMA | On, decay 0.9999 |
| Checkpoints | every 500 steps (keep last 3) + best.pt + final.pt |

3.28B tokens ≈ 2.8× Chinchilla-optimal for 68.7M (which suggests ~1.2B), so 50k steps is a solid, slightly compute-heavy budget.

### 4.5 Hardware & throughput

| Item | Value |
|---|---|
| GPU | 2× Tesla T4 (16 GB each), `NvidiaTeslaT4` in kernel metadata |
| Parallelism | `nn.DataParallel` (batch 4 per GPU) |
| Throughput | **~26.4k tok/s** (11.2k tok/s single-GPU before fixes) |
| Time per step | ~2.5 s |
| **Estimated full run** | ~34 h (3× 12 h Kaggle sessions) |
| Steps per 12 h session | ~17,300 |
| GPU memory used | ~2.2 GB / 15.6 GB per GPU |

### 4.6 Progress so far (current run, kernel v7)

| Step | Train loss | Val loss | Val PPL | Acc | lr |
|---|---|---|---|---|---|
| 0 | 10.43 | — | — | 0.01% | 0 |
| 500 | 6.87 | 6.88 | 974 | 10.1% | 7.5e-05 |
| 650 | 6.66 | — | — | 11.3% | 9.75e-05 |

Reference points: random accuracy on 32k vocab is 0.003%; a well-trained 68M model should reach ~30–45% top-1 next-token accuracy (val loss ~3.3–4.0).

### 4.7 How training is run & monitored

```bash
# Push the kernel (resumes from latest /kaggle/working checkpoint if any)
.venv/bin/kaggle kernels push -p .

# Status
.venv/bin/kaggle kernels status tomiokasan/research-v2-pretrain

# Live logs (streams while running)
PYTHONUNBUFFERED=1 .venv/bin/kaggle kernels logs tomiokasan/research-v2-pretrain --follow

# Download outputs when done
.venv/bin/kaggle kernels output tomiokasan/research-v2-pretrain -p /tmp/kaggle_out
```

### 4.8 Known constraints & caveats

- **12 h session limit** — Kaggle kills GPU kernels at 12 h; current speed means ~3 sessions for 50k steps.
- **`/kaggle/working` did not persist between kernel versions** — the v7 run started from step 0, not the step-500 checkpoint. To truly resume, checkpoints must be downloaded and re-uploaded as a dataset, or the kernel must finish in one session.
- **Total compute**: 3.28B tokens at ~26.4k tok/s ≈ 34 GPU-hours (well within Kaggle's weekly free quota of ~30 h/wk of GPU time).
- Accuracy at 11% (step 650) is normal — it climbs steeply after warmup ends (step 2000).
