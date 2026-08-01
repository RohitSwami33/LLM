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
