# Pretraining Progress — 68.7M TransformerLM + 70M Hybrid-MoE (v2)

Status snapshot and full history of the **research-v2 pretraining project** (training
a 68.7M-parameter decoder-only TransformerLM and the 70M Hybrid-MoE research model
on free Kaggle GPU kernels, managed entirely from the CLI).

> Companion docs: `docs/01_info.md` (deep postmortem of the early kernel crashes),
> `docs/KAGGLE_SETUP.md` (Kaggle CLI/account setup), `training/kaggle/kaggle_pipeline.py`
> (dense-68.7M kernel), `training/kaggle/kaggle_pipeline_v2.py` (MoE kernel).

---

## 1. Live Status (updated from the current kernel)

| Item | Value |
|---|---|
| Kernel | `roronoazoro3008/research-v2-pretrain` — **COMPLETE** ✅ |
| Result | **50,000 / 50,000 steps**, 3,276.8M tokens, 6.31 h (final 41k→50k push) |
| Final | Best val loss **4.5870** (PPL ≈ 98.2), final checkpoint `resume.pt` in dataset |
| Checkpoints | `roronoazoro3008/research-v2-checkpoints` (version refreshed at kernel end) |

**Pretraining is DONE** — the SFT/DPO base is the final `resume.pt`.

---

## 2. Model & Data

| Property | Value |
|---|---|
| Architecture | Dense decoder-only Transformer: RMSNorm, SwiGLU, RoPE (pure-real form), no bias, no weight tying |
| Size | `vocab 32k, d_model 576, 6 layers, 8 heads, d_ff 2304` → **68.7M params** |
| Context | 2048 tokens, gradient checkpointing ON, F.scaled_dot_product_attention |
| Optimizer | AdamW `lr 3e-4 → min 3e-5`, `wd 0.1`, `betas (0.9, 0.95)` (Muon dropped — see §4) |
| Precision | FP16 autocast + GradScaler; loss computed inside autocast (FP32 softmax) |
| Batch | effective 32 (batch 8 × grad accum 4) |
| Tokenizer | SentencePiece BPE, vocab 32,000 |
| Corpus | custom research/coding mixture (`corpus.jsonl` ~1.28 GB) — mounted from Kaggle dataset, pretokenized in-kernel to `tokens.bin` (12 min) |
| Budget | 50k steps × 2048 × 32 ≈ **3.28B tokens** (~10 epochs over the ~319M-token corpus) |

---

## 3. Metrics Progression

| Step | Train loss | Train acc | Val loss | Val PPL | Val acc |
|---|---|---|---|---|---|
| 8,000 (v9 boot) | — | — | — | — | 24.8% |
| 24,500 (v4 resume) | ~4.58 | ~26.0% | 4.6783 | 107.59 | 25.52% |
| 29,500 (now) | 4.49 | 26.9–27.6% | **4.6422** | **103.77** | **25.86%** |

---

## 4. Training History (why it took this long)

### 4.1 The early era — 7 failed kernel pushes (`tomiokasan`, 2x T4)

| Kernel | Failure → Fix |
|---|---|
| v1 | NaN (Muon on embeddings, FP16 loss) → AdamW, FP32 loss |
| v2 | corpus mount path wrong; P100 instead of T4 (`machine_shape: "T4"` invalid) |
| v3 | `tokens.bin` missing → pretokenize inside the same kernel |
| v4 | RoPE crash under DataParallel + checkpointing → pure-real rotate-half RoPE |
| v5 | DataParallel attribute error → keep `raw_model` for config/save/eval |
| v6 | CUDA OOM (2.1 GB fp32 logits copy) → loss inside autocast, 2.2 GB VRAM |
| v7 | **first healthy run** — 2x T4, ~26.4k tok/s (did NOT save checkpoints) |

Full postmortem: `docs/01_info.md`.

### 4.2 The checkpoint era (`tomiokasan`, 1x T4) — kernel v8/v9

- **v8** (commit `9ce1e5d`): added the upload/resume machinery — every 500 steps a
  `resume.pt` is staged via hardlink and pushed as a new version of the Kaggle dataset
  `research-v2-checkpoints` (`dataset_create_version`, `delete_old_versions=True`);
  on boot the kernel downloads it back and resumes. Crashed at ~8.5k on a **disk-full**
  bug (upload staging dirs leaked ~11 GB).
- **v9** (commit `a9307fc`): fixed the leak (`shutil.rmtree` on stage/download dirs) —
  disk steady 6.1 GB/21 GB for the whole 12 h; ran 8,000 → **24,525**, val loss 4.6783.
- Hit the weekly GPU quota (`Maximum weekly GPU quota of 30.00 hours reached`) → moved accounts.

### 4.3 The account saga (`roronoazoro3008`)

| Step | What happened |
|---|---|
| v1–v2 | Pushed under the new account → both crashed with `Failed to resolve 'api.kaggle.com'` — **sessions silently ran CPU-only, no internet** |
| Probe #1–#3 | Confirmed: CPU build of torch, all DNS dead, quota `hasEverRun: false`, 6 h/week |
| Root cause | Account not **identity-verified** — Kaggle (post-Dec-2025 accounts) requires Persona identity verification, not just a phone number, for GPU+internet sessions. Pushed metadata (`enable_gpu: true`) is *ignored* until then |
| After Persona verify | Probe #4: `torch 2.10.0+cu128`, `cuda: True`, **Tesla T4**, DNS OK, `kaggle AUTH: OK as roronoazoro3008`; quota now **30 h/week** |
| **v4 (current)** | Pushed with lazy-auth code → clean boot, auto-resume from step 24,500, uploads working |

Key code fix in this era (commit `9c78159`): **lazy Kaggle auth** — the client no longer
authenticates at module import (boot-time DNS outages used to kill the whole session);
it authenticates on demand at download/upload time, retried in background paths.

### 4.4 Lessons that would have saved weeks

1. Check GPU eligibility with a **probe kernel** (CUDA check + DNS check) *before* the real run — `training/kaggle/probe/`.
2. A fresh Kaggle account is **not GPU-eligible until phone AND identity verification** — verify before pushing anything.
3. Quota is account-level (`kaggle kernels` has no quota command; read it via the SDK `get_accelerator_quota_statistics`); new accounts start at 6 h/week until verified.
4. `kernels/list` `enableGpu` fields are unreliable (show `false` even for T4 sessions) — don't use them as a diagnostic.
5. In-kernel DNS can be down at boot: never authenticate at import time.

---

## 5. Checkpoint Upload/Resume Machinery (how it works)

All in `training/kaggle/kaggle_pipeline.py`:

1. **Boot (STEP 1):** pretokenize the mounted corpus → `tokens.bin` (~12 min).
2. **Resume:** `download_remote_checkpoint()` pulls the latest `research-v2-checkpoints`
   dataset version, copies `resume.pt` next to local checkpoints, loads highest step.
3. **Save cadence:** every 500 steps → `_enqueue_upload()` hardlinks `resume.pt` into
   `checkpoint_upload/upload_<step>/` + `latest_step.txt` + dataset metadata.
4. **Background `_upload_worker`:** `dataset_create_version(..., convert_to_csv=False,
   delete_old_versions=True)` then `rmtree`s the stage dir. Prints `[upload] X pushed in Ys`.
5. **End of session:** drains the queue (waits up to 30 min), so the final checkpoint
   always lands in the dataset before the kernel dies.
6. **Resume logic:** `resume.pt` has priority; otherwise the highest `step_*.pt`.
   Tokens already consumed are excluded from the tok/s counter.

`PP_SMOKE=1` forces `KAGGLE_AVAILABLE=False` so local smoke runs can never touch the
real datasets (smoke harness in `/var/folders/.../opencode/smoke_out/`, rebuildable).

Also available: `training/kaggle/colab_train.py` — same pipeline run on Colab
(env-aware paths, installs kaggle+sentencepiece, uses an uploaded `kaggle.json`).

---

## 6. Accounts, Datasets, Quota

| Account | Creds at | GPU quota | Status |
|---|---|---|---|
| `tomiokasan` (renamed from `roronoazoro3008`, 2026-08) | `~/.kaggle/kaggle.json` | 30 h/week (verified) | all current runs |

Kaggle datasets (all under `tomiokasan/`):

| Dataset | Contents | Role |
|---|---|---|
| `research-v2-corpus` | `corpus.jsonl` (1.28 GB) + `tokenizer/` | kernel input mount |
| `research-v2-checkpoints` | `resume.pt` (1.1 GB) + `latest_step.txt` | dense-68.7M checkpoint store |
| `research-moe-code` | `research_hybrid/*.py` (flat) | v2 kernel code mount (see §9.1) |
| `research-moe-checkpoints` | `resume.pt` + `latest_step.txt` | v2 MoE checkpoint store |

Kernel metadata: `kernel-metadata.json` (points at `tomiokasan/research-v2-70m`,
T4, internet on). Kernel owner is auto-detected from the mounted corpus path
(`KAGGLE_USERNAME` is NOT set inside kernels).

---

## 7. Ops Cheat-Sheet

```bash
# push the training kernel
.venv/bin/kaggle kernels push -p .
.venv/bin/kaggle kernels status tomiokasan/research-v2-70m

# stream live logs (SSE endpoint; plain `kaggle kernels logs` returns empty for running sessions)
U=tomiokasan; K=<key from ~/.kaggle/kaggle.json>
curl -s -u "$U:$K" "https://www.kaggle.com/api/v1/kernels/logs/stream/$U/research-v2-70m"

# check the saved checkpoints
.venv/bin/kaggle datasets files tomiokasan/research-v2-checkpoints
.venv/bin/kaggle datasets files tomiokasan/research-moe-checkpoints

# update the v2 code dataset after editing research_hybrid/
.venv/bin/python tools/push_code_dataset.py

# stop a bad run (CLI has no `cancel`; delete the kernel kills the session)
python - <<'EOF'
from kaggle.api.kaggle_api_extended import KaggleApi
KaggleApi().authenticate()  # then api.kernels_delete('owner/slug', no_confirm=True)
EOF

# GPU quota check
python -c "..."  # get_accelerator_quota_statistics via KaggleApi, see §4.3
```

---

## 8. Roadmap

1. ~~Finish 50k~~ — **DONE (2026-08-08):** 50,000 steps, 3.28B tokens, best val loss 4.5870.
2. **Post-training** (next): SFT on curated data, then preference tuning
   (DPO/KTO-style, following the Cursor post-training playbook) to lift accuracy/behavior.
   Base = final `resume.pt` from the checkpoint dataset.
3. Evals against the repo's evaluation harness on the way.

---

## 9. Research Track: 32K Hybrid-MoE-70M (v2)

Independent from the 68.7M dense run — a from-scratch, 32,768-token-context research
model (`research_hybrid/`), with its own design and report docs:

- `docs/research_report_32k.md` — the v2 literature review (2026 sweep) and verdicts.
- `docs/design.md` — v2 design: block-sparse attention (window 8192 + 128-token
  anchor, FlexAttention w/ chunked fallback), deepseek_moe + shared experts,
  MuonClip optimizer (QK-Clip per-head), 8K→32K curriculum, Mamba-2 ablation (off).

Status (2026-08-08): **training is LIVE on Kaggle** — kernel
`tomiokasan/research-v2-70m` (T4, mounts `research-v2-corpus` + `research-moe-code`).

### 9.1 Training pipeline (`research_hybrid/train.py` + `kaggle_pipeline_v2.py`)

- **Recipe** (design.md §7): MuonClip (Kimi K2 Algorithm 1) — 2D params Muon
  NS-5 / 1D AdamW, `lr 0.02 / lr_1d 0.01`, warmup 375, cosine, `wd 0.1`;
  QK-Clip τ=100 per-head (GQA min-γ over shared KV); fp16 autocast (fp32 weights,
  no GradScaler needed); EMA decay 0.999, evaluated for val; gradient
  checkpointing ON; curriculum 8K-causal → 32K-block_sparse at step 2435
  (4,870 steps × 65,536 tok/step ≈ 319M tokens ≈ 16 h on T4).
- **Boot:** STEP 0 assembles `research_hybrid/` into `/kaggle/working` from the
  flat files of the `research-moe-code` dataset; STEP 1 pretokenizes the corpus
  (reuses the v1 `tokens.bin` format, copies the tokenizer for generation);
  STEP 2 trains. Checkpoints → `tomiokasan/research-moe-checkpoints`
  (`resume.pt` + `latest_step.txt`, same upload/resume machinery as §5).

### 9.2 Verified fixes found while smoke-testing the pipeline

1. **MuonClip Newton–Schulz blowup** (critical): NS iterations are only stable
   for unit-norm inputs (`x·xᵀ@x` grows with `‖x‖`); the optimizer applied NS-5
   to the raw update and blew params up ~1e7× in one step (loss 20 → 1.9e10).
   Fixed by normalizing `update / (‖update‖ + 1e-8)` before NS-5 (K2's
   `G_Normalize`). `optim.py` + smoke regression.
2. **Tied-head logit scale**: `input_scale` was applied only at the input
   embedding; the tied LM head produced logits of RMS ~√d_model (init CE ~63 /
   ~300 at full scale). Baked `input_scale` into the embedding weight at init.
3. **QK-Clip NaN guard**: `gamma = clamp(tau/smax, max=1)` could go negative
   when all attention scores are negative → `sqrt` NaN. Now `(tau/smax).
   clamp_min(1e-8)` first.
4. **kaggle 2.2.4 dataset-create regression**: `_upload_file` never sets
   `UploadFile.description` (the dataset-side file name) → server rejects with
   "Dataset url's dataset slugs and hashlink are all null". `tools/
   push_code_dataset.py` monkeypatches it and pushes/updates `research-moe-code`
   (flat files; the kernel reassembles the package).
5. **Account rename**: the Kaggle account was renamed `roronoazoro3008` →
   `tomiokasan`; old-name refs 403 on new API endpoints. All refs (kernel,
   datasets, checkpoint store) now use `tomiokasan`.

Smoke validation: `.venv/bin/python -m research_hybrid.smoke_test` (9/9 PASS)
and `PP_SMOKE=1 .venv/bin/python -c "from research_hybrid.train import run; run()"`
(stable loss ~5.73→5.71 across the curriculum switch on synthetic data).

### 9.3 The CUDA OOM saga (kernels v4→v15) — how training finally went live

The first 11 v2 kernel pushes died in `train_step 0` with CUDA OOMs, each at a
different site; every fix surfaced the next bottleneck:

| Kernel | Crash site → Fix |
|---|---|
| v4 | `F.linear` in loss (512 MiB alloc) → still OOM: **sm75 T4 has no fused SDPA**; the `_flex_capable` (sm70+) gate was wrong — flash needs sm80+. Gate now `compute_capability[0] >= 8`, else chunked path |
| v5 | `torch.exp` 4.5 GiB — autocast blacklists exp (fp32); a full `(B,H,T,T)` fp32 exp is huge → **sub-blocked softmax**: pass 1 scans SUB=1024-wide score blocks for the exact per-row max `m`, pass 2 accumulates `exp*V` + `den`; peak ~900 MB/sub-block, numerically identical to dense fp32 |
| v7 | Same `F.linear` 12.68 GiB — inner per-chunk attention checkpoints **inside** the block checkpoint (nested non-reentrant) never freed memory → removed inner checkpoint |
| v8–v9 | OOM moved to `F.cross_entropy` — outer GC switched `use_reentrant=True` + `PP_MEM_DEBUG=1` diagnostics added |
| v10 | Diagnostics: only **3,136 MB live** after blocks but 13.2 GiB reserved → allocator fragmentation → `PYTORCH_ALLOC_CONF=expandable_segments:True` (module scope in pipeline + `train.run()`) |
| v11 | Still OOM at CE: each loss chunk retained its fp32 CE input/softmax temps in the autograd graph (~768 MB × 16 chunks ≈ 12 GB) → `loss_chunk` 1024→512, `empty_cache()` at loss start, **per-chunk loss checkpoint** (`_chunk_ce`, `use_reentrant=False`) |
| v14 | Forward finally flat (3,172 MB) — but OOM in `loss.backward()`: the reentrant block-checkpoint **recompute** rebuilt all 4 chunks × 8 sub-blocks of fp32 exp outputs (~18 GB) → **per-sub-block gradient checkpoints** inside `_attend` (pass 1 `_max_sub`, pass 2 `_exp_sub`) + outer GC back to `use_reentrant=False` |
| **v15** | **FIRST HEALTHY RUN** — step 25: `loss 23.23, acc 1.85%, tok/s 1436, gpu 2404 MB`; step 50: `loss 22.71`; memory flat ~3.8 GB live, no OOM through step 50+ |

Current live status: `tomiokasan/research-v2-70m` RUNNING, ~1,435 tok/s (~45 s/step).
At that rate a 12 h session covers ~950 steps — several runs with checkpoint
resume (`research-moe-checkpoints`) will be needed for the 4,870-step budget;
the first checkpoint lands at step 500.

---

*Last updated 2026-08-09 — **v2 MoE training LIVE on Kaggle** (loss 23.2 → 22.7, 1,435 tok/s, T4, no OOM since the v15 memory-budgeting fixes).**
