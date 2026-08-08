# Pretraining Progress — 68.7M TransformerLM

Status snapshot and full history of the **research-v2 pretraining project** (training
a 68.7M-parameter decoder-only TransformerLM on free Kaggle GPU kernels, managed
entirely from the CLI).

> Companion docs: `docs/01_info.md` (deep postmortem of the early kernel crashes),
> `docs/KAGGLE_SETUP.md` (Kaggle CLI/account setup), `training/kaggle/kaggle_pipeline.py`
> (the actual training kernel).

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
| `tomiokasan` | `~/.kaggle/kaggle.json` | 30 h/week, exhausted ~Aug 11 | old runs (v7–v9) |
| `roronoazoro3008` | `~/.kaggle2/kaggle.json` | **30 h/week** (verified) | current runs (v4+) |

Kaggle datasets (per account, same slugs):

| Dataset | Contents | Role |
|---|---|---|
| `research-v2-corpus` | `corpus.jsonl` (1.28 GB) + `tokenizer/` | kernel input mount |
| `research-v2-checkpoints` | `resume.pt` (1.1 GB) + `latest_step.txt` | checkpoint store |

Kernel metadata: `kernel-metadata.json` (points at `roronoazoro3008/research-v2-pretrain`,
T4, internet on). Kernel owner is auto-detected from the mounted corpus path
(`KAGGLE_USERNAME` is NOT set inside kernels).

---

## 7. Ops Cheat-Sheet

```bash
# push the training kernel (account matters!)
KAGGLE_CONFIG_DIR=~/.kaggle2 .venv/bin/kaggle kernels push -p .
KAGGLE_CONFIG_DIR=~/.kaggle2 .venv/bin/kaggle kernels status roronoazoro3008/research-v2-pretrain

# stream live logs (SSE endpoint; plain `kaggle kernels logs` returns empty for running sessions)
U=roronoazoro3008; K=<key from ~/.kaggle2/kaggle.json>
curl -s -u "$U:$K" "https://www.kaggle.com/api/v1/kernels/logs/stream/$U/research-v2-pretrain"

# check the saved checkpoints
KAGGLE_CONFIG_DIR=~/.kaggle2 .venv/bin/kaggle datasets files roronoazoro3008/research-v2-checkpoints

# stop a bad run (CLI has no `cancel`; delete the kernel kills the session)
python - <<'EOF'
import os; os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle2')
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

Status (2026-08-08): **code complete and verified** — all 9 CPU smoke tests pass
(attention patterns vs dense reference, MoBA, Mamba-2 SSD vs naive recurrence,
MuonClip/AdamW, KV-cache decode, EMA); `audit.py` asserts the active budget:
**total 139.7M, active 70.68M**, ≈200 MFLOPs/token forward / ≈600 train,
≈2.8 GB peak on T4 at B=2 × 32K. Next: training pipeline port to
block-sparse + MuonClip, then the Kaggle/Colab run.

---

*Last updated 2026-08-08 — **pretraining complete (50k/50k, val 4.5870)**; v2 research track green.**
