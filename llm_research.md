# LLM Research Project Context

> **Purpose:** Single source of truth for Rohit's small-LLM research project. This
> file replaces ephemeral chat memory — read it at the start of every session.
> Last updated: 2026-08-20.

---

## 1. Goal

Build an **efficient small LLM** that is **accurate on mathematics, general
knowledge, science, and tech** — competitive with SOTA on *those domains* (not
general language). Trained on free Kaggle compute (2× TPU per account, multiple
accounts).

Key constraint discovered empirically: a **68.7M generalist model failed** (PPL
98.2) because capacity was too small for a diverse corpus. The strategy pivot:
**distillation** (generate high-quality domain data with big teachers) +
**specialization** (train on focused domains) + **right-sized capacity**.

---

## 2. Hardware & Compute

| Resource | Spec | Notes |
|----------|------|-------|
| **Kaggle TPU** | 2× TPU per account | Free, multiple accounts used |
| **Kaggle GPU** | T4 (16GB, sm75) / P100 | Past runs; weekly 30h quota per verified account |
| **Local Mac** | M4 (16GB RAM, MPS) | Smoke tests, small runs |
| **Local GPU** | RTX 5070 Super 12GB | Fast bf16 (~110 TFLOPS), for 68M-300M comfortable, 1B tight, 2B needs quant/offload |

### Kaggle accounts (multi-account strategy — CRITICAL for continuous training)
- **`tomiokasan`** (renamed from `roronoazoro3008`, 2026-08) — creds at `~/.kaggle/kaggle.json`, 30h/week GPU quota, verified
- **Second account** + friends' accounts — each gives more weekly GPU quota
- **Quota model**: account-level, new accounts start 6h/week until phone + Persona identity verification
- **TPU note**: 2× TPU per account (v2-8 or similar) — potentially 2-4× faster than T4 for this workload
- **Checkpoint cadence: every 500 steps** (or ~every hour) — upload `resume.pt` + `latest_step.txt` to the shared checkpoint dataset so any account can resume

### Checkpoint/Resume across accounts (how it works)
- Shared Kaggle dataset `research-v2-checkpoints` / `research-moe-checkpoints` holds `resume.pt` + `latest_step.txt`
- Kernel boots → downloads latest checkpoint → resumes from highest step
- Upload every 500 steps in background worker (`_upload_worker`), drains queue at session end
- Any account can pick up where another left off → **continuous training across quota windows**

---

## 3. Architecture

### 3.1 Dense baseline (68.7M) — `training/`
- Vocab 32k, d_model 576, 6 layers, 8 heads, d_ff 2304 → **68.7M params**
- RMSNorm, SwiGLU, RoPE (pure-real rotate-half), no bias, no weight tying
- Context 2048, gradient checkpointing, `F.scaled_dot_product_attention`
- Optimizer: AdamW lr 3e-4 → min 3e-5, wd 0.1, betas (0.9, 0.95) — **Muon dropped** (NaN on embeddings)
- FP16 autocast + GradScaler (loss inside autocast, FP32 softmax)
- Effective batch 32 (8 × 4 accum), SentencePiece 32k BPE
- Training: `training/kaggle/kaggle_pipeline.py`

### 3.2 Hybrid-MoE-70M (v2, 32K context) — `research_hybrid/`
- **70M active params** (audit-enforced), 32,768-token context
- GQA: 9 q-heads / 3 kv-heads, head_dim 64, RoPE θ=1e6 (Mistral recipe)
- **Block-sparse attention**: window 8192 + 128-token anchor block (StreamingLLM sink), FlexAttention with chunked eager fallback for sm75/sm60
- **DeepSeekMoE FFN**: 1 shared expert (SwiGLU 576→1152) + 12 routed (576→832), top-4, capacity 1.25, balance loss + z-loss, jitter
- Optional off-by-default: Mamba-2 SSD layer, mHC, MLA, MoBA, YaRN extension
- Optimizer: **MuonClip** (Kimi K2 Algorithm 1) — 2D params Muon NS-5 / 1D AdamW, lr 0.02/0.01, QK-Clip τ=100 per-head
- Curriculum: ~50% steps @ 8K dense → ~50% @ 32K block-sparse (DeepSeek-V3 staged training)
- Budget: 4870 steps × 65,536 tok/step ≈ 319M tokens ≈ 16h on T4
- Training: `research_hybrid/train.py` + `training/kaggle/kaggle_pipeline_v2.py`

### Key implementation fixes (v2, hard-won)
1. **MuonClip NS blowup**: normalize update by ‖update‖ before Newton-Schulz
2. **Tied-head logit scale**: `input_scale` baked into embedding at init
3. **QK-Clip NaN guard**: `clamp_min(1e-8)` before sqrt
4. **CUDA OOM saga (v4→v15)**: expandable_segments, sub-blocked softmax (SUB=1024), per-chunk loss checkpoint, per-sub-block gradient checkpoints, `PYTORCH_ALLOC_CONF=expandable_segments:True` at module scope
5. **kaggle SDK 2.2.4 dataset-create regression**: monkeypatch `UploadFile.description` (`tools/push_code_dataset.py`)
6. **Lazy Kaggle auth**: never authenticate at import (boot-time DNS breaks sessions)

---

## 4. Data

### 4.0 Dataset strategy for the new run (2026-08-21)
**Next action per user: download the right dataset to compete on benchmarks.** The repo
has the full toolchain ready:
- `download_datasets.py` — HF streaming/local downloader (FineWeb, Wikipedia, CodeSearchNet, OpenStax, arXiv, FineMath, The Stack v2)
- `build_mixture.py` — builds a unified training mixture from YAML configs
- Configs in `configs/` (tiny/small/medium/large/research/research_v2/kaggle)

Target mix for **math + general knowledge + science + tech accuracy** (the goal):
| Dataset | Why | Size slice |
|---------|-----|-----------|
| **FineWeb-Edu** | High-quality web, edu-filtered | 100-500K docs |
| **FineMath** | Math reasoning | 50-100K docs |
| **Wikipedia** | General knowledge facts | 100-200K docs |
| **OpenStax** | Science textbooks | 50K docs |
| **arXiv (math/cs/physics)** | Research-grade STEM | 50K docs |
| **CodeSearchNet** | Code (tech) | 50K docs |
| **distilled_corpus** (ours) | Curated quality anchor | all 1,252 docs |
| **OpenWebMath** | Math web | optional |

This is the "right dataset" to actually compete on MMLU / GSM8K / MATH / HellaSwag
etc. for a small model. Exact recipe TBD when we run `build_mixture.py`.

### 4.1 Research corpus (old, for 68.7M + MoE runs)
- `corpus.jsonl` ~1.28 GB, ~319M tokens, research/coding mixture (FineWeb 250K + Wikipedia 70K + CodeSearchNet 40K + OpenStax/arXiv 25K + FineMath 15K ≈ 400K docs)
- Mounted on Kaggle as `research-v2-corpus`; pretokenized in-kernel to `tokens.bin` (~12 min)
- Tokenizer: SentencePiece 32k BPE

### 4.2 Distilled corpus v2 (new, the quality anchor) — `distilled_corpus/`
- **1,252 docs, ~3.07M tokens, 12.3M chars**, 100% English, 100% verified
- **Domains**: technology (380), general_knowledge (176), mathematics (180), engineering (120), biology (111), physics (108), chemistry (92), earth_space (85)
- **Difficulty**: 1→6 levels; **task types**: educational_explanation, encyclopedia_article, worked_example, proof_or_derivation, qa_problem_solving, comparison, long_synthesis, code_with_explanation, reasoning_example
- **Context lengths**: 256 → 32,768 tokens (1 doc @ 32K, 35 @ 16K)
- **Teachers**: nvidia/nemotron-3-super-120b-a12b (872 docs, 2.11M tok), nvidia/nemotron-3-ultra-550b-a55b (279, 645K), z-ai/glm-5.2 (72, 260K), poolside/laguna-s-2.1:free (23+6)
- Files: `final/final_corpus.jsonl` (training-ready), `verified/`, `deduplicated/`, `rejected/`, `metadata/`, `README.md`
- On GitHub: `RohitSwami33/LLM` (commit e62b707)

---

## 5. Training History & Results

### 68.7M dense run (COMPLETE)
- 50,000 steps × 65,536 tok/step = **3.28B tokens ≈ 10 epochs** over 319M-token corpus
- ~34.5h GPU across 2 accounts, best val loss **4.5870, PPL ≈ 98.2**
- **Result: FAILED as generalist** — 3-4 tokens plausible then repetition collapse (`isisisis...`, `nparray nparrayarray...`)
- **Diagnosis: underfitting, not overfitting** — 68M params too small to compress diverse corpus; capacity absorbed by token statistics
- Checkpoint: `roronoazoro3008/research-v2-checkpoints` → `resume.pt` (step 50,000) → now `tomiokasan/`

### Hybrid-MoE-70M v2 (training was LIVE as of 2026-08-09)
- Kernel `tomiokasan/research-v2-70m`, T4, loss 23.2 → 22.7 at step 50, ~1435 tok/s, no OOM after v15 fixes
- Checkpoint store: `tomiokasan/research-moe-checkpoints`

---

## 6. Pipeline (distillation → training)

```
generate.py (plan curriculum + prompts)
  → nvidia_runner.py / user_runner.py (teacher generation)
  → verify.py (LLM verifier + heuristics + code exec)
  → dedup.py (exact/near/semantic)
  → finalize.py (final_corpus.jsonl + stats)
  → train (Kaggle kernel or local 5070)
```

### Key operational lessons
- **Crash-safe runners**: shards must write incrementally (per-item), not at end — a kill loses content otherwise
- **verify_state over-claim bug**: state must reflect actual output files, or docs get marked done without being written
- **Reasoning models waste budget**: dots-3-note-preview burned all tokens on `reasoning` → use `reasoning: {effort: none}`; nemotron-super-120b needs `enable_thinking: false`
- Free API tiers (OpenRouter `:free`) throttle hard; NVIDIA NIM is reliable and fast (~183 tok/s for nemotron-3-super-120b)

---

## 6.5 DeepSeek-V4 Architecture — the implementation target (NEW, 2026-08-21)

We are implementing the attention/compression techniques from **DeepSeek-V4**:
*Towards Highly Efficient Million-Token Context Intelligence* (arXiv:2606.19348).
(These are the "mHC, MLA, DSA/DCA/CSA" the user refers to — DPSIC was a garbled
acronym for DeepSeek. MLA = DeepSeek-V2 multi-head latent attention; mHC = arXiv
2512.24880, already in `research_hybrid/mhc.py`.)

### What DeepSeek-V4 uses (all implementable at our scale)
1. **mHC — Manifold-Constrained Hyper-Connections** (arXiv:2512.24880). ✅ ALREADY IMPLEMENTED in `research_hybrid/mhc.py` (off by default). Constrains the residual mixing matrix to the Birkhoff polytope (doubly stochastic via Sinkhorn-Knopp) → spectral norm ≤ 1, stable identity mapping.
2. **CSA — Compressed Sparse Attention** (V4 §2.3.1, Eq. 9-19). Compress every **m** KV entries into one via learned weighted compression (dual Ca/Cb streams + overlapping indexer), then **DeepSeek Sparse Attention (DSA)** = lightning indexer with ReLU-gated top-k block selection. MQA shared KV + grouped output projection (GQA). + sliding-window local branch + attention sink + RoPE on last 64 dims + RMSNorm on Q/KV.
3. **HCA — Heavily Compressed Attention** (V4 §2.3.2, Eq. 20-26). Same as CSA but much larger compression rate m'≫m, no sparse attention, no overlapping compress. MQA + grouped output projection.
4. **MLA — Multi-head Latent Attention** (DeepSeek-V2, arXiv:2405.04434). ✅ ALREADY IMPLEMENTED in `research_hybrid/mla.py` (off by default). Low-rank latent KV compression.
5. **Muon optimizer** ✅ already in `research_hybrid/optim.py` (MuonClip, Algorithm 1 in V4).

### Implementation status in repo
| Technique | File | Status |
|-----------|------|--------|
| mHC | `research_hybrid/mhc.py` | ✅ Implemented, default OFF (off by design at 6 layers; V4 shows it helps at 3B+, but paper's open impl lets us ablate) |
| MLA | `research_hybrid/mla.py` | ✅ Implemented, default OFF |
| Muon | `research_hybrid/optim.py` | ✅ Alive in live training |
| block-sparse attn (proto-CSA) | `research_hybrid/attention.py` | ✅ Implemented (window+anchor+sink, chunked fallback) |
| **CSA/HCA** (new) | **TO IMPLEMENT** | ❌ Not yet — next work item |

### CSA/HCA implementation plan (faithful, no invented math)
Build new modules in `research_hybrid/`:
- `csa.py`: compress_kv (dual Ca/Cb + Z, overlapping softmax-weighted blend, Eq.11-12), lightning indexer (low-rank q^I via W_DQ/W_IUQ, ReLU-gated index scores Eq.13-16, top-k), core MQA attention (Eq.17-19), grouped output projection.
- `hca.py`: heavier single-stream compression (Eq.20-23), no sparse selection, MQA + grouped out.
- Both share: sliding-window local branch, attention sink (Eq.27), RoPE-on-last-64 + de-RoPE of outputs, Q/KV RMSNorm.
- Wire into `attention.py` / `transformer.py` behind a config flag (`cfg.attention.pattern: "csa"/"hca"` + params: m, indexer heads, top_k, d_c, window, sink).
- Open-source reference: `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`.

### Why CSA/HCA at our small scale
Original 32K design already had block-sparse attention; CSA/HCA generalize it with
learnable **compression + learned sparse selection** instead of a fixed window mask.
This is the path to real long-context (100K-1M) efficiency with sub-linear KV cache —
the core DSA/DCA/CSA value the user wants. Even at 70M-300M it should beat fixed block-sparse
on long-context recall (indexer learns *where* to look).

---

## 7. Strategy for the Next Model (the "fixed footsteps")

**What we're building now:** an efficient small LLM accurate on **math, general knowledge, science, tech**.

### Step 1 — Domain specialization (prove the approach)
- Pick ONE domain from distilled corpus (chemistry, math, etc.)
- 68M model, that domain's slice only, 3-5 epochs
- Success metric: **PPL < 20** on that domain's val set (vs 98.2 before)
- This proves capacity + specialization works before scaling

### Step 2 — Right-size capacity
- **300M** on full distilled corpus (3M × 3-5 epochs) → expect PPL < 25-30
- Or **1B** generalist (needs public slice + days)

### Step 3 — Epoch strategy (correct rules)
- Pretraining on big public data: **1 epoch** (Llama/GPT standard)
- Distilled corpus as mix: **1-2 epochs**
- Post-training/curriculum on distilled: **2-4 epochs**
- **Epochs don't create info** — 10 epochs on 3M tokens ≠ 30M unique tokens. More unique tokens > more epochs.

### Step 4 — Training budget on RTX 5070 Super 12GB
| Model | Throughput | 1 epoch (1B tokens) |
|-------|-----------|---------------------|
| 68M | 40-60k tok/s | 5-7h |
| 300M | 15-25k tok/s | 11-18h |
| 1B | 6-9k tok/s | 31-46h |
| 2B | 3-5k tok/s | 55-93h (needs quant/offload for 12GB) |

---

## 8. Evaluation

- `evaluation/eval_68m_final.py` — loads checkpoint with exact training arch, samples generations
- Metrics: val loss, PPL, accuracy, sample generations (temp 0.8, top-k 50, top-p 0.95, 200 tokens)
- Evaluation prompts: `def fibonacci(n):`, `The quick brown fox`, `import numpy as np`, `Explain quantum computing`, `What is machine learning?`

### 8.1 Target benchmarks (to compete with SOTA on math/science/gk/tech)
| Benchmark | Domain | Notes |
|-----------|--------|-------|
| **MMLU / MMLU-Pro** | General knowledge | 57 tasks, 5-shot |
| **GSM8K** | Grade-school math | arithmetic reasoning |
| **MATH** | Competition math | hardest |
| **HellaSwag / ARC** | Commonsense/science | |
| **MBPP / HumanEval** | Code (tech) | if including code |
| **Lambada / Pile** | Language modeling | perplexity baseline |

Small-model targets (realistic for 70M-300M with good distillation):
- English PPL < 20-30 (vs 98.2 baseline)
- GSM8K exact-match > 5-10% (small models struggle; 300M+ with math tuning does better)
- MMLU > 25-30% (random-ish = 25%, so this shows real learning)

These are modest but beat a 68M generalist (which scored ~0 language coherence).

---

## 9. Key Files

| Path | Purpose |
|------|---------|
| `distilled_corpus/` | The distilled dataset (1,252 docs) |
| `distill_pipeline/` | Distillation: generate → verify → dedup → finalize |
| `research_hybrid/` | 70M MoE 32K model + training |
| `training/` | Dense 68.7M stack (model, tokenizer, data, trainer) |
| `training/kaggle/` | Kaggle kernels + checkpoint machinery |
| `configs/` | Mixture + training configs |
| `docs/design.md` | v2 32K design doc |
| `docs/research_report*.md` | Literature reviews + verdicts |
| `README_PROGRESS.md` | Training progress log |
| `01_checkpoint.md` | Early project state |

---

## 10. Credentials & API Keys (for automation)

- **NVIDIA NIM**: `nvapi-...` (in ~/.nvidia_api_key or env) — nemotron models, fast
- **OpenRouter**: `sk-or-...` — dots-3-note-preview:free (reasoning must be disabled)
- **Kaggle**: `~/.kaggle/kaggle.json` (tomiokasan)
- **GitHub**: gh authed as RohitSwami33
- ⚠️ Never commit keys to git

---

## 11. Roadmap (current, 2026-08-21)

1. **Download dataset** — run `download_datasets.py` / `build_mixture.py` for the math+science+gk+tech mix (see §4.0) — next action
2. **Implement CSA/HCA** — new `research_hybrid/csa.py` + `hca.py` per V4 §2.3 (equations captured in §6.5)
3. **Enable mHC + MLA in config** — flip flags, ablate at 70M scale
4. **Train on Kaggle multi-account** — checkpoint every 500 steps, resume across accounts/TPUs
5. **Evaluate on MMLU/GSM8K/MATH** — vs the 98.2-PPL baseline
