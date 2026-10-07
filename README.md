# LLM Dataset Pipeline

A production-ready, configurable dataset pipeline for LLM pretraining and
post-training research. Supports **Transformer**, **Mamba**,
**Transformer+Mamba Hybrid**, **HRM**, and **TRM** architectures.

## Qwen3.5-4B Post-Training Benchmark (Full Training)

Full-training results for the completed ORPO-v4 adapter. All variants use the same NF4 configuration, deterministic decoding, task prompts and sampled items. The 7-task table is a screening run with 10 examples per task, not publication-quality scores.

| Benchmark | Base | Old adapter | Science-v2 | Replay-v3 | ORPO-v4 Final |
|---|---:|---:|---:|---:|---:|
| MMLU-Pro Computer Science | 0.00% | 10.00% | 0.00% | 20.00% | **30.00%** |
| TruthfulQA MC2 | **57.38%** | 49.24% | 49.31% | 48.50% | 43.86% |
| GSM8K | 0.00% | 40.00% | 0.00% | 40.00% | **60.00%** |
| HellaSwag | **70.00%** | 40.00% | 60.00% | 40.00% | 50.00% |
| BBH Logical Deduction (5 objects) | **30.00%** | 20.00% | 10.00% | **30.00%** | 20.00% |
| IFEval | 22.22% | 22.22% | 22.22% | **27.78%** | 22.22% |
| ARC Challenge | 30.00% | **40.00%** | 30.00% | **40.00%** | **40.00%** |
| **Selected-metric average** | 29.94% | 31.64% | 24.51% | 35.18% | **38.01% (+8.07 pp vs Base)** |

Final ORPO-v4 improved the selected-metric average by +8.07 pp over Base and +2.83 pp over Replay-v3. Strongest gains were GSM8K (0% → 60%) and MMLU-Pro CS (0% → 30%). TruthfulQA regressed (57.38% → 43.86%), so truthfulness remains unresolved.

See the [full comparison](qwen35_post_training/FINAL_POST_TRAINING_COMPARISON.md) and [raw final output](qwen35_post_training/benchmarks/v4-orpo-limit10.json).

### Paired 50-example check: GSM8K + IFEval (Base vs Final ORPO-v4)

Fifty real examples per task, seed 42, identical items for both models. Original Qwen3.5-4B vs local final ORPO-v4, NF4 4-bit, greedy decoding, 256-token cap. GSM8K 5-shot, IFEval 0-shot. Budget-constrained screening, not published full scores.

| Task / metric | Base Qwen | Final ORPO-v4 | Change |
|---|---:|---:|---:|
| GSM8K flexible accuracy | 6.00% | **66.00%** | **+60.00 pp** |
| GSM8K strict format accuracy | 0.00% | **60.00%** | **+60.00 pp** |
| IFEval prompt-level strict | 14.00% | 14.00% | 0.00 pp |
| IFEval instruction-level strict | 30.67% | 30.67% | 0.00 pp |
| IFEval prompt-level loose | 14.00% | **18.00%** | **+4.00 pp** |
| IFEval instruction-level loose | 30.67% | **33.33%** | **+2.67 pp** |

GSM8K shows a large paired gain; IFEval is flat on strict, small gain on loose. n=50 still has substantial sampling uncertainty.

See the [paired report](qwen35_post_training/quick_remaining_benchmarks/QUICK_REMAINING_COMPARISON.md) and raw outputs in `qwen35_post_training/quick_remaining_benchmarks/` (`base-gsm8k-n50.json`, `adapter-gsm8k-n50.json`, `base-ifeval-n50.json`, `adapter-ifeval-n50.json`).

## Qwen3.5-4B Post-Training: What We Did for Training

Goal: reduce hallucinations and improve factual / science QA on `Qwen/Qwen3.5-4B` without losing math and reasoning, on a single 12 GB GPU.

- Base: `Qwen/Qwen3.5-4B`, NF4 4-bit (`bnb_4bit_quant_type=nf4`, double-quant, `bfloat16` compute, `sdpa`), `trust_remote_code=True`
- GPU: NVIDIA RTX 4070 SUPER 12 GB, CUDA, `allow_tf32=True`, batch 1
- Method: QLoRA SFT → replay continuation → reference-free ORPO preference training, LoRA only
- Code: `qwen35_post_training/training/` (`build_science_sft.py`, `build_preference_data.py`, `train_4bit_qlora.py`, `train_orpo_4bit.py`)
- Eval: `qwen35_post_training/benchmarks/run_real_benchmarks.py` with `lm-evaluation-harness`, 7 tasks, deterministic greedy decoding, seed 42

### Stage 0 — Starting point: old fact-check adapter

- Public adapter: `RohitSwami33/qwen35-fc-adapter`, continued in later stages via `--init-adapter`
- Released cache: 25,037 sequences, 6.58M total tokens, 2.89M supervised tokens, `max_len=3072`, seed 1337
- Mixture: HotpotQA 9,000 + GSM8K 5,500 + OpenR1-Math 2,500 + SmolTalk 2,000 + synthetic search decisions 4,500 + synthetic hallucinations 1,840 + synthetic self-corrections 500; FEVER / AVeriTeC / ToolACE 0 rows in build
- Audit: `qwen35_post_training/dataset_audit/old-dataset-stats.json`

### Stage 1 — Science-v2 fresh QLoRA (did not help)

- Data: `training/build_science_sft.py` from train splits only — SciQ + PubMedQA `pqa_labeled` + OpenBookQA `main`, grounded in support passages, exact-text deduped, shuffled seed 3407
- Size: 16,430 usable examples, `max_length=768`, 1 epoch, 2,054 steps
- QLoRA: `r=16`, `alpha=32`, dropout 0.05, targets `q/k/v/o/gate/up/down_proj`, `paged_adamw_8bit`, cosine, warmup 50, `per_device=1`, `grad_accum=8`, `weight_decay=0.01`, `max_grad_norm=1.0`, `bf16/tf32`, gradient checkpointing
- Result: loss 0.03883, 1,528s, 10.75 samples/s — but screening regressed (avg 24.51% vs Base 29.94%)
- Lesson: SFT loss rewards imitation, not truthfulness; narrow data overwrote general behavior
- Metrics: `qwen35_post_training/model_science_v2/training_metrics.json`

### Stage 2 — Replay-v3 continuation (kept math, small gain)

- Init: old `fc_adapter` (not science-v2), low LR `5e-6`
- Data: 4,000 grounded science + 8,000 old-cache replay (7,999 used, 11,999 total), `max_length=768`, token-based prompt masking (`labels=-100` on prompt)
- Result: loss 0.08716, 2,652s, 4.52 samples/s, 1 epoch — screening avg 35.18% (+5.24 pp vs Base)
- Metrics: `qwen35_post_training/model_v3/training_metrics.json`
- Reproduce:
```powershell
python qwen35_post_training/training/train_4bit_qlora.py `
  --data path/to/science_sft.jsonl --limit 4000 `
  --replay-cache path/to/tokenized_cache --replay-limit 8000 `
  --init-adapter path/to/fc_adapter --learning-rate 5e-6 `
  --max-length 768 --output outputs/qwen35-fc-science-replay-v3
```

### Stage 3 — ORPO-v4 preference training (final adapter)

- Data: `training/build_preference_data.py` → 10,000 grounded chosen/rejected pairs, seed 7301: HaluEval-QA 5,200 + SciQ 2,500 + OpenBookQA 1,500 + PubMedQA 800. Chosen = evidence-grounded answer, rejected = hallucinated / overconfident answer
- Init: Replay-v3 `final_adapter`, ORPO `beta=0.05`, LR `2e-6`, cosine, warmup 40, batch 1x8, `max_completion_length=256`
- VRAM trick on 12 GB: fresh run `max_length=768`, resume with `max_length=384` so optimizer state fits; smoke-tested with 16 examples / 1 step first
- LoRA: `r=32`, `alpha=64`, dropout 0.05, targets `q/k/v/o/out/down/up/gate_proj`
- Result: loss 0.68678, 7,923s (~2.2h), 1.26 samples/s, 1 epoch, 10k examples — final screening avg 38.01% (+8.07 pp vs Base)
- Metrics: `qwen35_post_training/model_v4_orpo/training_metrics.json`, adapter: `qwen35_post_training/model_v4_orpo/`
- Pipeline: `work/run_final_preference_pipeline.py` trains → benchmarks limit-10 → writes `FINAL_POST_TRAINING_COMPARISON.md` → copies adapter + report into repo → pushes
- Reproduce:
```powershell
python qwen35_post_training/training/train_orpo_4bit.py `
  --data path/to/preference_10k.jsonl `
  --init-adapter path/to/replay-v3/final_adapter `
  --output outputs/qwen35-factual-orpo-v4 `
  --max-length 768 --learning-rate 2e-6 --beta 0.05
```

### How evaluation was run

- Runner: `qwen35_post_training/benchmarks/run_real_benchmarks.py` (`--variant base|adapter`, `--limit`, `--max-gen-toks 256`, `--tasks`, seed 42)
- 7-task screening: `mmlu_pro_computer_science`, `truthfulqa_mc2`, `gsm8k`, `hellaswag`, `bbh_cot_zeroshot_logical_deduction_five_objects`, `ifeval`, `arc_challenge`, `limit=10`, batch 1, `apply_chat_template`, greedy, logs VRAM + wall time
- Paired n50: `work/run_quick_remaining_comparison.py` — same 50 real GSM8K/IFEval items (seed 42) for both models, hash-checked, per-response cache, 256-token cap
- Full suite (resumable, HellaSwag excluded): `work/run_full_v4_suite.py` (`--limit 0`, manifest + skip-completed)
- Stack: PyTorch CUDA, Transformers, PEFT 0.21.0, bitsandbytes, Datasets, Accelerate, TRL ORPO, lm-evaluation-harness

Limitations: n=10 / n=50 are noisy screening runs, 256-token cap truncates long reasoning, TruthfulQA still regressed — do not claim broad improvement without larger held-out eval.

## Directory Structure

```
.
├── download_datasets.py        # Download / stream datasets from Hugging Face
├── preprocess.py                # Configurable text cleaning pipeline
├── build_mixture.py             # Build unified training mixtures from YAML configs
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── configs/                     # Mixture configuration files
│   ├── tiny.yaml                #   ~370k docs – quick prototyping (MacBook)
│   ├── small.yaml               #   ~3.7M docs – single-GPU experiments
│   ├── medium.yaml              #  ~37M   docs – multi-GPU training
│   └── large.yaml               # ~326M  docs – production training
│
└── datasets/                    # Created by the scripts
    ├── fineweb/                 #   General web pretraining corpus
    │   ├── data/                #     Arrow / JSONL data files
    │   ├── metadata.json        #     Reproducibility info
    │   ├── stats.json           #     Dataset statistics
    │   ├── README.md            #     Dataset documentation
    │   └── download_log.txt     #     Download log
    ├── fineweb_edu/
    ├── dolma/
    ├── the_stack_v2/
    ├── wikipedia/
    └── mixtures/                # Built by build_mixture.py
        ├── tiny/
        ├── small/
        ├── medium/
        └── large/
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Optional: for language filtering
pip install langdetect

# 2. Download / stream datasets
python download_datasets.py --mode stream

# 3. Build a mixture
python build_mixture.py configs/tiny.yaml
```

---

## 1. Downloading Datasets (`download_datasets.py`)

### Available Datasets

| Name | Hugging Face Path | Content | Est. Tokens | Est. Disk |
|---|---|---|---|---|
| `fineweb` | `HuggingFaceFW/fineweb` | General web text | ~10B | ~32 GB |
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` | Educational web text | ~10B | ~32 GB |
| `dolma` | `allenai/dolma` | Diverse open corpus | ~10B | ~22 GB |
| `the_stack_v2` | `bigcode/the-stack-v2` | Code (permissive license) | ~? | ~2-5 GB |
| `wikipedia` | `wikimedia/wikipedia` | English Wikipedia | ~2.5B | ~12 GB |

### Auto-Detection

Dataset versions are automatically detected:

- **Wikipedia**: the newest English snapshot (e.g. `20231101.en`)
- **FineWeb / FineWeb-Edu**: prefers `sample-10BT`, falls back to latest `CC-MAIN-YYYY-NN`
- **Dolma**: latest stable version (`v1_7`)

The resolved version is saved in `metadata.json`.

### Streaming vs Local Mode

```bash
# Streaming (default) – process without full download
python download_datasets.py --mode stream

# Local mode – download and cache to disk
python download_datasets.py --mode local
```

| Feature | Stream | Local |
|---|---|---|
| Disk space | Minimal (processed subset) | Full dataset |
| Speed | Slower first pass | Fast reload |
| Resume | Re-processes | Marker-based |
| Use case | Exploration, small mixtures | Production training |

### Commands

```bash
# List all datasets
python download_datasets.py --list

# Dry run (show what would be downloaded)
python download_datasets.py --dry-run

# Download specific datasets
python download_datasets.py --datasets fineweb wikipedia

# Force re-download
python download_datasets.py --force

# Custom output directory
python download_datasets.py --base-dir /mnt/data/datasets
```

### Per-Dataset Output

Each dataset directory contains:

| File | Description |
|---|---|
| `data/` | Arrow shards (local) or JSONL files (stream) |
| `metadata.json` | Full reproducibility info (versions, config, stats) |
| `stats.json` | Document count, tokens, chars, language distribution |
| `README.md` | Dataset description and usage notes |
| `download_log.txt` | Detailed download log |
| `.download_complete` | Marker for resume support |

### The Stack v2

The Stack v2 is **gated** – you must:

1. Accept terms at https://huggingface.co/datasets/bigcode/the-stack-v2
2. Run `huggingface-cli login`

By default, only permissively licensed files are downloaded (`license_type == "permissive"`)
for 8 popular languages: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++.

---

## 2. Preprocessing (`preprocess.py`)

A configurable, stateful cleaning pipeline used by both `download_datasets.py`
and `build_mixture.py`.

### Cleaning Steps

| Step | Config Key | Default | Description |
|---|---|---|---|
| Empty removal | `remove_empty` | `true` | Drop blank documents |
| HTML cleaning | `remove_html` | `true` | Strip tags, entities, URLs |
| Unicode norm. | `normalize_unicode` | `true` | NFKC normalization |
| Whitespace | `normalize_whitespace` | `true` | Collapse runs → single space |
| Deduplication | `deduplicate` | `true` | SHA-256 exact dedup |
| Length filter | `min_length` / `max_length` | `0` / `10M` | Drop/truncate out-of-range |
| Language filter | `filter_language` | `null` | Optional `langdetect` (ISO 639-1) |

### Standalone Usage

```bash
python preprocess.py input.jsonl --output cleaned.jsonl --config my_config.yaml
```

### Stats Collected

Each run produces a stats snapshot:
- Input/output document counts
- Removed counts per filter
- Retention rate, dedup ratio
- Total characters and estimated tokens
- Language distribution

---

## 3. Building Mixtures (`build_mixture.py`)

### Configuration Files

The `configs/` directory contains four mixture profiles:

#### `configs/tiny.yaml` (~370k documents, ~1-2 GB)

```
FineWeb:   200,000 docs
Wikipedia:  50,000 docs
TheStack:   20,000 docs  (Python, C++)
Dolma:     100,000 docs
```

Ideal for: Quick iteration, debugging, MacBook M4, Kaggle CPUs.

#### `configs/small.yaml` (~3.7M documents, ~15-20 GB)

Ideal for: Single GPU experiments, hyperparameter sweeps.

#### `configs/medium.yaml` (~37M documents, ~150-200 GB)

Ideal for: Multi-GPU training, architectural comparisons.

#### `configs/large.yaml` (~326M documents, ~1-2 TB)

Ideal for: Production training runs, final model training.

### Building

```bash
python build_mixture.py configs/tiny.yaml
python build_mixture.py configs/small.yaml --seed 123
python build_mixture.py configs/medium.yaml --dry-run
```

### What Happens

1. For each dataset, the latest version is auto-detected
2. Documents are loaded (streaming or local)
3. Preprocessing is applied (cleaning, filtering, dedup)
4. Documents are sampled (reservoir sampling for streaming)
5. All documents are concatenated and **deterministically shuffled** (fixed seed)
6. Output is written as sharded JSONL or Arrow
7. Tokenizer-ready `.txt` files are optionally generated
8. Full reproducibility metadata is saved

### Mixture Output

```
datasets/mixtures/tiny/
├── corpus.jsonl               # Unified corpus (single file)
├── shards/                    # Sharded version (10 shards)
│   ├── shard-00000.jsonl
│   ├── shard-00001.jsonl
│   └── ...
├── tokenizer_data/            # Clean UTF-8 text for tokenizer training
│   ├── text-00000.txt
│   ├── text-00001.txt
│   └── ...
├── metadata.json              # Full reproducibility info
└── mixture_config.yaml        # Copy of the YAML config
```

---

## 4. Reproducibility

Every output directory contains `metadata.json` with:

- Timestamp and random seed
- Dataset versions (resolved configs)
- Preprocessing configuration
- Per-dataset statistics
- Software versions (Python, datasets, PyYAML)
- Full config snapshot

The same YAML config + seed always produces the same corpus
(deterministic sampling + global shuffle).

---

## 5. Streaming Architecture

```
                         ┌─────────────┐
                         │  HuggingFace │
                         │   Hub/Datasets│
                         └──────┬──────┘
                                │ streaming=True
                                ▼
 ┌──────────────────────────────────────────────┐
 │           load_dataset(...)                   │
 │  (IterableDataset – no local storage)         │
 └──────────────────┬───────────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────────────┐
 │        PreprocessingPipeline                  │
 │  • Remove empty   • Clean HTML                │
 │  • Unicode NFKC   • Normalize whitespace      │
 │  • Deduplicate    • Filter by length          │
 │  • Language filter (optional)                 │
 └──────────────────┬───────────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────────────┐
 │         Reservoir Sampling                    │
 │  (select N documents, O(k) memory)            │
 └──────────────────┬───────────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────────────┐
 │     Deterministic Global Shuffle (seed=X)     │
 └──────────────────┬───────────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────────────┐
 │         Save: JSONL / Arrow / .txt           │
 └──────────────────────────────────────────────┘
```

---

## 6. Tokenizer Preparation

To train a tokenizer (SentencePiece or HuggingFace Tokenizers):

```bash
# Build a mixture with tokenizer data enabled
python build_mixture.py configs/tiny.yaml

# The tokenizer_data/ directory contains clean .txt files
ls datasets/mixtures/tiny/tokenizer_data/

# Train a SentencePiece tokenizer
spm_train \
  --input=datasets/mixtures/tiny/tokenizer_data/text-*.txt \
  --model_prefix=tokenizer \
  --vocab_size=32000 \
  --character_coverage=1.0 \
  --model_type=bpe

# Or train a HuggingFace tokenizer
python -c "
from tokenizers import Tokenizer, models, trainers
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=32000)
tokenizer.train([
    'datasets/mixtures/tiny/tokenizer_data/text-00000.txt',
], trainer)
tokenizer.save('tokenizer.json')
"
```

---

## 7. Disk Space Estimates

| Dataset | Stream Mode | Local Mode |
|---|---|---|
| `fineweb` (sample-10BT) | ~1-2 GB (200K docs) | ~32 GB |
| `fineweb_edu` (sample-10BT) | ~1-2 GB | ~32 GB |
| `dolma` (v1_6-sample) | ~1-2 GB (100K docs) | ~22 GB |
| `the_stack_v2` (subset) | ~2-5 GB (50K/lang) | ~2-5 GB |
| `wikipedia` (20231101.en) | ~300 MB (50K docs) | ~12 GB |

| Mixture | Stream (estimated) | Local (estimated) |
|---|---|---|
| `tiny` | ~2-3 GB | ~70 GB |
| `small` | ~15-20 GB | ~70 GB |
| `medium` | ~150-200 GB | ~70 GB (cached) |
| `large` | ~1-2 TB | ~70 GB (cached) |

> **Note**: In stream mode, only the sampled documents are saved.
> In local mode, the full datasets are cached and then subsets are sampled.

---

## 8. Architecture Support

This pipeline produces clean, tokenizer-ready text suitable for training:

| Architecture | Compatibility | Notes |
|---|---|---|
| **Transformer** | ✅ | Standard causal LM format |
| **Mamba** | ✅ | Same text format, no attention mask needed |
| **Mamba+Transformer Hybrid** | ✅ | Interleaved architectures use standard text |
| **HRM** (Hierarchical Reasoning Model) | ✅ | Needs document-separator tokens; use `<|endoftext|>` |
| **TRM** (Tiny Recursive Model) | ✅ | Same format; add recursion markers if needed |

---

## 9. Custom Configurations

Create a new YAML file and run `build_mixture.py`:

```yaml
# configs/my_mixture.yaml
mode: stream
seed: 12345

preprocessing:
  remove_empty: true
  remove_html: true
  normalize_unicode: true
  normalize_whitespace: true
  deduplicate: true
  min_length: 100
  max_length: 50000
  filter_language: en             # English only

datasets:
  FineWeb:
    documents: 500000
  Wikipedia:
    documents: 100000
  TheStack:
    languages:
      - Python
      - Rust
    documents: 50000
  Dolma:
    documents: 250000

output:
  format: jsonl
  path: datasets/mixtures/my_mixture
  save_tokenizer_data: true
```

```bash
python build_mixture.py configs/my_mixture.yaml
```
