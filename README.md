# LLM Dataset Pipeline

A production-ready, configurable dataset pipeline for LLM pretraining and
post-training research. Supports **Transformer**, **Mamba**,
**Transformer+Mamba Hybrid**, **HRM**, and **TRM** architectures.

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
