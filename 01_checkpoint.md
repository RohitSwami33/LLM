# Checkpoint 01 — Project State Summary

> **Date:** 2026-07-19  
> **Status:** MoE implementation complete, ready for training  
> **Hardware:** MacBook Air M4 (16 GB RAM), MPS backend

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Baseline Transformer Architecture](#3-baseline-transformer-architecture)
4. [MoE Architecture](#4-moe-architecture)
5. [Configuration Reference](#5-configuration-reference)
6. [Training System](#6-training-system)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Production Hardening](#8-production-hardening)
9. [Checkpoint Format](#9-checkpoint-format)
10. [Experiment Tracking](#10-experiment-tracking)
11. [How to Run](#11-how-to-run)
12. [Baseline Results](#12-baseline-results)
13. [Implementation Changelog](#13-implementation-changelog)

---

## 1. Project Overview

### Objective

Compare **Dense Transformer** vs **Sparse Mixture of Experts (MoE)** architectures under identical training conditions. The only independent variable is the model architecture.

### Key Design Principles

- **Configuration-only switching:** `architecture: transformer` vs `architecture: moe`
- **Scientific validity:** Identical tokenizer, dataset, optimizer, scheduler, token budget, evaluation intervals
- **Production-grade:** Health monitoring, auto-recovery, checkpointing, dashboard, reproducibility packages
- **MPS-first:** All settings tuned for MacBook Air M4 (16 GB), FP32, flash attention disabled

### Baseline Specs

| Metric | Value |
|--------|-------|
| Model | 68.7M params, 6L, 8H, 576d, 2304 FFN |
| Optimizer | Muon(lr=0.02) + AdamW(lr=1e-4) |
| Regularization | weight_decay=0.1, dropout=0.0, grad_clip=1.0 |
| Context | 1024 tokens |
| Token budget | 1B tokens (1 epoch of small mixture) |
| Tokenizer | SentencePiece, vocab=32768 |

---

## 2. Directory Structure

```
Sem5_project/
├── configs/                          # Dataset mixture configs
│   ├── tiny.yaml                     #   ~370K docs
│   ├── small.yaml                    #   ~3.7M docs (fineweb 2M + wikipedia 500K)
│   ├── medium.yaml                   #  ~37M docs
│   └── large.yaml                    # ~326M docs
│
├── datasets/                         # Built datasets
│   └── mixtures/small/               # corpus.jsonl + tokenizer_data/ + metadata.json
│
├── training/                         # Core training framework
│   ├── trainer.py                    # Main training loop
│   ├── ema.py                        # Exponential Moving Average
│   ├── monitor.py                    # Health monitoring daemon
│   ├── recovery.py                   # Auto-recovery system
│   ├── metadata.py                   # Experiment metadata collection
│   ├── dashboard.py                  # Live terminal dashboard
│   ├── artifacts.py                  # Artifact management
│   ├── analysis.py                   # Post-run analysis + leaderboard
│   ├── research_log.py               # EXPERIMENT_HISTORY.md management
│   │
│   ├── model/                        # Model architecture
│   │   ├── config.py                 # ModelConfig dataclass
│   │   ├── model.py                  # TransformerLM (supports transformer + moe)
│   │   ├── transformer.py            # TransformerBlock (dense)
│   │   ├── moe_transformer.py        # MoETransformerBlock (sparse)
│   │   ├── moe.py                    # MoE layer (router + experts)
│   │   ├── attention.py              # CausalSelfAttention + Flash
│   │   ├── mlp.py                    # SwiGLU / GELU MLP
│   │   ├── rmsnorm.py                # RMSNorm / LayerNorm
│   │   └── rope.py                   # Rotary Position Embeddings
│   │
│   ├── configs/                      # Training configs
│   │   ├── base.yaml                 # Base (CUDA, bf16, compile)
│   │   ├── pretrain_small.yaml       # Small mixture baseline (MPS, fp32)
│   │   ├── pretrain_moe_small.yaml   # Small mixture MoE
│   │   └── pretrain_tiny.yaml        # Tiny mixture quick test
│   │
│   ├── optimizer/                    # Optimizers
│   │   ├── builder.py                # Optimizer factory
│   │   └── muon.py                   # Muon + MuonWithAuxAdam
│   │
│   ├── data/                         # Dataset implementations
│   │   ├── dataset.py                # JsonlDataset, PackedDataset
│   │   └── collator.py               # DataCollator, PackedCollator
│   │
│   ├── tokenizer/                    # Tokenizer
│   │   ├── tokenizer.py              # SentencePiece / HuggingFace wrapper
│   │   └── small_tokenizer.model     # Trained tokenizer (32768 vocab)
│   │
│   ├── evaluation/                   # In-training evaluation
│   │   ├── evaluate.py               # val_loss, val_ppl, val_accuracy
│   │   └── generate.py               # Sample generation
│   │
│   ├── experiment/                   # Experiment management
│   │   ├── manager.py                # ExperimentManager
│   │   ├── reports.py                # Report generation
│   │   └── system_info.py            # System info collection
│   │
│   └── utils/
│       ├── checkpoint.py             # Checkpoint save/load/clean
│       └── logging.py                # TensorBoard/CSV/W&B
│
├── evaluation/                       # Post-training evaluation
│   ├── runner.py                     # EvalRunner orchestrator
│   ├── compare.py                    # Multi-checkpoint comparison
│   ├── generation.py                 # Greedy/top-k/top-p generation
│   ├── prompts.py                    # Standard evaluation prompts
│   ├── leaderboard.py                # CSV leaderboard
│   └── benchmarks/
│       ├── perplexity.py             # WikiText-2/103
│       ├── multiple_choice.py        # HellaSwag, ARC, BoolQ, PIQA, LAMBADA, Winogrande
│       └── efficiency.py             # Throughput, memory, latency, FLOPs
│
├── scripts/
│   ├── run_experiment.py             # Production experiment orchestrator
│   ├── run_small_pipeline.py         # End-to-end pipeline
│   ├── evaluate.py                   # CLI evaluation
│   └── compare.py                    # CLI comparison
│
└── experiments/                      # Experiment outputs
    ├── 2026-07-19_001/               # Training run
    ├── 2026-07-19_002/               # Training + full evaluation
    └── leaderboard.csv               # Experiment leaderboard
```

---

## 3. Baseline Transformer Architecture

### Model: TransformerLM

**File:** `training/model/model.py`

```
Token Embedding (32768 × 576)
  → Dropout (0.0)
  → [TransformerBlock × 6]
  → RMSNorm (576)
  → LM Head (576 × 32768, no bias)
```

### TransformerBlock

**File:** `training/model/transformer.py`

```
Input (576)
  → RMSNorm → CausalSelfAttention → Residual Add
  → RMSNorm → SwiGLU MLP → Residual Add
Output (576)
```

### CausalSelfAttention

**File:** `training/model/attention.py`

- Fused QKV projection: `Linear(576, 3×576)` = 3 matrices in one
- Output projection: `Linear(576, 576)`
- RoPE applied to Q and K
- 8 heads, head_dim = 72
- Flash Attention: disabled on MPS

### SwiGLU MLP

**File:** `training/model/mlp.py`

```
SwiGLU(x) = w_down(SiLU(w_gate(x)) × w_up(x))

w_gate: Linear(576, 2304)
w_up:   Linear(576, 2304)
w_down: Linear(2304, 576)
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Token embedding | 18.6M |
| 6 × TransformerBlock | 50.4M |
| Final RMSNorm | 576 |
| LM Head | 18.6M |
| **Total** | **68.7M** |
| **Trainable (excl. embedding)** | **50.1M** |

### ModelConfig

**File:** `training/model/config.py`

```python
@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 32000
    d_model: int = 576
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2304
    dropout: float = 0.0
    max_seq_len: int = 2048

    # Components
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    rope: bool = True
    rope_base: float = 10000.0
    flash_attention: bool = True
    bias: bool = False
    tie_weights: bool = False

    # Training
    gradient_checkpointing: bool = True

    # Architecture variant
    architecture: str = "transformer"  # "transformer" | "moe"

    # MoE configuration
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25
    moe_shared_expert: bool = False
    moe_load_balancing_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.001
    moe_router_temperature: float = 1.0
    moe_router_noise: float = 0.1

    # Future extensibility
    mla_latent_dim: int = 0
    mtp_heads: int = 1
    mhc: bool = False
```

---

## 4. MoE Architecture

### Design

Replaces each TransformerBlock's dense MLP with a Sparse Mixture of Experts. Attention remains identical.

```
Input (576)
  → RMSNorm → CausalSelfAttention → Residual Add
  → RMSNorm → MoELayer → Residual Add
Output (576) + moe_metrics dict
```

### MoELayer Components

**File:** `training/model/moe.py`

#### 1. TopKRouter

```python
gate = Linear(d_model, num_experts, bias=False)  # (576, 8)

# Forward:
logits = gate(x)                    # (B, T, 8)
logits = logits + noise(training)   # Gaussian noise during training
logits = logits / temperature       # Temperature scaling (1.0)
top_k_logits, top_k_indices = topk(logits, k=2)
weights = softmax(top_k_logits)     # Sparse weights (only top-k)
```

#### 2. ExpertGroup

8 × Expert networks, each a SwiGLU MLP:

```python
Expert = SwiGLU(d_model=576, d_ff=2304)
```

#### 3. Token Computation

```python
# For each token, compute weighted sum of top-k expert outputs:
output = sum(weight_i × Expert_i(token) for i in top_k)
```

#### 4. Capacity Constraints (Token Dropping)

```python
capacity = (B × T × top_k / num_experts) × capacity_factor
# capacity_factor = 1.25 → 25% buffer
# Tokens beyond capacity are dropped (first-come-first-served)
```

#### 5. Auxiliary Losses

**Load Balancing Loss:**
```python
f_i = fraction of tokens routed to expert i
P_i = average routing probability for expert i
L_balance = N × sum(f_i × P_i)
# Encourages equal expert utilization
```

**Router Z-Loss:**
```python
z_loss = mean(log(sum(weights))^2)
# Penalizes large router logits for stability
```

**Total Auxiliary Loss:**
```python
L_aux = 0.01 × L_balance + 0.001 × z_loss
# Added to main training loss
```

### MoE Metrics Tracked

| Metric | Description |
|--------|-------------|
| `expert_utilization` | Fraction of tokens per expert (should be ~1/num_experts) |
| `routing_entropy` | Higher = more uniform routing |
| `load_balance_loss` | Auxiliary loss encouraging equal utilization |
| `router_z_loss` | Penalizes large logits |
| `dropped_tokens_pct` | Percentage of tokens dropped due to capacity |
| `active_params` | Parameters active per token |
| `total_params` | All parameters (including inactive experts) |

### Parameter Comparison

| Metric | Transformer | MoE (8E, 2K) |
|--------|------------|---------------|
| Total params | 69.6M | 236.8M |
| Active params/token | 69.6M | 93.5M |
| Efficiency | 100% | 42.9% |
| FFN params | 1×SwiGLU | 8×SwiGLU + router |

---

## 5. Configuration Reference

### pretrain_small.yaml (Dense Transformer Baseline)

```yaml
model:
  vocab_size: 32768
  d_model: 576
  n_heads: 8
  n_layers: 6
  d_ff: 2304
  dropout: 0.0
  max_seq_len: 1024
  norm_type: "rmsnorm"
  activation: "swiglu"
  rope: true
  rope_base: 10000.0
  flash_attention: false
  gradient_checkpointing: true
  tie_weights: false
  bias: false

tokenizer:
  type: "sentencepiece"
  vocab_size: 32768
  model_path: "training/tokenizer/small_tokenizer.model"

dataset:
  path: "datasets/mixtures/small/corpus.jsonl"
  max_seq_len: 1024
  val_split: 0.02
  num_workers: 2
  pin_memory: false

training:
  optimizer: "muon"
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  momentum: 0.95
  grad_clip: 1.0
  scheduler: "cosine"
  warmup_steps: 500
  max_tokens: 1000000000       # 1B tokens
  max_steps: 31000
  batch_size: 4
  gradient_accumulation_steps: 8
  dtype: "fp32"
  compile: false
  use_ema: true
  ema_decay: 0.9999
  checkpoint_dir: "training/checkpoints"
  keep_last_n: 3
  save_every_tokens: 100000000  # 100M tokens
  eval_every_tokens: 50000000   # 50M tokens
  eval_steps: 100
  log_dir: "training/logs"
  experiment_dir: "experiments"
  tensorboard: true
  csv: true
  wandb: false
  log_every: 100
  early_stop_patience: 5
  early_stop_min_delta: 0.001
  nan_recovery: true
  nan_max_retries: 3
  nan_skip_batches: 10
  spike_detection: true
  spike_threshold: 2.0
  spike_window: 100
  spike_rollback: true
  seed: 42

evaluation:
  max_new_tokens: 200
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  num_samples: 5
  prompts:
    - "The history of artificial intelligence begins with"
    - "In quantum computing, the fundamental unit is"
    - "Python is a programming language that"
    - "The theory of general relativity describes"
    - "def fibonacci(n):"
```

### pretrain_moe_small.yaml (MoE — Only Changes)

```yaml
model:
  # ... all identical to baseline ...
  architecture: "moe"             # ← THE ONLY CHANGE

  moe_num_experts: 8
  moe_top_k: 2
  moe_capacity_factor: 1.25
  moe_shared_expert: false
  moe_load_balancing_weight: 0.01
  moe_router_z_loss_weight: 0.001
  moe_router_temperature: 1.0
  moe_router_noise: 0.1

# Everything else identical: tokenizer, dataset, training, evaluation
```

### Effective Batch Calculation

```
batch_size = 4
gradient_accumulation_steps = 8
effective_batch = 4 × 8 = 32 sequences
tokens_per_step = 32 × 1024 = 32,768 tokens
steps_for_1B = 1,000,000,000 / 32,768 ≈ 30,518 steps
```

---

## 6. Training System

### Training Loop (`training/trainer.py`)

**Initialization flow:**
1. Auto-detect device: MPS > CUDA > CPU
2. Set random seeds (torch, MPS, CUDA)
3. Create ExperimentManager (timestamped directory, save config/git/system info)
4. Build: tokenizer → model → optimizer → scheduler → scaler → EMA → dataloaders → logger
5. Optional: torch.compile (disabled on MPS)
6. Optional: resume from checkpoint

**Training step:**
```python
def train_step(batch):
    model.train()
    optimizer.zero_grad()
    
    logits, loss = model(input_ids, labels=labels)
    
    # Add MoE auxiliary losses if applicable
    if model.is_moe:
        loss = loss + model._moe_metrics["total_aux_loss"]
    
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    ema.update()
    scheduler.step()
    
    return {"loss", "lr", "grad_norm", "tokens_per_sec", "perplexity", "moe/*"}
```

**Stability features:**
- **NaN/Inf recovery:** Halve LR, skip batches, max retries
- **Loss spike detection:** Compare current loss to rolling mean (2× threshold), rollback to snapshot
- **Early stopping:** Monitor val_loss, patience=5, min_delta=0.001
- **Gradient clipping:** max_norm=1.0

### Optimizer: Muon

**File:** `training/optimizer/muon.py`

MuonWithAuxAdam:
- 2D params (linear weights): Muon with momentum + RMS normalization
- 1D params (biases, norms): AdamW

```python
# Muon update (2D params):
buf = momentum × buf + grad
update = grad + momentum × buf       # Nesterov
rms = sqrt(mean(update²))
update = update / rms                 # RMS normalize
p = p × (1 - lr × weight_decay) - lr × update
```

### Scheduler

Cosine annealing with linear warmup:

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max(min_lr / lr, 0.5 × (1 + cos(π × progress)))
```

---

## 7. Evaluation Framework

### In-Training Evaluation (`training/evaluation/`)

Runs during training every 50M tokens:
- **val_loss:** Cross-entropy (exclude padding/-100)
- **val_perplexity:** exp(loss), capped at 20
- **val_accuracy:** Per-token accuracy
- **Text generation:** 5 prompts × 200 tokens

### Post-Training Evaluation (`evaluation/`)

**Benchmarks** (auto-downloaded from HuggingFace):

| Benchmark | Metric | Type |
|-----------|--------|------|
| WikiText-2 | Perplexity | Perplexity |
| WikiText-103 | Perplexity | Perplexity |
| HellaSwag | Accuracy | Multiple Choice |
| ARC-Easy | Accuracy | Multiple Choice |
| ARC-Challenge | Accuracy | Multiple Choice |
| BoolQ | Accuracy | Multiple Choice |
| PIQA | Accuracy | Multiple Choice |
| LAMBADA | Accuracy | Multiple Choice |
| Winogrande | Accuracy | Multiple Choice |

**Efficiency Benchmark:**
- Throughput (tokens/sec)
- Inference latency (ms)
- Peak memory (GB)
- FLOPs estimate
- Model size (MB)

**Outputs:**
- `evaluation.json` — Machine-readable
- `evaluation.md` — Human-readable report
- `comparison.csv` — Spreadsheet analysis
- `generation_samples.txt` — Generated text
- `plots/radar_chart.png` — Performance radar
- `plots/efficiency.png` — Efficiency bars

---

## 8. Production Hardening

### Health Monitor (`training/monitor.py`)

Background daemon checking every 60s:
- Process alive status (build + training PIDs)
- MPS/CPU/system memory usage
- Disk free space
- Log file growth (stall detection: 5 min no growth)
- Checkpoint freshness (stale detection: 30 min)

**Thresholds:** CPU >95%, Memory >90%, MPS >14GB, Disk <5GB

### Recovery Manager (`training/recovery.py`)

- Process crash → auto-resume from latest valid checkpoint
- OOM → reduce batch size and retry
- Keyboard interrupt → graceful shutdown with checkpoint
- Corrupted checkpoint → fallback to previous
- Missing tokenizer/dataset → detection and reporting

### Metadata Collection (`training/metadata.py`)

Records: Git commit/branch/dirty/diff, Python/torch/datasets versions, OS/chip/RAM/CPU, MPS/CUDA status, config (sanitized), timestamps, seeds.

### Dashboard (`training/dashboard.py`)

- **TrainingDashboard (curses):** Full-screen terminal UI with ASCII loss curve, system stats, ETA
- **SimpleDashboard:** Single-line status updates to stdout

### Artifact Manager (`training/artifacts.py`)

- Checkpoint copying to experiment directory
- Log file organization
- Reproducibility package generation
- Artifact integrity verification

### Analysis (`training/analysis.py`)

- Training log parsing
- Comparison reports between experiments
- Leaderboard CSV management
- Matplotlib plots (loss curves, throughput, gradient norms)

### Research Log (`training/research_log.py`)

Manages `EXPERIMENT_HISTORY.md`: experiment entries, running observations, TODO tracking.

---

## 9. Checkpoint Format

### Standard Checkpoint

```python
{
    "step": int,                    # Global training step
    "epoch": int,                   # Current epoch
    "model": OrderedDict,           # Model state_dict
    "optimizer": dict,              # Optimizer state_dict
    "scheduler": dict,              # LR scheduler state_dict
    "timestamp": float,             # time.time() when saved
    "total_tokens": int,            # Total tokens processed
    "rng": {
        "torch": Tensor,            # torch RNG state
        "cuda": List[Tensor],       # Per-device CUDA states
    }
}
```

### MoE Checkpoint (Additional Key)

```python
{
    # ... standard keys ...
    "moe": {
        "config": {
            "num_experts": 8,
            "top_k": 2,
            "capacity_factor": 1.25,
            "shared_expert": false,
            "load_balancing_weight": 0.01,
            "router_z_loss_weight": 0.001,
        },
        "router_weights": {
            "blocks_0_moe_router": {"gate_weight": Tensor},
            "blocks_1_moe_router": {"gate_weight": Tensor},
            # ... one per MoE layer
        },
        "expert_utilization_history": [],
    }
}
```

### Special Checkpoints

| Name | Purpose |
|------|---------|
| `best_model` | Lowest validation loss |
| `early_stop_final` | Final checkpoint when early stopping triggers |
| `step_NNNNN` | Periodic checkpoints |
| `_spike_snapshot/snapshot.pt` | Rollback snapshot for loss spike recovery |

### Atomic Save

Write to `.tmp` file, then `os.rename()` to final path. Prevents corruption on crash.

---

## 10. Experiment Tracking

### ExperimentManager

Creates timestamped directory: `experiments/YYYY-MM-DD_NNN/`

Saves: config.yaml, git_commit.txt, system_info.json, model_summary.txt, tokenizer_info.json, dataset_stats.json

### Experiment Directory Structure

```
experiments/YYYY-MM-DD_NNN/
├── config.yaml
├── git_commit.txt
├── system_info.json
├── model_summary.txt
├── tokenizer_info.json
├── dataset_stats.json
├── train.log
├── checkpoints/
│   ├── step_NNNNN/
│   ├── best_model
│   ├── early_stop_final
│   └── _spike_snapshot/
├── logs/
│   ├── metrics.csv
│   └── tensorboard/
├── samples/
│   └── step_NNNNN.txt
├── evaluation/
│   ├── evaluation.json
│   ├── evaluation.md
│   ├── comparison.csv
│   ├── generation_samples.txt
│   └── plots/
├── report.json
├── report.md
└── README.md
```

### Leaderboard

CSV file: `experiments/leaderboard.csv`

Columns: name, architecture, total_params, active_params, optimizer, dataset, val_ppl, hellaswag_acc, wiki2_ppl, throughput_tok_s, peak_memory_gb, ...

---

## 11. How to Run

### Build Dataset Mixture

```bash
python build_mixture.py configs/small.yaml
```

### Train Tokenizer

```bash
python training/scripts/train_tokenizer.py \
    --input datasets/mixtures/small/corpus.jsonl \
    --vocab-size 32768 \
    --output training/tokenizer/small_tokenizer.model
```

### Run Full Pipeline

```bash
python scripts/run_small_pipeline.py
```

Phases: build_monitor → build_verify → tokenizer → validation → training → evaluation → reports

### Run Training Directly

```bash
# Dense Transformer (baseline)
python -m training.trainer training/configs/pretrain_small.yaml

# MoE Transformer
python -m training.trainer training/configs/pretrain_moe_small.yaml
```

### Production Orchestrator

```bash
# Dense Transformer with dashboard
python scripts/run_experiment.py \
    --config training/configs/pretrain_small.yaml \
    --dashboard

# MoE Transformer with dashboard
python scripts/run_experiment.py \
    --config training/configs/pretrain_moe_small.yaml \
    --dashboard
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint experiments/2026-07-19_002/checkpoints/best_model \
    --output-dir results/
```

### Compare Checkpoints

```bash
python scripts/compare.py \
    --checkpoints ckpt1 ckpt2 \
    --output-dir results/compare
```

---

## 12. Baseline Results

### Training (2000 steps on tiny mixture)

| Metric | Value |
|--------|-------|
| Best val loss | 9.95 |
| Val perplexity | 20,912 |
| Final accuracy | 3.9% |
| Throughput | ~2,300 tok/s |
| Training time | ~1 hour |

### Evaluation

| Benchmark | Value |
|-----------|-------|
| WikiText-2 PPL | 47,277 |
| HellaSwag | 23.4% |
| BoolQ | 37.5% |
| Winogrande | 50.5% |
| Throughput | 14.8K tok/s |

### Observations

- Baseline (12L, 4H, 64D): val_ppl=20,912, HellaSwag=23.4%
- All heads at layer 0 only: 67.1% speedup but +24% perplexity
- 6 dense heads + 2 early-exit: 29.8% speedup but +11% perplexity

---

## 13. Implementation Changelog

### Phase 1: Dataset Pipeline
- [x] HuggingFace datasets download (The Stack v2 gated, Dolma incompatible)
- [x] Text preprocessing pipeline (HTML removal, Unicode normalization, deduplication)
- [x] Mixture building (fineweb 2M + wikipedia 500K = 2.5M docs)
- [x] Pipeline resumability (state tracking, auto-resume)

### Phase 2: Training Framework
- [x] TransformerLM model (68.7M params)
- [x] MPS backend adaptation (FP32, no flash attention, no torch.compile)
- [x] Muon optimizer with AdamW auxiliary
- [x] Token-budget training (max_tokens, eval_every_tokens, save_every_tokens)
- [x] Checkpoint save/load with atomic writes
- [x] NaN/Inf recovery, loss spike detection/rollback
- [x] Early stopping with patience
- [x] ExperimentManager (directory, config, git, system info)

### Phase 3: Evaluation Framework
- [x] In-training evaluation (val_loss, val_ppl, val_accuracy)
- [x] Post-training benchmarks (WikiText-2, HellaSwag, BoolQ, Winogrande)
- [x] Efficiency benchmark (throughput, memory, latency, FLOPs)
- [x] Text generation (greedy, top-k, top-p)
- [x] Reports (JSON, Markdown, CSV, plots)
- [x] Leaderboard and comparison tools

### Phase 4: Production Hardening
- [x] Health monitoring daemon
- [x] Auto-recovery system
- [x] Metadata collection (git, system, config)
- [x] Live terminal dashboard
- [x] Artifact management
- [x] Post-run analysis
- [x] Research log management
- [x] Production experiment orchestrator

### Phase 5: MoE Architecture
- [x] TopKRouter (sparse routing, noisy gating, temperature scaling)
- [x] ExpertGroup (8 × SwiGLU MLPs)
- [x] MoELayer (routing + computation + capacity + auxiliary losses)
- [x] MoETransformerBlock (drop-in replacement for dense block)
- [x] Architecture switching in TransformerLM
- [x] MoE metrics tracking in trainer
- [x] MoE state in checkpoints (router weights, config)
- [x] MoE-specific evaluation metrics (active params, efficiency)
- [x] pretrain_moe_small.yaml (identical to baseline except architecture)
- [x] Backward compatibility (transformer checkpoints unaffected)

---

## End of Checkpoint 01
