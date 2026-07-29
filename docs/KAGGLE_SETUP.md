# Kaggle GPU Training Setup

Train your 68.7M parameter Transformer on Kaggle's free GPUs (T4/P100/L4).

## Quick Start (One Command After Setup)

```bash
# First time: deploy + submit
python pipeline.py --kaggle --submit

# Monitor training
python pipeline.py --kaggle --monitor

# Download results
python pipeline.py --kaggle --download
```

---

## 1. Kaggle Account Setup

1. Go to [kaggle.com](https://www.kaggle.com/account/login)
2. Create a free account
3. Go to **Settings** → scroll to **API** section
4. Click **Create New Token**
5. This downloads `kaggle.json`

## 2. API Authentication

### Option A: Automatic (Recommended)

Run any Kaggle script and it will prompt for credentials:
```bash
python pipeline.py --kaggle
```

### Option B: Manual Setup

```bash
# Move kaggle.json to ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Option C: Environment Variables

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

## 3. Install Kaggle CLI

```bash
pip install kaggle
```

Verify:
```bash
kaggle datasets list --csv | head
```

## 4. Upload Corpus Dataset

The training corpus (`datasets/research_v2/`) must be on Kaggle.

### First Time Upload

```bash
python deploy_kaggle.py --upload-dataset
```

This uploads `datasets/research_v2/` as a Kaggle dataset. Takes ~5-10 minutes for 1.3GB.

### Verify Upload

```bash
kaggle datasets list -s "research-v2" --csv
```

## 5. Start Training

### Single Command

```bash
python pipeline.py --kaggle --submit
```

### Step by Step

```bash
# Package project
python deploy_kaggle.py --package-only

# Submit to Kaggle
python train_kaggle.py --submit

# Or with GPU selection
python train_kaggle.py --submit --gpu P100
```

### Available Commands

| Command | Description |
|---------|-------------|
| `python pipeline.py --kaggle --submit` | Deploy + submit |
| `python train_kaggle.py --submit` | Submit training |
| `python train_kaggle.py --status` | Check status |
| `python train_kaggle.py --cancel` | Cancel run |
| `python train_kaggle.py --wait` | Submit + wait |

## 6. Monitor Training

### Real-Time Monitoring

```bash
python pipeline.py --kaggle --monitor
```

Shows: step, loss, eval loss, tok/s, GPU memory, elapsed time.

### Check Status Only

```bash
python monitor_kaggle.py --logs
```

### View Kernel Logs

```bash
python monitor_kaggle.py --logs --kernel research-v2-pretrain
```

## 7. Resume Training

If a Kaggle session ends before training completes:

### Auto-Resume (Default)

Checkpoints are saved every 500 steps to `/kaggle/working/checkpoints/`.
The next kernel run auto-detects and resumes from the latest checkpoint.

### Manual Resume

```bash
# Download checkpoints
python download_results.py --checkpoints

# The latest checkpoint is at training/checkpoints/kaggle/
# It will be included in the next upload automatically

# Resubmit
python train_kaggle.py --submit
```

## 8. Download Checkpoints

### All Results

```bash
python download_results.py
```

### Checkpoints Only

```bash
python download_results.py --checkpoints
```

### List Available Files

```bash
python download_results.py --list
```

### Download + Resume Instructions

```bash
python download_results.py --resume
```

## 9. GPU Options

| GPU | VRAM | Price | Speed | Notes |
|-----|------|-------|-------|-------|
| T4 | 16GB | Free | 1x | Default. Free tier. |
| P100 | 16GB | $5/hr | 1.5x | Faster. |
| L4 | 24GB | $10/hr | 2x | Best. More VRAM. |

Configure in `configs/kaggle.yaml`:
```yaml
kaggle:
  gpu_type: "T4"  # T4, P100, or L4
  max_runtime_hours: 12
```

Or pass on command line:
```bash
python train_kaggle.py --submit --gpu P100
```

## 10. Troubleshooting

### "Kaggle CLI not installed"
```bash
pip install kaggle
```

### "kaggle.json not found"
```bash
# Option 1: Run any pipeline script (auto-creates from env vars)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
python pipeline.py --kaggle

# Option 2: Manual
mkdir -p ~/.kaggle && chmod 600 ~/.kaggle
# Place your kaggle.json there
```

### "API credentials invalid"
Regenerate at https://www.kaggle.com/settings → Create New Token

### "Dataset not found"
Upload the corpus first:
```bash
python deploy_kaggle.py --upload-dataset
```

### "Kernel failed to start"
1. Check kernel logs: `python monitor_kaggle.py --logs`
2. Ensure dataset is uploaded and attached
3. Try a different GPU type

### "Out of memory (OOM)"
Reduce batch size in `configs/kaggle.yaml`:
```yaml
training:
  batch_size: 8   # Reduce from 16
```

### "Session timeout (12h)"
Kaggle limits GPU kernels to 12 hours. The training saves checkpoints every 500 steps, so you can resume across sessions.

### "torch.compile error"
Disable compilation in `configs/kaggle.yaml`:
```yaml
training:
  compile: false
```

---

## Configuration Reference

All Kaggle settings in `configs/kaggle.yaml`:

```yaml
kaggle:
  dataset_slug: "research-v2-corpus"    # Kaggle dataset
  kernel_name: "research-v2-pretrain"    # Kernel name
  gpu_type: "T4"                         # GPU type
  max_runtime_hours: 12                  # Time limit
  enable_internet: true                  # Internet access
  auto_resume: true                      # Auto-resume from checkpoint

training:
  batch_size: 16                         # Per-GPU batch size
  gradient_accumulation_steps: 2         # Effective: batch_size * accum
  max_steps: 50000                       # Training steps
  save_every: 500                        # Checkpoint interval
  eval_every: 500                        # Evaluation interval
```

---

## File Structure on Kaggle

```
/kaggle/working/                         # Your working directory
├── training/                            # Project code
│   ├── model/
│   ├── optimizer/
│   ├── data/
│   ├── tokenizer/
│   └── kaggle/
│       └── train.py                     # Entry point
├── configs/
│   └── kaggle.yaml
├── checkpoints/                         # Saved checkpoints
│   ├── step_500.pt
│   ├── step_1000.pt
│   ├── best.pt
│   └── final.pt
├── experiments/
│   └── kaggle_report.json
└── kernel-metadata.json

/kaggle/input/                           # Mounted datasets
└── research-v2-corpus/
    ├── corpus.jsonl
    └── tokenizer/
        ├── tokenizer.model
        └── tokenizer_config.json
```

## Architecture

```
Local Machine                    Kaggle GPU
─────────────────                ──────────────────
pipeline.py --kaggle             training/kaggle/train.py
    │                                │
    ├── deploy_kaggle.py             ├── Shared codebase
    ├── train_kaggle.py              │   (model, optimizer,
    ├── monitor_kaggle.py            │    data, tokenizer)
    └── download_results.py          │
                                     ├── Kaggle paths
                                     │   (/kaggle/input/*)
                                     │
                                     └── Checkpoints
                                         → downloadable
```

Both local and Kaggle pipelines share the same codebase:
- `training/model/` — Transformer architecture
- `training/optimizer/` — Muon, AdamW, etc.
- `training/data/` — Dataset, DataLoader, collators
- `training/tokenizer/` — SentencePiece tokenizer
- `training/evaluation/` — Loss, perplexity, generation
