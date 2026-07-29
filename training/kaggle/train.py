#!/usr/bin/env python3
"""Kaggle-optimized training script for Research Corpus v2.

This script runs inside a Kaggle kernel. It uses the shared codebase
(training.model, training.optimizer, training.data, training.tokenizer)
with Kaggle-specific path handling, mixed precision, and checkpoint recovery.

Entry point for Kaggle kernel submission.
"""
import os
import sys
import time
import math
import json
import yaml
from pathlib import Path

# ── Kaggle path setup ─────────────────────────────────────────────────────────
# When running on Kaggle, the project is extracted to /kaggle/working/
# The dataset is mounted at /kaggle/input/<dataset-slug>/

def detect_kaggle() -> bool:
    """Detect if running inside a Kaggle kernel."""
    return os.path.exists("/kaggle/working") or os.path.exists("/kaggle/input")

KAGGLE = detect_kaggle()

if KAGGLE:
    # Find project root (contains training/ directory)
    WORKING_DIR = Path("/kaggle/working")
    # The project zip is extracted here, so training/ is at /kaggle/working/training/
    PROJECT_ROOT = WORKING_DIR
    # Add project root to Python path so we can import training.*
    sys.path.insert(0, str(PROJECT_ROOT))

    # Find the dataset
    INPUT_DIRS = list(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
    CORPUS_DIR = INPUT_DIRS[0] if INPUT_DIRS else None
else:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    CORPUS_DIR = PROJECT_ROOT / "datasets" / "research_v2"

# ── Imports (from shared codebase) ────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.model.config import ModelConfig
from training.model.model import TransformerLM
from training.optimizer.builder import build_optimizer
from training.tokenizer.tokenizer import build_tokenizer
from training.data.dataset import JsonlDataset, PackedDataset, build_dataloader
from training.data.collator import DataCollator, PackedCollator
from training.ema import EMA
from training.utils.checkpoint import save_checkpoint, load_checkpoint
from training.utils.logging import LoggerManager
from training.evaluation.evaluate import evaluate
from training.evaluation.generate import generate_samples


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load training config, adjusting paths for Kaggle."""
    # Try kaggle.yaml first, fall back to pretrain_research_v2.yaml
    config_candidates = [
        PROJECT_ROOT / "configs" / "kaggle.yaml",
        PROJECT_ROOT / "training" / "configs" / "pretrain_research_v2.yaml",
    ]

    config = None
    for path in config_candidates:
        if path.exists():
            with open(path) as f:
                config = yaml.safe_load(f)
            print(f"  Config: {path}")
            break

    if config is None:
        raise FileNotFoundError("No config found")

    # Adjust paths for Kaggle
    if KAGGLE and CORPUS_DIR:
        config["dataset"]["path"] = str(CORPUS_DIR / "corpus.jsonl")
        config["tokenizer"]["model_path"] = str(CORPUS_DIR / "tokenizer" / "tokenizer.model")
    else:
        # Local relative paths
        ds_path = Path(config["dataset"]["path"])
        if not ds_path.exists():
            # Try relative to project root
            alt = PROJECT_ROOT / ds_path
            if alt.exists():
                config["dataset"]["path"] = str(alt)

        tok_path = Path(config["tokenizer"]["model_path"])
        if not tok_path.exists():
            alt = PROJECT_ROOT / tok_path
            if alt.exists():
                config["tokenizer"]["model_path"] = str(alt)

    # Kaggle-specific overrides
    if KAGGLE:
        config["training"]["checkpoint_dir"] = "/kaggle/working/checkpoints"
        config["training"]["experiment_dir"] = "/kaggle/working/experiments"
        config["training"]["log_every"] = 25
        config["training"]["save_every"] = 500

    return config


# ── Auto-resume from checkpoint ───────────────────────────────────────────────

def find_resume_checkpoint(config: dict) -> str:
    """Find the latest checkpoint for auto-resume.

    Checks:
    1. /kaggle/working/checkpoints/ (current session)
    2. Local checkpoint directory
    """
    ckpt_dir = Path(config["training"].get("checkpoint_dir", "checkpoints"))

    if not ckpt_dir.exists():
        return None

    # Find latest step_*.pt or final.pt
    ckpt_files = list(ckpt_dir.glob("step_*.pt")) + list(ckpt_dir.glob("final.pt"))
    if not ckpt_files:
        return None

    # Sort by step number or modification time
    def get_step(f):
        name = f.stem
        if name.startswith("step_"):
            try:
                return int(name.split("_")[1])
            except ValueError:
                return 0
        return float("inf") if name == "final" else 0

    ckpt_files.sort(key=get_step, reverse=True)
    return str(ckpt_files[0])


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    """Kaggle-optimized training loop."""
    print("=" * 60)
    print("LLM PRETRAINING — KAGGLE GPU")
    print("=" * 60)
    print(f"  Kaggle mode: {KAGGLE}")
    print(f"  Project root: {PROJECT_ROOT}")
    if CORPUS_DIR:
        print(f"  Corpus dir: {CORPUS_DIR}")

    # Load config
    config = load_config()
    training_config = config.get("training", {})

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  Warning: No GPU available, using CPU")

    # Seed
    seed = training_config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Build tokenizer
    print("\n  Building tokenizer...")
    tok_config = config.get("tokenizer", {})
    tokenizer = build_tokenizer(tok_config, data_path=config.get("dataset", {}).get("path"))
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Build model
    print("  Building model...")
    model_config = ModelConfig.from_dict(config.get("model", {}))
    model_config.vocab_size = tokenizer.vocab_size
    model = TransformerLM(model_config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params / 1e6:.1f}M params on {device}")

    # torch.compile (works well on CUDA)
    if training_config.get("compile", True) and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile failed: {e}")

    # Build optimizer
    optimizer = build_optimizer(model, training_config)

    # Build scheduler
    lr = training_config.get("learning_rate", 3e-4)
    min_lr = training_config.get("min_lr", 3e-5)
    warmup_steps = training_config.get("warmup_steps", 500)
    max_steps = training_config.get("max_steps", 50000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = None
    if training_config.get("use_ema", True):
        ema = EMA(model, decay=training_config.get("ema_decay", 0.9999))
        print("  EMA enabled")

    # Gradient scaler for mixed precision
    use_amp = training_config.get("dtype", "bf16") in ("bf16", "fp16")
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    amp_dtype = torch.bfloat16 if training_config.get("dtype") == "bf16" else torch.float16

    # Build dataset and dataloader
    print("\n  Loading dataset...")
    dataset_path = config["dataset"]["path"]
    if not Path(dataset_path).exists():
        print(f"  Error: Dataset not found at {dataset_path}")
        print("  Ensure the corpus dataset is attached to this kernel")
        sys.exit(1)

    dataset = JsonlDataset(
        data_path=dataset_path,
        max_seq_len=config["dataset"].get("max_seq_len", 2048),
    )

    pack_factor = config["dataset"].get("pack_factor", 4)
    packed = PackedDataset(dataset, max_seq_len=config["dataset"].get("max_seq_len", 2048), pack_factor=pack_factor)

    # Split train/val
    train_split = config["dataset"].get("train_split", 0.95)
    n_train = int(len(packed) * train_split)
    n_val = len(packed) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        packed, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    collator = PackedCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=config["dataset"].get("max_seq_len", 2048),
    )

    batch_size = training_config.get("batch_size", 16)
    num_workers = config["dataset"].get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collator, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
        collate_fn=collator,
    )

    print(f"  Train: {n_train:,} examples | Val: {n_val:,} examples")
    print(f"  Batch size: {batch_size}")

    # Setup logging
    log_dir = Path(training_config.get("experiment_dir", "experiments"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = LoggerManager(log_dir=str(log_dir), tensorboard=True, csv=True)

    # Auto-resume
    global_step = 0
    total_tokens = 0
    best_val_loss = float("inf")

    resume_path = find_resume_checkpoint(config)
    if resume_path and training_config.get("auto_resume", True):
        print(f"\n  Resuming from: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler)
        global_step = ckpt.get("step", 0)
        total_tokens = ckpt.get("total_tokens", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        print(f"  Resumed at step {global_step}, {total_tokens / 1e6:.1f}M tokens")

    # Grad accumulation
    grad_accum = training_config.get("gradient_accumulation_steps", 2)
    grad_clip = training_config.get("grad_clip", 1.0)
    log_every = training_config.get("log_every", 25)
    save_every = training_config.get("save_every", 500)
    eval_every = training_config.get("eval_every", 500)

    # Checkpoint dir
    ckpt_dir = Path(training_config.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"  TRAINING")
    print(f"  Max steps: {max_steps}")
    print(f"  Grad accum: {grad_accum} | Effective batch: {batch_size * grad_accum}")
    print(f"{'=' * 60}\n")

    model.train()
    train_start = time.time()
    step_loss = 0.0
    step_count = 0

    while global_step < max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward with mixed precision
            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp and device.type == "cuda",
            ):
                logits, loss = model(input_ids=input_ids, labels=labels)
                loss = loss / grad_accum

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss += loss.item() * grad_accum
            step_count += 1

            # Optimizer step
            if (batch_idx + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                if ema:
                    ema.update()

                global_step += 1
                total_tokens += input_ids.numel()

                # Log
                if global_step % log_every == 0:
                    avg_loss = step_loss / step_count
                    lr_now = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - train_start
                    tok_per_sec = total_tokens / max(elapsed, 1)

                    print(
                        f"  Step {global_step:>6d} | "
                        f"tok {total_tokens / 1e6:>8.1f}M | "
                        f"loss: {avg_loss:.4f} | "
                        f"lr: {lr_now:.2e} | "
                        f"tok/s: {tok_per_sec:.0f} | "
                        f"gpu: {torch.cuda.memory_allocated() / 1e6:.0f}MB"
                        if torch.cuda.is_available() else ""
                    )

                    logger.log_metrics({
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "train/tokens_per_sec": tok_per_sec,
                        "train/total_tokens": total_tokens,
                        "train/gpu_memory_mb": torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
                    }, global_step)

                    step_loss = 0.0
                    step_count = 0

                # Save checkpoint
                if global_step % save_every == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                    save_checkpoint(
                        str(ckpt_path), model, optimizer, scheduler,
                        step=global_step, total_tokens=total_tokens,
                        extra={"best_val_loss": best_val_loss},
                    )
                    print(f"    Checkpoint saved: {ckpt_path.name}")

                    # Clean old checkpoints
                    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
                    keep_n = training_config.get("keep_last_n", 3)
                    for old in ckpt_files[:-keep_n]:
                        old.unlink()

                # Evaluate
                if global_step % eval_every == 0:
                    print(f"\n  --- Evaluation at step {global_step} ---")
                    model.eval()
                    eval_loss = 0
                    eval_steps = 0
                    with torch.no_grad():
                        for eval_batch in val_loader:
                            if eval_steps >= training_config.get("eval_steps", 50):
                                break
                            eids = eval_batch["input_ids"].to(device)
                            elabels = eval_batch["labels"].to(device)
                            _, eloss = model(input_ids=eids, labels=elabels)
                            eval_loss += eloss.item()
                            eval_steps += 1

                    avg_eval_loss = eval_loss / max(eval_steps, 1)
                    ppl = math.exp(min(avg_eval_loss, 20))
                    print(f"  Eval loss: {avg_eval_loss:.4f} | Perplexity: {ppl:.2f}")

                    logger.log_metrics({
                        "val/loss": avg_eval_loss,
                        "val/perplexity": ppl,
                    }, global_step)

                    if avg_eval_loss < best_val_loss:
                        best_val_loss = avg_eval_loss
                        best_path = ckpt_dir / "best.pt"
                        save_checkpoint(
                            str(best_path), model, optimizer, scheduler,
                            step=global_step, total_tokens=total_tokens,
                            extra={"best_val_loss": best_val_loss},
                        )
                        print(f"    New best: {best_path.name}")

                    model.train()

    # Final save
    final_path = ckpt_dir / "final.pt"
    save_checkpoint(
        str(final_path), model, optimizer, scheduler,
        step=global_step, total_tokens=total_tokens,
        extra={"best_val_loss": best_val_loss},
    )
    print(f"\n  Final checkpoint: {final_path}")

    # Generate samples
    print(f"\n  --- Sample Generation ---")
    model.eval()
    prompts = config.get("evaluation", {}).get("prompts", ["def fibonacci(n):"])
    samples = generate_samples(
        model, tokenizer, prompts,
        max_new_tokens=200,
        temperature=config.get("evaluation", {}).get("temperature", 0.8),
        device=device,
    )
    for i, (prompt, sample) in enumerate(zip(prompts, samples)):
        print(f"\n  Prompt {i+1}: {prompt}")
        print(f"  Output: {sample[:200]}...")

    # Save report
    elapsed = time.time() - train_start
    report = {
        "total_steps": global_step,
        "total_tokens": total_tokens,
        "final_loss": step_loss / max(step_count, 1),
        "best_val_loss": best_val_loss,
        "elapsed_hours": elapsed / 3600,
        "tokens_per_sec": total_tokens / max(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }

    report_path = Path(training_config.get("experiment_dir", "experiments")) / "kaggle_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Steps: {global_step} | Tokens: {total_tokens / 1e6:.1f}M")
    print(f"  Time: {elapsed / 3600:.2f}h | Best val loss: {best_val_loss:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
