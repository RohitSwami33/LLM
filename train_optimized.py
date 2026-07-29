#!/usr/bin/env python3
"""Optimized training script for Apple Silicon (M4).

Features:
- Pre-tokenized cache (first run tokenizes, subsequent runs are instant)
- Auto-tuned batch size, workers, prefetching
- MPS for all compute
- torch.compile for Apple Silicon
- Async CPU/GPU pipeline
- Comprehensive profiling
- Performance report generation
"""

import sys
import os
import time
import json
import math
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.model.config import ModelConfig
from training.model.model import TransformerLM
from training.tokenizer.tokenizer import build_tokenizer
from training.data.optimized_dataset import (
    OptimizedPackedDataset, auto_tune, get_system_info
)
from training.data.collator import PackedCollator
from training.optimizer.builder import build_optimizer
from training.utils.checkpoint import save_checkpoint
from training.utils.logging import LoggerManager
from training.experiment.manager import ExperimentManager


class PerformanceProfiler:
    """Track and report training performance metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self._last_batch_time = time.time()
        self._data_wait_total = 0.0
        self._forward_total = 0.0
        self._backward_total = 0.0
        self._optim_step_total = 0.0
        self._batch_count = 0

    def start_batch(self):
        self._batch_start = time.time()

    def end_data_load(self):
        self._data_wait_total += time.time() - self._batch_start

    def end_forward(self):
        self._forward_start = time.time()

    def end_backward(self):
        self._forward_total += time.time() - self._forward_start
        self._backward_start = time.time()

    def end_optim(self):
        self._backward_total += time.time() - self._backward_start
        self._optim_start = time.time()

    def end_step(self):
        self._optim_step_total += time.time() - self._optim_start
        self._batch_count += 1

    def log(self, step: int, loss: float, lr: float, tokens: int):
        elapsed = time.time() - self.start_time
        self.metrics['step'].append(step)
        self.metrics['loss'].append(loss)
        self.metrics['lr'].append(lr)
        self.metrics['tokens'].append(tokens)
        self.metrics['elapsed'].append(elapsed)

    def get_stats(self) -> Dict[str, float]:
        if self._batch_count == 0:
            return {}

        total_time = time.time() - self.start_time
        total_tokens = sum(self.metrics['tokens'])

        return {
            'total_time_s': total_time,
            'total_tokens': total_tokens,
            'tokens_per_sec': total_tokens / max(total_time, 1e-6),
            'samples_per_sec': self._batch_count / max(total_time, 1e-6),
            'avg_data_wait_ms': (self._data_wait_total / self._batch_count) * 1000,
            'avg_forward_ms': (self._forward_total / self._batch_count) * 1000,
            'avg_backward_ms': (self._backward_total / self._batch_count) * 1000,
            'avg_optim_ms': (self._optim_step_total / self._batch_count) * 1000,
            'data_wait_pct': (self._data_wait_total / total_time) * 100,
            'compute_pct': ((self._forward_total + self._backward_total + self._optim_step_total) / total_time) * 100,
            'peak_memory_mb': torch.mps.current_allocated_memory() / 1e6 if torch.backends.mps.is_available() else 0,
        }

    def report(self) -> str:
        stats = self.get_stats()
        if not stats:
            return "No data collected yet."

        lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"Total training time: {stats['total_time_s']:.1f}s",
            f"Total tokens processed: {stats['total_tokens']/1e6:.1f}M",
            "",
            "--- Throughput ---",
            f"Tokens/sec: {stats['tokens_per_sec']:.0f}",
            f"Samples/sec: {stats['samples_per_sec']:.2f}",
            "",
            "--- Timing Breakdown ---",
            f"Avg data load time: {stats['avg_data_wait_ms']:.1f}ms",
            f"Avg forward pass: {stats['avg_forward_ms']:.1f}ms",
            f"Avg backward pass: {stats['avg_backward_ms']:.1f}ms",
            f"Avg optimizer step: {stats['avg_optim_ms']:.1f}ms",
            "",
            "--- Utilization ---",
            f"Data loading: {stats['data_wait_pct']:.1f}%",
            f"Compute: {stats['compute_pct']:.1f}%",
            f"Idle: {100 - stats['data_wait_pct'] - stats['compute_pct']:.1f}%",
            "",
            "--- Memory ---",
            f"Peak MPS memory: {stats['peak_memory_mb']:.0f} MB",
        ]

        # Add suggestions
        lines.extend(["", "--- Suggestions ---"])
        if stats['data_wait_pct'] > 30:
            lines.append("GPU is waiting for data. Increase num_workers or prefetch_factor.")
        elif stats['compute_pct'] > 90:
            lines.append("GPU is fully utilized. Good!")
        else:
            lines.append("Balanced CPU/GPU utilization.")

        if stats['peak_memory_mb'] > 8000:
            lines.append("High MPS memory usage. Consider reducing batch_size.")

        lines.append("=" * 60)
        return "\n".join(lines)


def generate_performance_report(
    experiment_dir: str,
    profiler: PerformanceProfiler,
    system_info: Dict,
    config: Dict,
    model_params: int,
):
    """Generate performance_report.md."""
    stats = profiler.get_stats()

    report = f"""# Performance Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## System

| Property | Value |
|----------|-------|
| Platform | {system_info['platform']} |
| Architecture | {system_info['machine']} |
| CPU cores | {system_info['cpu_count']} |
| RAM | {system_info['ram_gb']:.1f} GB |
| MPS available | {system_info['mps']} |
| PyTorch | {torch.__version__} |

## Model

| Property | Value |
|----------|-------|
| Parameters | {model_params/1e6:.1f}M |
| Architecture | Transformer (Dense) |
| Sequence length | {config.get('model', {}).get('max_seq_len', 2048)} |

## Training Configuration

| Property | Value |
|----------|-------|
| Optimizer | {config.get('training', {}).get('optimizer', 'muon')} |
| Learning rate | {config.get('training', {}).get('learning_rate', 3e-4)} |
| Batch size | {config.get('training', {}).get('batch_size', 4)} |
| Gradient accumulation | {config.get('training', {}).get('gradient_accumulation_steps', 8)} |
| Effective batch size | {config.get('training', {}).get('batch_size', 4) * config.get('training', {}).get('gradient_accumulation_steps', 8) * config.get('model', {}).get('max_seq_len', 2048):,} tokens |
| Mixed precision | {config.get('training', {}).get('dtype', 'bf16')} |
| torch.compile | {config.get('training', {}).get('compile', True)} |

## Performance

| Metric | Value |
|--------|-------|
| Total training time | {stats.get('total_time_s', 0):.1f}s |
| Total tokens processed | {stats.get('total_tokens', 0)/1e6:.1f}M |
| **Throughput** | **{stats.get('tokens_per_sec', 0):.0f} tokens/sec** |
| Samples/sec | {stats.get('samples_per_sec', 0):.2f} |

### Timing Breakdown

| Stage | Avg Time | % of Total |
|-------|----------|------------|
| Data loading | {stats.get('avg_data_wait_ms', 0):.1f}ms | {stats.get('data_wait_pct', 0):.1f}% |
| Forward pass | {stats.get('avg_forward_ms', 0):.1f}ms | - |
| Backward pass | {stats.get('avg_backward_ms', 0):.1f}ms | - |
| Optimizer step | {stats.get('avg_optim_ms', 0):.1f}ms | - |
| **Compute total** | **{stats.get('avg_forward_ms', 0) + stats.get('avg_backward_ms', 0) + stats.get('avg_optim_ms', 0):.1f}ms** | **{stats.get('compute_pct', 0):.1f}%** |

### Memory

| Metric | Value |
|--------|-------|
| Peak MPS memory | {stats.get('peak_memory_mb', 0):.0f} MB |
| Model size (FP32) | {model_params * 4 / 1e6:.0f} MB |

## Optimization Suggestions

"""

    # Generate suggestions
    suggestions = []
    if stats.get('data_wait_pct', 0) > 30:
        suggestions.append("1. **Increase dataloader workers** - GPU is waiting for data")
        suggestions.append("2. **Increase prefetch_factor** - Allow more background loading")
        suggestions.append("3. **Use pre-tokenized cache** - Eliminate tokenization overhead")
    if stats.get('peak_memory_mb', 0) > 8000:
        suggestions.append("4. **Reduce batch_size** - MPS memory is high")
        suggestions.append("5. **Increase gradient_accumulation** - Maintain throughput with smaller batches")
    if stats.get('compute_pct', 0) < 80:
        suggestions.append("6. **Check CPU-GPU sync points** - May be blocking pipeline")

    if not suggestions:
        suggestions.append("Training is well-optimized. No changes needed.")

    report += "\n".join(suggestions)
    report += "\n\n## Bottleneck Analysis\n\n"

    if stats.get('data_wait_pct', 0) > stats.get('compute_pct', 0):
        report += "**Primary bottleneck: Data loading** - The GPU is idle waiting for data.\n"
    else:
        report += "**Primary bottleneck: Compute** - The GPU is fully utilized.\n"

    # Save report
    report_path = os.path.join(experiment_dir, "performance_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nPerformance report saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Optimized training for Apple Silicon")
    parser.add_argument("--config", type=str, default="training/configs/pretrain_research_v2.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto-tune", action="store_true", help="Auto-tune batch size and workers")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.max_tokens:
        config['training']['max_tokens'] = args.max_tokens

    # System info
    system_info = get_system_info()
    print(f"\n{'='*60}")
    print(f"OPTIMIZED TRAINING FOR APPLE SILICON")
    print(f"{'='*60}")
    print(f"Platform: {system_info['platform']}")
    print(f"Cores: {system_info['cpu_count']} | RAM: {system_info['ram_gb']:.1f}GB | MPS: {system_info['mps']}")

    # Build tokenizer
    print("\nBuilding tokenizer...")
    tok_config = config.get("tokenizer", {})
    tokenizer = build_tokenizer(tok_config)

    # Build dataset
    print("\nBuilding dataset...")
    data_config = config.get("dataset", {})
    t0 = time.time()
    dataset = OptimizedPackedDataset(
        path=data_config.get("path", "datasets/research_v2/corpus.jsonl"),
        tokenizer=tokenizer,
        max_seq_len=config.get("model", {}).get("max_seq_len", 2048),
        pack_factor=data_config.get("pack_factor", 8),
    )
    print(f"Dataset ready in {time.time()-t0:.1f}s")

    # Auto-tune
    if args.auto_tune:
        model_params = 68_721_984  # Known from profiling
        tuned = auto_tune(len(dataset), model_params)
        config['training']['batch_size'] = tuned['batch_size']
        config['training']['gradient_accumulation_steps'] = tuned['gradient_accumulation_steps']
        config['dataset']['num_workers'] = tuned['num_workers']
        config['dataset']['prefetch_factor'] = tuned['prefetch_factor']
        data_config['pack_factor'] = tuned['pack_factor']

    # Build model
    print("\nBuilding model...")
    model_config = ModelConfig.from_dict(config.get("model", {}))
    model_config.vocab_size = tokenizer.vocab_size
    model = TransformerLM(model_config)

    device = torch.device("mps" if system_info['mps'] else "cpu")
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.1f}M params on {device}")

    # torch.compile on MPS
    if config.get("training", {}).get("compile", True) and system_info['mps']:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # Build optimizer
    optimizer = build_optimizer(model, config.get("training", {}))

    # Build scheduler
    lr = config.get("training", {}).get("learning_rate", 3e-4)
    min_lr = config.get("training", {}).get("min_lr", 3e-5)
    warmup_steps = config.get("training", {}).get("warmup_steps", 2000)
    max_steps = config.get("training", {}).get("max_steps", 50000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Build dataloader
    batch_size = config.get("training", {}).get("batch_size", 4)
    num_workers = data_config.get("num_workers", 4)
    prefetch_factor = data_config.get("prefetch_factor", 4)

    collator = PackedCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=model_config.max_seq_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # PackedDataset returns one packed sequence
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=system_info['machine'] != 'arm64',  # Not useful on MPS
        shuffle=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Experiment tracking
    experiment = ExperimentManager(
        base_dir=config.get("training", {}).get("experiment_dir", "experiments"),
        config=config,
    )

    # Logger
    logger = LoggerManager(
        log_dir=experiment.get_log_dir(),
        tensorboard=config.get("training", {}).get("tensorboard", True),
        csv=True,
        wandb=False,
    )

    # Profiler
    profiler = PerformanceProfiler()

    # Training loop
    max_steps = config.get("training", {}).get("max_steps", 50000)
    max_tokens = config.get("training", {}).get("max_tokens", None)
    log_every = config.get("training", {}).get("log_every", 50)
    save_every = config.get("training", {}).get("save_every", 1000)
    eval_every = config.get("training", {}).get("eval_every", 1000)
    grad_clip = config.get("training", {}).get("grad_clip", 1.0)
    grad_accum = config.get("training", {}).get("gradient_accumulation_steps", 1)

    total_tokens = 0
    global_step = 0
    train_acc_sum = 0.0
    train_acc_count = 0

    print(f"\n{'='*60}")
    print(f"TRAINING START")
    print(f"{'='*60}")
    print(f"Dataset: {len(dataset):,} examples")
    print(f"Batch size: {batch_size} | Grad accum: {grad_accum}")
    print(f"Effective batch: {batch_size * grad_accum * model_config.max_seq_len:,} tokens/step")
    print(f"Max steps: {max_steps}")
    if max_tokens:
        print(f"Max tokens: {max_tokens/1e6:.0f}M")
    print(f"Workers: {num_workers} | Prefetch: {prefetch_factor}")
    print(f"{'='*60}\n")

    train_start = time.time()

    while global_step < max_steps:
        if max_tokens and total_tokens >= max_tokens:
            break

        for batch_idx, batch in enumerate(dataloader):
            if global_step >= max_steps:
                break
            if max_tokens and total_tokens >= max_tokens:
                break

            profiler.start_batch()

            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tokens = (labels != -100).sum().item() if (labels != -100).any() else input_ids.numel()

            profiler.end_data_load()

            # Forward pass
            model.train()
            optimizer.zero_grad()

            dtype = config.get("training", {}).get("dtype", "bf16")
            use_amp = dtype in ("bf16", "fp16") and device.type == "cuda"

            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
                enabled=use_amp,
            ):
                logits, loss = model(input_ids=input_ids, labels=labels)

            profiler.end_forward()

            # Compute training accuracy
            with torch.no_grad():
                mask = labels != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct = (preds[mask] == labels[mask]).float().sum().item()
                    total = mask.float().sum().item()
                    step_acc = correct / total
                else:
                    step_acc = 0.0
                train_acc_sum += step_acc
                train_acc_count += 1

            # Backward pass
            loss.backward()
            profiler.end_backward()

            # Optimizer step
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            profiler.end_optim()

            profiler.end_step()

            total_tokens += tokens
            global_step += 1

            # Log
            loss_val = loss.item()
            lr_val = optimizer.param_groups[0]['lr']
            profiler.log(global_step, loss_val, lr_val, tokens)

            if global_step % log_every == 0:
                stats = profiler.get_stats()
                elapsed = time.time() - train_start
                pct = (total_tokens / max_tokens * 100) if max_tokens else 0
                avg_train_acc = train_acc_sum / max(train_acc_count, 1)

                print(
                    f"Step {global_step:>6d} | "
                    f"tok {total_tokens/1e6:>8.1f}M"
                    + (f"/{max_tokens/1e6:.0f}M ({pct:.1f}%)" if max_tokens else "") + " | "
                    f"loss: {loss_val:.4f} | "
                    f"acc: {avg_train_acc:.4f} | "
                    f"lr: {lr_val:.2e} | "
                    f"tok/s: {stats['tokens_per_sec']:.0f} | "
                    f"data: {stats['avg_data_wait_ms']:.0f}ms | "
                    f"fwd: {stats['avg_forward_ms']:.0f}ms | "
                    f"bwd: {stats['avg_backward_ms']:.0f}ms | "
                    f"mem: {stats['peak_memory_mb']:.0f}MB"
                )

                # Log to tensorboard
                logger.log_metrics({
                    'train/loss': loss_val,
                    'train/acc': avg_train_acc,
                    'train/lr': lr_val,
                    'train/tokens_per_sec': stats['tokens_per_sec'],
                    'train/total_tokens': total_tokens,
                    'perf/data_wait_ms': stats['avg_data_wait_ms'],
                    'perf/forward_ms': stats['avg_forward_ms'],
                    'perf/backward_ms': stats['avg_backward_ms'],
                    'perf/peak_memory_mb': stats['peak_memory_mb'],
                }, global_step)

                # Reset accuracy accumulators for next interval
                train_acc_sum = 0.0
                train_acc_count = 0

            # Save checkpoint
            if global_step % save_every == 0:
                ckpt_dir = os.path.join(experiment.experiment_dir, "checkpoints")
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler,
                    step=global_step, total_tokens=total_tokens,
                )
                print(f"  Checkpoint saved: step_{global_step}")

            # Evaluate
            if global_step % eval_every == 0:
                print(f"\n--- Evaluation at step {global_step} ---")
                model.eval()
                eval_loss = 0
                eval_acc_sum = 0.0
                eval_steps = 0
                with torch.no_grad():
                    for eval_batch in dataloader:
                        if eval_steps >= 50:
                            break
                        input_ids = eval_batch["input_ids"].to(device)
                        labels = eval_batch["labels"].to(device)
                        logits, loss = model(input_ids=input_ids, labels=labels)
                        eval_loss += loss.item()
                        # Compute eval accuracy
                        mask = labels != -100
                        if mask.any():
                            preds = logits.argmax(dim=-1)
                            correct = (preds[mask] == labels[mask]).float().sum().item()
                            total = mask.float().sum().item()
                            eval_acc_sum += correct / total
                        eval_steps += 1
                avg_eval_loss = eval_loss / max(eval_steps, 1)
                avg_eval_acc = eval_acc_sum / max(eval_steps, 1)
                print(f"  Eval loss: {avg_eval_loss:.4f} | Perplexity: {math.exp(min(avg_eval_loss, 20)):.2f} | Accuracy: {avg_eval_acc:.4f}")

                logger.log_metrics({
                    'val/loss': avg_eval_loss,
                    'val/acc': avg_eval_acc,
                    'val/perplexity': math.exp(min(avg_eval_loss, 20)),
                }, global_step)
                model.train()

    # Final save
    ckpt_dir = os.path.join(experiment.experiment_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "final")
    save_checkpoint(
        ckpt_path, model, optimizer, scheduler,
        step=global_step, total_tokens=total_tokens,
    )

    # Generate performance report
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(profiler.report())

    generate_performance_report(
        experiment.experiment_dir,
        profiler,
        system_info,
        config,
        total_params,
    )

    print(f"\nExperiment: {experiment.experiment_dir}")


if __name__ == "__main__":
    main()
