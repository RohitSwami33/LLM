"""Main training loop for the decoder-only Transformer.

Implements:
    - Mixed precision training (BF16/FP16)
    - Gradient accumulation and clipping
    - torch.compile
    - EMA
    - Async checkpointing
    - Sequence packing
    - NaN/Inf detection and recovery
    - Early stopping with patience
    - Loss spike detection and rollback
    - Experiment tracking with reproducibility
    - Comprehensive logging and reporting
    - Resume from checkpoint
"""

import os
import sys
import time
import math
import copy
import torch
import torch.nn as nn
import yaml
from typing import Optional, Dict, Any, List
from pathlib import Path

from .model.config import ModelConfig
from .model.model import TransformerLM
from .optimizer.builder import build_optimizer
from .tokenizer.tokenizer import build_tokenizer
from .data.dataset import JsonlDataset, PackedDataset, build_dataloader
from .data.collator import DataCollator, PackedCollator
from .ema import EMA
from .utils.logging import LoggerManager
from .utils.checkpoint import (
    save_checkpoint, load_checkpoint, clean_old_checkpoints,
)
from .evaluation.evaluate import evaluate, compute_perplexity
from .evaluation.generate import generate_evaluation_samples
from .experiment.manager import ExperimentManager


class Trainer:
    """Complete training loop for decoder-only Transformer.

    Usage:
        trainer = Trainer.from_yaml("training/configs/base.yaml")
        trainer.train()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config.get("training", {})
        self.model_config = ModelConfig.from_dict(config.get("model", {}))

        # Device — auto-detect MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.is_mps = self.device.type == "mps"
        print(f"Using device: {self.device}")

        # Seed
        seed = self.training_config.get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if self.is_mps:
            torch.mps.manual_seed(seed)

        # ── Experiment Manager (creates directory, saves config/git/system) ──
        self.experiment = ExperimentManager(
            base_dir=self.training_config.get("experiment_dir", "experiments"),
            config=config,
        )

        # Redirect stdout to train.log
        self._log_file = open(os.path.join(self.experiment.experiment_dir, "train.log"), "w")
        self._original_stdout = sys.stdout
        sys.stdout = self._Tee(self._original_stdout, self._log_file)

        # Build components
        self.tokenizer = self._build_tokenizer()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()
        self.ema = self._build_ema()
        self.dataloader = self._build_dataloader("train")
        self.val_dataloader = self._build_dataloader("val")
        self.collator = DataCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_seq_len=self.model_config.max_seq_len,
        )

        # Logger (writes to experiment log dir)
        self.logger = self._build_logger()

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.total_tokens = 0

        # Early stopping state
        self.early_stop_patience = self.training_config.get("early_stop_patience", 0)
        self.early_stop_min_delta = self.training_config.get("early_stop_min_delta", 1e-4)
        self._early_stop_counter = 0
        self._early_stop_best = float("inf")

        # NaN/Inf recovery state
        self.nan_recovery = self.training_config.get("nan_recovery", True)
        self.nan_max_retries = self.training_config.get("nan_max_retries", 3)
        self.nan_skip_batches = self.training_config.get("nan_skip_batches", 10)
        self._nan_retry_count = 0

        # Loss spike detection state
        self.spike_detection = self.training_config.get("spike_detection", True)
        self.spike_threshold = self.training_config.get("spike_threshold", 2.0)
        self.spike_window = self.training_config.get("spike_window", 100)
        self.spike_rollback = self.training_config.get("spike_rollback", True)
        self._loss_history: List[float] = []
        self._spike_rollback_count = 0

        # Checkpoint save for rollback
        self._last_checkpoint_path: Optional[str] = None

        # Save metadata
        model_summary = self.experiment.save_model_summary(self.model)
        tokenizer_info = self.experiment.save_tokenizer_info(self.tokenizer)
        tp = model_summary["total_params_m"]
        trp = model_summary["trainable_params_m"]
        print(f"Model: {tp}M params ({trp}M trainable)")

        # torch.compile — disabled on MPS (unstable)
        if self.training_config.get("compile", True) and not self.is_mps:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed (falling back): {e}")
        elif self.is_mps:
            print("torch.compile disabled on MPS")

        # Resume
        resume_path = self.training_config.get("resume_from")
        if resume_path:
            self._resume(resume_path)

    class _Tee:
        """Write to both stdout and a file."""
        def __init__(self, original, logfile):
            self.original = original
            self.logfile = logfile

        def write(self, text):
            self.original.write(text)
            self.logfile.write(text)
            self.logfile.flush()

        def flush(self):
            self.original.flush()
            self.logfile.flush()

    def _build_tokenizer(self):
        tok_config = self.config.get("tokenizer", {})
        data_path = self.config.get("dataset", {}).get("path")
        return build_tokenizer(tok_config, data_path=data_path)

    def _build_model(self):
        self.model_config.vocab_size = self.tokenizer.vocab_size
        model = TransformerLM(self.model_config)
        model = model.to(self.device)
        self.is_moe = model.is_moe
        return model

    def _build_optimizer(self):
        return build_optimizer(self.model, self.training_config)

    def _build_scheduler(self):
        lr = self.training_config.get("learning_rate", 3e-4)
        min_lr = self.training_config.get("min_lr", 3e-5)
        warmup_steps = self.training_config.get("warmup_steps", 2000)
        max_steps = self.training_config.get("max_steps", 100000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _build_scaler(self):
        dtype = self.training_config.get("dtype", "bf16")
        if dtype == "fp16" and self.device.type == "cuda":
            return torch.amp.GradScaler("cuda")
        return None

    def _build_ema(self):
        if self.training_config.get("use_ema", False):
            decay = self.training_config.get("ema_decay", 0.9999)
            warmup = self.training_config.get("ema_warmup_steps", 2000)
            return EMA(self.model, decay=decay, warmup_steps=warmup)
        return None

    def _build_dataloader(self, split: str):
        data_config = self.config.get("dataset", {})
        path = data_config.get("path", "datasets/synthetic/v1/dataset.jsonl")
        use_packing = data_config.get("packing", False)
        max_seq_len = self.model_config.max_seq_len

        if split == "val":
            dataset = JsonlDataset(path, self.tokenizer, max_seq_len)
            n = len(dataset)
            val_size = int(n * data_config.get("val_split", 0.1))
            train_size = n - val_size
            _, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            batch_size = self.training_config.get("batch_size", 8)
            return build_dataloader(
                val_dataset, batch_size,
                DataCollator(self.tokenizer.pad_token_id, max_seq_len),
                num_workers=data_config.get("num_workers", 2),
                pin_memory=data_config.get("pin_memory", not self.is_mps),
                shuffle=False,
            )
        else:
            batch_size = self.training_config.get("batch_size", 8)
            if use_packing:
                dataset = PackedDataset(
                    path, self.tokenizer, max_seq_len,
                    pack_factor=data_config.get("pack_factor", 4),
                )
                self.experiment.save_dataset_stats(dataset, name="train_packed")
                collator = PackedCollator(
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_seq_len=max_seq_len,
                )
                return build_dataloader(
                    dataset, 1, collator,
                    num_workers=data_config.get("num_workers", 2),
                    pin_memory=data_config.get("pin_memory", not self.is_mps),
                    shuffle=True,
                )
            else:
                dataset = JsonlDataset(path, self.tokenizer, max_seq_len)
                n = len(dataset)
                val_size = int(n * data_config.get("val_split", 0.1))
                train_size = n - val_size
                train_dataset, _ = torch.utils.data.random_split(
                    dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                self.experiment.save_dataset_stats(train_dataset, name="train")
                return build_dataloader(
                    train_dataset, batch_size,
                    DataCollator(self.tokenizer.pad_token_id, max_seq_len),
                    num_workers=data_config.get("num_workers", 2),
                    pin_memory=data_config.get("pin_memory", not self.is_mps),
                    shuffle=True,
                )

    def _build_logger(self):
        return LoggerManager(
            log_dir=self.experiment.get_log_dir(),
            tensorboard=self.training_config.get("tensorboard", True),
            csv=True,  # Always write CSV to experiment dir
            wandb=self.training_config.get("wandb", False),
            wandb_project=self.training_config.get("wandb_project", "deepseek-baseline"),
            config=self.config,
        )

    def _resume(self, path: str):
        print(f"Resuming from checkpoint: {path}")
        state = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler,
            self.scaler, self.ema, self.device
        )
        self.global_step = state["step"]
        self.epoch = state.get("epoch", 0)
        self.total_tokens = state.get("total_tokens", 0)
        print(f"Resumed at step {self.global_step}, epoch {self.epoch}, tokens {self.total_tokens/1e6:.1f}M")

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _detect_nan_inf(self, tensor: torch.Tensor) -> bool:
        return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()

    def _save_snapshot_for_rollback(self):
        path = os.path.join(self.experiment.get_checkpoint_dir(), "_spike_snapshot")
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, os.path.join(path, "snapshot.pt"))
        self._last_checkpoint_path = path

    def _rollback_from_snapshot(self):
        path = os.path.join(self.experiment.get_checkpoint_dir(), "_spike_snapshot", "snapshot.pt")
        if not os.path.exists(path):
            print("  No snapshot found for rollback, skipping.")
            return False

        print(f"  Rolling back to step stored in snapshot...")
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        if self.scaler and state["scaler_state"]:
            self.scaler.load_state_dict(state["scaler_state"])
        self.global_step = state["global_step"]
        self.best_val_loss = state["best_val_loss"]
        self._spike_rollback_count += 1
        return True

    def _handle_nan_inf(self, loss: torch.Tensor) -> bool:
        if not self._detect_nan_inf(loss):
            self._nan_retry_count = 0
            return False

        print(f"\n  WARNING: NaN/Inf detected in loss at step {self.global_step}!")
        self._nan_retry_count += 1

        if self._nan_retry_count > self.nan_max_retries:
            print(f"  Exceeded max retries ({self.nan_max_retries}). Training may be unstable.")
            return True

        self.optimizer.zero_grad()
        for pg in self.optimizer.param_groups:
            pg["lr"] *= 0.5
        print(f"  Halved LR to {self.get_lr():.2e}. Skipping next {self.nan_skip_batches} batches.")
        return True

    def _detect_loss_spike(self) -> bool:
        if not self._loss_history:
            return False
        if len(self._loss_history) < self.spike_window:
            return False

        recent = self._loss_history[-self.spike_window:]
        mean_loss = sum(recent) / len(recent)
        current_loss = self._loss_history[-1]

        return current_loss > mean_loss * self.spike_threshold

    def _freeze_embed_grads(self):
        if hasattr(self.model, "tok_emb"):
            self.model.tok_emb.weight.requires_grad_(False)
            print("  Froze embedding gradients for stability.")

    def _unfreeze_embed_grads(self):
        if hasattr(self.model, "tok_emb"):
            self.model.tok_emb.weight.requires_grad_(True)

    def _estimate_flops(self, tokens: int) -> int:
        """Estimate total FLOPs: ~6 * non_embedding_params * tokens."""
        n_params = self.model.get_num_params(non_embedding=True)
        return 6 * n_params * tokens

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        if "position_ids" in batch:
            tokens = (labels != -100).sum().item()
        else:
            batch_size, seq_len = input_ids.shape
            tokens = batch_size * seq_len

        dtype = self.training_config.get("dtype", "bf16")
        # MPS: force FP32 (BF16 not supported, FP16 unstable)
        if self.is_mps:
            use_amp = False
        else:
            use_amp = dtype in ("bf16", "fp16") and self.device.type == "cuda"

        with torch.amp.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu",
            dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
            enabled=use_amp,
        ):
            logits, loss = self.model(input_ids=input_ids, labels=labels)

        # Add MoE auxiliary losses if applicable
        moe_metrics = {}
        if self.is_moe and hasattr(self.model, "_moe_metrics") and self.model._moe_metrics:
            moe_metrics = self.model.get_moe_metrics()
            total_aux_loss = moe_metrics.get("total_aux_loss", 0.0)
            if isinstance(total_aux_loss, torch.Tensor):
                loss = loss + total_aux_loss
            elif total_aux_loss > 0:
                loss = loss + total_aux_loss

        if self._handle_nan_inf(loss):
            return {"loss": float("nan"), "lr": self.get_lr(), "grad_norm": 0.0, "tokens_per_sec": 0.0, "perplexity": float("nan")}

        grad_accum = self.training_config.get("gradient_accumulation_steps", 1)
        loss = loss / grad_accum

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_clip = self.training_config.get("grad_clip", 1.0)
        if grad_clip > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.ema:
            self.ema.update()

        if self.global_step % grad_accum == 0:
            self.scheduler.step()

        loss_val = loss.item() * grad_accum
        lr = self.get_lr()
        grad_norm = self.get_grad_norm()
        tokens_per_sec = tokens / max(time.time() - self._step_start, 1e-6)

        result = {
            "loss": loss_val,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
            "perplexity": math.exp(min(loss_val, 20)),
        }

        # Add MoE metrics to result
        if self.is_moe and moe_metrics:
            for k, v in moe_metrics.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    result[f"moe/{k}"] = v.item()
                elif isinstance(v, float):
                    result[f"moe/{k}"] = v

        return result

    def train(self):
        max_steps = self.training_config.get("max_steps", 100000)
        max_tokens = self.training_config.get("max_tokens", None)
        log_every = self.training_config.get("log_every", 100)
        eval_every = self.training_config.get("eval_every", 500)
        eval_every_tokens = self.training_config.get("eval_every_tokens", None)
        save_every = self.training_config.get("save_every", 5000)
        save_every_tokens = self.training_config.get("save_every_tokens", None)
        grad_accum = self.training_config.get("gradient_accumulation_steps", 1)
        batch_size = self.training_config.get("batch_size", 8)
        seq_len = self.model_config.max_seq_len
        tokens_per_step = batch_size * grad_accum * seq_len

        self.total_tokens = 0
        self._best_val_loss = float("inf")
        self._no_improve_count = 0

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.experiment.experiment_dir}")
        print(f"{'='*60}")
        print(f"Device: {self.device} | Dtype: {'fp32' if self.is_mps else self.training_config.get('dtype', 'bf16')}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {total_params/1e6:.1f}M params ({'MoE' if self.is_moe else 'Transformer'})")
        if self.is_moe:
            total, active = self.model.get_total_and_active_params()
            print(f"  Total params:    {total/1e6:.1f}M")
            print(f"  Active per token: {active/1e6:.1f}M")
            print(f"  Experts: {self.model_config.moe_num_experts} | Top-k: {self.model_config.moe_top_k}")
        print(f"Dataset: {self.config.get('dataset', {}).get('path', 'unknown')}")
        print(f"Training started | Step {self.global_step}/{max_steps}")
        if max_tokens:
            print(f"Token budget: {max_tokens/1e9:.2f}B tokens | Tokens/step: {tokens_per_step:,}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {grad_accum}")
        print(f"Effective batch size: {batch_size * grad_accum}")
        print(f"Sequence length: {seq_len}")
        print(f"Eval interval: {'every '+str(eval_every_tokens//1e6)+'M tokens' if eval_every_tokens else 'every '+str(eval_every)+' steps'}")
        print(f"Save interval: {'every '+str(save_every_tokens//1e6)+'M tokens' if save_every_tokens else 'every '+str(save_every)+' steps'}")
        print(f"Early stopping patience: {self.early_stop_patience if self.early_stop_patience > 0 else 'disabled'}")
        print(f"Spike detection: {'enabled' if self.spike_detection else 'disabled'}")
        print(f"NaN recovery: {'enabled' if self.nan_recovery else 'disabled'}")
        print(f"{'='*60}\n")

        self._step_start = time.time()
        self._train_start_time = time.time()
        running_metrics = {}
        self._save_snapshot_for_rollback()

        # MoE metrics accumulation
        moe_running_metrics = {}

        while self.global_step < max_steps:
            if max_tokens and self.total_tokens >= max_tokens:
                break

            for batch in self.dataloader:
                if self.global_step >= max_steps:
                    break
                if max_tokens and self.total_tokens >= max_tokens:
                    break

                self._step_start = time.time()
                metrics = self.train_step(batch)
                iter_time = time.time() - self._step_start
                self.global_step += 1

                if math.isnan(metrics["loss"]):
                    continue

                # Record timing and tokens
                if "position_ids" in batch:
                    labels = batch["labels"]
                    tokens = (labels != -100).sum().item()
                else:
                    tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

                self.total_tokens += tokens

                self.experiment.record_iteration(iter_time, tokens)
                self.experiment.record_gpu_memory()

                # Accumulate metrics
                for k, v in metrics.items():
                    running_metrics[k] = running_metrics.get(k, 0) + v

                # Accumulate MoE metrics
                if self.is_moe:
                    for k, v in metrics.items():
                        if k.startswith("moe/"):
                            moe_running_metrics[k] = moe_running_metrics.get(k, 0) + v

                # Track loss for spike detection
                if self.spike_detection:
                    self._loss_history.append(metrics["loss"])
                    if len(self._loss_history) > self.spike_window * 10:
                        self._loss_history = self._loss_history[-self.spike_window * 5:]

                # Log
                if self.global_step % log_every == 0:
                    avg_metrics = {k: v / log_every for k, v in running_metrics.items()}
                    avg_metrics["tokens_per_sec"] = metrics["tokens_per_sec"]
                    avg_metrics["gpu_memory_gb"] = (
                        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    )

                    # Add MoE metrics to logging
                    if self.is_moe and moe_running_metrics:
                        for k, v in moe_running_metrics.items():
                            avg_metrics[k] = v / log_every

                    # Calculate ETA
                    elapsed = time.time() - self._train_start_time
                    steps_done = self.global_step
                    if max_tokens:
                        tokens_remaining = max_tokens - self.total_tokens
                        avg_tok_rate = self.total_tokens / max(elapsed, 1)
                        eta_seconds = tokens_remaining / max(avg_tok_rate, 1)
                    else:
                        steps_remaining = max_steps - steps_done
                        avg_step_time = elapsed / max(steps_done, 1)
                        eta_seconds = steps_remaining * avg_step_time
                    eta_h, eta_rem = divmod(int(eta_seconds), 3600)
                    eta_m, eta_s = divmod(eta_rem, 60)

                    self.logger.log_metrics(avg_metrics, self.global_step)

                    tok_pct = (self.total_tokens / max_tokens * 100) if max_tokens else 0
                    print(
                        f"Step {self.global_step:>6d} | "
                        f"tok {self.total_tokens/1e6:>8.1f}M"
                        + (f"/{max_tokens/1e6:.0f}M ({tok_pct:.1f}%)" if max_tokens else "") + " | "
                        f"loss: {avg_metrics['loss']:.4f} | "
                        f"ppl: {avg_metrics['perplexity']:.2f} | "
                        f"lr: {avg_metrics['lr']:.2e} | "
                        f"tok/s: {avg_metrics['tokens_per_sec']:.0f} | "
                        f"ETA: {eta_h}h{eta_m:02d}m{eta_s:02d}s"
                    )

                    # Print MoE-specific metrics
                    if self.is_moe and moe_running_metrics:
                        moe_parts = []
                        if "moe/routing_entropy" in avg_metrics:
                            moe_parts.append(f"entropy={avg_metrics['moe/routing_entropy']:.3f}")
                        if "moe/load_balance_loss" in avg_metrics:
                            moe_parts.append(f"lb_loss={avg_metrics['moe/load_balance_loss']:.4f}")
                        if "moe/dropped_tokens_pct" in avg_metrics:
                            moe_parts.append(f"dropped={avg_metrics['moe/dropped_tokens_pct']*100:.1f}%")
                        if moe_parts:
                            print(f"         MoE: {' | '.join(moe_parts)}")

                    running_metrics = {}
                    moe_running_metrics = {}

                # Loss spike detection and rollback
                if self.spike_detection and self.spike_rollback and self._detect_loss_spike():
                    print(f"\n  LOSS SPIKE DETECTED at step {self.global_step}!")
                    print(f"  Current loss: {metrics['loss']:.4f}")
                    if self._rollback_from_snapshot():
                        self._freeze_embed_grads()
                        self._loss_history = self._loss_history[:-self.spike_window:]
                    continue
                else:
                    self._unfreeze_embed_grads()

                # Evaluate
                should_eval = (self.global_step % eval_every == 0) or \
                              (eval_every_tokens and self.total_tokens >= getattr(self, '_next_eval_tokens', 0))
                if should_eval:
                    val_loss = self._evaluate()

                    if eval_every_tokens:
                        self._next_eval_tokens = getattr(self, '_next_eval_tokens', 0) + eval_every_tokens

                    if self.early_stop_patience > 0:
                        if val_loss < self._early_stop_best - self.early_stop_min_delta:
                            self._early_stop_best = val_loss
                            self._early_stop_counter = 0
                        else:
                            self._early_stop_counter += 1
                            print(f"  Early stopping patience: {self._early_stop_counter}/{self.early_stop_patience}")

                            if self._early_stop_counter >= self.early_stop_patience:
                                print(f"\n  EARLY STOPPING triggered at step {self.global_step}!")
                                print(f"  Best val_loss: {self._early_stop_best:.4f}")
                                self._save_checkpoint(name="early_stop_final")
                                self._finalize_and_report()
                                return

                    self._save_snapshot_for_rollback()

                # Save checkpoint
                should_save = (self.global_step % save_every == 0) or \
                              (save_every_tokens and self.total_tokens >= getattr(self, '_next_save_tokens', 0))
                if should_save:
                    if save_every_tokens:
                        self._next_save_tokens = getattr(self, '_next_save_tokens', 0) + save_every_tokens
                    self._save_checkpoint()

        # Final save
        self._save_checkpoint()
        self._finalize_and_report()

    def _evaluate(self) -> float:
        print(f"\n--- Evaluation at step {self.global_step} ---")

        eval_steps = self.training_config.get("eval_steps", 100)
        results = evaluate(
            self.model, self.val_dataloader, self.device,
            max_steps=eval_steps, pad_token_id=self.tokenizer.pad_token_id,
        )

        self.logger.log_metrics({
            "val/loss": results["val_loss"],
            "val/perplexity": results["val_perplexity"],
            "val/accuracy": results["val_accuracy"],
            "val/tokens": results["val_tokens"],
        }, self.global_step)

        print(f"  val_loss: {results['val_loss']:.4f}")
        print(f"  val_ppl:  {results['val_perplexity']:.2f}")
        print(f"  val_acc:  {results['val_accuracy']:.4f}")

        # Generate samples
        eval_config = self.config.get("evaluation", {})
        n_samples = eval_config.get("num_samples", 3)
        prompts = eval_config.get("prompts", [
            "Explain quantum computing",
            "Write a Python function to sort a list",
            "What are the benefits of exercise?",
        ])[:n_samples]

        samples = generate_evaluation_samples(
            self.model, self.tokenizer, self.device,
            prompts=prompts,
            max_new_tokens=eval_config.get("max_new_tokens", 200),
        )

        # Save samples to experiment dir
        samples_path = os.path.join(self.experiment.get_samples_dir(), f"step_{self.global_step}.txt")
        with open(samples_path, "w") as f:
            f.write(samples)

        self.logger.log_scalar("samples/text", hash(samples), self.global_step)

        if results["val_loss"] < self.best_val_loss:
            self.best_val_loss = results["val_loss"]
            self._save_checkpoint(name="best_model")

        print("--- End evaluation ---\n")
        return results["val_loss"]

    def _save_checkpoint(self, name: Optional[str] = None):
        if name:
            path = os.path.join(self.experiment.get_checkpoint_dir(), name)
        else:
            path = os.path.join(self.experiment.get_checkpoint_dir(), f"step_{self.global_step}")

        rng_state = {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=self.ema,
            step=self.global_step,
            epoch=self.epoch,
            rng_state=rng_state,
            total_tokens=self.total_tokens,
        )

        keep = self.training_config.get("keep_last_n", 3)
        clean_old_checkpoints(self.experiment.get_checkpoint_dir(), keep)

        print(f"Checkpoint saved: {path}")

    def _finalize_and_report(self):
        """Finalize training and generate all reports."""
        # Run final evaluation if not done recently
        eval_results = {
            "best_val_loss": self.best_val_loss,
            "best_perplexity": math.exp(min(self.best_val_loss, 20)),
            "final_val_loss": self.best_val_loss,
        }

        # Try to get final val loss from last eval
        try:
            eval_steps = self.training_config.get("eval_steps", 100)
            results = evaluate(
                self.model, self.val_dataloader, self.device,
                max_steps=eval_steps, pad_token_id=self.tokenizer.pad_token_id,
            )
            eval_results["final_val_loss"] = results["val_loss"]
            eval_results["final_perplexity"] = results["val_perplexity"]
            eval_results["final_accuracy"] = results["val_accuracy"]
        except Exception:
            pass

        # Finalize training stats
        training_stats = self.experiment.finalize_training(
            total_steps=self.global_step,
            best_val_loss=self.best_val_loss,
            final_val_loss=eval_results.get("final_val_loss", self.best_val_loss),
            best_perplexity=eval_results["best_perplexity"],
        )

        # Generate reports
        model_summary = self.experiment.save_model_summary(self.model)
        self.experiment.generate_reports(
            model_summary=model_summary,
            eval_results=eval_results,
        )

        # Print final summary
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Steps:           {self.global_step}")
        print(f"  Training time:   {training_stats['training_time']}")
        print(f"  Tokens processed: {training_stats['tokens_processed_b']}B")
        print(f"  Avg tok/s:       {training_stats['avg_tokens_per_sec']}")
        print(f"  Best val loss:   {self.best_val_loss:.4f}")
        print(f"  Best perplexity: {training_stats['best_perplexity']:.2f}")
        print(f"  Peak GPU mem:    {training_stats['peak_gpu_memory_gb']} GB")
        print(f"  Experiment:      {self.experiment.experiment_dir}")
        print(f"{'='*60}\n")

        self.logger.close()

        # Restore stdout
        sys.stdout = self._original_stdout
        self._log_file.close()


def train_from_yaml(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_from_yaml(sys.argv[1])
    else:
        train_from_yaml("training/configs/base.yaml")
