"""Experiment manager for reproducible training runs.

Creates a unique experiment directory per run with all metadata needed
for exact reproducibility and cross-run comparison.
"""

import os
import json
import time
import shutil
import yaml
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from .system_info import SystemInfo
from .reports import ReportGenerator


class ExperimentManager:
    """Manages experiment directory structure and metadata collection.

    Directory structure:
        experiments/
            YYYY-MM-DD_NNN/
                config.yaml
                git_commit.txt
                model_summary.txt
                train.log
                metrics.csv
                tensorboard/
                checkpoints/
                samples/
                report.json
                report.md
                README.md

    Args:
        base_dir: Root experiments directory.
        experiment_name: Optional custom name. Auto-generated if None.
        config: Full config dict to save.
    """

    def __init__(
        self,
        base_dir: str = "experiments",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = base_dir
        self.config = config or {}
        self.start_time = time.time()

        # Create experiment directory
        if experiment_name:
            self.experiment_dir = os.path.join(base_dir, experiment_name)
        else:
            self.experiment_dir = self._create_unique_dir()

        os.makedirs(self.experiment_dir, exist_ok=True)

        # Create subdirectories
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.samples_dir = os.path.join(self.experiment_dir, "samples")
        self.tb_dir = os.path.join(self.experiment_dir, "tensorboard")

        for d in [self.checkpoint_dir, self.log_dir, self.samples_dir, self.tb_dir]:
            os.makedirs(d, exist_ok=True)

        # Collect system info
        self.system_info = SystemInfo.collect_all()

        # Save static metadata
        self._save_config()
        self._save_git_commit()
        self._save_system_info()

        # Report generator
        self.report_gen = ReportGenerator(self.experiment_dir)

        # Training stats accumulator
        self._training_stats: Dict[str, Any] = {
            "total_steps": 0,
            "tokens_processed": 0,
            "iteration_times": [],
            "peak_gpu_memory_gb": 0.0,
        }

        print(f"Experiment: {self.experiment_dir}")

    def _create_unique_dir(self) -> str:
        """Create a uniquely named experiment directory."""
        os.makedirs(self.base_dir, exist_ok=True)

        # Find next available index
        existing = []
        for entry in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, entry)):
                parts = entry.split("_")
                if len(parts) >= 2:
                    try:
                        existing.append(int(parts[-1]))
                    except ValueError:
                        continue

        idx = max(existing, default=0) + 1
        date_str = time.strftime("%Y-%m-%d")
        name = f"{date_str}_{idx:03d}"
        return os.path.join(self.base_dir, name)

    def _save_config(self) -> None:
        """Save full YAML config."""
        path = os.path.join(self.experiment_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def _save_git_commit(self) -> None:
        """Save git commit info."""
        path = os.path.join(self.experiment_dir, "git_commit.txt")
        SystemInfo.save_git_commit_file(path)

    def _save_system_info(self) -> None:
        """Save system info as JSON."""
        path = os.path.join(self.experiment_dir, "system_info.json")
        with open(path, "w") as f:
            json.dump(self.system_info, f, indent=2, default=str)

    def save_model_summary(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Generate and save model summary with parameter counts."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        embedding_params = 0
        if hasattr(model, "tok_emb"):
            embedding_params = model.tok_emb.weight.numel()

        summary = {
            "total_params": total_params,
            "total_params_m": round(total_params / 1e6, 2),
            "trainable_params": trainable_params,
            "trainable_params_m": round(trainable_params / 1e6, 2),
            "embedding_params": embedding_params,
            "non_embedding_params": total_params - embedding_params,
            "num_layers": getattr(model, "n_layers", "N/A"),
            "d_model": getattr(model, "d_model", "N/A"),
            "n_heads": getattr(model, "n_heads", "N/A"),
            "vocab_size": getattr(model, "vocab_size", "N/A"),
        }

        # Also save as text
        txt_path = os.path.join(self.experiment_dir, "model_summary.txt")
        with open(txt_path, "w") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        return summary

    def save_dataset_stats(self, dataset, name: str = "train") -> Dict[str, Any]:
        """Collect and save dataset statistics."""
        stats = {
            "name": name,
            "num_examples": len(dataset),
            "max_seq_len": getattr(dataset, "max_seq_len", "N/A"),
        }

        # Sample a few examples to get length distribution
        lengths = []
        sample_size = min(1000, len(dataset))
        indices = range(0, len(dataset), max(1, len(dataset) // sample_size))[:sample_size]
        for i in indices:
            item = dataset[i]
            if "input_ids" in item:
                lengths.append(len(item["input_ids"]))

        if lengths:
            stats["avg_length"] = round(sum(lengths) / len(lengths), 1)
            stats["min_length"] = min(lengths)
            stats["max_length"] = max(lengths)
            stats["median_length"] = sorted(lengths)[len(lengths) // 2]

        # Save
        path = os.path.join(self.experiment_dir, "dataset_stats.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def save_tokenizer_info(self, tokenizer) -> Dict[str, Any]:
        """Save tokenizer information."""
        info = {
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "type": type(tokenizer).__name__,
        }

        path = os.path.join(self.experiment_dir, "tokenizer_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

        return info

    def log_training_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Log a training step's metrics."""
        # Track tokens processed
        if "tokens_per_sec" in metrics:
            # Approximate: we don't know exact tokens from metrics alone,
            # but we can track cumulative from the caller
            pass

    def record_iteration(self, iteration_time: float, tokens: int) -> None:
        """Record iteration timing and token count."""
        self._training_stats["iteration_times"].append(iteration_time)
        self._training_stats["tokens_processed"] += tokens

    def record_gpu_memory(self) -> None:
        """Record peak GPU memory usage (CUDA or MPS)."""
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            self._training_stats["peak_gpu_memory_gb"] = max(
                self._training_stats["peak_gpu_memory_gb"], mem
            )
        elif torch.backends.mps.is_available():
            # MPS: report current allocated memory (no max_memory_allocated API)
            mem = torch.mps.current_allocated_memory() / 1e9 if hasattr(torch.mps, 'current_allocated_memory') else 0
            self._training_stats["peak_gpu_memory_gb"] = max(
                self._training_stats["peak_gpu_memory_gb"], mem
            )

    def finalize_training(
        self,
        total_steps: int,
        best_val_loss: float,
        final_val_loss: float,
        best_perplexity: float,
    ) -> Dict[str, Any]:
        """Compile final training statistics."""
        elapsed = time.time() - self.start_time
        iter_times = self._training_stats["iteration_times"]

        stats = {
            "total_steps": total_steps,
            "training_time": self._format_time(elapsed),
            "training_time_seconds": elapsed,
            "tokens_processed": self._training_stats["tokens_processed"],
            "tokens_processed_b": round(self._training_stats["tokens_processed"] / 1e9, 3),
            "avg_tokens_per_sec": round(
                self._training_stats["tokens_processed"] / max(elapsed, 1), 1
            ),
            "avg_iteration_time": round(
                sum(iter_times) / max(len(iter_times), 1), 4
            ),
            "min_iteration_time": round(min(iter_times), 4) if iter_times else 0,
            "max_iteration_time": round(max(iter_times), 4) if iter_times else 0,
            "peak_gpu_memory_gb": round(self._training_stats["peak_gpu_memory_gb"], 2),
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "best_perplexity": best_perplexity,
        }

        self._training_stats.update(stats)
        return stats

    def generate_reports(
        self,
        model_summary: Dict[str, Any],
        eval_results: Dict[str, Any],
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate all report files."""
        self.report_gen.generate_all(
            config=self.config,
            system_info=self.system_info,
            training_stats=self._training_stats,
            eval_results=eval_results,
            model_summary=model_summary,
            dataset_stats=dataset_stats,
        )
        print(f"Reports saved to {self.experiment_dir}")

    def get_checkpoint_dir(self) -> str:
        """Return the checkpoint directory path."""
        return self.checkpoint_dir

    def get_log_dir(self) -> str:
        """Return the log directory path."""
        return self.log_dir

    def get_tb_dir(self) -> str:
        """Return the TensorBoard log directory path."""
        return self.tb_dir

    def get_samples_dir(self) -> str:
        """Return the samples directory path."""
        return self.samples_dir

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = seconds % 60
            return f"{m}m {s:.0f}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"
