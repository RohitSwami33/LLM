"""Report generation for experiment results.

Generates:
    - report.json: Machine-readable experiment summary
    - report.md: Human-readable experiment summary
    - README.md: Model card for the experiment
"""

import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path


class ReportGenerator:
    """Generates experiment reports in multiple formats."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir

    def generate_all(
        self,
        config: Dict[str, Any],
        system_info: Dict[str, Any],
        training_stats: Dict[str, Any],
        eval_results: Dict[str, Any],
        model_summary: Dict[str, Any],
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate all report files."""
        self._generate_report_json(
            config, system_info, training_stats, eval_results,
            model_summary, dataset_stats
        )
        self._generate_report_md(
            config, system_info, training_stats, eval_results,
            model_summary, dataset_stats
        )
        self._generate_readme(
            config, system_info, training_stats, eval_results,
            model_summary, dataset_stats
        )

    def _generate_report_json(
        self,
        config: Dict, system_info: Dict, training_stats: Dict,
        eval_results: Dict, model_summary: Dict,
        dataset_stats: Optional[Dict] = None,
    ) -> None:
        report = {
            "experiment": {
                "directory": self.experiment_dir,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "config": config,
            "system": system_info,
            "model": model_summary,
            "dataset": dataset_stats or {},
            "training": training_stats,
            "evaluation": eval_results,
        }

        path = os.path.join(self.experiment_dir, "report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def _generate_report_md(
        self,
        config: Dict, system_info: Dict, training_stats: Dict,
        eval_results: Dict, model_summary: Dict,
        dataset_stats: Optional[Dict] = None,
    ) -> None:
        lines = ["# Experiment Report\n"]

        # System
        lines.append("## System\n")
        lines.append(f"| Key | Value |")
        lines.append(f"|-----|-------|")
        lines.append(f"| OS | {system_info.get('os', {}).get('platform', 'N/A')} |")
        lines.append(f"| Python | {system_info.get('python', {}).get('version_short', 'N/A')} |")
        lines.append(f"| PyTorch | {system_info.get('pytorch', {}).get('version', 'N/A')} |")
        cuda = system_info.get("cuda", {})
        if cuda.get("available"):
            lines.append(f"| CUDA | {cuda.get('version', 'N/A')} |")
            lines.append(f"| GPU | {cuda.get('device_name', 'N/A')} |")
            lines.append(f"| GPU Memory | {cuda.get('total_memory_gb', 'N/A')} GB |")
        lines.append("")

        # Git
        git = system_info.get("git", {})
        if git.get("available"):
            lines.append("## Git\n")
            lines.append(f"- **Commit**: `{git.get('commit_hash', 'N/A')[:12]}`")
            lines.append(f"- **Branch**: {git.get('branch', 'N/A')}")
            lines.append(f"- **Message**: {git.get('commit_message', 'N/A')}")
            lines.append(f"- **Dirty**: {git.get('dirty', 'N/A')}")
            lines.append("")

        # Model
        lines.append("## Model\n")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        for key, val in model_summary.items():
            lines.append(f"| {key} | {val} |")
        lines.append("")

        # Dataset
        if dataset_stats:
            lines.append("## Dataset\n")
            lines.append(f"| Statistic | Value |")
            lines.append(f"|-----------|-------|")
            for key, val in dataset_stats.items():
                lines.append(f"| {key} | {val} |")
            lines.append("")

        # Training
        lines.append("## Training\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for key, val in training_stats.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")
        lines.append("")

        # Evaluation
        lines.append("## Evaluation\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for key, val in eval_results.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")
        lines.append("")

        path = os.path.join(self.experiment_dir, "report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _generate_readme(
        self,
        config: Dict, system_info: Dict, training_stats: Dict,
        eval_results: Dict, model_summary: Dict,
        dataset_stats: Optional[Dict] = None,
    ) -> None:
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        data_cfg = config.get("dataset", {})
        tok_cfg = config.get("tokenizer", {})

        lines = ["# Model Card\n"]

        # Overview
        lines.append("## Overview\n")
        lines.append(f"- **Architecture**: Decoder-only Transformer")
        lines.append(f"- **Parameters**: {model_summary.get('total_params', 'N/A')}")
        lines.append(f"- **Vocabulary Size**: {model_cfg.get('vocab_size', 'N/A')}")
        lines.append(f"- **Context Length**: {model_cfg.get('max_seq_len', 'N/A')} tokens")
        lines.append("")

        # Architecture details
        lines.append("## Architecture\n")
        lines.append(f"| Component | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| d_model | {model_cfg.get('d_model', 'N/A')} |")
        lines.append(f"| n_heads | {model_cfg.get('n_heads', 'N/A')} |")
        lines.append(f"| n_layers | {model_cfg.get('n_layers', 'N/A')} |")
        lines.append(f"| d_ff | {model_cfg.get('d_ff', 'N/A')} |")
        lines.append(f"| Activation | {model_cfg.get('activation', 'N/A')} |")
        lines.append(f"| Normalization | {model_cfg.get('norm_type', 'N/A')} |")
        lines.append(f"| RoPE | {model_cfg.get('rope', 'N/A')} |")
        lines.append(f"| Flash Attention | {model_cfg.get('flash_attention', 'N/A')} |")
        lines.append(f"| Gradient Checkpointing | {model_cfg.get('gradient_checkpointing', 'N/A')} |")
        lines.append("")

        # Optimizer
        lines.append("## Optimizer\n")
        lines.append(f"| Hyperparameter | Value |")
        lines.append(f"|----------------|-------|")
        lines.append(f"| Optimizer | {train_cfg.get('optimizer', 'N/A')} |")
        lines.append(f"| Learning Rate | {train_cfg.get('learning_rate', 'N/A')} |")
        lines.append(f"| Min LR | {train_cfg.get('min_lr', 'N/A')} |")
        lines.append(f"| Weight Decay | {train_cfg.get('weight_decay', 'N/A')} |")
        lines.append(f"| Beta1 | {train_cfg.get('beta1', 'N/A')} |")
        lines.append(f"| Beta2 | {train_cfg.get('beta2', 'N/A')} |")
        lines.append(f"| Grad Clip | {train_cfg.get('grad_clip', 'N/A')} |")
        lines.append(f"| Scheduler | {train_cfg.get('scheduler', 'N/A')} |")
        lines.append(f"| Warmup Steps | {train_cfg.get('warmup_steps', 'N/A')} |")
        lines.append(f"| Batch Size | {train_cfg.get('batch_size', 'N/A')} |")
        lines.append(f"| Grad Accum Steps | {train_cfg.get('gradient_accumulation_steps', 'N/A')} |")
        lines.append(f"| Effective Batch | {train_cfg.get('batch_size', 0) * train_cfg.get('gradient_accumulation_steps', 1)} |")
        lines.append(f"| Mixed Precision | {train_cfg.get('dtype', 'N/A')} |")
        lines.append(f"| EMA | {train_cfg.get('use_ema', 'N/A')} (decay={train_cfg.get('ema_decay', 'N/A')}) |")
        lines.append("")

        # Tokenizer
        lines.append("## Tokenizer\n")
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        lines.append(f"| Type | {tok_cfg.get('type', 'N/A')} |")
        lines.append(f"| Vocab Size | {tok_cfg.get('vocab_size', 'N/A')} |")
        lines.append(f"| Model Path | {tok_cfg.get('model_path', 'N/A')} |")
        lines.append("")

        # Dataset
        lines.append("## Dataset\n")
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        lines.append(f"| Path | {data_cfg.get('path', 'N/A')} |")
        lines.append(f"| Max Seq Len | {data_cfg.get('max_seq_len', 'N/A')} |")
        lines.append(f"| Train/Val Split | {data_cfg.get('train_split', 'N/A')}/{data_cfg.get('val_split', 'N/A')} |")
        lines.append(f"| Packing | {data_cfg.get('packing', False)} |")
        if dataset_stats:
            for key, val in dataset_stats.items():
                lines.append(f"| {key} | {val} |")
        lines.append("")

        # Training duration
        lines.append("## Training\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Steps | {training_stats.get('total_steps', 'N/A')} |")
        lines.append(f"| Training Time | {training_stats.get('training_time', 'N/A')} |")
        lines.append(f"| Tokens Processed | {training_stats.get('tokens_processed', 'N/A')} |")
        lines.append(f"| Avg Tokens/sec | {training_stats.get('avg_tokens_per_sec', 'N/A')} |")
        lines.append(f"| Avg Iteration Time | {training_stats.get('avg_iteration_time', 'N/A')} |")
        lines.append("")

        # Evaluation
        lines.append("## Evaluation\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for key, val in eval_results.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")
        lines.append("")

        # System
        lines.append("## Environment\n")
        lines.append(f"- **OS**: {system_info.get('os', {}).get('platform', 'N/A')}")
        lines.append(f"- **Python**: {system_info.get('python', {}).get('version_short', 'N/A')}")
        lines.append(f"- **PyTorch**: {system_info.get('pytorch', {}).get('version', 'N/A')}")
        cuda = system_info.get("cuda", {})
        if cuda.get("available"):
            lines.append(f"- **CUDA**: {cuda.get('version', 'N/A')}")
            lines.append(f"- **GPU**: {cuda.get('device_name', 'N/A')}")
        git = system_info.get("git", {})
        if git.get("available"):
            lines.append(f"- **Git Commit**: `{git.get('commit_hash', 'N/A')[:12]}`")
            lines.append(f"- **Git Branch**: {git.get('branch', 'N/A')}")
        lines.append("")

        path = os.path.join(self.experiment_dir, "README.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
