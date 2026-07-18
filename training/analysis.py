"""Post-run analysis and leaderboard management.

Handles:
  - Post-run analysis (loss curves, gradient norms, memory usage)
  - Comparison report generation
  - Leaderboard CSV update
  - Visualization generation (plots saved to experiment dir)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger(__name__)


class AnalysisManager:
    """Post-run analysis and comparison."""

    def __init__(self, project_root: Path, experiment_dir: Path):
        self.project_root = Path(project_root)
        self.experiment_dir = Path(experiment_dir)
        self.leaderboard_path = project_root / "experiments" / "leaderboard.csv"

    def analyze_training_run(self, log_file: Path) -> dict:
        """Parse training log and extract metrics."""
        metrics = {
            "steps": [],
            "train_loss": [],
            "val_loss": [],
            "val_ppl": [],
            "tok_per_sec": [],
            "grad_norm": [],
            "learning_rate": [],
            "timestamp": [],
        }

        if not log_file.exists():
            LOG.warning("Log file not found: %s", log_file)
            return metrics

        try:
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse common log formats
                    try:
                        data = json.loads(line)
                        if "step" in data:
                            metrics["steps"].append(data["step"])
                        if "loss" in data:
                            metrics["train_loss"].append(data["loss"])
                        if "val_loss" in data:
                            metrics["val_loss"].append(data["val_loss"])
                        if "val_ppl" in data:
                            metrics["val_ppl"].append(data["val_ppl"])
                        if "tok_per_sec" in data:
                            metrics["tok_per_sec"].append(data["tok_per_sec"])
                        if "grad_norm" in data:
                            metrics["grad_norm"].append(data["grad_norm"])
                        if "learning_rate" in data:
                            metrics["learning_rate"].append(data["learning_rate"])
                        if "timestamp" in data:
                            metrics["timestamp"].append(data["timestamp"])
                    except json.JSONDecodeError:
                        # Try line-based parsing
                        pass
        except Exception as e:
            LOG.error("Failed to parse log: %s", e)

        return metrics

    def generate_comparison_report(
        self,
        baseline_dir: Path,
        current_dir: Path,
    ) -> dict:
        """Generate a comparison report between two experiments."""
        baseline_eval = self._load_eval_results(baseline_dir)
        current_eval = self._load_eval_results(current_dir)

        if not baseline_eval or not current_eval:
            return {"error": "Missing evaluation results"}

        # Compare benchmarks
        benchmarks = {}
        for bench_name in ["wikitext2", "hellaswag", "boolq", "winogrande"]:
            base = baseline_eval.get(bench_name, {})
            curr = current_eval.get(bench_name, {})
            if base and curr:
                base_val = base.get("perplexity", base.get("accuracy", 0))
                curr_val = curr.get("perplexity", curr.get("accuracy", 0))

                # For perplexity, lower is better; for accuracy, higher is better
                is_ppl = "perplexity" in base
                if is_ppl:
                    improvement = ((base_val - curr_val) / base_val) * 100
                else:
                    improvement = ((curr_val - base_val) / base_val) * 100

                benchmarks[bench_name] = {
                    "baseline": base_val,
                    "current": curr_val,
                    "improvement_pct": round(improvement, 2),
                    "metric": "perplexity" if is_ppl else "accuracy",
                }

        # Compare efficiency
        baseline_eff = self._load_efficiency(baseline_dir)
        current_eff = self._load_efficiency(current_dir)

        efficiency = {}
        if baseline_eff and current_eff:
            base_tok = baseline_eff.get("tokens_per_second", 0)
            curr_tok = current_eff.get("tokens_per_second", 0)
            if base_tok > 0:
                efficiency["throughput_change_pct"] = round(((curr_tok - base_tok) / base_tok) * 100, 2)

            base_mem = baseline_eff.get("peak_memory_gb", 0)
            curr_mem = current_eff.get("peak_memory_gb", 0)
            efficiency["memory_change_gb"] = round(curr_mem - base_mem, 3)

        return {
            "benchmarks": benchmarks,
            "efficiency": efficiency,
            "baseline_experiment": str(baseline_dir),
            "current_experiment": str(current_dir),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def _load_eval_results(self, experiment_dir: Path) -> Optional[dict]:
        """Load evaluation results from experiment directory."""
        eval_path = experiment_dir / "eval_results.json"
        if eval_path.exists():
            try:
                with open(eval_path) as f:
                    return json.load(f)
            except Exception as e:
                LOG.warning("Failed to load eval results from %s: %s", eval_path, e)
        return None

    def _load_efficiency(self, experiment_dir: Path) -> Optional[dict]:
        """Load efficiency results from experiment directory."""
        eff_path = experiment_dir / "efficiency.json"
        if eff_path.exists():
            try:
                with open(eff_path) as f:
                    return json.load(f)
            except Exception as e:
                LOG.warning("Failed to load efficiency results: %s", e)
        return None

    def update_leaderboard(self, entry: dict):
        """Update the leaderboard CSV with a new entry."""
        import csv

        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing entries
        entries = []
        if self.leaderboard_path.exists():
            try:
                with open(self.leaderboard_path, "r") as f:
                    reader = csv.DictReader(f)
                    entries = list(reader)
            except Exception:
                entries = []

        # Add or update entry
        name = entry.get("name", "")
        updated = False
        for i, e in enumerate(entries):
            if e.get("name") == name:
                entries[i] = entry
                updated = True
                break

        if not updated:
            entries.append(entry)

        # Write back
        if entries:
            fieldnames = list(entries[0].keys())
            with open(self.leaderboard_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(entries)

        LOG.info("Leaderboard updated: %s (%s)", name, "updated" if updated else "added")

    def generate_plots(self, metrics: dict, output_dir: Path):
        """Generate analysis plots and save to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Loss curves
            if metrics.get("train_loss") or metrics.get("val_loss"):
                fig, ax = plt.subplots(figsize=(10, 6))
                if metrics.get("train_loss"):
                    ax.plot(metrics["train_loss"], label="Train Loss", alpha=0.8)
                if metrics.get("val_loss"):
                    ax.plot(metrics["val_loss"], label="Val Loss", alpha=0.8)
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss Curves")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                LOG.info("Saved loss curves plot")

            # Throughput
            if metrics.get("tok_per_sec"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(metrics["tok_per_sec"], alpha=0.7)
                ax.set_xlabel("Step")
                ax.set_ylabel("Tokens/sec")
                ax.set_title("Training Throughput")
                ax.grid(True, alpha=0.3)
                fig.savefig(output_dir / "throughput.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                LOG.info("Saved throughput plot")

            # Gradient norms
            if metrics.get("grad_norm"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(metrics["grad_norm"], alpha=0.7)
                ax.set_xlabel("Step")
                ax.set_ylabel("Gradient Norm")
                ax.set_title("Gradient Norms")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                fig.savefig(output_dir / "grad_norms.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                LOG.info("Saved gradient norms plot")

        except ImportError:
            LOG.warning("matplotlib not available, skipping plots")
        except Exception as e:
            LOG.error("Failed to generate plots: %s", e)

    def generate_report(self, experiment_dir: Path) -> str:
        """Generate a human-readable experiment report."""
        eval_results = self._load_eval_results(experiment_dir)
        metadata_path = experiment_dir / "experiment_metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                pass

        lines = []
        lines.append("=" * 60)
        lines.append("EXPERIMENT REPORT")
        lines.append("=" * 60)
        lines.append("")

        # System info
        sys_info = metadata.get("system", {})
        lines.append(f"System: {sys_info.get('chip', 'unknown')} | {sys_info.get('ram_gb', '?')} GB RAM")
        lines.append(f"OS: {sys_info.get('os', '?')} {sys_info.get('os_release', '?')}")
        lines.append("")

        # Git info
        git_info = metadata.get("git", {})
        lines.append(f"Git commit: {git_info.get('commit', 'N/A')[:12]}")
        lines.append(f"Branch: {git_info.get('branch', 'N/A')}")
        lines.append(f"Dirty: {git_info.get('dirty', 'N/A')}")
        lines.append("")

        # Evaluation results
        if eval_results:
            lines.append("EVALUATION RESULTS")
            lines.append("-" * 40)

            # Benchmarks
            for bench_name in ["wikitext2", "hellaswag", "boolq", "winogrande"]:
                bench = eval_results.get(bench_name, {})
                if bench:
                    if "perplexity" in bench:
                        lines.append(f"  {bench_name:20s}: PPL={bench['perplexity']:.2f}")
                    if "accuracy" in bench:
                        lines.append(f"  {bench_name:20s}: Acc={bench['accuracy']:.1f}%")

            lines.append("")

            # Efficiency
            eff = eval_results.get("efficiency", {})
            if eff:
                lines.append("EFFICIENCY")
                lines.append("-" * 40)
                lines.append(f"  Throughput: {eff.get('tokens_per_second', 'N/A'):.0f} tok/s")
                lines.append(f"  Peak memory: {eff.get('peak_memory_gb', 'N/A'):.3f} GB")
                lines.append(f"  Model params: {eff.get('model_params', 'N/A')}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
