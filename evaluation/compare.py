"""Compare multiple model checkpoints side by side."""

import os
import sys
import json
import csv
import torch
from typing import List, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.runner import EvalRunner
from evaluation.benchmarks.base import EvalResults


def compare_checkpoints(checkpoint_paths: List[str], benchmark_names: List[str] = None,
                        config_path: str = None, output_dir: str = None,
                        max_samples: int = None, device: torch.device = None,
                        generate: bool = False) -> Dict[str, EvalResults]:
    """Evaluate and compare multiple checkpoints.

    Returns dict mapping checkpoint_path -> EvalResults.
    """
    all_results = {}

    for ckpt_path in checkpoint_paths:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {ckpt_path}")
        print(f"{'#'*60}")

        runner = EvalRunner.from_checkpoint(ckpt_path, config_path, device)
        results = runner.evaluate(
            benchmarks=benchmark_names, generate=generate,
            efficiency=True, max_samples=max_samples,
        )
        all_results[ckpt_path] = results

        if output_dir:
            name = os.path.splitext(os.path.basename(ckpt_path))[0]
            runner.save_results(results, os.path.join(output_dir, name))

    if output_dir:
        _save_comparison_table(all_results, output_dir)
        _save_comparison_csv(all_results, output_dir)

    return all_results


def _get_label(path: str) -> str:
    """Extract a clean label from a checkpoint path."""
    name = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    if parent == "checkpoints":
        parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return f"{parent}/{name}"


def _save_comparison_table(all_results: Dict[str, EvalResults], output_dir: str):
    """Save a markdown comparison table."""
    labels = [_get_label(path) for path in all_results.keys()]
    results_list = list(all_results.values())

    # Collect all metrics
    all_metrics = []
    seen = set()
    for results in results_list:
        for name, bench in results.benchmarks.items():
            for metric in bench.metrics:
                key = f"{name}/{metric}"
                if key not in seen:
                    all_metrics.append((name, metric))
                    seen.add(key)

    lines = ["# Model Comparison\n"]
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|--------|" + "|".join(["------:" for _ in labels]) + "|"
    lines.append(header)
    lines.append(sep)

    for bench_name, metric in all_metrics:
        row = f"| {bench_name}/{metric} |"
        for results in results_list:
            if bench_name in results.benchmarks:
                val = results.benchmarks[bench_name].metrics.get(metric, "N/A")
                if isinstance(val, float):
                    row += f" {val:.4f} |"
                else:
                    row += f" {val} |"
            else:
                row += " - |"
        lines.append(row)

    # Efficiency
    eff_metrics = set()
    for results in results_list:
        eff_metrics.update(results.efficiency.keys())
    eff_metrics.discard("error")

    if eff_metrics:
        lines.append("")
        lines.append("## Efficiency\n")
        header = "| Metric | " + " | ".join(labels) + " |"
        sep = "|--------|" + "|".join(["------:" for _ in labels]) + "|"
        lines.append(header)
        lines.append(sep)
        for metric in sorted(eff_metrics):
            row = f"| {metric} |"
            for results in results_list:
                val = results.efficiency.get(metric, "N/A")
                if isinstance(val, float):
                    row += f" {val:.4f} |"
                else:
                    row += f" {val} |"
            lines.append(row)

    with open(os.path.join(output_dir, "comparison.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\nComparison table saved to {output_dir}/comparison.md")


def _save_comparison_csv(all_results: Dict[str, EvalResults], output_dir: str):
    """Save comparison as CSV."""
    labels = [_get_label(path) for path in all_results.keys()]
    results_list = list(all_results.values())

    all_metrics = []
    seen = set()
    for results in results_list:
        for name, bench in results.benchmarks.items():
            for metric in bench.metrics:
                key = f"{name}/{metric}"
                if key not in seen:
                    all_metrics.append(key)
                    seen.add(key)
    for results in results_list:
        for metric in results.efficiency:
            if metric != "error":
                key = f"efficiency/{metric}"
                if key not in seen:
                    all_metrics.append(key)
                    seen.add(key)

    with open(os.path.join(output_dir, "comparison.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + labels)

        for metric_key in all_metrics:
            row = [metric_key]
            for results in results_list:
                parts = metric_key.split("/", 1)
                if len(parts) == 2:
                    section, metric = parts
                    if section == "efficiency":
                        val = results.efficiency.get(metric, "")
                    else:
                        bench = results.benchmarks.get(section)
                        val = bench.metrics.get(metric, "") if bench else ""
                else:
                    val = ""
                row.append(val)
            writer.writerow(row)
