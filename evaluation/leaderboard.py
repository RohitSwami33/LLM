"""Local leaderboard for tracking and ranking model experiments."""

import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Optional, Any


LEADERBOARD_FILE = "leaderboard.csv"
LEADERBOARD_FIELDS = [
    "experiment_id", "architecture", "total_params_m", "optimizer",
    "dataset", "checkpoint_step", "wikitext2_ppl", "hellaswag_acc",
    "arc_easy_acc", "arc_challenge_acc", "boolq_acc", "piqa_acc",
    "lambada_acc", "winogrande_acc", "val_loss", "throughput_tok_s",
    "peak_memory_gb", "eval_time", "timestamp",
]


def _ensure_leaderboard(output_dir: str) -> str:
    path = os.path.join(output_dir, LEADERBOARD_FILE)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
            writer.writeheader()
    return path


def add_to_leaderboard(output_dir: str, eval_results: Dict[str, Any],
                       architecture: str = "transformer", optimizer: str = "unknown",
                       dataset: str = "unknown", experiment_id: str = None):
    """Add an evaluation result to the leaderboard."""
    path = _ensure_leaderboard(output_dir)

    if experiment_id is None:
        experiment_id = os.path.basename(eval_results.get("checkpoint", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    benchmarks = eval_results.get("benchmarks", {})
    efficiency = eval_results.get("efficiency", {})
    model_info = eval_results.get("model_info", {})

    row = {
        "experiment_id": experiment_id,
        "architecture": architecture,
        "total_params_m": model_info.get("total_params_m", ""),
        "optimizer": optimizer,
        "dataset": dataset,
        "checkpoint_step": model_info.get("checkpoint_step", ""),
        "wikitext2_ppl": benchmarks.get("wikitext2", {}).get("metrics", {}).get("perplexity", ""),
        "hellaswag_acc": benchmarks.get("hellaswag", {}).get("metrics", {}).get("accuracy", ""),
        "arc_easy_acc": benchmarks.get("arc_easy", {}).get("metrics", {}).get("accuracy", ""),
        "arc_challenge_acc": benchmarks.get("arc_challenge", {}).get("metrics", {}).get("accuracy", ""),
        "boolq_acc": benchmarks.get("boolq", {}).get("metrics", {}).get("accuracy", ""),
        "piqa_acc": benchmarks.get("piqa", {}).get("metrics", {}).get("accuracy", ""),
        "lambada_acc": benchmarks.get("lambada", {}).get("metrics", {}).get("accuracy", ""),
        "winogrande_acc": benchmarks.get("winogrande", {}).get("metrics", {}).get("accuracy", ""),
        "val_loss": benchmarks.get("wikitext2", {}).get("metrics", {}).get("loss", ""),
        "throughput_tok_s": efficiency.get("throughput_tok_s", ""),
        "peak_memory_gb": efficiency.get("peak_memory_gb", ""),
        "eval_time": "",
        "timestamp": datetime.now().isoformat(),
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writerow(row)

    print(f"Added to leaderboard: {experiment_id}")
    return row


def load_leaderboard(output_dir: str) -> List[Dict[str, str]]:
    """Load the leaderboard as a list of dicts."""
    path = os.path.join(output_dir, LEADERBOARD_FILE)
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_leaderboard(output_dir: str, sort_by: str = "wikitext2_ppl",
                    ascending: bool = True) -> str:
    """Get a formatted leaderboard string, sorted by the specified metric."""
    entries = load_leaderboard(output_dir)
    if not entries:
        return "Leaderboard is empty."

    def sort_key(entry):
        val = entry.get(sort_by, "")
        try:
            return float(val)
        except (ValueError, TypeError):
            return float("inf") if ascending else float("-inf")

    entries.sort(key=sort_key, reverse=not ascending)

    lines = []
    lines.append(f"# Leaderboard (sorted by {sort_by}, {'ascending' if ascending else 'descending'})\n")
    lines.append("| Rank | Experiment | Architecture | Params (M) | Optimizer | W2 PPL | HellaSwag | ARC-E | ARC-C | Throughput | Mem (GB) |")
    lines.append("|------|-----------|-------------|-----------|-----------|--------|-----------|-------|-------|-----------|---------|")

    for i, entry in enumerate(entries, 1):
        lines.append(
            f"| {i} | {entry.get('experiment_id', '')} | {entry.get('architecture', '')} "
            f"| {entry.get('total_params_m', '')} | {entry.get('optimizer', '')} "
            f"| {entry.get('wikitext2_ppl', '')} | {entry.get('hellaswag_acc', '')} "
            f"| {entry.get('arc_easy_acc', '')} | {entry.get('arc_challenge_acc', '')} "
            f"| {entry.get('throughput_tok_s', '')} | {entry.get('peak_memory_gb', '')} |"
        )

    return "\n".join(lines)
