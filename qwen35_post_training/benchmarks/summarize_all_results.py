"""Create a compact four/five-way Markdown comparison from lm-eval JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = {
    "mmlu_pro_computer_science": "exact_match,custom-extract",
    "truthfulqa_mc2": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "hellaswag": "acc_norm,none",
    "bbh_cot_zeroshot_logical_deduction_five_objects": "exact_match,flexible-extract",
    "ifeval": "inst_level_strict_acc,none",
    "arc_challenge": "acc_norm,none",
}


def load(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {task: float(raw["results"][task][metric]) for task, metric in METRICS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", nargs=2, metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    variants = [(name, Path(path), load(Path(path))) for name, path in args.variant]
    base = variants[0][2]
    lines = [
        "# Qwen3.5-4B Local Post-Training Comparison",
        "",
        "All variants were evaluated with the same NF4 configuration, deterministic decoding, "
        "task prompts and sampled items. Each task contains only 10 examples, so this is a "
        "screening comparison rather than a publication-quality estimate.",
        "",
        "| Task | " + " | ".join(v[0] for v in variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]
    for task in METRICS:
        lines.append("| " + task + " | " + " | ".join(f"{100*v[2][task]:.2f}%" for v in variants) + " |")
    lines += ["", "## Selected-metric average", "", "| Variant | Average | Change vs base |", "|---|---:|---:|"]
    for name, _, scores in variants:
        avg = sum(scores.values()) / len(scores)
        base_avg = sum(base.values()) / len(base)
        lines.append(f"| {name} | {100*avg:.2f}% | {100*(avg-base_avg):+.2f} pp |")
    lines += [
        "",
        "## Interpretation rule",
        "",
        "A final improvement claim requires gains on the targeted factual/truthfulness tasks without "
        "material regression on general reasoning. Because n=10 per task is noisy, any promising "
        "adapter should receive a larger held-out evaluation before publication.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
