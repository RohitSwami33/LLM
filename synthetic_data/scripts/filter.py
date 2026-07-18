#!/usr/bin/env python3
"""Filter a generated dataset with the configured quality pipeline.

Reads one or more JSONL files produced by ``generate.py``, runs every filter,
and writes the accepted samples to a new JSONL file.  Per-reason rejection
counts and acceptance rate are logged.

Examples
--------
    python -m synthetic_data.scripts.filter -c configs/generation.yaml \
        -i data/generated/reasoning.jsonl -o data/filtered/reasoning.jsonl
    python -m synthetic_data.scripts.filter -c configs/generation.yaml \
        --input-dir data/generated --output-dir data/filtered
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthetic_data.core.config import load_config  # noqa: E402
from synthetic_data.core.telemetry import Metrics, setup_logger  # noqa: E402
from synthetic_data.core.schema import Sample, loads, dumps  # noqa: E402
from synthetic_data.filtering import build_filters  # noqa: E402

logger = setup_logger("filter")


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield loads(line)


def process(input_paths, output_path, config) -> Metrics:
    metrics = Metrics()
    pipeline = build_filters(config.filters)

    total = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for in_path in input_paths:
            for sample in _iter_jsonl(in_path):
                total += 1
                metrics.attempted += 1
                keep, reason = pipeline.check(sample)
                if keep:
                    metrics.accepted += 1
                    fout.write(dumps(sample) + "\n")
                else:
                    metrics.record_rejection(reason or "unknown")

    # Enforce optional global acceptance cap.
    cap = config.filters.max_acceptance_rate
    if cap is not None and total > 0:
        rate = metrics.accepted / total
        if rate > cap:
            logger.warning(
                "Acceptance rate %.2f exceeds cap %.2f (consider tightening filters)",
                rate, cap,
            )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Filter synthetic data")
    parser.add_argument("-c", "--config", default="configs/generation.yaml")
    parser.add_argument("-i", "--input", nargs="+", default=None, help="Input JSONL file(s)")
    parser.add_argument("--input-dir", default=None, help="Filter every *.jsonl in a dir")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file")
    parser.add_argument("--output-dir", default=None, help="Output dir (with --input-dir)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists() and (ROOT / args.config).exists():
        cfg_path = ROOT / args.config
    config = load_config(str(cfg_path))

    if args.input:
        inputs = [Path(p) for p in args.input]
        out = Path(args.output or "data/filtered/accepted.jsonl")
    elif args.input_dir:
        inputs = sorted(Path(args.input_dir).glob("*.jsonl"))
        out_dir = Path(args.output_dir or "data/filtered")
        out = out_dir / "accepted.jsonl"
    else:
        # Default: filter everything under the config's generation output dir.
        gen_dir = Path(config.generation.output_dir)
        inputs = sorted(gen_dir.glob("*.jsonl"))
        out = Path("data/filtered") / "accepted.jsonl"

    if not inputs:
        logger.error("No input JSONL files found.")
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Filtering %d file(s) → %s", len(inputs), out)
    metrics = process(inputs, out, config)
    total = metrics.attempted
    rate = (metrics.accepted / total * 100) if total else 0.0
    logger.info(
        "Filtering complete: %d processed, %d accepted (%.1f%%), %d rejected",
        total, metrics.accepted, rate, metrics.rejected,
    )
    for reason, count in metrics.rejected_reasons.items():
        logger.info("  rejected[%s]: %d", reason, count)


if __name__ == "__main__":
    main()
