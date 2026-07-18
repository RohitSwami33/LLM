#!/usr/bin/env python3
"""Export a (filtered) dataset to JSONL / Parquet / Arrow / Hugging Face.

Reads JSONL samples (the native generator output) and writes them in the
requested columnar or hub format.  Large datasets are sharded for Parquet/Arrow.

Examples
--------
    python -m synthetic_data.scripts.export -c configs/generation.yaml \
        -i data/filtered/accepted.jsonl --format parquet
    python -m synthetic_data.scripts.export -i data/filtered/accepted.jsonl \
        --format hf --dataset-name my-org/synthetic --push-to-hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthetic_data.core.config import load_config  # noqa: E402
from synthetic_data.core.telemetry import setup_logger  # noqa: E402
from synthetic_data.core.schema import Sample, loads  # noqa: E402
from synthetic_data.exporters import export, read_jsonl  # noqa: E402

logger = setup_logger("export")


def _read_samples(path: Path):
    for s in read_jsonl(path):
        yield s


def main():
    parser = argparse.ArgumentParser(description="Export synthetic data")
    parser.add_argument("-c", "--config", default="configs/generation.yaml")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("--format", default=None,
                        choices=["jsonl", "parquet", "arrow", "hf"],
                        help="Output format (defaults to config.export.format)")
    parser.add_argument("-o", "--output", default=None, help="Output path/dir")
    parser.add_argument("--dataset-name", default=None, help="HF dataset name")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists() and (ROOT / args.config).exists():
        cfg_path = ROOT / args.config
    config = load_config(str(cfg_path))

    fmt = args.format or config.export.format
    in_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        ext = {"jsonl": "jsonl", "parquet": "parquet", "arrow": "arrow",
               "hf": "hf_dataset"}[fmt]
        out_path = Path(config.export.output_dir) / f"{in_path.stem}.{ext}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = _read_samples(in_path)
    logger.info("Exporting %s → %s (format=%s)", in_path, out_path, fmt)

    count = export(
        samples,
        fmt=fmt,
        path=out_path,
        push_to_hub=args.push_to_hub or config.export.push_to_hub,
        dataset_name=args.dataset_name or config.export.dataset_name,
    )
    logger.info("Exported %d samples to %s", count, out_path)


if __name__ == "__main__":
    main()
