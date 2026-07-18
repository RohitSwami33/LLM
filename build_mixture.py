#!/usr/bin/env python3
"""
build_mixture.py – Configurable dataset mixture builder for LLM training
==========================================================================

Reads a YAML mixture specification, samples documents from each dataset,
applies preprocessing, deterministically shuffles, and writes a unified
training corpus ready for tokenizer training or model training.

Usage
-----
    python build_mixture.py configs/tiny.yaml
    python build_mixture.py configs/small.yaml  --dry-run
    python build_mixture.py configs/medium.yaml --seed 123

Pipeline
--------
    1. For each dataset in the config, detect the latest version (or use
       the explicit version if given) and load it in streaming or local mode.
    2. Apply preprocessing (cleaning, filtering, dedup).
    3. Sample up to ``max_documents`` from each dataset (using reservoir
       sampling for streaming, random subset for local).
    4. Concatenate all sampled documents.
    5. Deterministically shuffle with a fixed random seed.
    6. Save as sharded JSONL or Arrow files.
    7. Save reproducibility metadata + config snapshot.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
from tqdm import tqdm

from download_datasets import DATASET_REGISTRY, DatasetConfig, detect_config
from preprocess import PreprocessingPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("build_mixture")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"


def _setup_logger():
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        log.addHandler(ch)
    return log


_setup_logger()

# ---------------------------------------------------------------------------
# Reservoir sampling (for streaming datasets)
# ---------------------------------------------------------------------------

def reservoir_sample(iterator: Iterator[Any], k: int, rng: random.Random) -> list[Any]:
    """
    Reservoir sampling: select *k* random items from an iterable of unknown
    length using a single pass and O(k) memory.
    """
    reservoir: list[Any] = []
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def resolve_dataset_config(ds_name: str, config_yaml: dict) -> DatasetConfig:
    """Find the DatasetConfig for *ds_name* in the registry and patch it with YAML overrides."""
    registry_map = {d.name: d for d in DATASET_REGISTRY}
    if ds_name not in registry_map:
        raise ValueError(f"Unknown dataset '{ds_name}'. Available: {list(registry_map.keys())}")

    cfg = registry_map[ds_name]
    max_docs = config_yaml.get("documents", config_yaml.get("files", None))
    languages = config_yaml.get("languages", None)

    return DatasetConfig(
        name=cfg.name,
        hf_path=cfg.hf_path,
        hf_config=cfg.hf_config,
        purpose=cfg.purpose,
        splits=cfg.splits,
        max_samples=max_docs,
        mode=cfg.mode,
        per_language=cfg.per_language,
        language_filter=languages if languages else cfg.language_filter,
        license_filter_field=cfg.license_filter_field,
        license_filter_values=cfg.license_filter_values,
        preprocess_config=cfg.preprocess_config,
        text_column=cfg.text_column,
        extra_columns=cfg.extra_columns,
        estimated_tokens=cfg.estimated_tokens,
        estimated_disk=cfg.estimated_disk,
    )


def load_dataset_for_mixture(
    cfg: DatasetConfig,
) -> tuple[list[dict], str, dict]:
    """
    Load (or stream) a dataset, apply preprocessing, sample, and return
    ``(examples, resolved_config, stats)``.

    Returns a *concrete list* of dicts (materialised in memory) for the
    sampled subset.  This is necessary for deterministic global shuffle.
    """
    resolved = detect_config(cfg.hf_path, cfg.hf_config)
    log.info("  Resolved %s → %s", cfg.name, resolved)

    pipeline = PreprocessingPipeline(cfg.preprocess_config) if cfg.preprocess_config else None
    rng = random.Random(42)  # deterministic per-dataset seed

    if cfg.name == "the_stack_v2":
        return _load_stack_v2(cfg, resolved, pipeline, rng)
    return _load_single(cfg, resolved, pipeline, rng)


def _load_single(
    cfg: DatasetConfig, resolved: str, pipeline: Optional[PreprocessingPipeline], rng: random.Random,
) -> tuple[list[dict], str, dict]:
    """Load a single-configuration dataset (streaming or local)."""
    streaming = cfg.mode == "stream"
    split_name = cfg.splits[0] if isinstance(cfg.splits, list) else cfg.splits
    ds = load_dataset(cfg.hf_path, name=resolved, split=split_name, streaming=streaming)

    if streaming:
        source = ds if isinstance(ds, IterableDataset) else ds[split_name]
    else:
        source = ds[split_name] if isinstance(ds, (Dataset, DatasetDict)) else ds

    # Sample & preprocess
    collected: list[dict] = []
    max_docs = cfg.max_samples or 10_000_000
    lang_counter: Counter = Counter()

    for ex in tqdm(source, desc=f"  Loading {cfg.name}", unit="ex"):
        text = ex.get(cfg.text_column, "")
        if pipeline is not None:
            meta = {k: v for k, v in ex.items() if k != cfg.text_column}
            cleaned = pipeline.process(text, meta)
            if cleaned is None:
                continue
            ex = {cfg.text_column: cleaned, **meta}
            if "language" in ex:
                lang_counter[ex["language"]] += 1

        collected.append(ex)
        if len(collected) >= max_docs:
            break

    # If not streaming, we can randomly subsample instead of taking first N
    if not streaming and len(collected) > max_docs:
        rng.shuffle(collected)
        collected = collected[:max_docs]

    log.info("  Collected %s docs from %s", f"{len(collected):,}", cfg.name)

    stats = {
        "num_documents": len(collected),
        "estimated_tokens": sum(len(e.get(cfg.text_column, "")) // 4 for e in collected),
        "total_chars": sum(len(e.get(cfg.text_column, "")) for e in collected),
        "languages": dict(lang_counter.most_common()),
    }
    if pipeline:
        stats["preprocessing"] = pipeline.get_stats()

    return collected, resolved, stats


def _load_stack_v2(
    cfg: DatasetConfig, resolved: str, pipeline: Optional[PreprocessingPipeline], rng: random.Random,
) -> tuple[list[dict], str, dict]:
    """Load per-language configs from The Stack v2 with license filtering."""
    collected: list[dict] = []
    per_lang: dict[str, int] = {}
    max_per_lang = cfg.max_samples or 50_000

    for lang in (cfg.language_filter or []):
        try:
            ds = load_dataset(cfg.hf_path, name=lang, split="train", streaming=True)
        except Exception as e:
            log.warning('  Config "%s" not found: %s', lang, e)
            continue

        count = 0
        for ex in tqdm(ds, desc=f"  {cfg.name}/{lang}", unit="ex", leave=False):
            if cfg.license_filter_field and cfg.license_filter_values:
                if ex.get(cfg.license_filter_field) not in cfg.license_filter_values:
                    continue

            text = ex.get(cfg.text_column, "")
            if pipeline is not None:
                meta = {k: v for k, v in ex.items() if k != cfg.text_column}
                cleaned = pipeline.process(text, meta)
                if cleaned is None:
                    continue
                ex = {cfg.text_column: cleaned, **meta}

            collected.append(ex)
            count += 1
            if count >= max_per_lang:
                break

        per_lang[lang] = count
        log.info("    %s: %s docs", lang, f"{count:,}")

    stats = {
        "num_documents": len(collected),
        "per_language": per_lang,
        "estimated_tokens": sum(len(e.get(cfg.text_column, "")) // 4 for e in collected),
        "total_chars": sum(len(e.get(cfg.text_column, "")) for e in collected),
    }
    if pipeline:
        stats["preprocessing"] = pipeline.get_stats()

    return collected, resolved, stats


# ---------------------------------------------------------------------------
# Mixture assembly
# ---------------------------------------------------------------------------

def build_mixture(mixture_config: dict, output_dir: Path, seed: int, dry_run: bool = False):
    """
    Build a unified training mixture from the given config.

    Steps:
        1. Load & preprocess each dataset
        2. Concatenate
        3. Deterministic global shuffle
        4. Write output
    """
    rng = random.Random(seed)

    # Resolve per-dataset overrides from the mixture config
    dataset_overrides: dict = mixture_config.get("datasets", {})
    global_preprocess_config: dict = mixture_config.get("preprocessing", {})
    output_cfg: dict = mixture_config.get("output", {})
    mode: str = mixture_config.get("mode", "stream")
    save_tokenizer_data: bool = output_cfg.get("save_tokenizer_data", False)
    fmt: str = output_cfg.get("format", "jsonl")

    # Build a PreprocessingPipeline for the global config (used when loading)
    global_pipeline = PreprocessingPipeline(global_preprocess_config) if global_preprocess_config else None

    if dry_run:
        log.info("=" * 55)
        log.info("DRY RUN – mixture config: %s", mixture_config.get("mixture", {}).get("name", "unnamed"))
        log.info("Datasets:")
        total_est = 0
        for ds_name, ds_cfg in dataset_overrides.items():
            max_docs = ds_cfg.get("documents", ds_cfg.get("files", "?"))
            log.info("  %-20s %s docs", ds_name, f"{max_docs:,}" if isinstance(max_docs, int) else max_docs)
            total_est += max_docs if isinstance(max_docs, int) else 0
        log.info("Total estimated documents: %s", f"{total_est:,}" if total_est else "?")
        log.info("Output: %s", output_cfg.get("path", "datasets/mixtures/<name>"))
        return

    # ── Phase 1: Load each dataset ──────────────────────────────────────
    all_examples: list[dict] = []
    per_dataset_stats: dict[str, dict] = {}
    resolved_versions: dict[str, str] = {}

    for ds_name, ds_config_yaml in dataset_overrides.items():
        log.info("Processing dataset: %s", ds_name)

        # Build a DatasetConfig (merge YAML overrides with registry defaults)
        base = resolve_dataset_config(ds_name, ds_config_yaml)

        # Apply global mode override
        if mode:
            base.mode = mode

        # Apply global preprocessing (but allow per-dataset overrides)
        if global_preprocess_config and base.preprocess_config:
            merged = {**global_preprocess_config, **base.preprocess_config}
            base.preprocess_config = merged
        elif global_preprocess_config:
            base.preprocess_config = global_preprocess_config

        max_docs = ds_config_yaml.get("documents", ds_config_yaml.get("files", None))
        if max_docs is not None:
            base.max_samples = max_docs

        examples, resolved, stats = load_dataset_for_mixture(base)
        all_examples.extend(examples)
        per_dataset_stats[ds_name] = stats
        resolved_versions[ds_name] = resolved

    if not all_examples:
        log.error("No documents collected from any dataset!")
        sys.exit(1)

    log.info("Total documents before global shuffle: %s", f"{len(all_examples):,}")

    # ── Phase 2: Deterministic shuffle ──────────────────────────────────
    log.info("Shuffling with seed=%d …", seed)
    rng.shuffle(all_examples)

    # ── Phase 3: Write output ───────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    text_column = "text"  # unified column name for the mixture
    # Rename content → text for consistency
    for ex in all_examples:
        if "content" in ex and "text" not in ex:
            ex["text"] = ex.pop("content")

    if fmt == "jsonl":
        shard_size = max(1, len(all_examples) // 10)  # 10 shards
        shards_dir = output_dir / "shards"
        shards_dir.mkdir(exist_ok=True)
        shard_paths = []
        for i in range(0, len(all_examples), shard_size):
            shard_idx = i // shard_size
            shard_path = shards_dir / f"shard-{shard_idx:05d}.jsonl"
            with open(shard_path, "w") as f:
                for ex in all_examples[i:i + shard_size]:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            shard_paths.append(str(shard_path.name))

        # Also a single-file version
        merged_path = output_dir / "corpus.jsonl"
        with open(merged_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("Wrote %s docs to %s (%d shards)", f"{len(all_examples):,}", merged_path, len(shard_paths))

    elif fmt == "arrow":
        # Save as Arrow using datasets.Dataset
        from datasets import Dataset as HFDataset, Features, Value

        # Determine features from first example
        first = all_examples[0]
        features_dict = {}
        for k, v in first.items():
            if isinstance(v, str):
                features_dict[k] = Value("string")
            elif isinstance(v, (int, float)):
                features_dict[k] = Value("float32") if isinstance(v, float) else Value("int64")
            else:
                features_dict[k] = Value("string")  # fallback
        features = Features(features_dict)

        hf_dataset = HFDataset.from_list(all_examples, features=features)
        hf_dataset.save_to_disk(str(output_dir / "data"))
        log.info("Wrote %s docs to %s/data (Arrow)", f"{len(all_examples):,}", output_dir)

    else:
        raise ValueError(f"Unknown output format: {fmt}")

    # ── Tokenizer preparation: simple concatenated text files ───────────
    if save_tokenizer_data:
        tokenizer_dir = output_dir / "tokenizer_data"
        tokenizer_dir.mkdir(exist_ok=True)
        texts = [ex.get("text", ex.get("content", "")) for ex in all_examples]
        # Write 10 shards for parallel tokenizer training
        t_shard_size = max(1, len(texts) // 10)
        for i in range(0, len(texts), t_shard_size):
            shard_idx = i // t_shard_size
            path = tokenizer_dir / f"text-{shard_idx:05d}.txt"
            with open(path, "w", encoding="utf-8") as f:
                for t in texts[i:i + t_shard_size]:
                    f.write(t.strip() + "\n")
        log.info("Tokenizer data: %s files in %s", len(list(tokenizer_dir.glob("*.txt"))), tokenizer_dir)

    # ── Phase 4: Save reproducibility metadata ──────────────────────────
    total_chars = sum(len(ex.get("text", "")) for ex in all_examples)
    total_tokens = sum(max(1, len(ex.get("text", "")) // 4) for ex in all_examples)
    avg_doc_len = round(total_chars / len(all_examples), 1) if all_examples else 0

    reproducibility = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mixture_name": output_dir.name,
        "random_seed": seed,
        "total_documents": len(all_examples),
        "total_chars": total_chars,
        "estimated_tokens": total_tokens,
        "avg_doc_length_chars": avg_doc_len,
        "dataset_versions": resolved_versions,
        "per_dataset_stats": per_dataset_stats,
        "preprocessing_config": global_preprocess_config,
        "software_versions": _software_versions(),
        "config_snapshot": mixture_config,
        "output_format": fmt,
        "tokenizer_data_saved": save_tokenizer_data,
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(reproducibility, f, indent=2, default=str)
    log.info("Wrote %s", meta_path)

    # Also copy the original YAML config
    config_copy_path = output_dir / "mixture_config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(mixture_config, f, default_flow_style=False)
    log.info("Wrote %s", config_copy_path)

    log.info("")
    log.info("✔ Mixture built: %s", output_dir)
    log.info("  Documents: %s", f"{len(all_examples):,}")
    log.info("  Estimated tokens: %s", f"{total_tokens:,}")
    log.info("  Estimated disk: %s", _size_on_disk(output_dir))


def _software_versions() -> dict[str, str]:
    return {
        "python": sys.version,
        "datasets": __import__("datasets").__version__,
        "preprocess": "0.1.0",
        "pyyaml": yaml.__version__,
    }


def _size_on_disk(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.2f} {unit}"
        total /= 1024
    return f"{total:.2f} PB"


# ===================================================================
# CLI
# ===================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dataset mixture for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Path to YAML mixture config")
    parser.add_argument("--output", "-o", default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Preview without building")
    parser.add_argument("--mode", choices=["stream", "local"], default=None, help="Override streaming mode")
    return parser.parse_args()


def main():
    args = _parse_args()

    # Load YAML config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        mixture_config = yaml.safe_load(f)

    seed = args.seed or mixture_config.get("seed", 42)
    output_path = args.output or mixture_config.get("output", {}).get("path", "")
    if not output_path:
        # Auto name from config filename
        output_path = f"datasets/mixtures/{config_path.stem}"
    output_dir = Path(output_path)

    if args.mode:
        mixture_config["mode"] = args.mode

    log.info("Building mixture: %s", config_path)
    log.info("Output: %s", output_dir.resolve())

    build_mixture(mixture_config, output_dir, seed=seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
