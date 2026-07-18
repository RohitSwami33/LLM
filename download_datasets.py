#!/usr/bin/env python3
"""
download_datasets.py – Production-ready Hugging Face dataset downloader
=======================================================================

Downloads / streams Hugging Face datasets for LLM pretraining and
post-training research.  Features:

  • Auto-detection of latest dataset versions (Wikipedia snapshot,
    FineWeb CC-MAIN, Dolma, …)
  • Streaming mode – process without fully downloading
  • Local cached mode – full download via ``save_to_disk``
  • Per-dataset ``stats.json`` with document counts, token estimates,
    language distributions, SHA-256 checksums
  • Reproducibility tracking (software versions, config snapshots, seed)
  • Integration with :mod:`preprocess` for optional inline cleaning
  • Rich logging: progress bars, speed, ETA, skipped-file summaries

Usage
-----
    python download_datasets.py                           # download all
    python download_datasets.py --datasets fineweb         # single dataset
    python download_datasets.py --mode stream              # force streaming
    python download_datasets.py --dry-run                  # preview
    python download_datasets.py --list                     # list datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import (
    get_dataset_config_names,
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from tqdm import tqdm

from preprocess import PreprocessingPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path("datasets")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
GATED_DATASETS = {"the_stack_v2"}
PERMISSIVE_LICENSES = [
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSD-4-Clause",
    "ISC", "MIT-0", "CC0-1.0", "CC-BY-4.0", "Unlicense", "Zlib",
    "Python-2.0", "PostgreSQL", "ICU", "Unicode-DFS-2016", "NCSA",
    "WTFPL", "BlueOak-1.0.0", "0BSD",
]
STACK_V2_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C", "C++",
]

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_log: logging.Logger = logging.getLogger("download")


def _setup_logger() -> logging.Logger:
    _log.setLevel(logging.DEBUG)
    if not _log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        _log.addHandler(ch)
    return _log


_log = _setup_logger()


def _add_file_logger(logger: logging.Logger, path: Path) -> logging.Handler:
    fh = logging.FileHandler(str(path), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(fh)
    return fh


# ===================================================================
# Auto-detection of latest dataset versions
# ===================================================================

def _parse_wiki_date(config: str) -> tuple[int, ...]:
    """Parse ``YYYYMMDD`` from config names like ``20231101.en``."""
    m = re.match(r"(\d{8})", config)
    return tuple(map(int, m.groups())) if m else (0,)


def _parse_fineweb_date(config: str) -> tuple[int, ...]:
    """Parse ``YYYY-NN`` from config names like ``CC-MAIN-2024-10``."""
    m = re.match(r"CC-MAIN-(\d{4})-(\d+)", config)
    return tuple(map(int, m.groups())) if m else (0, 0)


def detect_latest_wikipedia() -> str:
    """Return the newest English Wikipedia snapshot config name."""
    configs = get_dataset_config_names("wikimedia/wikipedia")
    en_configs = [c for c in configs if c.endswith(".en")]
    if not en_configs:
        _log.warning("No English Wikipedia config found; falling back to 20231101.en")
        return "20231101.en"
    best = max(en_configs, key=_parse_wiki_date)
    _log.info("Detected latest Wikipedia: %s", best)
    return best


def detect_latest_fineweb(prefer_sample: Optional[str] = "sample-10BT") -> str:
    """Return a stable FineWeb config (prefer sample, fall back to latest CC)."""
    configs = get_dataset_config_names("HuggingFaceFW/fineweb")
    if prefer_sample and prefer_sample in configs:
        _log.info("Detected FineWeb: %s", prefer_sample)
        return prefer_sample
    cc_configs = [c for c in configs if c.startswith("CC-MAIN-")]
    if cc_configs:
        best = max(cc_configs, key=_parse_fineweb_date)
        _log.info("Detected FineWeb (latest CC): %s", best)
        return best
    _log.warning("Fallback FineWeb config to sample-10BT")
    return "sample-10BT"


def detect_latest_fineweb_edu(prefer_sample: Optional[str] = "sample-10BT") -> str:
    """Return a stable FineWeb-Edu config."""
    configs = get_dataset_config_names("HuggingFaceFW/fineweb-edu")
    if prefer_sample and prefer_sample in configs:
        _log.info("Detected FineWeb-Edu: %s", prefer_sample)
        return prefer_sample
    cc_configs = [c for c in configs if c.startswith("CC-MAIN-")]
    if cc_configs:
        best = max(cc_configs, key=_parse_fineweb_date)
        _log.info("Detected FineWeb-Edu (latest CC): %s", best)
        return best
    return "sample-10BT"


def detect_latest_dolma() -> str:
    """Return the latest stable Dolma config."""
    configs = get_dataset_config_names("allenai/dolma")
    # Known version ordering
    version_order = ["v1", "v1_5", "v1_5-sample", "v1_6", "v1_6-sample", "v1_7"]
    for v in reversed(version_order):
        if v in configs:
            _log.info("Detected Dolma: %s", v)
            return v
    _log.warning("Fallback Dolma config to v1_6-sample")
    return "v1_6-sample"


def detect_config(hf_path: str, hf_config: str) -> str:
    """Auto-detect the latest config if *hf_config* is ``"auto"``."""
    DISPATCH = {
        "wikimedia/wikipedia": detect_latest_wikipedia,
        "HuggingFaceFW/fineweb": detect_latest_fineweb,
        "HuggingFaceFW/fineweb-edu": detect_latest_fineweb_edu,
        "allenai/dolma": detect_latest_dolma,
    }
    if hf_config != "auto":
        return hf_config
    detector = DISPATCH.get(hf_path)
    if detector is None:
        raise ValueError(f"No auto-detection logic for {hf_path}")
    return detector()


# ===================================================================
# Dataset configuration
# ===================================================================

@dataclass
class DatasetConfig:
    """Specification for a single dataset to download / stream."""

    # Core identity
    name: str
    hf_path: str
    hf_config: str = "auto"         # "auto" = detect latest
    purpose: str = ""

    # Splits & limits
    splits: list = field(default_factory=lambda: ["train"])
    max_samples: Optional[int] = None

    # Mode
    mode: str = "stream"            # "stream" or "local"

    # License / language filtering
    per_language: bool = False
    language_filter: Optional[list] = None
    license_filter_field: Optional[str] = None
    license_filter_values: Optional[list] = None

    # Preprocessing (applied during download if non-empty)
    preprocess_config: Optional[dict] = None

    # Column names
    text_column: str = "text"
    extra_columns: Optional[list] = None

    # Human estimates
    estimated_tokens: str = "N/A"
    estimated_disk: str = "N/A"


# -----------------------------------------------------------------------
# Registry of datasets
# -----------------------------------------------------------------------

DATASET_REGISTRY: list[DatasetConfig] = [
    DatasetConfig(
        name="fineweb",
        hf_path="HuggingFaceFW/fineweb",
        hf_config="auto",
        purpose="General web pretraining corpus (10B token sample)",
        estimated_tokens="~10B tokens",
        estimated_disk="~27.6 GB (parquet); ~32 GB (arrow)",
        text_column="text",
        extra_columns=["id"],
        preprocess_config={
            "remove_html": True,
            "normalize_unicode": True,
            "normalize_whitespace": True,
        },
    ),
    DatasetConfig(
        name="fineweb_edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_config="auto",
        purpose="High-quality educational web corpus (10B token sample)",
        estimated_tokens="~10B tokens",
        estimated_disk="~27.6 GB (parquet); ~32 GB (arrow)",
        text_column="text",
        extra_columns=["id", "score"],
        preprocess_config={
            "remove_html": True,
            "normalize_unicode": True,
            "normalize_whitespace": True,
        },
    ),
    DatasetConfig(
        name="dolma",
        hf_path="allenai/dolma",
        hf_config="auto",
        purpose="Large open pretraining corpus (10B token sample)",
        estimated_tokens="~10B tokens",
        estimated_disk="~16.4 GB (gzip); ~22 GB (arrow)",
        text_column="text",
        extra_columns=["id", "source", "added", "created"],
        preprocess_config={
            "normalize_unicode": True,
            "normalize_whitespace": True,
        },
    ),
    DatasetConfig(
        name="the_stack_v2",
        hf_path="bigcode/the-stack-v2",
        hf_config="default",
        purpose="Code pretraining (permissively licensed subset)",
        estimated_tokens="~? tokens (50K × 8 languages)",
        estimated_disk="~2-5 GB (JSONL subset)",
        text_column="content",
        extra_columns=[
            "language", "license_type", "max_stars_count",
            "max_stars_repo_path",
        ],
        max_samples=50_000,
        mode="stream",
        per_language=True,
        language_filter=STACK_V2_LANGUAGES,
        license_filter_field="license_type",
        license_filter_values=["permissive"],
        preprocess_config={
            "remove_html": True,
            "normalize_unicode": True,
            "normalize_whitespace": True,
        },
    ),
    DatasetConfig(
        name="wikipedia",
        hf_path="wikimedia/wikipedia",
        hf_config="auto",
        purpose="English Wikipedia corpus for factual knowledge",
        estimated_tokens="~2.5B tokens",
        estimated_disk="~6-8 GB (parquet); ~9-12 GB (arrow)",
        text_column="text",
        extra_columns=["id", "title", "url"],
        preprocess_config={
            "normalize_unicode": True,
            "normalize_whitespace": True,
        },
    ),
]


# ===================================================================
# Utilities
# ===================================================================

DATASET_DIR = Path("datasets")


def _dataset_dir(info: DatasetConfig) -> Path:
    return DATASET_DIR / info.name


def _marker_path(info: DatasetConfig) -> Path:
    return _dataset_dir(info) / ".download_complete"


def _error_marker(info: DatasetConfig) -> Path:
    return _dataset_dir(info) / ".download_error"


def is_complete(info: DatasetConfig) -> bool:
    return _marker_path(info).exists()


def _size_on_disk(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.2f} {unit}"
        total /= 1024
    return f"{total:.2f} PB"


# ===================================================================
# Reproducibility snapshot
# ===================================================================

def _software_versions() -> dict[str, str]:
    import datasets as ds
    import preprocess as pp

    return {
        "python": sys.version,
        "datasets": ds.__version__,
        "preprocess": pp.__version__ if hasattr(pp, "__version__") else "0.1.0",
    }


def _build_reproducibility_info(
    info: DatasetConfig,
    resolved_config: str,
    stats: dict,
    seed: int = 42,
) -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "random_seed": seed,
        "dataset": {
            "name": info.name,
            "hf_path": info.hf_path,
            "resolved_config": resolved_config,
            "mode": info.mode,
            "max_samples": info.max_samples,
            "per_language": info.per_language,
            "language_filter": info.language_filter,
            "license_filter": {
                "field": info.license_filter_field,
                "values": info.license_filter_values,
            },
        },
        "preprocessing_config": info.preprocess_config,
        "software_versions": _software_versions(),
        "stats": stats,
    }


# ===================================================================
# Stats computation
# ===================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words + punctuation ≈ 1 token per 4 chars."""
    return max(1, len(text) // 4)


def _compute_example_stats(examples: list[dict], text_key: str) -> dict:
    """Compute aggregate stats from a list of processed examples."""
    if not examples:
        return {
            "num_documents": 0,
            "total_chars": 0,
            "estimated_tokens": 0,
            "avg_doc_length_chars": 0,
            "avg_tokens_per_doc": 0,
        }
    total_chars = sum(len(e.get(text_key, "")) for e in examples)
    total_tokens = sum(_estimate_tokens(e.get(text_key, "")) for e in examples)
    n = len(examples)
    return {
        "num_documents": n,
        "total_chars": total_chars,
        "estimated_tokens": total_tokens,
        "avg_doc_length_chars": round(total_chars / n, 1),
        "avg_tokens_per_doc": round(total_tokens / n, 1),
    }


def _compute_sha256(path: Path, sample_bytes: int = 1_000_000) -> str:
    """Compute SHA-256 of the first *sample_bytes* (for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(sample_bytes))
    return h.hexdigest()


# ===================================================================
# Download implementations
# ===================================================================

def _download_single(
    info: DatasetConfig,
    resolved_config: str,
    force: bool,
    seed: int,
):
    """Download a non-per-language dataset (Arrow or JSONL)."""
    dd = _dataset_dir(info)
    dd.mkdir(parents=True, exist_ok=True)
    fh = _add_file_logger(_log, dd / "download_log.txt")

    if not force and is_complete(info):
        _log.info("[%s] Already downloaded; skipping (use --force to re-download)", info.name)
        return

    try:
        # Build preprocessing pipeline
        pipeline = PreprocessingPipeline(info.preprocess_config) if info.preprocess_config else None

        _log.info(
            "[%s] Loading %s config=%s mode=%s …",
            info.name, info.hf_path, resolved_config, info.mode,
        )

        streaming = info.mode == "stream"
        dataset = load_dataset(
            info.hf_path,
            name=resolved_config,
            split=info.splits,
            streaming=streaming,
        )

        data_dir = dd / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        if streaming:
            # ---- streaming path: write JSONL while iterating ------------------
            target = data_dir / f"{info.name}.jsonl"
            count = 0
            start_time = time.time()
            processed_examples: list[dict] = []
            lang_counter: dict[str, int] = {}

            source_iter: Iterator
            if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
                source_iter = dataset if isinstance(dataset, IterableDataset) else dataset[info.splits[0]]
            else:
                source_iter = dataset[info.splits[0]]

            pbar = tqdm(
                source_iter,
                desc=f"  {info.name}",
                unit="ex",
                smoothing=0.1,
                mininterval=0.5,
            )
            with open(target, "w") as out:
                for ex in pbar:
                    text = ex.get(info.text_column, "")
                    if pipeline is not None:
                        meta = {k: v for k, v in ex.items() if k != info.text_column}
                        cleaned = pipeline.process(text, meta)
                        if cleaned is None:
                            continue
                        ex = {info.text_column: cleaned, **meta}
                        if "language" in ex:
                            lang_counter[ex["language"]] = lang_counter.get(ex["language"], 0) + 1

                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    count += 1
                    processed_examples.append(ex)
                    if len(processed_examples) > 1000:
                        processed_examples = processed_examples[-500:]

                    if info.max_samples and count >= info.max_samples:
                        break

                    # Speed & ETA on progress bar
                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        speed = count / elapsed if elapsed > 0 else 0
                        pbar.set_postfix(
                            speed=f"{speed:.0f} ex/s",
                            kept=count,
                        )

            elapsed = time.time() - start_time
            _log.info(
                "  Wrote %s examples to %s in %.1fs (%.0f ex/s)",
                f"{count:,}", target, elapsed, count / elapsed if elapsed > 0 else 0,
            )

            # Stats
            stats = _compute_example_stats(
                processed_examples[-1000:] if processed_examples else [],
                info.text_column,
            )
            stats["num_documents"] = count
            stats["download_mode"] = "stream"
            if pipeline:
                pipe_stats = pipeline.get_stats()
                stats["preprocessing"] = pipe_stats
                _log.info("  Cleaning stats: %s docs kept / %s input",
                          pipe_stats["output_documents"], pipe_stats["input_documents"])

        else:
            # ---- local mode: save_to_disk (Arrow) ----------------------------
            if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
                _log.warning("  Cannot save_to_disk for streaming dataset without caching; converting to list …")
                dataset = list(dataset[info.splits[0]])

            if pipeline is not None:
                _log.info("  Applying preprocessing before saving …")
                cleaned_records = []
                for ex in tqdm(dataset, desc="  Cleaning", unit="ex"):
                    text = ex.get(info.text_column, "")
                    meta = {k: v for k, v in ex.items() if k != info.text_column}
                    cleaned = pipeline.process(text, meta)
                    if cleaned is not None:
                        ex[info.text_column] = cleaned
                        cleaned_records.append(ex)

                # Save cleaned data as JSONL
                target = data_dir / f"{info.name}.jsonl"
                with open(target, "w") as f:
                    for ex in cleaned_records:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                _log.info("  Saved %s cleaned documents to %s", f"{len(cleaned_records):,}", target)
                stats = {"num_documents": len(cleaned_records), "download_mode": "local_cleaned"}
                if pipeline:
                    stats["preprocessing"] = pipeline.get_stats()
            else:
                if isinstance(dataset, (Dataset, DatasetDict)):
                    dataset.save_to_disk(str(data_dir))
                    _log.info("  Saved to %s", data_dir)
                    stats = {
                        "num_documents": len(dataset) if isinstance(dataset, Dataset) else sum(len(s) for s in dataset.values()),
                        "download_mode": "local_arrow",
                    }
                else:
                    _log.warning("  Unexpected dataset type; writing JSONL fallback")
                    target = data_dir / f"{info.name}.jsonl"
                    count = 0
                    for ex in tqdm(dataset, desc="  Writing", unit="ex"):
                        with open(target, "a") as f:
                            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        count += 1
                    stats = {"num_documents": count, "download_mode": "local_jsonl"}

        # Language distribution
        if pipeline and pipeline.stats.languages:
            stats["language_distribution"] = dict(pipeline.stats.languages.most_common())

        # Reproducibility
        repro = _build_reproducibility_info(info, resolved_config, stats, seed)

        # Write metadata & reproducibility
        meta_path = dd / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(repro, f, indent=2, default=str)
        _log.info("  Wrote %s", meta_path)

        # Statistics file
        stats_path = dd / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        _log.info("  Wrote %s", stats_path)

        # README generation (concise)
        _generate_readme(info, resolved_config, stats, dd)

        # Mark complete
        _marker_path(info).touch()
        _log.info("[%s] ✔ Done", info.name)

    except Exception:
        _log.exception("[%s] Failed", info.name)
        _error_marker(info).write_text(
            f"Failed at {datetime.utcnow().isoformat()}Z\n"
        )
        raise
    finally:
        _log.removeHandler(fh)


def _download_stack_v2(
    info: DatasetConfig,
    resolved_config: str,
    force: bool,
    seed: int,
):
    """Download The Stack v2: per-language configs + permissive filter."""
    dd = _dataset_dir(info)
    dd.mkdir(parents=True, exist_ok=True)
    fh = _add_file_logger(_log, dd / "download_log.txt")

    if not force and is_complete(info):
        _log.info("[%s] Already downloaded; skipping", info.name)
        return

    try:
        pipeline = PreprocessingPipeline(info.preprocess_config) if info.preprocess_config else None
        data_dir = dd / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        per_lang: dict[str, int] = {}
        features: dict = {}
        all_cleaned: list[dict] = []
        start_time = time.time()

        for lang in info.language_filter:
            target = data_dir / f"{lang}.jsonl"
            count = 0

            _log.info('[%s] Loading %s config="%s" …', info.name, info.hf_path, lang)
            try:
                ds = load_dataset(info.hf_path, name=lang, split="train", streaming=True)
            except Exception as e:
                _log.warning('  Config "%s" not found: %s', lang, e)
                continue

            for ex in tqdm(ds, desc=f"  {lang}", unit="ex", leave=False):
                # License filter
                lf = info.license_filter_field
                lv = info.license_filter_values
                if lf and lv and ex.get(lf) not in lv:
                    continue

                text = ex.get(info.text_column, "")
                if pipeline is not None:
                    meta = {k: v for k, v in ex.items() if k != info.text_column}
                    cleaned = pipeline.process(text, meta)
                    if cleaned is None:
                        continue
                    ex = {info.text_column: cleaned, **meta}

                if not features:
                    features = {k: type(v).__name__ for k, v in ex.items()}
                all_cleaned.append(ex)
                with open(target, "a") as f:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
                total += 1
                if info.max_samples and count >= info.max_samples:
                    break

            per_lang[lang] = count
            _log.info("  %s: %s docs", lang, f"{count:,}")

        elapsed = time.time() - start_time
        _log.info(
            "  Total: %s docs in %.1fs (%.0f ex/s)",
            f"{total:,}", elapsed, total / elapsed if elapsed > 0 else 0,
        )

        stats = {
            "num_documents": total,
            "per_language": per_lang,
            "languages_downloaded": info.language_filter,
            "max_samples_per_language": info.max_samples,
            "download_mode": "stream",
            "file_format": "jsonl",
        }
        if pipeline:
            stats["preprocessing"] = pipeline.get_stats()

        # SHA-256 for data directory
        if data_dir.exists():
            stats["sha256_sample"] = _compute_sha256(
                next(data_dir.rglob("*.jsonl"), Path(""))
            )

        repro = _build_reproducibility_info(info, resolved_config, stats, seed)
        repro["dataset"]["license_filter"] = {
            "field": info.license_filter_field,
            "values": info.license_filter_values,
        }

        meta_path = dd / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(repro, f, indent=2, default=str)

        stats_path = dd / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        _generate_readme(info, resolved_config, stats, dd)
        _marker_path(info).touch()
        _log.info("[%s] ✔ Done (%s examples total)", info.name, f"{total:,}")

    except Exception:
        _log.exception("[%s] Failed", info.name)
        _error_marker(info).write_text(
            f"Failed at {datetime.utcnow().isoformat()}Z\n"
        )
        raise
    finally:
        _log.removeHandler(fh)


# ===================================================================
# README generation
# ===================================================================

def _generate_readme(info: DatasetConfig, resolved: str, stats: dict, dd: Path):
    lines = [
        f"# {info.name}",
        "",
        f"**Purpose:** {info.purpose}",
        f"**Source:** [{info.hf_path}](https://huggingface.co/datasets/{info.hf_path})",
        f"**Config:** `{resolved}`",
        f"**Mode:** {info.mode}",
        f"**Download date:** {datetime.utcnow().isoformat()}Z",
        "",
        f"**Number of documents:** {stats.get('num_documents', 'N/A'):,}",
        f"**Estimated tokens:** {info.estimated_tokens}",
        f"**Estimated disk:** {info.estimated_disk}",
        "",
        "## Features",
    ]
    if info.extra_columns:
        lines.append(f"- `{info.text_column}` (primary text)")
        for col in info.extra_columns:
            lines.append(f"- `{col}`")
    lines.append("")
    lines.append("## Preprocessing")
    if info.preprocess_config:
        for k, v in info.preprocess_config.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("None configured.")
    lines.append("")
    lines.append("## Suggested LLM preprocessing pipeline")
    lines.append("1. Extract `text` / `content` field")
    lines.append("2. Normalize Unicode (NFKC)")
    lines.append("3. Strip HTML & control characters")
    lines.append("4. Collapse whitespace")
    lines.append("5. Exact SHA-256 dedup")
    lines.append("6. Filter by length (50 – 100k chars)")
    lines.append("7. Optional language detection + filtering")
    lines.append("8. Shard into JSONL or Arrow for training")

    readme = dd / "README.md"
    with open(readme, "w") as f:
        f.write("\n".join(lines) + "\n")
    _log.info("  Wrote %s", readme)


# ===================================================================
# Public API
# ===================================================================

def download(
    info: DatasetConfig,
    force: bool = False,
    mode: Optional[str] = None,
    seed: int = 42,
):
    """Download or stream a single dataset."""
    if mode:
        info.mode = mode

    resolved = detect_config(info.hf_path, info.hf_config)
    _log.info("─" * 55)
    _log.info("[%s] Resolved config: %s", info.name, resolved)

    if info.name == "the_stack_v2":
        _download_stack_v2(info, resolved, force, seed)
    else:
        _download_single(info, resolved, force, seed)


def download_all(
    force: bool = False,
    mode: Optional[str] = None,
    seed: int = 42,
):
    """Download all datasets in the registry."""
    for info in DATASET_REGISTRY:
        download(info, force=force, mode=mode, seed=seed)


# ===================================================================
# CLI
# ===================================================================

def _dry_run(selected: list[DatasetConfig]):
    header = f"{'Dataset':<20} {'Config':<25} {'Mode':<10} {'Est. Disk':<25} {'Est. Tokens':<20}"
    _log.info(header)
    _log.info("─" * len(header))
    for ds in selected:
        cfg = ds.hf_config if ds.hf_config != "auto" else "(auto-detect)"
        _log.info("%-20s %-25s %-10s %-25s %-20s", ds.name, cfg, ds.mode, ds.estimated_disk, ds.estimated_tokens)
    _log.info("")
    _log.info("Datasets to download: %d", len(selected))
    _log.info("Output directory:     %s", DATASET_DIR.resolve())


def _list_datasets():
    _log.info("%-20s %-40s %-10s %s", "Name", "HF Path", "Mode", "Config")
    _log.info("─" * 85)
    for ds in DATASET_REGISTRY:
        cfg = ds.hf_config if ds.hf_config != "auto" else "auto-detect"
        _log.info("%-20s %-40s %-10s %s", ds.name, ds.hf_path, ds.mode, cfg)


def _login_if_needed(selected: list[DatasetConfig]):
    if any(d.name in GATED_DATASETS for d in selected):
        _log.info("Checking Hugging Face authentication …")
        try:
            from huggingface_hub import whoami
            whoami()
            _log.info("Authenticated.")
        except Exception:
            _log.warning("Not authenticated. Run: huggingface-cli login")
            _log.warning("The Stack v2 may fail to download.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download / stream Hugging Face datasets for LLM research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-d", "--datasets", nargs="+",
        choices=[d.name for d in DATASET_REGISTRY],
        default=[d.name for d in DATASET_REGISTRY],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--mode", choices=["stream", "local"], default=None,
        help="Override mode for all datasets",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if previously completed",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--base-dir", type=str, default=str(DATASET_DIR),
        help=f"Output root (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    global DATASET_DIR
    DATASET_DIR = Path(args.base_dir)

    if args.list:
        _list_datasets()
        return

    selected = [d for d in DATASET_REGISTRY if d.name in args.datasets]

    if args.dry_run:
        _dry_run(selected)
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _log.info("Output directory: %s", DATASET_DIR.resolve())
    _login_if_needed(selected)

    for ds in selected:
        download(ds, force=args.force, mode=args.mode, seed=args.seed)

    _log.info("")
    _log.info("=" * 55)
    _log.info("All done!")
    _log.info("=" * 55)


if __name__ == "__main__":
    main()
