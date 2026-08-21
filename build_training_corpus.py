#!/usr/bin/env python3
"""Build the 200M-model training corpus: math + general knowledge + science + tech.

Downloads (streaming) each source, extracts the text field properly (handles
schema quirks), samples to the configured size, and writes ONE combined JSONL
plus a mixture report. NO coding datasets.

Usage:
    env -u PYTHONPATH .venv/bin/python build_training_corpus.py [--dry-run]

Output: datasets/train_corpus/corpus.jsonl  ({"text": "...", "source": "..."})
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

OUT_DIR = Path("datasets/train_corpus")
SEED = 42

# name -> (hf_path, config, text_extractor, max_samples)
# text_extractor: callable(row) -> str
SOURCES = {
    "fineweb_edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_config": "sample-10BT",
        "max_samples": 900_000,
        "text": lambda r: str(r.get("text", "")),
    },
    "wikipedia": {
        "hf_path": "wikimedia/wikipedia",
        "hf_config": "20231101.en",
        "max_samples": 300_000,
        "text": lambda r: str(r.get("text", "")),
    },
    "finemath": {
        "hf_path": "HuggingFaceTB/finemath",
        "hf_config": "finemath-4plus",
        "max_samples": 100_000,
        "text": lambda r: str(r.get("text", "")),
    },
    "openwebmath": {
        "hf_path": "open-web-math/open-web-math",
        "hf_config": None,
        "max_samples": 120_000,
        "text": lambda r: str(r.get("text", "")),
    },
    "openstax": {
        "hf_path": "HuggingFaceTB/openstax_paragraphs",
        "hf_config": None,
        "max_samples": 40_000,
        # nested: chapters[].sections[].paragraph
        "text": lambda r: "\n\n".join(
            s.get("paragraph", "") or ""
            for ch in r.get("chapters", []) or []
            for s in ch.get("sections", []) or []
        ),
    },
    "arxiv": {
        "hf_path": "open-index/open-arxiv",
        "hf_config": None,
        "max_samples": 100_000,
        "text": lambda r: str(r.get("abstract", "") or r.get("text", "")),
    },
}


def iter_rows(hf_path: str, hf_config, max_samples: int) -> Iterator[dict]:
    """Stream rows from HF, yield up to max_samples (best-effort)."""
    try:
        ds = load_dataset(hf_path, name=hf_config, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            yield row
    except Exception as e:
        print(f"  ! stream failed for {hf_path}: {e}")
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None, help="override total docs")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    all_rows = []  # (source, text)
    report = {}

    for name, cfg in SOURCES.items():
        print(f"\n=== {name} (max {cfg['max_samples']:,}) ===")
        t0 = time.time()
        got = 0
        kept = 0
        for row in iter_rows(cfg["hf_path"], cfg["hf_config"], cfg["max_samples"]):
            got += 1
            text = cfg["text"](row).strip()
            if len(text) < 200:      # drop too-short docs
                continue
            if len(text) > 200_000:  # drop absurd docs
                continue
            all_rows.append((name, text))
            kept += 1
            if args.max_docs and len(all_rows) >= args.max_docs:
                break
        dt = time.time() - t0
        report[name] = {"got": got, "kept": kept, "secs": round(dt, 1)}
        print(f"  got={got:,} kept={kept:,} in {dt:.1f}s")

    if args.dry_run:
        print("\nDRY RUN — no file written. Report:", json.dumps(report, indent=2))
        return

    # shuffle deterministically
    rng.shuffle(all_rows)

    # write combined JSONL
    out_file = OUT_DIR / "corpus.jsonl"
    distilled_added = 0
    distilled_src = Path("distilled_corpus/final/final_corpus.jsonl")
    with open(out_file, "w") as f:
        for src, text in all_rows:
            f.write(json.dumps({"text": text, "source": src}, ensure_ascii=False) + "\n")
        # append distilled corpus (quality anchor)
        if distilled_src.exists():
            for line in open(distilled_src):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                f.write(json.dumps({"text": r["text"], "source": "distilled_corpus"}, ensure_ascii=False) + "\n")
                distilled_added += 1

    total_chars = sum(len(t) for _, t in all_rows)
    print(f"\nWROTE {len(all_rows):,} docs (+ {distilled_added:,} distilled) to {out_file}")
    print(f"total chars: {total_chars:,} (~{total_chars//4:,} tokens est, + distilled)")
    with open(OUT_DIR / "mixture_report.json", "w") as f:
        json.dump({"total_docs": len(all_rows) + distilled_added, "total_chars": total_chars,
                   "distilled_added": distilled_added, "report": report}, f, indent=2)
    print("report: datasets/train_corpus/mixture_report.json")


if __name__ == "__main__":
    main()
