"""Assemble the complete distilled corpus into a clean, publishable dataset.

Combines:
  - verified/verified.jsonl  (all KEEP + REVIEW docs from verify)
  - rejected/rejected.jsonl (skipped — not included)
into final/ with:
  - dataset.jsonl            (the plain text docs, one per line: {"text": ...})
  - dataset_metadata.jsonl   (full records with metadata + provenance)
  - dataset_stats.json       (counts, token estimates, source breakdown)
"""
import json, os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "distilled_corpus"
VERIFIED = CORPUS / "verified" / "verified.jsonl"
FINAL = CORPUS / "final"
FINAL.mkdir(parents=True, exist_ok=True)

CHARS_PER_TOKEN = 4.0

def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)

def main():
    recs = []
    for line in open(VERIFIED):
        line = line.strip()
        if line:
            recs.append(json.loads(line))

    print(f"verified records: {len(recs)}")

    # dedupe by id just in case
    seen = set()
    uniq = []
    for r in recs:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    recs = uniq
    print(f"after dedupe: {len(recs)}")

    # Sort by id for stable output
    recs.sort(key=lambda r: r["id"])

    # Plain-text dataset (training-ready)
    with open(FINAL / "dataset.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")

    # Full metadata version
    with open(FINAL / "dataset_metadata.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    total_chars = sum(len(r["text"]) for r in recs)
    total_tokens = sum(est_tokens(r["text"]) for r in recs)
    by_source = Counter(r["metadata"].get("source", "?") for r in recs)
    by_teacher = Counter(r["metadata"].get("teacher", "?") for r in recs)
    by_task = Counter(r["metadata"].get("task_type", "?") for r in recs)
    by_diff = Counter(str(r["metadata"].get("difficulty", "?")) for r in recs)

    stats = {
        "records": len(recs),
        "total_chars": total_chars,
        "estimated_tokens": total_tokens,
        "by_source": dict(by_source),
        "by_teacher": dict(by_teacher),
        "by_task_type": dict(by_task),
        "by_difficulty": dict(by_diff),
        "chars_per_token_estimate": CHARS_PER_TOKEN,
    }
    with open(FINAL / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
