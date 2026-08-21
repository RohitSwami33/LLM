#!/usr/bin/env python3
"""Append the distilled corpus (quality anchor) to the training corpus.jsonl."""
import json
from pathlib import Path

SRC = Path("distilled_corpus/final/final_corpus.jsonl")
OUT = Path("datasets/train_corpus/corpus.jsonl")
TMP = Path("datasets/train_corpus/corpus_with_distilled.jsonl")

added = 0
with open(TMP, "w") as fout:
    # copy existing corpus
    for line in open(OUT):
        fout.write(line)
    # append distilled
    for line in open(SRC):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fout.write(json.dumps({"text": r["text"], "source": "distilled_corpus"}, ensure_ascii=False) + "\n")
        added += 1

print(f"append {added} distilled docs")
import os
os.replace(TMP, OUT)
# report size
tot = sum(1 for _ in open(OUT))
print(f"total docs now: {tot:,}")