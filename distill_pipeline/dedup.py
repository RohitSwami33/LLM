"""Deduplication: exact, normalized, near-duplicate (n-gram Jaccard), semantic (TF-IDF cosine).

Reads verified/verified.jsonl, removes duplicates keeping the highest quality_score,
writes deduplicated/deduplicated.jsonl + reports/dedup_report.json.
"""
from __future__ import annotations
import json
import re
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "distilled_corpus"
IN = CORPUS / "verified" / "verified.jsonl"
OUT = CORPUS / "deduplicated" / "deduplicated.jsonl"
REPORT = CORPUS / "reports" / "dedup_report.json"


def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def shingles(t: str, k: int = 5) -> set:
    tokens = t.split()
    if len(tokens) < k:
        return set(tokens)
    return set(" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run(near_thresh: float = 0.85, sem_thresh: float = 0.92):
    recs = []
    for line in open(IN):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    print(f"loaded {len(recs)} verified records")

    seen_exact = set()
    seen_norm = set()
    kept = []
    dropped = defaultdict(int)

    # pass 1: exact + normalized dedup
    stage1 = []
    for r in recs:
        h = hashlib.sha256(r["text"].encode()).hexdigest()
        nh = hashlib.sha256(normalize_text(r["text"]).encode()).hexdigest()
        if h in seen_exact:
            dropped["exact"] += 1
            continue
        if nh in seen_norm:
            dropped["normalized"] += 1
            continue
        seen_exact.add(h)
        seen_norm.add(nh)
        stage1.append(r)

    # pass 2: near-duplicate via shingles (within same domain to keep it fast)
    stage2 = []
    by_domain = defaultdict(list)
    for r in stage1:
        by_domain[r["metadata"]["domain"]].append(r)
    for dom, items in by_domain.items():
        sh = [shingles(normalize_text(r["text"])) for r in items]
        keep_idx = []
        for i in range(len(items)):
            dup = False
            for j in keep_idx:
                if jaccard(sh[i], sh[j]) >= near_thresh:
                    dup = True
                    break
            if dup:
                dropped["near_duplicate"] += 1
            else:
                keep_idx.append(i)
                stage2.append(items[i])

    # pass 3: semantic dedup via TF-IDF cosine (global)
    if len(stage2) > 1:
        texts = [normalize_text(r["text"]) for r in stage2]
        try:
            tf = TfidfVectorizer(max_features=20000, dtype=np.float32)
            X = tf.fit_transform(texts)
            sim = cosine_similarity(X)
            keep = []
            for i in range(len(stage2)):
                dup = False
                for j in keep:
                    if i != j and sim[i, j] >= sem_thresh:
                        # keep the higher quality one
                        if stage2[i]["metadata"]["quality_score"] <= stage2[j]["metadata"]["quality_score"]:
                            dup = True
                            break
                if not dup:
                    keep.append(i)
                else:
                    dropped["semantic"] += 1
            final = [stage2[i] for i in keep]
        except Exception as e:
            print("semantic dedup skipped:", e)
            final = stage2
    else:
        final = stage2

    # sort by quality desc for stable output
    final.sort(key=lambda r: r["metadata"]["quality_score"], reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "input": len(recs),
        "after_exact_normalized": len(stage1),
        "after_near_duplicate": len(stage2),
        "after_semantic": len(final),
        "dropped": dict(dropped),
        "total_dropped": sum(dropped.values()),
        "near_thresh": near_thresh,
        "sem_thresh": sem_thresh,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
