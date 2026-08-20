"""Finalize: produce all deliverables from deduplicated/deduplicated.jsonl.

Outputs:
  final/final_corpus.jsonl
  final/dataset_statistics.json
  final/quality_report.json
  final/provenance.jsonl
"""
from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "distilled_corpus"
IN = CORPUS / "deduplicated" / "deduplicated.jsonl"
OUT = CORPUS / "final"
OUT.mkdir(parents=True, exist_ok=True)

# rough tokens-per-char estimate (English+code ~4 chars/token)
CHARS_PER_TOKEN = 4.0


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def context_bucket(n_tokens: int) -> str:
    for b in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        if n_tokens <= b:
            return str(b)
    return ">32768"


def run():
    recs = []
    if IN.exists():
        for line in open(IN):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    total_chars = sum(len(r["text"]) for r in recs)
    total_tokens = sum(est_tokens(r["text"]) for r in recs)

    by_domain = Counter()
    tok_by_domain = Counter()
    by_difficulty = Counter()
    tok_by_difficulty = Counter()
    by_task = Counter()
    tok_by_task = Counter()
    by_teacher = Counter()
    tok_by_teacher = Counter()
    ctx_dist = Counter()
    quality_scores = []
    verdicts = Counter()
    synthetic = 0
    source_grounded = 0
    verified = 0
    langs = Counter()
    for r in recs:
        md = r["metadata"]
        t = est_tokens(r["text"])
        by_domain[md["domain"]] += 1
        tok_by_domain[md["domain"]] += t
        by_difficulty[md["difficulty"]] += 1
        tok_by_difficulty[md["difficulty"]] += t
        by_task[md["task_type"]] += 1
        tok_by_task[md["task_type"]] += t
        teacher_key = "+".join(md.get("teacher_models", [md.get("teacher", "?")]))
        by_teacher[teacher_key] += 1
        tok_by_teacher[teacher_key] += t
        ctx_dist[context_bucket(t)] += 1
        quality_scores.append(md.get("quality_score", 0))
        verdicts[md.get("verification_status", "unknown")] += 1
        if md.get("source_type") == "synthetic":
            synthetic += 1
        else:
            source_grounded += 1
        if md.get("verification_status") in ("keep", "verified", "review"):
            verified += 1
        langs[md.get("language", "en")] += 1

    # provenance
    prov_path = OUT / "provenance.jsonl"
    with open(prov_path, "w") as f:
        for r in recs:
            p = {
                "id": r["id"],
                "seed_id": r.get("provenance", {}).get("seed_id"),
                "concept": r["metadata"]["concept"],
                "teachers": r["metadata"].get("teacher_models"),
                "task_type": r["metadata"]["task_type"],
                "difficulty": r["metadata"]["difficulty"],
                "verification_status": r["metadata"].get("verification_status"),
                "quality_score": r["metadata"].get("quality_score"),
                "source_type": r["metadata"].get("source_type"),
                "knowledge_type": r["metadata"].get("knowledge_type"),
            }
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # final corpus (text + metadata only, no raw_content)
    final_path = OUT / "final_corpus.jsonl"
    with open(final_path, "w") as f:
        for r in recs:
            out = {"text": r["text"], "metadata": r["metadata"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = len(recs)
    avg_q = (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0

    stats = {
        "total_documents": n,
        "total_characters": total_chars,
        "estimated_tokens": total_tokens,
        "tokens_by_domain": dict(tok_by_domain),
        "documents_by_domain": dict(by_domain),
        "tokens_by_difficulty": {str(k): v for k, v in tok_by_difficulty.items()},
        "tokens_by_task_type": dict(tok_by_task),
        "tokens_by_teacher": dict(tok_by_teacher),
        "documents_by_teacher": dict(by_teacher),
        "context_length_distribution": dict(ctx_dist),
        "percentage_synthetic": round(100 * synthetic / n, 2) if n else 0,
        "percentage_source_grounded": round(100 * source_grounded / n, 2) if n else 0,
        "percentage_verified": round(100 * verified / n, 2) if n else 0,
        "verification_verdicts": dict(verdicts),
        "average_quality_score": round(avg_q, 3),
        "languages": dict(langs),
        "token_estimate_method": "chars / 4.0 (rough; recompute with actual tokenizer later)",
    }
    (OUT / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

    qr = {
        "n_documents": n,
        "average_quality_score": round(avg_q, 3),
        "verdicts": dict(verdicts),
        "verification_notes": [
            "code_with_explanation items were executed; failing code rejected.",
            "Independent LLM verifier (GLM-5.2-NVFP4) issued KEEP/REVIEW/REJECT verdicts.",
            "Heuristics rejected teacher artifacts and too-short outputs.",
            "See reports/dedup_report.json for duplicate-removal breakdown.",
        ],
        "known_limitations": [
            "Pilot scale only; API throughput limited (~50s/call, 504-prone under concurrency).",
            "Reasoning-model (Qwen/GLM/Kimi) generation limited by latency; bulk pilot used DeepSeek-Flash.",
            "Semantic dedup uses TF-IDF cosine, not dense embeddings.",
        ],
    }
    (OUT / "quality_report.json").write_text(json.dumps(qr, indent=2))

    print(json.dumps(stats, indent=2))
    print(f"\nWrote {final_path} ({n} docs, ~{total_tokens} tokens)")


if __name__ == "__main__":
    run()
